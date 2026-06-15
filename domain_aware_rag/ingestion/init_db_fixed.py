# Main script for database initialization - compatible without pgvector
import psycopg2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_HOST = "localhost"
DB_NAME = "rag_system"
DB_USER = "postgres"
DB_PASS = "root"
DB_PORT = 5432


def init_database():
    conn = None
    try:
        # Connect to rag_system database
        logger.info("Connecting to rag_system database...")
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT
        )

        with conn.cursor() as cur:

            logger.info("Ensuring extensions exist...")
            has_vector = False
            
            # Try to create vector extension with savepoint
            try:
                cur.execute("SAVEPOINT try_vector")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                has_vector = True
                logger.info("✓ Vector extension enabled")
            except Exception as e:
                cur.execute("ROLLBACK TO SAVEPOINT try_vector")
                logger.warning(f"⚠ Vector extension not available: {e}")
                logger.info("  Will proceed without vector support (embeddings will be stored as text)")
            
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

            logger.info("Creating schema 'rag'...")
            cur.execute("CREATE SCHEMA IF NOT EXISTS rag;")

            logger.info("Creating table 'rag.documents'...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS rag.documents (
                    id UUID PRIMARY KEY,
                    external_id TEXT UNIQUE,
                    title TEXT,
                    version TEXT,
                    category TEXT,
                    subcategory TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Check if vector extension is available
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            has_vector = cur.fetchone() is not None

            if has_vector:
                logger.info("Creating table 'rag.nodes' with vector support...")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS rag.nodes (
                        id UUID PRIMARY KEY,
                        document_id UUID REFERENCES rag.documents(id) ON DELETE CASCADE,
                        parent_id UUID REFERENCES rag.nodes(id) ON DELETE CASCADE,
                        level INTEGER,
                        title TEXT,
                        content TEXT,
                        text TEXT,
                        embedding vector(384),
                        metadata JSONB,
                        keywords TEXT[],
                        is_leaf BOOLEAN,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            else:
                logger.info("Creating table 'rag.nodes' without vector support...")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS rag.nodes (
                        id UUID PRIMARY KEY,
                        document_id UUID REFERENCES rag.documents(id) ON DELETE CASCADE,
                        parent_id UUID REFERENCES rag.nodes(id) ON DELETE CASCADE,
                        level INTEGER,
                        title TEXT,
                        content TEXT,
                        text TEXT,
                        embedding TEXT,  -- Store as JSON string when no vector support
                        metadata JSONB,
                        keywords TEXT[],
                        is_leaf BOOLEAN,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)

            logger.info("Creating table 'rag.entities'...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS rag.entities (
                    id SERIAL PRIMARY KEY,
                    document_id UUID REFERENCES rag.documents(id) ON DELETE CASCADE,
                    node_id UUID REFERENCES rag.nodes(id) ON DELETE CASCADE,
                    canonical_name TEXT,
                    entity_type TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)

            if has_vector:
                logger.info("Creating table 'rag.doc_tables' with vector support...")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS rag.doc_tables (
                        id SERIAL PRIMARY KEY,
                        document_id UUID REFERENCES rag.documents(id) ON DELETE CASCADE,
                        node_id UUID REFERENCES rag.nodes(id) ON DELETE CASCADE,
                        data JSONB,
                        tabular_text TEXT,
                        nrows INTEGER,
                        ncols INTEGER,
                        metadata JSONB,
                        embedding vector(384),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            else:
                logger.info("Creating table 'rag.doc_tables' without vector support...")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS rag.doc_tables (
                        id SERIAL PRIMARY KEY,
                        document_id UUID REFERENCES rag.documents(id) ON DELETE CASCADE,
                        node_id UUID REFERENCES rag.nodes(id) ON DELETE CASCADE,
                        data JSONB,
                        tabular_text TEXT,
                        nrows INTEGER,
                        ncols INTEGER,
                        metadata JSONB,
                        embedding TEXT,  -- Store as JSON string when no vector support
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)

            logger.info("Creating indexes from hybrid_retrieval.py...")

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_nodes_keywords_gin
                ON rag.nodes USING GIN(keywords);
            """)

            cur.execute("""
                CREATE OR REPLACE FUNCTION rag.immutable_array_to_string(text[], text)
                RETURNS text AS $$
                    SELECT array_to_string($1, $2);
                $$ LANGUAGE sql IMMUTABLE;
            """)

            cur.execute("""
                ALTER TABLE rag.nodes
                ADD COLUMN IF NOT EXISTS keywords_tsv tsvector
                GENERATED ALWAYS AS (
                    to_tsvector('simple', rag.immutable_array_to_string(keywords, ' '))
                ) STORED;
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_nodes_keywords_tsv
                ON rag.nodes USING GIN (keywords_tsv);
            """)

            cur.execute("""
                ALTER TABLE rag.entities
                ADD COLUMN IF NOT EXISTS canonical_name_tsv tsvector
                GENERATED ALWAYS AS (
                    to_tsvector('simple', canonical_name)
                ) STORED;
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_canonical_name_tsv
                ON rag.entities USING GIN (canonical_name_tsv);
            """)

            # Only create vector index if vector extension is available
            if has_vector:
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_nodes_embedding_hnsw
                    ON rag.nodes USING hnsw (embedding vector_cosine_ops);
                """)
                logger.info("✓ Vector index created")
            else:
                logger.info("⚠ Skipping vector index (no vector extension)")

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_canonical
                ON rag.entities(canonical_name);
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_nodes_parent_id
                ON rag.nodes(parent_id);
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_nodes_doc_id
                ON rag.nodes(document_id);
            """)

            cur.execute("ALTER TABLE rag.documents ADD COLUMN IF NOT EXISTS category TEXT;")
            cur.execute("ALTER TABLE rag.documents ADD COLUMN IF NOT EXISTS subcategory TEXT;")
            cur.execute("ALTER TABLE rag.nodes ADD COLUMN IF NOT EXISTS category TEXT;")
            cur.execute("ALTER TABLE rag.nodes ADD COLUMN IF NOT EXISTS subcategory TEXT;")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_category ON rag.documents(category);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_subcategory ON rag.documents(subcategory);")

            logger.info("Altering table 'rag.doc_tables' to add columns if missing...")

            if has_vector:
                cur.execute("""
                    ALTER TABLE rag.doc_tables
                    ADD COLUMN IF NOT EXISTS embedding vector(384),
                    ADD COLUMN IF NOT EXISTS tabular_text TEXT;
                """)
            else:
                cur.execute("""
                    ALTER TABLE rag.doc_tables
                    ADD COLUMN IF NOT EXISTS embedding TEXT,
                    ADD COLUMN IF NOT EXISTS tabular_text TEXT;
                """)

            conn.commit()
            logger.info("✅ Database initialization successful!")
            logger.info(f"   Vector support: {'✓ Enabled' if has_vector else '⚠ Disabled (stored as text)'}")

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        if conn:
            conn.rollback()

    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    init_database()
