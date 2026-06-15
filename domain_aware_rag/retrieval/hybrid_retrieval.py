"""
Hybrid Retrieval System for RAG
Implements: Semantic + Keyword + Entity + Parent Document Retrieval

Architecture:
1. S1: Retrieval Prep - Query expansion, anchor keyword extraction, and metadata-based filtering (Keywords/Entities)
2. S2: Retrieval - pgvector Semantic Search combined with Keyword Search
3. S3: Ranking & Context Assembly - Re-ranking, parent document retrieval, and hierarchical context assembly

Flow:
User Query -> Prep (Expansions, Metadata Filtering) -> Search (Hybrid Vector+Keyword) -> Re-rank -> Parent Climb -> Context Assembly
"""

import psycopg2
import logging
import re
import os
import json
from typing import List, Dict, Any, Optional, Tuple, Set, cast
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
import numpy as np

# Configure logging
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "log.txt")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Common English stop words
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
    'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

def debug_print(function_name: str, title: str, data: Any):
    """Enhanced helper for formatted debug printing and logging"""
    separator = "=" * 30
    header = f"\n{separator} STEP start: [{function_name}] {title} {separator}"
    footer = f"{separator} STEP end: [{function_name}] {title} {separator}\n"

    msg_parts = [header]

    if data is None:
        msg_parts.append("Data: None")
    elif isinstance(data, (list, set)):
        items: List[Any] = list(data)
        msg_parts.append(f"Type: {type(data).__name__}, Count: {len(items)}")
        if len(items) > 0:
            if isinstance(items[0], tuple):
                msg_parts.append("Data (Rows):")
                for i, row in enumerate(items): # Show up to 20 rows
                    if i >= 20:
                        break
                    msg_parts.append(f"  [{i}] {row}")
                if len(items) > 20:
                    msg_parts.append(f"  ... and {len(items) - 20} more rows")
            else:
                msg_parts.append(f"Data: {items}")
    elif isinstance(data, dict):
        msg_parts.append(f"Type: dict, Keys Count: {len(data)}")
        msg_parts.append(f"Data: {data}")
    else:
        str_data = str(data)
        if len(str_data) > 1000:
            msg_parts.append(f"Data (Truncated): {str_data[:1000]}... [Total Length: {len(str_data)}]")
        else:
            msg_parts.append(f"Data: {str_data}")

    msg_parts.append(footer)
    full_msg = "\n".join(msg_parts)
    logger.info(full_msg)

# Database config
DB_HOST = "localhost"
DB_NAME = "rag_system"
DB_USER = "postgres"
DB_PASS = "root"
DB_PORT = 5432

# Load embedding model (same as ingestion)
# model = SentenceTransformer('all-MiniLM-L6-v2')

@dataclass
class SearchResult:
    """Represents a single search result with all signals"""
    node_id: Any  # Can be str or uuid.UUID
    title: str
    content: str
    text: str
    level: int
    parent_id: Optional[Any]
    vector_score: float
    bm25_score: float
    final_score: float
    metadata: Dict[str, Any]
    hierarchy_path: List[str]

@dataclass
class RetrievalConfig:
    """Configuration for retrieval behavior"""
    top_k_semantic: int = 80
    similarity_threshold: float = 0.0

    vector_weight: float = 0.7
    bm25_weight: float = 0.3

    top_k_final: int = 3

    include_parents: bool = True
    max_parent_levels: int = 4


@dataclass
class RetrievalResult:
    """Structured retrieval output - consumed by app.py and answer_generator."""
    context_text: str                                       # Clean text context for LLM
    top_results: List[Dict[str, Any]] = field(default_factory=list)   # Top-k ranked nodes with scores
    enriched_nodes: List[Dict[str, Any]] = field(default_factory=list)  # All nodes incl. parents
    subgraph_edges: List[Dict[str, Any]] = field(default_factory=list)  # Parent->child edges
    source_documents: List[Dict[str, Any]] = field(default_factory=list)  # Unique doc references

class HybridRetriever:
    """
    Main retrieval class implementing hybrid search strategy
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self._model: Optional[SentenceTransformer] = None # Lazy loaded
        self.conn = self._get_db_connection()
        self._ensure_indexes()

    def _get_db_connection(self):
        """Establish database connection"""
        return psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT
        )

    def _ensure_indexes(self):
        """Ensure required indexes exist for optimal performance"""
        # Create indexes individually to handle failures gracefully
        
        # GIN index for keyword array search
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_nodes_keywords_gin
                    ON rag.nodes USING GIN(keywords);
                """)
                self.conn.commit()
                logger.info("Created idx_nodes_keywords_gin")
        except Exception as e:
            logger.warning(f"Could not create idx_nodes_keywords_gin: {e}")
            self.conn.rollback()

        # --- FTS Migration ---
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    CREATE OR REPLACE FUNCTION rag.immutable_array_to_string(text[], text)
                    RETURNS text AS $$
                        SELECT array_to_string($1, $2);
                    $$ LANGUAGE sql IMMUTABLE;
                """)
                self.conn.commit()
        except Exception as e:
            logger.warning(f"Could not create immutable_array_to_string function: {e}")
            self.conn.rollback()

        # Keywords FTS
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    ALTER TABLE rag.nodes
                    ADD COLUMN IF NOT EXISTS keywords_tsv tsvector
                    GENERATED ALWAYS AS (to_tsvector('simple', rag.immutable_array_to_string(keywords, ' '))) STORED;
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_keywords_tsv ON rag.nodes USING GIN (keywords_tsv);")
                self.conn.commit()
                logger.info("Created keywords_tsv column and index")
        except Exception as e:
            logger.warning(f"Could not create keywords_tsv: {e}")
            self.conn.rollback()

        # Entities FTS
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    ALTER TABLE rag.entities
                    ADD COLUMN IF NOT EXISTS canonical_name_tsv tsvector
                    GENERATED ALWAYS AS (to_tsvector('simple', canonical_name)) STORED;
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_entities_canonical_name_tsv ON rag.entities USING GIN (canonical_name_tsv);")
                self.conn.commit()
                logger.info("Created canonical_name_tsv column and index")
        except Exception as e:
            logger.warning(f"Could not create canonical_name_tsv: {e}")
            self.conn.rollback()

        # HNSW index (requires pgvector) - wrapped in try-except
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_nodes_embedding_hnsw
                    ON rag.nodes USING hnsw (embedding vector_cosine_ops);
                """)
                self.conn.commit()
                logger.info("Created HNSW index")
        except Exception as e:
            logger.warning(f"Could not create HNSW index (pgvector may not be available): {e}")
            self.conn.rollback()

        # Content FTS
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    ALTER TABLE rag.nodes
                    ADD COLUMN IF NOT EXISTS content_tsv tsvector
                    GENERATED ALWAYS AS (
                        setweight(to_tsvector('simple', COALESCE(title, '')), 'A') ||
                        setweight(to_tsvector('simple', COALESCE(text, '')), 'B') ||
                        setweight(to_tsvector('simple', rag.immutable_array_to_string(keywords, ' ')), 'C')
                    ) STORED;
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_content_tsv ON rag.nodes USING GIN (content_tsv);")
                self.conn.commit()
                logger.info("Created content_tsv column and index")
        except Exception as e:
            logger.warning(f"Could not create content_tsv: {e}")
            self.conn.rollback()

        # Index on entities for fast lookup
        try:
            with self.conn.cursor() as cur:
                cur.execute("CREATE INDEX IF NOT EXISTS idx_entities_canonical ON rag.entities(canonical_name);")
                self.conn.commit()
                logger.info("Created idx_entities_canonical")
        except Exception as e:
            logger.warning(f"Could not create idx_entities_canonical: {e}")
            self.conn.rollback()

    def _check_vector_extension(self) -> bool:
        """Check if pgvector extension is available"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                return cur.fetchone() is not None
        except Exception:
            return False

    def _semantic_search_python(self, query_vec: List[float], document_id: Optional[str], 
                                candidate_ids: Optional[Set[str]], document_cat: Optional[str] = None, 
                                document_subcat: Optional[str] = None) -> List[tuple]:
        """Compute semantic similarity in Python when pgvector is not available"""
        results = []
        query_vec_np = np.array(query_vec)
        
        with self.conn.cursor() as cur:
            sql = """
                SELECT n.id, n.title, n.content, n.text, n.level, n.parent_id, n.metadata, n.keywords, n.embedding
                FROM rag.nodes as n
                INNER JOIN rag.documents as d ON n.document_id = d.id
                WHERE n.embedding IS NOT NULL
            """
            params: List[Any] = []

            if document_id:
                sql += " AND n.document_id = %s::uuid"
                params.append(document_id)

            if document_cat and document_subcat:
                sql += " AND (LOWER(d.category) = LOWER(%s) OR LOWER(d.subcategory) = LOWER(%s))"
                params.extend([document_cat, document_subcat])
            elif document_cat:
                sql += " AND LOWER(d.category) = LOWER(%s)"
                params.append(document_cat)
            elif document_subcat:
                sql += " AND LOWER(d.subcategory) = LOWER(%s)"
                params.append(document_subcat)

            if candidate_ids is not None:
                sql += " AND n.id = ANY(%s::uuid[])"
                params.append(list(candidate_ids))

            cur.execute(sql, params)
            rows = cur.fetchall()
            
            for row in rows:
                embedding_text = row[8]
                if embedding_text:
                    try:
                        import json
                        node_vec = json.loads(embedding_text)
                        node_vec_np = np.array(node_vec)
                        # Cosine similarity
                        dot_product = np.dot(query_vec_np, node_vec_np)
                        query_norm = np.linalg.norm(query_vec_np)
                        node_norm = np.linalg.norm(node_vec_np)
                        if query_norm > 0 and node_norm > 0:
                            similarity = dot_product / (query_norm * node_norm)
                            results.append((row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], float(similarity)))
                    except Exception:
                        continue
        
        # Sort by similarity and limit to top_k
        results.sort(key=lambda x: x[8], reverse=True)
        return results[:self.config.top_k_semantic]

    def _detect_query_type(self, query: str, ontology_config=None) -> str:
        query_lower = query.lower()
        
        if ontology_config and ontology_config.retrieval_hints:
            service_indicators = ontology_config.retrieval_hints.service_indicators
            exclusion_indicators = ontology_config.retrieval_hints.exclusion_indicators
            process_indicators = ontology_config.retrieval_hints.process_indicators
        else:
            service_indicators = ['copay', 'cost', 'pay', 'covered', 'coverage', 'amount',
                                 'ambulance', 'mental health', 'dme', 'dental', 'hospice',
                                 'emergency', 'urgent', 'inpatient', 'outpatient', 'therapy',
                                 'how much', 'what do i pay', 'my costs']
            exclusion_indicators = ['not covered', 'excluded', 'limitations', 'exclusions',
                                   'never covered', 'does not cover']
            process_indicators = ['how to', 'how do i', 'how can i', 'contact', 'call',
                                 'reach', 'phone', 'address', 'where to', 'who to']

        if service_indicators and any(indicator in query_lower for indicator in service_indicators):
            return 'specific_service'

        if exclusion_indicators and any(indicator in query_lower for indicator in exclusion_indicators):
            return 'exclusion'

        if process_indicators and any(indicator in query_lower for indicator in process_indicators):
            return 'process'

        return 'general'

    def retrieve(
        self,
        query: str,
        query_expansions: Optional[List[str]] = None,
        anchor_entities: Optional[List[str]] = None,
        document_id: Optional[str] = None,
        model: Optional[SentenceTransformer] = None,
        document_subcat: Optional[str] = None,
        document_cat: Optional[str] = None,
        ontology_config=None
    ) -> 'RetrievalResult':
        logger.info(f"Retrieving for query: {query}")

        # Ensure model is initialized
        if model is None:
            if not hasattr(self, '_model'):
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
            model = self._model

        if ontology_config is None:
            try:
                from domain_aware_rag.ontology.registry import OntologyRegistry
                ontology_config = OntologyRegistry().get_active_ontology()
            except Exception:
                ontology_config = None

        # --- S1: Prep ---
        expansions, entities = self._retrieval_prep(query, query_expansions, anchor_entities)
        candidate_ids = self._apply_metadata_filters(entities, document_id)

        query_type = self._detect_query_type(query, ontology_config)
        active_config = RetrievalConfig(
            top_k_semantic=self.config.top_k_semantic,
            similarity_threshold=self.config.similarity_threshold,
            top_k_final=self.config.top_k_final,
            include_parents=self.config.include_parents,
            max_parent_levels=self.config.max_parent_levels
        )

        if query_type == 'specific_service':
            active_config.vector_weight = 0.6
            active_config.bm25_weight = 0.4
        elif query_type == 'exclusion':
            active_config.vector_weight = 0.5
            active_config.bm25_weight = 0.5
        else:
            active_config.vector_weight = self.config.vector_weight
            active_config.bm25_weight = self.config.bm25_weight

        # --- S2: Retrieval ---
        semantic_results = self._semantic_search(query, expansions, document_id, candidate_ids, model, document_cat, document_subcat)
        bm25_results = self._bm25_search(query, document_id, candidate_ids, document_cat, document_subcat)

        # --- S3: Ranking ---
        ranked_results = self._rerank(semantic_results, bm25_results, query=query, query_expansions=expansions, config=active_config, ontology_config=ontology_config)
        top_k_results = ranked_results[:self.config.top_k_final]


        if self.config.include_parents and top_k_results:
            top_node_ids = [r.node_id for r in top_k_results]
            enriched_nodes = self._fetch_parents(top_node_ids)
        else:
            enriched_nodes = [
                {
                    'node_id': r.node_id,
                    'parent_id': r.parent_id,
                    'title': r.title,
                    'content': r.content,
                    'text': r.text,
                    'level': r.level,
                    'metadata': r.metadata,
                    'hierarchy_path': r.hierarchy_path
                } for r in top_k_results
            ]

        # Build structured outputs
        context_text = self._build_context(enriched_nodes, top_k_results)
        subgraph_edges = self._build_subgraph_edges(enriched_nodes)
        source_documents = self._extract_source_documents(enriched_nodes)

        top_results_dicts = [
            {
                'rank': i,
                'node_id': str(r.node_id),
                'title': r.title,
                'level': r.level,
                'parent_id': str(r.parent_id) if r.parent_id else None,
                'vector_score': r.vector_score,
                'bm25_score': r.bm25_score,
                'final_score': r.final_score,
                'hierarchy_path': r.hierarchy_path,
                'metadata': r.metadata,
                'content': r.content,
                'text': r.text,
            } for i, r in enumerate(top_k_results, 1)
        ]

        enriched_nodes_out = [
            {
                'node_id': str(n['node_id']),
                'parent_id': str(n['parent_id']) if n.get('parent_id') else None,
                'title': n['title'],
                'content': n['content'],
                'text': n.get('text', ''),
                'level': n['level'],
                'metadata': n.get('metadata', {}),
                'hierarchy_path': n.get('hierarchy_path', []),
            } for n in enriched_nodes
        ]

        logger.info(f"Retrieval complete: {len(top_k_results)} top results, "
                     f"{len(enriched_nodes)} enriched nodes, "
                     f"{len(subgraph_edges)} subgraph edges, "
                     f"{len(source_documents)} source docs")
        print("context:",context_text)
        return RetrievalResult(
            context_text=context_text,
            top_results=top_results_dicts,
            enriched_nodes=enriched_nodes_out,
            subgraph_edges=subgraph_edges,
            source_documents=source_documents,
        )

    def _retrieval_prep(self, query: str, query_expansions: Optional[List[str]], anchor_entities: Optional[List[str]]) -> Tuple[List[str], List[str]]:
        return query_expansions or [query], anchor_entities or []

    def _to_tsquery_string(self, terms: List[str]) -> str:
        if not terms: return ""
        all_tokens = []
        for term in terms:
            tokens = re.split(r'[^a-z0-9_]', term.lower())
            for t in tokens:
                if t: all_tokens.append(f"{t}:*")
        unique_tokens = list(dict.fromkeys(all_tokens))
        return " | ".join(unique_tokens)

    def _apply_metadata_filters(self, entities: List[str], document_id: Optional[str]) -> Optional[Set[str]]:
        if not entities: return None
        node_ids = set()
        lower_entities = [e.lower() for e in entities]
        with self.conn.cursor() as cur:
            conditions: List[str] = []
            params: List[Any] = []
            if document_id:
                conditions.append("document_id = %s::uuid")
                params.append(document_id)

            or_expr = []
            or_expr.append("EXISTS (SELECT 1 FROM jsonb_array_elements_text(metadata->'canonical_names') c WHERE lower(c) = ANY(%s))")
            params.append(lower_entities)
            or_expr.append("EXISTS (SELECT 1 FROM jsonb_array_elements_text(metadata->'entity_types') t WHERE lower(t) = ANY(%s))")
            params.append(lower_entities)
            or_expr.append("EXISTS (SELECT 1 FROM jsonb_array_elements_text(metadata->'keywords') t WHERE lower(t) = ANY(%s))")
            params.append(lower_entities)
            or_expr.append("lower(text) LIKE ANY(%s)")
            params.append([f"%{e}%" for e in lower_entities])

            conditions.append(f"({' OR '.join(or_expr)})")
            sql = f"SELECT id FROM rag.nodes WHERE {' AND '.join(conditions)}"
            cur.execute(sql, params)
            for row in cur.fetchall():
                node_ids.add(row[0])
        return node_ids

    def _semantic_search(self, query: str, expansions: Optional[List[str]], document_id: Optional[str], candidate_ids: Optional[Set[str]], model:Optional[SentenceTransformer] = None,document_cat: Optional[str] = None, document_subcat: Optional[str] = None) -> List[Dict[str, Any]]:
        queries = [query] + (expansions or [])
        if candidate_ids is not None and not candidate_ids: return []

        all_candidates: Dict[Any, Any] = {}
        
        # Check if we have vector extension available
        has_vector = self._check_vector_extension()
        
        for q in queries:
            if model is None: continue 
            query_vec = model.encode(q).tolist()
            
            # If no vector extension, compute similarity in Python
            if not has_vector:
                candidates = self._semantic_search_python(query_vec, document_id, candidate_ids, document_cat, document_subcat)
                for row in candidates:
                    node_id = row[0]
                    if node_id not in all_candidates or row[8] > all_candidates[node_id][8]:
                        all_candidates[node_id] = row
            else:
                with self.conn.cursor() as cur:
                    sql = """
                        SELECT n.id, n.title, n.content, n.text, n.level, n.parent_id, n.metadata, n.keywords,
                            1 - (embedding <=> %s::vector) AS score
                        FROM rag.nodes as n
                        INNER JOIN rag.documents as d ON n.document_id = d.id
                        WHERE 1=1
                    """
                    params: List[Any] = [query_vec]

                    if document_id:
                        sql += " AND n.document_id = %s::uuid"
                        params.append(document_id)

                    if document_cat and document_subcat:
                        sql += " AND (LOWER(d.category) = LOWER(%s) OR LOWER(d.subcategory) = LOWER(%s))"
                        params.extend([document_cat, document_subcat])
                    elif document_cat:
                        sql += " AND LOWER(d.category) = LOWER(%s)"
                        params.append(document_cat)
                    elif document_subcat:
                        sql += " AND LOWER(d.subcategory) = LOWER(%s)"
                        params.append(document_subcat)

                    if candidate_ids is not None:
                        sql += " AND n.id = ANY(%s::uuid[])"
                        params.append(list(candidate_ids))

                    sql += " ORDER BY n.embedding <=> %s::vector LIMIT %s"
                    params.extend([query_vec, self.config.top_k_semantic])

                    cur.execute(sql, params)
                    for row in cur.fetchall():
                        node_id = row[0]
                        if node_id not in all_candidates or row[8] > all_candidates[node_id][8]:
                            all_candidates[node_id] = row

        results = []
        for key, row in all_candidates.items():
            if row[8] >= self.config.similarity_threshold:
                results.append({
                    'node_id': row[0], 'title': row[1], 'content': row[2], 'text': row[3],
                    'level': row[4], 'parent_id': row[5], 'metadata': row[6] or {},
                    'keywords': row[7] or [], 'vector_score': float(row[8])
                })
        return results

    def _bm25_search(self, query: str, document_id: Optional[str], candidate_ids: Optional[Set[str]], document_cat: Optional[str] = None, document_subcat: Optional[str] = None) -> Dict[str, float]:
        tokens = self._preprocess_text(query)
        if not tokens: return {}
        tsquery = self._to_tsquery_string(tokens)
        bm25_hits: Dict[Any, float] = {}
        with self.conn.cursor() as cur:
            sql = """
                SELECT n.id, ts_rank_cd(n.content_tsv, to_tsquery('simple', %s)) as rank
                FROM rag.nodes as n
                INNER JOIN rag.documents as d ON n.document_id = d.id
                WHERE n.content_tsv @@ to_tsquery('simple', %s)
            """
            params: List[Any] = [tsquery, tsquery]

            if document_id:
                sql += " AND n.document_id = %s::uuid"
                params.append(document_id)

            if document_cat and document_subcat:
                sql += " AND (LOWER(d.category) = LOWER(%s) OR LOWER(d.subcategory) = LOWER(%s))"
                params.extend([document_cat, document_subcat])
            elif document_cat:
                sql += " AND LOWER(d.category) = LOWER(%s)"
                params.append(document_cat)
            elif document_subcat:
                sql += " AND LOWER(d.subcategory) = LOWER(%s)"
                params.append(document_subcat)

            if candidate_ids is not None:
                sql += " AND n.id = ANY(%s::uuid[])"
                params.append(list(candidate_ids))

            cur.execute(sql, params)
            for row in cur.fetchall():
                bm25_hits[row[0]] = float(row[1])

        if bm25_hits:
            max_s = max(bm25_hits.values())
            if max_s > 0:
                for k in bm25_hits: bm25_hits[k] /= max_s
        return bm25_hits

    def _preprocess_text(self, text: str) -> List[str]:
        if not text: return []
        tokens = re.split(r'[^a-zA-Z0-9\-]+', text.lower())
        return [t for t in tokens if t and t not in STOP_WORDS]

    def _tokenise(self, s: str) -> Set[str]:
        return set(self._preprocess_text(s))

    def _rerank(self, semantic_results: List[Dict[str, Any]], bm25_hits: Dict[str, float], *, query: str, query_expansions: Optional[List[str]], config: Optional[RetrievalConfig], ontology_config=None) -> List[SearchResult]:
        active_config = config or self.config
        final_candidates = {}

        for res in semantic_results:
            node_id = res['node_id']
            final_candidates[node_id] = res
            final_candidates[node_id]['bm25_score'] = bm25_hits.get(node_id, 0.0)

        bm25_only_ids = [nid for nid in bm25_hits.keys() if nid not in final_candidates]
        if bm25_only_ids:
            with self.conn.cursor() as cur:
                sql = "SELECT id, title, content, text, level, parent_id, metadata, keywords FROM rag.nodes WHERE id = ANY(%s::uuid[])"
                cur.execute(sql, [bm25_only_ids])
                for row in cur.fetchall():
                    node_id = row[0]
                    final_candidates[node_id] = {
                        'node_id': node_id, 'title': row[1], 'content': row[2], 'text': row[3],
                        'level': row[4], 'parent_id': row[5], 'metadata': row[6] or {},
                        'keywords': row[7] or [], 'vector_score': 0.0, 'bm25_score': bm25_hits[node_id]
                    }

        q_all = self._tokenise(query) | self._tokenise(" ".join(query_expansions or []))
        results = []
        for node_id, cand in final_candidates.items():
            vec_s = cand.get('vector_score', 0.0)
            bm25_s = cand.get('bm25_score', 0.0)
            final_score = (vec_s * active_config.vector_weight + bm25_s * active_config.bm25_weight)

            meta = cand.get('metadata', {}) or {}
            node_kws = {k.lower() for k in (meta.get('keywords') or cand.get('keywords') or [])}
            kw_overlap = len(q_all & node_kws)
            bonus = 0.05 if kw_overlap > 0 else 0.0

            if meta.get('depth', 0) >= 3: bonus += 0.03

            hierarchy_path = meta.get('hierarchy_path', [])
            h_str = ' > '.join(hierarchy_path).lower()
            
            # Dynamic hierarchy boost
            boost_paths = ontology_config.retrieval_hints.hierarchy_boost_paths if (ontology_config and ontology_config.retrieval_hints) else ['medical benefits', 'section 2']
            if boost_paths:
                if any(bp.lower() in h_str for bp in boost_paths):
                    indicators = ontology_config.retrieval_hints.service_indicators if (ontology_config and ontology_config.retrieval_hints) else {'ambulance', 'mental health', 'dme', 'dental', 'hospice', 'emergency', 'urgent care', 'inpatient', 'outpatient', 'therapy'}
                    if any(term.lower() in (query + " ".join(query_expansions or [])).lower() for term in indicators):
                        bonus += 0.04

            final_score += bonus
            results.append(SearchResult(
                node_id=node_id, title=cand['title'], content=cand['content'], text=cand['text'],
                level=cand['level'], parent_id=cand['parent_id'], vector_score=vec_s,
                bm25_score=bm25_s, final_score=final_score, metadata=cand['metadata'],
                hierarchy_path=hierarchy_path
            ))


        results.sort(key=lambda x: x.final_score, reverse=True)
        return results

    def _fetch_parents(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        with self.conn.cursor() as cur:
            sql = """
                WITH RECURSIVE parents AS (
                    SELECT id, parent_id, title, content, text, level, metadata, 0 as depth
                    FROM rag.nodes WHERE id = ANY(%s::uuid[])
                    UNION
                    SELECT n.id, n.parent_id, n.title, n.content, n.text, n.level, n.metadata, p.depth + 1
                    FROM rag.nodes n JOIN parents p ON n.id = p.parent_id WHERE p.depth < %s
                )
                SELECT DISTINCT id, parent_id, title, content, text, level, metadata FROM parents ORDER BY level ASC;
            """
            cur.execute(sql, [node_ids, self.config.max_parent_levels])
            return [{'node_id': r[0], 'parent_id': r[1], 'title': r[2], 'content': r[3], 'text': r[4], 'level': r[5], 'metadata': r[6] or {}, 'hierarchy_path': (r[6] or {}).get('hierarchy_path', [])} for r in cur.fetchall()]

    def _build_context(self, enriched_nodes: List[Dict[str, Any]], top_results: List[SearchResult]) -> str:
        table_rows = [f"| {i} | {r.final_score:.3f} | {r.vector_score:.3f} | {r.bm25_score:.3f} | {(r.title[:47]+'...') if len(r.title)>50 else r.title} |" for i, r in enumerate(top_results, 1)]
        ranking = "## Top Ranked Candidates\n\n| Rank | Score | Vector | BM25 | Title |\n|---|---|---|---|---|\n" + "\n".join(table_rows) + "\n\n"

        seen = set()
        context_parts = ["## Context Documents\n"]
        context_parts = ["## Context Documents\n"]
        for node in sorted(enriched_nodes, key=lambda x: int(x.get('level', 0))):
            if node['node_id'] in seen: continue
            seen.add(node['node_id'])
            h_str = " > ".join(cast(List[str], node['hierarchy_path'])) if node.get('hierarchy_path') else node['title']
            context_parts.append(f"### [{node['level']}] {h_str}\n*Node ID:* {node['node_id']}\n\n{node['content']}\n\n---\n")

        return f"# Retrieval Results\n\n{ranking}{''.join(context_parts)}"

    def _build_subgraph_edges(self, enriched_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build parent->child edge list from enriched nodes for subgraph visualization."""
        node_map = {str(n['node_id']): n for n in enriched_nodes}
        edges = []
        for node in enriched_nodes:
            pid = str(node['parent_id']) if node.get('parent_id') else None
            if pid and pid in node_map:
                edges.append({
                    'source': pid,
                    'source_title': node_map[pid]['title'],
                    'source_level': node_map[pid]['level'],
                    'target': str(node['node_id']),
                    'target_title': node['title'],
                    'target_level': node['level'],
                })
        return edges

    def _extract_source_documents(self, enriched_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract unique document references from retrieved nodes."""
        seen_docs: Dict[Any, Dict[str, Any]] = {}
        for node in enriched_nodes:
            meta: Dict[str, Any] = node.get('metadata', {}) or {}
            doc_id = meta.get('document_id') or meta.get('doc_id')
            doc_name = meta.get('document_name') or meta.get('source_file') or 'Unknown Document'
            if doc_id and doc_id not in seen_docs:
                seen_docs[doc_id] = {
                    'document_id': str(doc_id),
                    'document_name': str(doc_name),
                    'node_ids': cast(List[str], []),
                }
            if doc_id:
                cast(List[str], seen_docs[doc_id]['node_ids']).append(str(node['node_id']))
        return list(seen_docs.values())

    def close(self):
        if self.conn: self.conn.close()

def retrieve_for_query(
    query: str,
    query_expansions: Optional[List[str]] = None,
    anchor_entities: Optional[List[str]] = None,
    document_id: Optional[str] = None,
    document_cat: Optional[str] = None,
    document_subcat: Optional[str] = None,
    config: Optional[RetrievalConfig] = None,
    ontology_config=None
) -> 'RetrievalResult':
    retriever = HybridRetriever(config)
    try:
        return retriever.retrieve(
            query=query,
            query_expansions=query_expansions,
            anchor_entities=anchor_entities,
            document_id=document_id,
            document_cat=document_cat,
            document_subcat=document_subcat,
            ontology_config=ontology_config
        )

    finally:
        retriever.close()

if __name__ == "__main__":
    logger.info("Testing hybrid retrieval system...")
    result = retrieve_for_query(query="What plan is Personal Choice 65 SM Medical-Only and what does it cover?", query_expansions=['What is the Dementia Support Program and who is eligible?'])
    print("Context:", result.context_text[:500])
    print(f"\nTop results: {len(result.top_results)}")
    print(f"Enriched nodes: {len(result.enriched_nodes)}")
    print(f"Subgraph edges: {len(result.subgraph_edges)}")
    print(f"Source documents: {len(result.source_documents)}")
