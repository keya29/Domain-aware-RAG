import json
import psycopg2
import uuid
import logging
import os
from pathlib import Path
from psycopg2.extras import execute_values, Json
from sentence_transformers import SentenceTransformer

# Import URLPDFDownloader - try both relative and absolute imports
try:
    from .url_pdf_downloader import URLPDFDownloader
except ImportError:
    from url_pdf_downloader import URLPDFDownloader
# Optional pgvector adapter import (safe if not installed)
try:
    from pgvector.psycopg2 import register_vector  # [NEW]
    _HAS_PGVECTOR = True
except Exception:
    _HAS_PGVECTOR = False
# ---------------------------
# Configuration / Globals
# ---------------------------
# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Load embedding model
logger.info("Use pytorch device_name: cpu")
logger.info("Load pretrained SentenceTransformer: all-MiniLM-L6-v2")
model = SentenceTransformer('all-MiniLM-L6-v2')
# Database config
DB_HOST = "localhost"
DB_NAME = "rag_system"
DB_USER = "postgres"
DB_PASS = "root"
DB_PORT = 5432
# File paths (relative to script directory) - DEFAULTS FOR BACKWARD COMPATIBILITY
# These are overridden when calling ingest_optimized() with parameters
DEFAULT_INPUT_FILE = "outputs/2026-pc65-medical-only-eoc.custom.json"
DEFAULT_ENTITIES_FILE = "outputs/hierarchy_extraction_results.json"
DEFAULT_KEYWORDS_FILE = "outputs/entity_output_keywords_103.json"

INPUT_FILE = DEFAULT_INPUT_FILE
ENTITIES_FILE = DEFAULT_ENTITIES_FILE
KEYWORDS_FILE = DEFAULT_KEYWORDS_FILE
# Cache to avoid duplicate nodes (reset per document ingestion)
def create_node_cache():
    """Create a fresh node cache for a new document."""
    return {}

node_cache = create_node_cache()
# Behavior toggles
FORCE_EMBED_ALL = False  # [NEW] Set True to embed all nodes for testing
LOG_DEBUG_NO_ENTITY_MISSES = True  # [NEW]
# Stats counters [NEW]
_nodes_total = 0
_nodes_with_entities = 0
_embeddings_attempted = 0
_embeddings_non_null = 0
# Runtime column handling [NEW]
_EMBEDDING_COL_TYPE = None   # 'vector' | 'jsonb' | 'json' | 'unknown'
_KEYWORDS_COL_TYPE = None    # 'jsonb' | 'json' | 'text' | 'unknown'
def norm(s: str) -> str:
    """Normalize titles for matching: lowercase + collapse whitespace."""
    return " ".join(str(s).split()).strip().lower()
def to_markdown_table(table_data):
    """Converts list of dicts or list of lists to a markdown table string."""
    if not table_data or not isinstance(table_data, list):
        return ""
   
    if not table_data[0]:
        return ""
       
    if isinstance(table_data[0], dict):
        headers = list(table_data[0].keys())
        rows = table_data
    elif isinstance(table_data[0], (list, tuple)):
        headers = [str(cell) if cell is not None else "" for cell in table_data[0]]
        rows = []
        for r in table_data[1:]:
            row_dict = {}
            for i, val in enumerate(r):
                if i < len(headers):
                    h = headers[i] if headers[i] else f"Col_{i}"
                    row_dict[h] = val
            rows.append(row_dict)
    else:
        return ""
    if not headers:
        return ""
    # Clean headers and escape pipes
    headers = [str(h).replace("\n", " ").replace("|", "\\|").strip() for h in headers]
   
    # Create markdown
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
   
    lines = [header_row, separator_row]
    for r in rows:
        row_cells = []
        for h in headers:
            val = r.get(h, "")
            if val is None: val = ""
            cell_str = str(val).replace("\n", " ").replace("|", "\\|").strip()
            row_cells.append(cell_str)
        lines.append("| " + " | ".join(row_cells) + " |")
   
    return "\n".join(lines)
def remove_duplicate_table_text(text, table_data):
    """
    Identifies if the table data is present in the text as plain text
    (not formatted as a markdown table) and removes it.
    """
    if not table_data or not text:
        return text
    # Extract clean strings to look for (headers + first few cells of data)
    search_terms = []
    if isinstance(table_data[0], dict):
        search_terms.extend([str(k).strip() for k in table_data[0].keys()])
        search_terms.extend([str(v).strip() for v in table_data[0].values()])
    elif isinstance(table_data[0], (list, tuple)) and len(table_data) > 0:
        search_terms.extend([str(c).strip() for c in table_data[0]])
        if len(table_data) > 1:
            row1 = table_data[1]
            if isinstance(row1, (list, tuple)):
                search_terms.extend([str(c).strip() for c in row1])
            elif isinstance(row1, dict):
                search_terms.extend([str(v).strip() for v in row1.values()])
   
    # Filter out empty or too short strings
    search_terms = [t for t in search_terms if t and len(t) > 2]
    if not search_terms:
        return text
    first_term = search_terms[0]
    start_idx = text.find(first_term)
    if start_idx == -1:
        return text
   
    matches = 1
    current_pos = start_idx + len(first_term)
    last_match_pos = current_pos
   
    for term in search_terms[1:15]:  # check more terms for higher confidence
        pos = text.find(term, current_pos)
        if pos != -1 and (pos - current_pos) < 300:
            matches += 1
            current_pos = pos + len(term)
            last_match_pos = current_pos
       
    if matches >= 3:
        segment = text[start_idx:last_match_pos]
        if segment.count("|") > matches:
            return text
       
        last_row = table_data[-1]
        last_cells = []
        if isinstance(last_row, dict):
            last_cells = [str(v).strip() for v in last_row.values() if v]
        elif isinstance(last_row, (list, tuple)):
            last_cells = [str(c).strip() for c in last_row if c]
           
        final_end_pos = last_match_pos
        if last_cells:
            for lc in reversed(last_cells):
                if not lc or len(str(lc)) < 3: continue
                lpos = text.find(str(lc), last_match_pos)
                if lpos != -1 and (lpos - last_match_pos) < 2000:
                    final_end_pos = lpos + len(str(lc))
                    break
        processed_text = text[:start_idx].rstrip() + "\n\n" + text[final_end_pos:].lstrip()
        return processed_text.strip()
   
    return text
def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT
    )
    return conn
# ---------------------------
# DB helpers for column handling [NEW]
# ---------------------------
def detect_column_type(conn, schema: str, table: str, column: str):
    """
    Returns ('jsonb'|'json'|'vector'|'text'|'unknown', udt_name) for a column.
    """
    q = """
    SELECT data_type, udt_name
    FROM information_schema.columns
    WHERE table_schema=%s AND table_name=%s AND column_name=%s
    """
    with conn.cursor() as cur:
        cur.execute(q, (schema, table, column))
        row = cur.fetchone()
        if not row:
            return "unknown", None
        data_type, udt_name = row
        # For pgvector, data_type is usually 'USER-DEFINED' and udt_name='vector'
        if (udt_name or "").lower() == "vector":
            return "vector", udt_name
        if data_type.lower() in ("jsonb", "json"):
            return data_type.lower(), udt_name
        if data_type.lower() in ("text", "character varying"):
            return "text", udt_name
        return "unknown", udt_name
def prepare_param_for_column(value, col_type: str):
    """
    Wrap values appropriately based on detected column type.
    """
    if value is None:
        return None
    if col_type in ("jsonb", "json"):
        return Json(value)
    # For 'vector', psycopg2 with pgvector adapter can accept python lists.
    # For 'text', cast to string if needed.
    if col_type == "text":
        return json.dumps(value)  # store JSON as text (stringified)
    return value  # unknown/vector: pass-through, hope adapter works
# ---------------------------
# Document / Node helpers
# ---------------------------
def get_or_create_document(conn, external_id, title, version):
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM rag.documents WHERE external_id = %s", (external_id,))
        res = cur.fetchone()
        if res:
            return res[0]
       
        doc_id = str(uuid.uuid4())
        cur.execute("""
            INSERT INTO rag.documents (id, external_id, title, version)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (doc_id, external_id, title, version))
        conn.commit()
        return cur.fetchone()[0]
def generate_embedding(text):
    if not text:
        return None
    try:
        return model.encode(text).tolist()
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None
def get_or_create_node(conn, doc_id, parent_id, level, title, content, is_leaf,
                       text=None, embedding=None, metadata=None, keywords=None):
    cache_key = (doc_id, parent_id, title, level)
   
    # Column handling [NEW]
    emb_param = prepare_param_for_column(embedding, _EMBEDDING_COL_TYPE)
    kw_param = prepare_param_for_column(keywords, _KEYWORDS_COL_TYPE)
    if cache_key in node_cache:
        curr_node_id = node_cache[cache_key]
        if (content or text or embedding is not None or metadata or keywords is not None):
            with conn.cursor() as cur:
                update_fields = []
                params = []
                if content:
                    update_fields.append("content = %s")
                    params.append(content)
                if text:
                    update_fields.append("text = %s")
                    params.append(text)
                if embedding is not None:
                    update_fields.append("embedding = %s")
                    params.append(emb_param)
                if metadata:
                    update_fields.append("metadata = %s")
                    params.append(Json(metadata))
                if keywords is not None:
                    update_fields.append("keywords = %s")
                    params.append(kw_param)
                params.append(curr_node_id)
                if update_fields:
                    sql = f"UPDATE rag.nodes SET {', '.join(update_fields)} WHERE id = %s"
                    cur.execute(sql, tuple(params))
        return curr_node_id
    with conn.cursor() as cur:
        if parent_id:
            query = "SELECT id FROM rag.nodes WHERE parent_id = %s AND title = %s LIMIT 1"
            params = (parent_id, title)
        else:
            query = "SELECT id FROM rag.nodes WHERE document_id = %s AND parent_id IS NULL AND title = %s LIMIT 1"
            params = (doc_id, title)
        cur.execute(query, params)
        row = cur.fetchone()
       
        if row:
            node_id = row[0]
            update_fields = []
            params = []
            if content:
                update_fields.append("content = %s")
                params.append(content)
            if text:
                update_fields.append("text = %s")
                params.append(text)
            if embedding is not None:
                update_fields.append("embedding = %s")
                params.append(emb_param)
            if metadata:
                update_fields.append("metadata = %s")
                params.append(Json(metadata))
            if keywords is not None:
                update_fields.append("keywords = %s")
                params.append(kw_param)
            params.append(node_id)
            if update_fields:
                sql = f"UPDATE rag.nodes SET {', '.join(update_fields)} WHERE id = %s"
                cur.execute(sql, tuple(params))
        else:
            node_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO rag.nodes
                (id, document_id, parent_id, level, title, content, text, embedding, metadata, keywords)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                node_id, doc_id, parent_id, level, title, content,
                text, emb_param, Json(metadata) if metadata else None, kw_param
            ))
    node_cache[cache_key] = node_id
    return node_id
# --------------------------
# LOAD ENTITIES
# --------------------------
def load_entities_map():
    from collections import defaultdict
    entity_map = defaultdict(list)
    if not os.path.exists(ENTITIES_FILE):
        logger.warning(f"{ENTITIES_FILE} missing.")
        return entity_map
    try:
        with open(ENTITIES_FILE, "r", encoding="utf-8") as f:
            lines = [line for line in f if not line.strip().startswith("#")]
            content = "".join(lines).strip()
            if not content:
                logger.warning(f"{ENTITIES_FILE} is empty or only contains comments.")
                return entity_map
            data = json.loads(content)
        # Handle new NER format: {"levels": {"Level": {"entity_name": {"scope_id": "value"}}}}
        if isinstance(data, dict) and "levels" in data:
            levels = data.get("levels", {})
            for level, entity_map_by_level in levels.items():
                if not isinstance(entity_map_by_level, dict):
                    continue
                for entity_name, section_map in entity_map_by_level.items():
                    if not isinstance(section_map, dict):
                        continue
                    for scope_id, entity_value in section_map.items():
                        if entity_name and entity_value:
                            path_key = (1, norm(level))  # Simple path for now
                            entity_map[path_key].append({
                                "category": level,
                                "type": entity_name,
                                "value": entity_value
                            })
        # Handle legacy format: array of items with level, entity_category, entity_value
        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                category = item.get("level")
                entity_type = item.get("entity_category")
                entity_val = item.get("entity_value")
                hierarchy_path = item.get("hierarchy_path", {})
     
                if not entity_type or not entity_val:
                    continue
     
                titles = {}
                current = hierarchy_path
                depth = 1
     
                # Extract normalized path titles
                while isinstance(current, dict):
                    level_key = f"Level_{depth}"
                    if level_key in current:
                        inner = current[level_key]
                        if isinstance(inner, dict) and inner:
                            raw_title = list(inner.keys())[0]
                            title = norm(raw_title)  # normalize for matching
                            titles[depth] = title
                            current = inner[raw_title]
                            depth += 1
                        else:
                            break
                    else:
                        break
     
                if titles:
                    path_key = tuple(sorted(titles.items()))
                    entity_map[path_key].append({"category": category, "type": entity_type, "value": entity_val})
    except Exception as e:
        logger.error(f"Entities map error: {e}")
    return entity_map
# --------------------------
# LOAD KEYWORDS (UNCHANGED)
# --------------------------
def load_keywords_map():
    keywords_map = {}
    if not os.path.exists(KEYWORDS_FILE):
        logger.warning(f"{KEYWORDS_FILE} missing.")
        return keywords_map
    try:
        with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
            lines = [line for line in f if not line.strip().startswith("#")]
            content = "".join(lines).strip()
            if not content:
                logger.warning(f"{KEYWORDS_FILE} is empty or only contains comments.")
                return keywords_map
            data = json.loads(content)
        kw = data.get("keywords", {})
        for sec, kw_list in kw.items():
            # Normalize title key (spacing only, keep case as-is for compatibility)
            normalized_key = " ".join(sec.split()).strip()
            keywords_map[normalized_key] = kw_list
    except Exception as e:
        logger.error(f"Keywords map error: {e}")
    return keywords_map


# --------------------------
# URL HANDLER - Process PDF from URL or local path
# --------------------------
def process_pdf_input(input_source: str) -> str:
    """
    Process PDF input from either a URL or local file path.
    For now, validates the PDF and returns the default JSON path.
    Full extraction pipeline requires hierarchical package setup.
    
    Args:
        input_source: Either a URL (http/https) or local file path to PDF
        
    Returns:
        Path to JSON file for ingestion
        
    Raises:
        ValueError: If input is invalid
    """
    input_source = input_source.strip() if input_source else ""
    
    # Check if it's a URL
    if input_source.lower().startswith(('http://', 'https://')):
        logger.info(f"Detected URL input: {input_source}")
        try:
            pdf_path, filename = URLPDFDownloader.download(input_source)
            logger.info(f"✓ PDF downloaded successfully: {filename}")
            # For now, use the default JSON (extraction requires hierarchical package)
            logger.info("Note: Full PDF extraction pipeline requires additional setup. Using default dataset.")
            return INPUT_FILE
        except Exception as e:
            logger.error(f"Failed to download PDF from URL: {str(e)}")
            raise
    
    # Assume it's a local file path
    else:
        if not os.path.exists(input_source):
            raise ValueError(f"File not found: {input_source}")
        if not input_source.lower().endswith('.pdf'):
            raise ValueError(f"Input file is not a PDF: {input_source}")
        logger.info(f"✓ Local PDF file validated: {input_source}")
        logger.info("Note: Full PDF extraction pipeline requires additional setup. Using default dataset.")
        return INPUT_FILE


# --------------------------
# INGEST MAIN
# --------------------------
def ingest_optimized(pdf_input: str = None, doc_id: str = None, doc_title: str = None,
                     doc_version: str = None, input_file: str = None, entities_file: str = None,
                     keywords_file: str = None):
    """
    Main ingestion pipeline. Processes PDF from URL or local path with document-specific metadata.
    
    Args:
        pdf_input: Optional URL (http/https) or local file path to PDF for validation
        doc_id: Unique document ID (if None, extracted from pdf_input or uses default)
        doc_title: Document title for rag.documents table (default: "Untitled Document")
        doc_version: Document version (default: "1.0")
        input_file: Path to hierarchical JSON (default: DEFAULT_INPUT_FILE)
        entities_file: Path to entities/hierarchy JSON (default: DEFAULT_ENTITIES_FILE)
        keywords_file: Path to keywords JSON (default: DEFAULT_KEYWORDS_FILE)
    """
    global INPUT_FILE, ENTITIES_FILE, KEYWORDS_FILE, node_cache
    
    # Set file paths from parameters or defaults
    INPUT_FILE = input_file or DEFAULT_INPUT_FILE
    ENTITIES_FILE = entities_file or DEFAULT_ENTITIES_FILE
    KEYWORDS_FILE = keywords_file or DEFAULT_KEYWORDS_FILE
    
    # Reset node cache for this document ingestion
    node_cache = create_node_cache()
    
    # Set document metadata from parameters or defaults
    if not doc_id:
        doc_id = "default_document"
    if not doc_title:
        doc_title = f"Document: {doc_id}"
    if not doc_version:
        doc_version = "1.0"
    
    logger.info(f"[Ingestion] Document ID: {doc_id}")
    logger.info(f"[Ingestion] Document Title: {doc_title}")
    logger.info(f"[Ingestion] Document Version: {doc_version}")

  
    
    # Handle PDF input (URL or local file)
    if pdf_input:
        logger.info(f"Processing PDF input: {pdf_input}")
        try:
            pdf_path = process_pdf_input(pdf_input)
        except Exception as e:
            logger.error(f"Failed to process PDF input: {str(e)}")
            raise
    else:
        pdf_path = INPUT_FILE
        logger.info(f"Using default INPUT_FILE: {pdf_path}")
    
    # Log absolute paths [NEW]
    logger.info(f"PDF path absolute: {os.path.abspath(pdf_path)}")
    logger.info(f"ENTITIES_FILE absolute: {os.path.abspath(ENTITIES_FILE)}")
    logger.info(f"KEYWORDS_FILE absolute: {os.path.abspath(KEYWORDS_FILE)}")
    # Sanity: model test embedding [NEW]
    try:
        _test_vec = generate_embedding("test embedding")
        logger.info(f"Test embedding length: {len(_test_vec) if _test_vec else None}")
    except Exception as e:
        logger.warning(f"Could not generate test embedding: {e}")
    conn = get_db_connection()
    # Detect column types & register pgvector if available [NEW]
    global _EMBEDDING_COL_TYPE, _KEYWORDS_COL_TYPE
    try:
        _EMBEDDING_COL_TYPE, emb_udt = detect_column_type(conn, "rag", "nodes", "embedding")
        _KEYWORDS_COL_TYPE, kw_udt = detect_column_type(conn, "rag", "nodes", "keywords")
        logger.info(f"Detected column types: embedding={_EMBEDDING_COL_TYPE} (udt={emb_udt}), keywords={_KEYWORDS_COL_TYPE} (udt={kw_udt})")
        if _EMBEDDING_COL_TYPE == "vector":
            if _HAS_PGVECTOR:
                try:
                    register_vector(conn)
                    logger.info("pgvector adapter registered on connection.")
                except Exception as e:
                    logger.warning(f"Failed to register pgvector adapter: {e}")
            else:
                logger.warning("pgvector python package not available; embedding insert may fail for 'vector' type.")
    except Exception as e:
        logger.warning(f"Could not detect column types: {e}")
        _EMBEDDING_COL_TYPE = _EMBEDDING_COL_TYPE or "unknown"
        _KEYWORDS_COL_TYPE = _KEYWORDS_COL_TYPE or "unknown"
    # Stats [NEW]
    global _nodes_total, _nodes_with_entities, _embeddings_attempted, _embeddings_non_null
    _nodes_total = _nodes_with_entities = _embeddings_attempted = _embeddings_non_null = 0
    try:
        entity_map = load_entities_map()
        logger.info(f"Loaded entity map: {len(entity_map)} paths")
        keywords_map = load_keywords_map()
        logger.info(f"Loaded {len(keywords_map)} keyword sections")
        with open(pdf_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded main content")
        doc_uuid = get_or_create_document(
            conn,
            doc_id,  # Use parameterized doc_id
            doc_title,  # Use parameterized doc_title
            doc_version  # Use parameterized doc_version
        )
        def traverse(level_data, current_level, parent_id, path_titles):
            global _nodes_total, _nodes_with_entities, _embeddings_attempted, _embeddings_non_null
            for title, node_data in level_data.items():
                _nodes_total += 1
                # Keep original title for storage; use normalized title for matching [NEW]
                norm_title_for_match = norm(title)
                # Update path for matching
                current_path_titles = path_titles.copy()
                current_path_titles[current_level] = norm_title_for_match  # normalized [NEW]
                # Get text/table
                content_obj = node_data.get("content", {})
                text_content = content_obj.get("text", "")
                tables = content_obj.get("table", [])
                # Add tables to text column in markdown format
                if tables:
                    for tbl in tables:
                        text_content = remove_duplicate_table_text(text_content, tbl)
                        md_table = to_markdown_table(tbl)
                        if md_table:
                            if md_table not in text_content:
                                text_content = (text_content.strip() + "\n\n" + md_table).strip()
                # Entities: match using normalized path key [NEW]
                path_key = tuple(sorted(current_path_titles.items()))
                entities = entity_map.get(path_key, [])
                if entities:
                    _nodes_with_entities += 1
                # Hierarchy string (keep original titles for readability)
                hierarchy_list = []
                # Build hierarchy list from path_titles BUT use original titles if available
                # Since we only stored normalized titles in current_path_titles for matching,
                # we reconstruct display hierarchy using a copy of incoming titles per level if passed.
                # Here we cannot retrieve originals from map, so we will build from prior path context
                # For display, use the existing parent path (if provided in metadata we keep titles),
                # else fallback to current title.
                # Simpler: keep a separate display dict with originals [NEW]
                # -- Implement: maintain a parallel dict
                # But to keep code minimally invasive, we'll assemble hierarchy_list progressively:
                if path_titles:
                    # path_titles contains previous levels' normalized values; we cannot recover originals from there.
                    # So we will build display using only current node's title.
                    # Keep previous behavior for hierarchy as before (original code relied on current_path_titles order),
                    # but we still need a hierarchy_string. We'll use only the titles we know (current title).
                    pass
                # Original behavior constructed hierarchy_list from current_path_titles; we'll adapt:
                # We can't recover originals, so use normalized titles for parents and original for current.
                # This is acceptable for content concatenation; original 'title' is still stored in node record.
                hierarchy_list = [t for _, t in sorted(current_path_titles.items())]
                # Replace the last with the original current title for better readability
                if hierarchy_list:
                    hierarchy_list[-1] = title
                hierarchy_string = " > ".join(hierarchy_list)  # keep readable '>' [CHANGED from &gt;]
                content_parts = [hierarchy_string] if hierarchy_string else []
                if text_content:
                    content_parts.append(text_content)
                if entities:
                    parts = ", ".join([f"{e['value']} ({e['type']})" for e in entities])
                    content_parts.append(parts)
                full_content = "\n".join(content_parts)
                # Embedding and text per rule (gated by entities unless FORCE_EMBED_ALL)
                if FORCE_EMBED_ALL or entities:
                    embedding_vector = generate_embedding(full_content)
                    _embeddings_attempted += 1
                    if embedding_vector is not None:
                        _embeddings_non_null += 1
                    text_for_column = text_content
                else:
                    embedding_vector = None
                    text_for_column = None
                    if LOG_DEBUG_NO_ENTITY_MISSES:
                        logger.debug(f"No entities for hierarchy='{hierarchy_string}', skipping embeddings and text column")

                # Is leaf?
                next_level_key = f"Level_{current_level+1}"
                has_children = next_level_key in node_data and node_data[next_level_key]
                is_leaf = not has_children

                # Keywords lookup (keep original logic)
                hierarchy_string_for_kw = " ".join(hierarchy_string.split()).strip()
                section_keywords = keywords_map.get(hierarchy_string_for_kw) \
                                   or keywords_map.get(" ".join(title.split()).strip()) \
                                   or []

                # Metadata
                page_number = node_data.get("page")
                node_metadata = {
                    "hierarchy_path": hierarchy_list,
                    "hierarchy_string": hierarchy_string,
                    "depth": current_level,
                    "is_leaf": is_leaf,
                    "has_tables": bool(tables),
                    "has_entities": bool(entities),
                    "entity_types": list({e["type"] for e in entities}) if entities else [],
                    "canonical_names": list({e["value"] for e in entities}) if entities else [],
                    "keywords": section_keywords,
                    "has_keywords": bool(section_keywords),
                    "page_num": page_number,
                    "section_title": title
                }

                # Insert node
                node_id = get_or_create_node(
                    conn, doc_uuid, parent_id, current_level,
                    title, "\n".join(content_parts), is_leaf,  # keep full_content for content
                    text=text_for_column,
                    embedding=embedding_vector,
                    metadata=node_metadata,
                    keywords=section_keywords
                )
                # Insert entities
                if entities:
                    with conn.cursor() as cur:
                        for ent in entities:
                            ent_md = {
                                "related_node_title": title,
                                "hierarchy_context": hierarchy_string,
                                "source": "hierarchy_extraction",
                                "category": ent.get("category", "")
                            }
                            cur.execute("""
                                INSERT INTO rag.entities (document_id, node_id, canonical_name, entity_type, metadata)
                                VALUES (%s, %s, %s, %s, %s)
                            """, (
                                doc_uuid, node_id,
                                ent["value"], ent["type"],
                                Json(ent_md)
                            ))
                # Recurse
                if has_children:
                    traverse(node_data[next_level_key], current_level+1, node_id, current_path_titles)
        # Start
        if "Level_1" in data:
            traverse(data["Level_1"], 1, None, {})
        # Stats [NEW]
        logger.info(f"Stats: nodes_total={_nodes_total}, nodes_with_entities={_nodes_with_entities}, "
                    f"embeddings_attempted={_embeddings_attempted}, embeddings_non_null={_embeddings_non_null}")
        conn.commit()
        logger.info("Ingestion complete.")
    except Exception as e:
        import traceback
        logger.error(f"Ingestion failed: {traceback.format_exc()}")
        conn.rollback()
    finally:
        conn.close()
if __name__ == "__main__":
    ingest_optimized()