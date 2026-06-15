import os
import json
import re
from typing import List, Dict, Any
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
 
MAX_ANCHORS = 25 
MAX_VARIANTS = 3 
 
# Current embedder: 384-dim. If your DB column is VECTOR(1536), 
# either ALTER the column to 384, or switch to Azure embeddings. 
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 
 
# LOGGING PATHS 
LOG_PATH_JSONL = "logs/query_expansion.jsonl"   # append-only, robust 
LOG_PATH_XLSX = "outputs/query_expansion.xlsx"     # maintained incrementally 

# ANCHORS PATH 
ANCHOR_PATH = "data/query_anchors.json"  # Path to the anchors file 

try:
    import pandas as pd
except ImportError:
    pd = None

 
# ===================================================== 
# AZURE OPENAI (YOUR SETUP) 
# ===================================================== 
 
# Load .env from parent directory
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path) 
 
client = AzureOpenAI( 
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_KEY"), 
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"), 
) 
 
MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT") 
 
if not client or not MODEL: 
    raise RuntimeError("Azure OpenAI configuration missing") 
 
# ===================================================== 
# EMBEDDING MODEL 
# ===================================================== 
 
_embedder = None 
 
def load_embedder(): 
    """ 
    Lazily load the sentence-transformers embedder. 
    NOTE: Produces 384-dim vectors for all-MiniLM-L6-v2. 

    Align your DB schema accordingly if you plan to store these vectors. 
    """ 
    global _embedder 
    if _embedder is None: 
        _embedder = SentenceTransformer(EMBED_MODEL) 
    return _embedder 
 
def embed_texts(texts: List[str]) -> List[List[float]]: 
    model = load_embedder() 
    vecs = model.encode(texts, batch_size=32, normalize_embeddings=True) 
    return [v.tolist() for v in vecs] 
 
# ===================================================== 
# TEXT UTILS 
# ===================================================== 
 
TOKEN_RE = re.compile(r"\b[\w\-]+\b", re.UNICODE) 
 
def normalize(text: str) -> str: 
    return re.sub(r"\s+", " ", text.strip().lower()) 
 
def tokenize(text: str): 
    return set(TOKEN_RE.findall(normalize(text))) 
 
# ===================================================== 
# LOAD ANCHORS 
# ===================================================== 
 
def load_anchors(path=ANCHOR_PATH): 
    try: 
        with open(path, "r", encoding="utf-8") as f: 
            data = json.load(f) 
        return data if isinstance(data, dict) else {} 
    except (FileNotFoundError, json.JSONDecodeError): 
        return {} 
 
ANCHORS = load_anchors() 
 
# ===================================================== 
# ANCHOR SELECTION (token-aware) 
# ===================================================== 
 
def select_anchors(query: str, anchors: dict): 
    """ 

    Selects buckets by token-aware synonym match or phrase presence (word 
boundaries). 
    Returns (selected_terms[:MAX_ANCHORS], matched_bucket_names). 
    """ 
    q_norm = normalize(query) 
    q_tokens = tokenize(query) 
    matched_buckets = set() 
    selected_terms = [] 
 
    for bucket, data in anchors.items(): 
        syns = [normalize(s) for s in data.get("synonyms", [])] 
        phrases = [normalize(p) for p in data.get("phrases", [])] 
 
        # Match any synonym or phrase as a phrase (with word boundaries) 
        found = False 
        for s in syns: 
            if re.search(rf"\b{re.escape(s)}\b", q_norm): 
                found = True 
                break 
        if not found: 
            for p in phrases: 
                if re.search(rf"\b{re.escape(p)}\b", q_norm): 
                    found = True 
                    break 
        if found: 
            matched_buckets.add(bucket) 
 
    # Collect anchor terms 
    for b in matched_buckets: 
        d = anchors[b] 
        selected_terms.append(d.get("canonical", "")) 
        selected_terms.extend(d.get("phrases", [])[:3]) 
        selected_terms.extend(d.get("synonyms", [])[:2]) 
 
    # Deduplicate while preserving order 
    seen, clean = set(), [] 
    for t in selected_terms: 
        t = normalize(t) 
        if t and t not in seen: 
            seen.add(t) 
            clean.append(t) 
 
    return clean[:MAX_ANCHORS], list(matched_buckets) 
 
# ===================================================== 
# PROMPT BUILDER (strict-JSON) 
# ===================================================== 
 

def build_prompt(query: str, anchors: List[str], ontology_config=None) -> List[dict]: 
    anchor_text = ", ".join(anchors) 
    
    if ontology_config and ontology_config.query_expansion:
        sys_role = ontology_config.query_expansion.system_role
    else:
        sys_role = "You translate informal member questions into formal Medicare insurance policy search queries."

    sys = ( 
        f"{sys_role}\n" 
        "Return ONLY a JSON array of strings. No prose." 
    ) 
    user = f"""User question: 
{query} 
 
Approved policy terms: 
{anchor_text} 
 
Instructions: 
- Use the approved terms when relevant 
- Do not introduce new topics 
- Do not add explanations 
- Do not hallucinate 
 
Rewrite into {MAX_VARIANTS} short search queries. 
 
JSON schema: 
[ 
  "string", 
  "string", 
  "string" 
]""" 
    return [{"role": "system", "content": sys}, {"role": "user", "content": 
user}] 

 
# ===================================================== 
# LLM EXPANSION (JSON mode with fallback) 
# ===================================================== 
 
def expand_with_llm(query: str, anchors: List[str], ontology_config=None) -> List[str]: 
    messages = build_prompt(query, anchors, ontology_config) 
    # First, try with JSON mode if your Azure deployment supports it 
    try: 
        resp = client.chat.completions.create( 
            model=MODEL, 
            messages=messages, 
            temperature=0, 
            response_format={"type": "json_object"}, 
        ) 
        content = resp.choices[0].message.content.strip() 
    except Exception as e: 
        # Fallback without response_format 
        print(f"Error: {e}") 
 
        """print(f"WARN: First LLM attempt failed (JSON mode): {e}") 
        resp = client.chat.completions.create( 
            model=MODEL, 
            messages=messages, 
            temperature=0 
        ) 
        content = resp.choices[0].message.content.strip()""" 
 
    # Handle accidental code fences 
    content = content.strip("` \n") 
    try: 
        data = json.loads(content) 
        # Allow either a top-level array or {"queries": [...]} 
        if isinstance(data, list): 
            variants = data 
        elif isinstance(data, dict) and "queries" in data and isinstance(data["queries"], list):
            variants = data["queries"] 
        else: 
            return [] 
        return [v for v in variants if isinstance(v, str) and v.strip()][:MAX_VARIANTS]
    except Exception: 
        return [] 

 
# ===================================================== 
# MAIN PIPELINE 
# ===================================================== 
 
def expand_and_embed(query: str, ontology_config=None): 
    if ontology_config is None:
        try:
            from domain_aware_rag.ontology.registry import OntologyRegistry
            ontology_config = OntologyRegistry().get_active_ontology()
        except Exception as e:
            logger.warning(f"Could not load active ontology config: {e}")
            ontology_config = None

    anchors_data = ontology_config.anchor_taxonomy if ontology_config else ANCHORS
    if not anchors_data:
        anchors_data = ANCHORS

    # 1. Anchor selection 
    anchor_terms, buckets = select_anchors(query, anchors_data) 
 
    # DEBUG: Print anchor selection inputs and outputs 
    print('DEBUG: Query:', query) 
    print('DEBUG: ANCHORS keys:', list(anchors_data.keys())[:5], '...') 
    anchor_terms, buckets = select_anchors(query, anchors_data) 
    print('DEBUG: anchor_terms:', anchor_terms) 
    print('DEBUG: buckets:', buckets) 
 
    # 2. LLM expansion 
    llm_variants = expand_with_llm(query, anchor_terms, ontology_config) 
 
    # 3. Final queries (original + up to MAX_VARIANTS unique) 
    queries = [query] 
    queries.extend(llm_variants) 
    queries = list(dict.fromkeys(queries))[:MAX_VARIANTS + 1] 
 
    # 4. Embedding (384-dim for MiniLM unless you switch models) 
    vectors = embed_texts(queries) 
 
    # 5. Return keywords (for entity/keyword search) 
    return { 
        "queries": queries, 
        "vectors": vectors, 
        "anchors": anchor_terms, 
        "buckets": buckets, 
        "keywords": anchor_terms,  # for compatibility with entity/keyword search
    } 

 
# ===================================================== 
# LOGGING (JSONL is append-only) 
# ===================================================== 
 
def append_log_jsonl(turn_id: int, user_text: str, result: Dict[str, Any], 
role: str = "User"): 
    """ 
    Appends a single line of JSON to query_expansion.jsonl. 
    Each line is a complete JSON object (append-safe). 
    """ 
    entry = { 
        "turn": turn_id, 
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), 
        "role": role, 
        "user_query": user_text, 
        "anchors": result.get("anchors", []), 
        "buckets": result.get("buckets", []), 
        "expanded_queries": result.get("queries", []), 
        "vectors": result.get("vectors", []),  # remove if file grows too large
    } 
    with open(LOG_PATH_JSONL, "a", encoding="utf-8") as f: 
        f.write(json.dumps(entry, ensure_ascii=False)) 
        f.write("\n") 
 
# ===================================================== 
# EXCEL HELPERS 
# ===================================================== 
 
def append_row_to_excel(xlsx_path: str, rows: List[Dict[str, Any]]) -> None: 
    """ 
    Appends rows to an Excel file by reading it (if exists), concatenating, 
    de-duplicating on (turn, user_query), and writing back. 
    """ 

    df_new = pd.DataFrame(rows) 
 
    if os.path.exists(xlsx_path) and os.path.getsize(xlsx_path) > 0: 
        try: 
            df_old = pd.read_excel(xlsx_path, engine="openpyxl") 
        except Exception: 
            df_old = pd.DataFrame() 
        df_all = pd.concat([df_old, df_new], ignore_index=True) 
        df_all.drop_duplicates(subset=["turn", "user_query"], keep="last", 
inplace=True) 
    else: 
        df_all = df_new 
 
    df_all.sort_values(by=["turn", "timestamp_utc"], inplace=True) 
    df_all.to_excel(xlsx_path, index=False, engine="openpyxl") 
 
def log_to_excel_from_result( 
    turn_id: int, 
    user_text: str, 
    result: Dict[str, Any], 
    role: str = "User", 
    xlsx_path: str = LOG_PATH_XLSX, 
    include_vectors: bool = False, 
    vectors_as_length_only: bool = True 
): 
    """ 
    Converts the result to a row and appends it to Excel. 
    By default, doesn't embed full vectors (keeps file small). 
    """ 
    row = { 
        "turn": turn_id, 
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), 
        "role": role, 
        "user_query": user_text, 
        "anchors": ", ".join(result.get("anchors", [])), 
        "buckets": ", ".join(result.get("buckets", [])), 
        "expanded_queries": " | ".join(result.get("queries", [])), 
    } 
    if include_vectors: 
        vecs = result.get("vectors", []) 
        if vectors_as_length_only: 
            row["vectors_count"] = len(vecs) 
            row["vector_dim"] = len(vecs[0]) if vecs and isinstance(vecs[0], 
list) else 0 
        else: 
            row["vectors_json"] = json.dumps(vecs, ensure_ascii=False) 
 
    append_row_to_excel(xlsx_path, [row]) 

 
def export_jsonl_to_excel( 
    jsonl_path: str = LOG_PATH_JSONL, 
    xlsx_path: str = LOG_PATH_XLSX, 
    include_vectors: bool = False, 
    vectors_as_length_only: bool = True 
) -> None: 
    """ 
    Converts the append-only JSONL log into a clean Excel snapshot. 
    Useful if you prefer rebuilding Excel periodically. 
    """ 
    if not os.path.exists(jsonl_path) or os.path.getsize(jsonl_path) == 0: 
        print(f"No data to export: {jsonl_path} is missing or empty.") 
        return 
 
    rows: List[Dict[str, Any]] = [] 
    with open(jsonl_path, "r", encoding="utf-8") as f: 
        for line in f: 
            line = line.strip() 
            if not line: 
                continue 
            try: 
                obj = json.loads(line) 
            except json.JSONDecodeError: 
                continue 
 
            row = { 
                "turn": obj.get("turn"), 
                "timestamp_utc": obj.get("timestamp_utc"), 
                "role": obj.get("role"), 
                "user_query": obj.get("user_query"), 
                "anchors": ", ".join(obj.get("anchors", [])), 
                "buckets": ", ".join(obj.get("buckets", [])), 
                "expanded_queries": " | ".join(obj.get("expanded_queries", 
[])), 
            } 
 
            if include_vectors: 
                vecs = obj.get("vectors", []) 
                if vectors_as_length_only: 
                    dim = len(vecs[0]) if vecs and isinstance(vecs[0], list) else 0
                    row["vectors_count"] = len(vecs) 
                    row["vector_dim"] = dim 
                else: 
                    row["vectors_json"] = json.dumps(vecs, ensure_ascii=False) 
 
            rows.append(row) 

 
    df = pd.DataFrame(rows).sort_values(by=["turn", "timestamp_utc"]) 
    df.to_excel(xlsx_path, index=False, engine="openpyxl") 
    print(f"Exported {len(df)} rows to {xlsx_path}") 
 
# ===================================================== 
# TURN COUNTER 
# ===================================================== 
 
def compute_next_turn(jsonl_path: str = LOG_PATH_JSONL) -> int: 
    """ 
    Reads the last line of JSONL and returns last_turn + 1. 
    Starts at 1 if file doesn't exist. 
    """ 
    if not os.path.exists(jsonl_path) or os.path.getsize(jsonl_path) == 0: 
        return 1 
    last = None 
    with open(jsonl_path, "rb") as f: 
        try: 
            f.seek(-2, os.SEEK_END) 
            while f.read(1) != b"\n": 
                f.seek(-2, os.SEEK_CUR) 
        except OSError: 
            f.seek(0) 
        last = f.readline().decode("utf-8", errors="ignore").strip() 
    try: 
        obj = json.loads(last) 
        return int(obj.get("turn", 0)) + 1 
    except Exception: 
        return 1 
 
# ===================================================== 
# CLI / BATCH 
# ===================================================== 
 
def process_single_cli(): 
    q = input("Enter query: ").strip() 
    result = expand_and_embed(q) 
 
    print("\n========== RESULT ==========\n") 
    print("Anchors:\n", result["anchors"]) 
    print("\nBuckets:\n", result["buckets"]) 
    print("\nExpanded Queries:") 
    for x in result["queries"]: 
        print("-", x) 
    print("\nVectors:", len(result["vectors"])) 
 

    # Log as next turn to JSONL and Excel 
    next_turn = compute_next_turn(LOG_PATH_JSONL) 
    append_log_jsonl(next_turn, q, result, role="User") 
    log_to_excel_from_result(next_turn, q, result, role="User", 
xlsx_path=LOG_PATH_XLSX) 
 
def process_batch(questions: List[str], start_turn: int = None): 
    """ 
    Processes a list of user questions and appends each result to JSONL and 
Excel. 
    """ 
    turn = start_turn if start_turn is not None else compute_next_turn(LOG_PATH_JSONL)
    for q in questions: 
        q = q.strip() 
        if not q: 
            continue 
        result = expand_and_embed(q) 
        append_log_jsonl(turn, q, result, role="User") 
        log_to_excel_from_result(turn, q, result, role="User", xlsx_path=LOG_PATH_XLSX)
        turn += 1 
 
if __name__ == "__main__": 
    # ----- Option A: interactive single-run ----- 
    # process_single_cli() 
 
    # ----- Option B: batch process your list ----- 
    questions = [ 
        "What is the capital of France?", 
        "How do I make a cup of tea?", 
        "Explain quantum computing in simple terms.", 
    ] 
    process_batch(questions) 
    # If you prefer to rebuild Excel from JSONL at the end instead of per-row: 
    # export_jsonl_to_excel(jsonl_path=LOG_PATH_JSONL, xlsx_path=LOG_PATH_XLSX, include_vectors=False)
 
