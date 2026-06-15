# Query validation logic 
import os 
import json 
import re 
from typing import List, Dict, Any 

from datetime import datetime, timezone 
from dotenv import load_dotenv 
from openai import AzureOpenAI 
from sentence_transformers import SentenceTransformer 
import pandas as pd 
 
# ===================================================== 
# CONFIG 
# ===================================================== 
ANCHOR_PATH = "data/query_anchors.json" 
MAX_ANCHORS = 25 
MAX_VARIANTS = 3 
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 
LOG_PATH_JSONL = "logs/query_expansion.jsonl" 
LOG_PATH_XLSX = "outputs/query_expansion.xlsx" 
 
# ===================================================== 
# AZURE OPENAI 
# ===================================================== 
load_dotenv() 
client = AzureOpenAI( 
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_KEY"), 
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"), 
    timeout=20.0, 
) 
MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT") 
if not client or not MODEL: 
    raise RuntimeError("Azure OpenAI configuration missing") 
 
# ===================================================== 
# EMBEDDING MODEL 
# ===================================================== 
_embedder = None 
 
def load_embedder(): 
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
            return json.load(f) 
    except (FileNotFoundError, json.JSONDecodeError): 
        return {} 

ANCHORS = load_anchors() 
 
# ===================================================== 
# ANCHOR SELECTION 
# ===================================================== 
def select_anchors(query: str, anchors: dict): 
    q_tokens = tokenize(query) 
    matched_buckets = set() 
    selected_terms = [] 
    for bucket, data in anchors.items(): 
        syns = [normalize(s) for s in data.get("synonyms", [])] 
        phrases = [normalize(p) for p in data.get("phrases", [])] 
        if any(s in q_tokens for s in syns): 
            matched_buckets.add(bucket) 
        elif any(re.search(rf"\b{re.escape(p)}\b", normalize(query)) for p in 
phrases): 
            matched_buckets.add(bucket) 
    for b in matched_buckets: 
        d = anchors[b] 
        selected_terms.append(d.get("canonical", "")) 
        selected_terms.extend(d.get("phrases", [])[:3]) 
        selected_terms.extend(d.get("synonyms", [])[:2]) 
    seen, clean = set(), [] 
    for t in selected_terms: 
        t = normalize(t) 
        if t and t not in seen: 
            seen.add(t) 
            clean.append(t) 
    return clean[:MAX_ANCHORS], list(matched_buckets) 
 
# ===================================================== 
# PROMPT BUILDER: 
# ===================================================== 
 

def build_unified_prompt(query: str, anchors: List[str], max_variants: int = 3, ontology_config=None) -> List[dict]: 
    """ 
    Unified, production-grade prompt: 
    - Stage 1: Validate active domain relevance and clarity. 
    - Stage 2: Expand into formal, concise policy search queries (translator behavior). 
    - Deterministic JSON output with hard constraints. 
    - Uses approved policy terms when relevant; no hallucinations. 
    """ 
    # Normalize anchors: strip, dedupe while preserving order, cap to 15 for token discipline
    seen = set() 
    clean_anchors = [] 
    for a in anchors or []: 
        if not a or not isinstance(a, str): 
            continue 
        s = a.strip() 
        if s and s not in seen: 
            seen.add(s) 
            clean_anchors.append(s) 
    anchor_text = ", ".join(clean_anchors[:15]) if clean_anchors else "[No domain terms matched]"
 
    if ontology_config:
        domain_display_name = ontology_config.display_name
        validation_scope = ontology_config.query_expansion.validation_scope
        domain_context = ontology_config.query_expansion.domain_context
    else:
        domain_display_name = "Medicare"
        validation_scope = "healthcare, Medicare, or health insurance contexts"
        domain_context = "Medicare or healthcare policy documentation"

    sys = (
        f"You are a {domain_display_name} policy query validator and expander that rewrites informal member questions "
        f"into formal, concise {domain_display_name} policy search queries.\n\n"
        "Behavioral contract:\n"
        "- Perform two steps, in order: (1) Validate the input query; (2) If and only if valid, generate formal search expansions.\n"
        "- Output must be a single valid JSON object that conforms exactly to the specified schema.\n"
        "- Return ONLY JSON. No prose, no comments, no markdown.\n"
        "- Use approved policy terms when relevant; do not force irrelevant terms.\n"
        "- Preserve original intent; do not introduce new medical or policy concepts.\n"
        f"- Do not infer {domain_display_name} intent unless explicit in the user query or approved policy terms.\n"
        "- Be conservative: if meaning is unclear or off-topic, mark invalid.\n\n"
        "JSON schema (must match exactly):\n"
        "{\n"
        '  "validation": {\n'
        '    "is_valid": true or false,\n'
        '    "reason": string or null,\n'
        '    "confidence": "high" or "medium" or "low",\n'
        '    "intent": string or null\n'
        "  },\n"
        '  "queries": array of strings\n'
        "}\n\n"
        "Hard constraints:\n"
        f"- If validation.is_valid = false -> queries must be [].\n"
        f"- If validation.is_valid = true -> queries must contain exactly {max_variants} strings.\n"
        '- "reason" must be a brief explanation when invalid; must be null when valid.\n'
        '- "intent" must be a brief description when valid; must be null when invalid.\n'
        "- JSON must be syntactically valid (no trailing commas; no extra fields)."
    )
 
    user = (
        f'INPUT QUERY:\n"{query}"\n\n'
        f"APPROVED POLICY TERMS:\n{anchor_text}\n\n"
        "STAGE 1 - VALIDATION\n"
        "Reject as invalid if any of the following apply:\n"
        '- Only emojis, symbols, random strings, or mostly noise (e.g., "???", "asdf%%%").\n'
        f"- Unrelated to {validation_scope}.\n"
        "- Contains hate, harmful, violent, sexually explicit, or otherwise unsafe content.\n"
        "- Is a bare URL, URL-like string, or code snippet with no meaningful query intent.\n"
        f"- Too ambiguous to determine {domain_display_name}-related intent without guessing.\n\n"
        "Accept as valid if all of the following are true:\n"
        f"- Relates to {domain_display_name} coverage/benefits, eligibility, costs, billing, policy rules, or clearly references relevant services/providers.\n"
        "- Intent is understandable without guessing or inventing context.\n"
        f"- Can be answered using {domain_context}.\n\n"
        "STAGE 2 - EXPANSION (apply only if valid)\n"
        f"Generate exactly {max_variants} short, formal, policy-oriented search queries that:\n"
        "- Maintain semantic equivalence with the user's intent.\n"
        "- Naturally incorporate APPROVED POLICY TERMS when relevant.\n"
        "- Are concise (aim 5-12 words), unambiguous, and optimized for semantic/document search.\n"
        "- Do not add new medical, legal, or policy concepts; do not hallucinate.\n\n"
        "OUTPUT\n"
        "Return one JSON object matching the schema defined in the system message. No text outside the JSON object."
    )
 
    return [ 
        {"role": "system", "content": sys}, 
        {"role": "user", "content": user}, 
    ] 
# ===================================================== 
# LLM CALL WITH PRE-VALIDATION 
# ===================================================== 
def validate_and_expand_with_llm(query: str, anchors: List[str], ontology_config=None) -> Dict[str, Any]: 
    """ 
    Validates and expands query with pre-flight checks. 
    """ 
    stripped = query.strip() 
     
    # Pre-flight checks (optimize API costs) 
    if len(stripped) < 3: 
        return { 
            "validation": { 
                "is_valid": False, 
                "reason": "Query too short (<2 characters)", 
                "confidence": "high", 
                "intent": None 
            }, 
            "queries": [] 
        } 
     
    if not re.search(r'[a-zA-Z]{2,}', stripped): 
        return { 
            "validation": { 
                "is_valid": False, 
                "reason": "No meaningful words detected", 
                "confidence": "high", 
                "intent": None 
            }, 
            "queries": [] 
        } 
     
    if re.match(r'^https?://\S+$', stripped): 
        return { 
            "validation": { 
                "is_valid": False, 
                "reason": "URL-only input", 
                "confidence": "high", 
                "intent": None 
            }, 
            "queries": [] 
 
        } 
     
    # LLM validation + expansion 
    messages = build_unified_prompt(query, anchors, ontology_config=ontology_config) 

     
    try: 
        resp = client.chat.completions.create( 
            model=MODEL, 
            messages=messages, 
            temperature=0, 
            response_format={"type": "json_object"}, 
            max_tokens=400, 
        ) 
        content = resp.choices[0].message.content.strip() 
    except Exception: 
        try: 
            resp = client.chat.completions.create( 
                model=MODEL, 
                messages=messages, 
                temperature=0, 
                max_tokens=400 
            ) 
            content = resp.choices[0].message.content.strip() 
        except Exception as e: 
            return { 
                "validation": { 
                    "is_valid": False, 
                    "reason": f"API error: {str(e)[:80]}", 
                    "confidence": "low", 
                    "intent": None 
                }, 
                "queries": [] 
            } 
     
    # Parse response 
    content = content.strip("` \n") 
    if content.startswith("json"): 
        content = content[4:].strip() 
     
    try: 
        result = json.loads(content) 
        validation = result.get("validation", {}) 
        queries = result.get("queries", []) 
         
        return { 
            "validation": { 
                "is_valid": validation.get("is_valid", False), 
                "reason": validation.get("reason"), 

                "confidence": validation.get("confidence", "medium"), 
                "intent": validation.get("intent") 
            }, 
            "queries": [q for q in queries if isinstance(q, str) and 
q.strip()][:MAX_VARIANTS] 
        } 
    except json.JSONDecodeError as e: 
        return { 
            "validation": { 
                "is_valid": False, 
                "reason": f"Response parsing failed: {str(e)[:80]}", 
                "confidence": "low", 
                "intent": None 
            }, 
            "queries": [] 
        } 
 
# ===================================================== 
# MAIN PIPELINE 
# ===================================================== 
def expand_and_embed(query: str, ontology_config=None): 
    """ 
    Complete pipeline: validation → expansion → embedding. 
    """ 
    if ontology_config is None:
        try:
            from domain_aware_rag.ontology.registry import OntologyRegistry
            ontology_config = OntologyRegistry().get_active_ontology()
        except Exception:
            ontology_config = None

    anchors_data = ontology_config.anchor_taxonomy if ontology_config else ANCHORS
    if not anchors_data:
        anchors_data = ANCHORS

    anchor_terms, buckets = select_anchors(query, anchors_data) 
    llm_result = validate_and_expand_with_llm(query, anchor_terms, ontology_config=ontology_config) 
     
    validation = llm_result["validation"] 
    if not validation["is_valid"]: 
        return { 
            "queries": [], 
            "vectors": [], 
            "anchors": anchor_terms, 
            "buckets": buckets, 
            "validation": validation, 
            "rejected": True 
        } 
     
    llm_variants = llm_result["queries"] 
    queries = [query] + llm_variants 
    queries = list(dict.fromkeys(queries))[:MAX_VARIANTS + 1] 
     
    vectors = embed_texts(queries) 
     
    return { 
        "queries": queries, 
        "vectors": vectors, 
        "anchors": anchor_terms, 
        "buckets": buckets, 
        "validation": validation, 
        "rejected": False 
    } 
 
# ===================================================== 
# LOGGING 
# ===================================================== 
def append_log_jsonl(turn_id: int, user_text: str, result: Dict[str, Any], 
role: str = "User"): 
    entry = { 
        "turn": turn_id, 
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), 
        "role": role, 
        "user_query": user_text, 
        "rejected": result.get("rejected", False), 
        "validation": result.get("validation", {}), 
        "anchors": result.get("anchors", []), 
        "buckets": result.get("buckets", []), 
        "expanded_queries": result.get("queries", []), 
        "vectors": result.get("vectors", []), 
    } 
    with open(LOG_PATH_JSONL, "a", encoding="utf-8") as f: 
        f.write(json.dumps(entry, ensure_ascii=False)) 
        f.write("\n") 
 
def append_row_to_excel(xlsx_path: str, rows: List[Dict[str, Any]]) -> None: 
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
    validation = result.get("validation", {}) 
     
    row = { 
        "turn": turn_id, 
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), 
        "role": role, 
        "user_query": user_text, 
        "valid": not result.get("rejected", False), 
        "rejection_reason": validation.get("reason"), 
        "confidence": validation.get("confidence"), 
        "detected_intent": validation.get("intent"), 
        "anchors": ", ".join(result.get("anchors", [])[:10]), 
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
             
            validation = obj.get("validation", {}) 

            row = { 
                "turn": obj.get("turn"), 
                "timestamp_utc": obj.get("timestamp_utc"), 
                "role": obj.get("role"), 
                "user_query": obj.get("user_query"), 
                "valid": not obj.get("rejected", False), 
                "rejection_reason": validation.get("reason"), 
                "confidence": validation.get("confidence"), 
                "detected_intent": validation.get("intent"), 
                "anchors": ", ".join(obj.get("anchors", [])[:10]), 
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
    print(f"✅ Exported {len(df)} rows to {xlsx_path}") 
 
def compute_next_turn(jsonl_path: str = LOG_PATH_JSONL) -> int: 
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
     
    print("\n" + "="*70) 
    validation = result.get("validation", {}) 
     
    if result.get("rejected"): 
        print("❌ REJECTED") 
        print(f"   Reason: {validation.get('reason')}") 
        print(f"   Confidence: {validation.get('confidence')}") 
    else: 
        print("✅ VALIDATED") 
        print(f"   Intent: {validation.get('intent')}") 
        print(f"   Confidence: {validation.get('confidence')}") 
        print(f"\n📋 Expanded Queries ({len(result['queries'])}):") 
        for i, x in enumerate(result["queries"], 1): 
            print(f"   {i}. {x}") 
        print(f"\n🎯 Anchors: {len(result['anchors'])} terms matched") 
        print(f"🔢 Vectors: {len(result['vectors'])} x {len(result['vectors'][0]) if result['vectors'] else 0}D")
     
    print("="*70 + "\n") 
     
    next_turn = compute_next_turn(LOG_PATH_JSONL) 
    append_log_jsonl(next_turn, q, result, role="User") 
    log_to_excel_from_result(next_turn, q, result, role="User", xlsx_path=LOG_PATH_XLSX)
 
def process_batch(questions: List[str], start_turn: int = None): 
    turn = start_turn if start_turn is not None else compute_next_turn(LOG_PATH_JSONL)
    stats = {"total": 0, "valid": 0, "rejected": 0, "high_conf": 0} 
     
    print(f"\n{'='*70}") 
    print(f"BATCH PROCESSING: {len(questions)} queries") 
    print(f"{'='*70}\n") 
     
    for q in questions: 
        q = q.strip() 
        if not q: 
            continue 
         
        stats["total"] += 1 
        result = expand_and_embed(q) 
         
        validation = result.get("validation", {}) 

        if result.get("rejected"): 
            stats["rejected"] += 1 
            reason = validation.get("reason", "unknown") 
            print(f"❌ T{turn}: {q[:50]}... → {reason}") 
        else: 
            stats["valid"] += 1 
            if validation.get("confidence") == "high": 
                stats["high_conf"] += 1 
         
        append_log_jsonl(turn, q, result, role="User") 
        log_to_excel_from_result(turn, q, result, role="User", xlsx_path=LOG_PATH_XLSX)
        turn += 1 
     
    print(f"\n{'='*70}") 
    print(f"✅ COMPLETE: {stats['valid']}/{stats['total']} valid ({stats['valid']/stats['total']*100:.1f}%)")
    if stats['valid'] > 0: 
        print(f"   High confidence: {stats['high_conf']}/{stats['valid']} ({stats['high_conf']/stats['valid']*100:.1f}%)")
    print(f"   Rejected: {stats['rejected']} ({stats['rejected']/stats['total']*100:.1f}%)")
    print(f"{'='*70}\n") 
 
if __name__ == "__main__":
    QUESTIONS = [
        "How does the plan coordinate with my other insurance?",
        "What are the eligibility requirements to enroll?",
        "What happens if I move out of the plan service area?",
    ]
    process_batch(QUESTIONS)
    print("Done!")
