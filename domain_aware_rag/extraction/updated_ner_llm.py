import os 
import json 
from collections import defaultdict 
from typing import Dict, Any, List, Tuple 
from pathlib import Path
import pandas as pd 
from dotenv import load_dotenv 
from openai import AzureOpenAI 
 
############################################################### 
# 1) PATHS 
############################################################### 
 
# Default paths (for backward compatibility; will be overridden by function parameters) 
DEFAULT_LEVELLER_JSON_PATH = "outputs/2026-pc65-medical-only-eoc.custom.json" 
DEFAULT_CATALOGUE_PATH     = "data/entity_catalogue_table_V2.xlsx" 
DEFAULT_OUTPUT_JSON        = "outputs/entity_output_updated_103.json" 

DEFAULT_KEYWORDS_JSON      = "outputs/entity_output_keywords_103.json" 
 
# These will be set by main() function 
LEVELLER_JSON_PATH = DEFAULT_LEVELLER_JSON_PATH 
CATALOGUE_PATH     = DEFAULT_CATALOGUE_PATH 
OUTPUT_JSON        = DEFAULT_OUTPUT_JSON 
KEYWORDS_JSON      = DEFAULT_KEYWORDS_JSON 
 
############################################################### 
# 2) AZURE OPENAI CLIENT 
############################################################### 
# Load .env from parent directory
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path) 


client = AzureOpenAI( 
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_KEY"), 
    api_version=os.getenv("AZURE_OPENAI_API_VERSION") ) 

MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT")

if not MODEL:
    raise RuntimeError("Azure OpenAI deployment name not found in environment variables") 
 
############################################################### 
# 3) LOAD ENTITY CATALOGUE 
############################################################### 
 
def load_catalogue(path: str) -> List[Dict[str, str]]:
    """
    V2 loader: read only the 4 fields needed for context and level.
      - 'Entity Category (High Level)'  -> used as level
      - 'Entity Sub Category'
      - 'Entity Name'
      - 'Description (from EOC)'
    
    Returns empty list if file doesn't exist.
    """
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except FileNotFoundError:
        print(f"[WARNING] Catalogue file not found: {path}. Using empty catalogue.")
        return []
    except Exception as e:
        print(f"[ERROR] Failed to load catalogue {path}: {e}. Using empty catalogue.")
        return []

    # Keep only needed columns, clean whitespace
    cols = [
        "Entity Category (High Level)",
        "Entity Sub Category",
        "Entity Name",
        "Description (from EOC)"
    ]
    for c in cols:
        if c not in df.columns:
            print(f"[WARNING] Missing column in catalogue: {c}. Skipping catalogue.")
            return []

    df = df[cols].copy()
    for c in cols:
        df[c] = df[c].astype(str).str.strip()

    # Drop empty names
    df = df[df["Entity Name"].str.len() > 0]

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "entity_category_high": r["Entity Category (High Level)"],  # will become level
            "entity_sub_category": r["Entity Sub Category"],
            "entity_name": r["Entity Name"],
            "description": r["Description (from EOC)"]
        })
    return rows 
############################################################### 
# 4) LOAD LEVELLER JSON 
############################################################### 
 
def load_leveller(path: str) -> Dict[str, Any]: 
    with open(path, "r", encoding="utf-8") as f: 
        return json.load(f) 
 
############################################################### 
# 5) FULLY RECURSIVE LEAF SCOPE EXTRACTION 
############################################################### 
 
def extract_scopes(node: Dict[str, Any], path=None): 
    """ 
    Walk the JSON tree and collect scopes at the lowest nodes that contain text. 
    scope = { parent_section, scope_id, text } 
    """ 
    if path is None: 
        path = [] 
 
    scopes = [] 
    current_text = node.get("content", {}).get("text", "").strip() 
    child_has_text = False 
 
    for key, value in node.items(): 
        if key == "content": 
            continue 
 

        if isinstance(value, dict): 
            # Flatten any "level_*" wrappers 
            if key.lower().startswith("level_"): 
                scopes.extend(extract_scopes(value, path)) 
            else: 
                child_scopes = extract_scopes(value, path + [key]) 
                if child_scopes: 
                    child_has_text = True 
                scopes.extend(child_scopes) 
 
    # If this node has text and no child with text, treat as leaf scope 
    if current_text and not child_has_text: 
        scopes.append({ 
            "parent_section": path[1] if len(path) > 1 else "", 
            "scope_id": path[-1] if path else "", 
            "text": current_text 
        }) 
 
    return scopes 
 
############################################################### 
# 6) GROUP SCOPES BY SAME PARENT SECTION (SAFE BATCHING) 
############################################################### 
 
def group_scopes(scopes: List[Dict]): 
    grouped = defaultdict(list) 
    for s in scopes: 
        grouped[s["parent_section"]].append(s) 
    return grouped 
 
############################################################### 
# 7) BATCHED GPT CALL (explicit rules, plus entity_name) 
############################################################### 
def scopes_without_entities(all_scopes: List[Dict], entity_levels: Dict[str, Any]) -> List[Dict]: 
    """ 
    Compare all leaf scopes vs scope_ids present in the entity output. 
    Return list of scope dicts (parent_section, scope_id, text) that have zero entities. 
    """ 
    # All scope_ids from leveller (true leafs) 
    scope_index = {s["scope_id"]: s for s in all_scopes} 
 
    # Collect scope_ids that appeared in output under any level/entity_name 
    seen = set() 
    for lvl, ent_map in (entity_levels or {}).items(): 
        for ent_name, sec_map in (ent_map or {}).items(): 
            for sid in (sec_map or {}).keys(): 

                # Your aggregator sometimes disambiguates duplicates as "Section X (2)"; strip that suffix 
                base = sid.split(" (")[0].strip() 
                seen.add(base) 
 
    # Missing = leaf scopes that never showed up in the entity output 
    missing = [scope_index[sid] for sid in scope_index.keys() if sid not in seen] 
    return missing 
 
def call_gpt_batched(parent_section: str, scope_batch: List[Dict], catalogue_json: str, ontology_config=None): 
    """ 
    Batches scopes and prompts the LLM to: 
      - Use entity_name EXACTLY as in the catalogue. 
      - Use entity_value ONLY from the scope text. 
      - Return 'level' EXACTLY as the catalogue's "Entity Category (High Level)" for that entity. 
    """ 
 
    scopes_payload = [{"scope_id": s["scope_id"], "text": s["text"]} for s in scope_batch] 
    scopes_json = json.dumps(scopes_payload, ensure_ascii=False) 
 
    schema_obj = [ 
        { 
            "scope_id": "...", 
            "entities": [ 
                { 
                    "entity_name": "...",             # exact "Entity Name" (catalogue) 
                    "entity_category": "...",         # you may use Entity Sub Category here 
                    "entity_value": "...",            # extracted from scope text only 
                    "level": "Plan"                   # MUST equal 'Entity Category (High Level)' from catalogue 
                } 
            ] 
        } 
    ] 
    schema_json = json.dumps(schema_obj, ensure_ascii=False, indent=2) 
 
    # Load examples dynamically if available
    if ontology_config and ontology_config.entity_examples:
        examples_str = ""
        for idx, ex in enumerate(ontology_config.entity_examples, 1):
            ex_json = json.dumps(ex, ensure_ascii=False, indent=2)
            entity_name = ex.get("expected_entity", {}).get("entity_name", "Example")
            level = ex.get("expected_entity", {}).get("level", "Plan")
            examples_str += f"EXAMPLE {idx} - {entity_name} (High Level={level})\n{ex_json}\n\n"
    else:
        # Examples aligned to V2 (High Level comes from the first column) 
        ex1 = { 
            "catalogue": { 
                "Entity Category (High Level)": "Payer", 
                "Entity Sub Category": "Medicare (CMS)", 
                "Entity Name": "Medicare", 
                "Description (from EOC)": "Federal agency overseeing Medicare; benefits, plan comparisons, complaints." 
            }, 
            "scope_text_contains": "... contact Medicare ...", 
            "expected_entity": { 
                "entity_name": "Medicare", 
                "entity_category": "Medicare (CMS)", 
                "entity_value": "Medicare", 
                "level": "Payer" 
            } 
        } 
        ex2 = { 
            "catalogue": { 
                "Entity Category (High Level)": "Appeals and Grievances", 
                "Entity Sub Category": "Appeals", 
                "Entity Name": "Appeals", 
                "Description (from EOC)": "Multi-level appeals with standard/fast timelines." 
            }, 
            "scope_text_contains": "... you can make an appeal ...", 
            "expected_entity": { 
                "entity_name": "Appeals", 
                "entity_category": "Appeals", 
                "entity_value": "appeal", 
                "level": "Appeals and Grievances" 
            } 
        } 
        ex3 = { 
            "catalogue": { 
                "Entity Category (High Level)": "Provider", 
                "Entity Sub Category": "Provider Directory", 
                "Entity Name": "Provider/Pharmacy Directory", 
                "Description (from EOC)": "Online and print directory of in-network providers, facilities, and DME suppliers." 
            }, 
            "scope_text_contains": "... see our Provider/Pharmacy Directory ...", 
            "expected_entity": { 
                "entity_name": "Provider/Pharmacy Directory", 
                "entity_category": "Provider Directory", 
                "entity_value": "Provider/Pharmacy Directory", 
                "level": "Provider" 
            } 
        } 
        ex4 = { 
            "catalogue": { 
                "Entity Category (High Level)": "Plan", 
                "Entity Sub Category": "Plan Premium", 
                "Entity Name": "Monthly Plan Premium", 
                "Description (from EOC)": "Monthly plan premium owed to the plan (separate from Part B premium)." 
            }, 
            "scope_text_contains": "For 2026, your monthly plan premium is $119.", 
            "expected_entity": { 
                "entity_name": "Monthly Plan Premium", 
                "entity_category": "Plan Premium", 
                "entity_value": "$119", 
                "level": "Plan" 
            } 
        } 
     
        ex1_json = json.dumps(ex1, ensure_ascii=False, indent=2) 
        ex2_json = json.dumps(ex2, ensure_ascii=False, indent=2) 
        ex3_json = json.dumps(ex3, ensure_ascii=False, indent=2) 
        ex4_json = json.dumps(ex4, ensure_ascii=False, indent=2) 

        examples_str = (
            "EXAMPLE 1 - Medicare (High Level=Payer)\n" + ex1_json + "\n\n" 
            "EXAMPLE 2 - Appeals (High Level=Appeals and Grievances)\n" + ex2_json + "\n\n" 
            "EXAMPLE 3 - Provider/Pharmacy Directory (High Level=Provider)\n" + ex3_json + "\n\n" 
            "EXAMPLE 4 - Monthly Plan Premium (High Level=Plan)\n" + ex4_json + "\n\n"
        )
 
    # PROMPT - only 4 columns are sent as catalogue context; level is taken from the first column 
    prompt = ( 
        "You are an expert policy and regulatory analyst.\n\n" 
        "Goal\n" 
        "For EACH scope independently, extract entities that appear in the catalogue provided below (only 4 fields).\n" 
        "For every returned entity:\n" 
        "  • Use entity_name EXACTLY as it appears in the catalogue (no paraphrasing).\n" 
        "  • Use entity_category to reflect the catalogue's 'Entity Sub Category' for the matched row.\n" 
        "  • Set level EXACTLY to the catalogue's 'Entity Category (High Level)' for the matched row (DO NOT infer).\n" 
        "  • Extract entity_value ONLY from the SCOPE TEXT (never from the catalogue description). Use the exact surface string(s).\n\n" 
        "Return STRICT JSON in this exact format:\n" + schema_json + "\n\n" 
        "CATALOGUE CONTEXT (ONLY these fields):\n" 
        "  - Entity Category (High Level)\n" 
        "  - Entity Sub Category\n" 
        "  - Entity Name\n" 
        "  - Description (from EOC)\n\n" 
        "CATALOGUE_ROWS:\n" + catalogue_json + "\n\n" 
        "SCOPES:\n" + scopes_json + "\n\n" 
        "------------------------------------------------------------\n" 
        "VALUE EXTRACTION RULES\n" 
        "- entity_value MUST be extracted from the SCOPE TEXT.\n" 
        "- Capture exact surface strings: currency ($119), percentages (50%), dates/periods (2026, days 1-6), names/titles (Medicare).\n" 
        "- Do NOT copy the catalogue description as the value.\n" 

        "- If the scope references the concept but no explicit value appears, you MAY use a concise canonical label from the catalogue as a fallback.\n\n" 
        "------------------------------------------------------------\n" 
        "LEVEL RULE (MANDATORY)\n" 
        "- Set 'level' to the catalogue's 'Entity Category (High Level)' for the matched entity_name. Do NOT infer.\n\n" 
        "------------------------------------------------------------\n" 
        "ILLUSTRATIVE EXAMPLES (aligned to the catalogue's 'High Level')\n\n" 
        + examples_str +
        "------------------------------------------------------------\n" 
        "OUTPUT HYGIENE\n" 
        "- Treat each scope independently; do not mix content across scopes.\n" 
        "- Ensure uniqueness per scope; no duplicate entities.\n" 
        "- If no matches for a scope, return: \"entities\": [].\n" 
    ) 
 
    resp = client.chat.completions.create( 
        model=MODEL, 
        messages=[{"role": "user", "content": prompt}], 
        temperature=0 
    ) 
     
    # DEBUG: Print LLM response 
    llm_response = resp.choices[0].message.content 
    print(f"\n[DEBUG] Entity Extraction LLM Response:") 
    print(f"Raw response (first 500 chars):") 
    try: 
        print(llm_response[:500]) 
    except UnicodeEncodeError: 
        print(llm_response[:500].encode('utf-8', 
errors='replace').decode('utf-8')) 
    print(f"Full response:") 
    try: 
        print(llm_response) 
    except UnicodeEncodeError: 
        print(llm_response.encode('utf-8', errors='replace').decode('utf-8')) 
    
    # Strip code block markers if present
    if llm_response.startswith('```json'):
        llm_response = llm_response[7:]  # Remove ```json
    if llm_response.startswith('```'):
        llm_response = llm_response[3:]   # Remove ```
    if llm_response.endswith('```'):
        llm_response = llm_response[:-3]  # Remove ending ```
    llm_response = llm_response.strip()
    
    try: 
        result = json.loads(llm_response) 
        print(f"[DEBUG] Parsed JSON successfully. Entities found:") 
        if isinstance(result, list): 

            for item in result: 
                if isinstance(item, dict): 
                    entities_count = len(item.get('entities', [])) 
                    print(f"  - {item.get('scope_id', 'unknown')}: {entities_count} entities") 
        return result 
    except Exception as e: 
        print(f"[DEBUG] Failed to parse JSON: {e}") 
        return [] 
############################################################### 
# 8) LEVEL NORMALIZATION 
############################################################### 
def call_gpt_keywords(parent_section: str, scope_batch: List[Dict], *, top_k: int = 12, ontology_config=None): 
    """ 
    For each scope in the batch, extract ranked keyword phrases from SCOPE 
    TEXT ONLY. 
    Returns a list of dicts: [{ "scope_id": "...", "keywords": ["...", ...] }, ...] 
    """ 
    # ----- Payload ----- 
    scopes_payload = [{"scope_id": s["scope_id"], "text": s["text"]} for s in scope_batch if (s.get("text") or "").strip()] 
    if not scopes_payload: 
        return [] 
 
    scopes_json = json.dumps(scopes_payload, ensure_ascii=False) 
 
    # ----- Output schema ----- 
    # One entry per scope, strictly: 
    # { 
    #   "scope_id": "<scope id as provided>", 
    #   "keywords": ["keyword phrase 1", "keyword phrase 2", ...]   # 2-6 words, lowercase, ranked 
    # } 
    schema_obj = [ 
        { 
            "scope_id": "...", 
            "keywords": ["...", "..."] 
        } 
    ] 
    schema_json = json.dumps(schema_obj, ensure_ascii=False, indent=2) 
 
    if ontology_config:
        domain_display_name = ontology_config.display_name
        acronyms = ontology_config.keyword_acronyms
    else:
        domain_display_name = "Medicare Advantage Evidence of Coverage (EOC)"
        acronyms = ["eft", "qio", "rrb", "cms", "msp", "moop", "hmo", "ppo", "dme", "esrd"]

    # ----- Prompt (domain-aware, robust, not overfitted) ----- 
    prompt = ( 
        f"You are a domain-aware keyword extractor for {domain_display_name} documents.\n\n" 
        "TASK\n" 
        f"For EACH scope independently, extract the TOP {top_k} KEY DOMAIN PHRASES from THE SCOPE TEXT ONLY.\n" 
        "The goal is to capture phrases that best represent the meaning of the scope for retrieval.\n\n" 
        "QUALITY BAR (Mandatory):\n" 
        "- Phrases MUST be 2-6 words long (except standardized acronyms).\n" 
        "- Phrases MUST reflect domain meaning: rules, requirements, timelines, limits, deadlines, processes, exceptions, coverage boundaries, costs, payments, entities/agencies.\n" 
        "- Prioritize:\n" 
        "  • SERVICE TYPES or Core domain subjects\n" 
        "  • COST indicators (copay, coinsurance, deductibles, premium, etc.)\n" 
        "  • TEMPORAL qualifiers (per day, per year, monthly, benefit periods, etc.)\n" 
        "  • obligations & conditions (prior authorization, notification, eligibility rules)\n" 
        "  • rights & processes (appeals, filing complaints, claims)\n" 
        "  • exclusions or limitations (not covered, exclusions, limitations)\n" 
        "- Phrases MUST be surface strings or precise compressions from the SAME sentence.\n" 
        "- DO NOT invent terms; DO NOT use description-like generalizations.\n" 
        "- All output MUST be lowercase, no trailing punctuation.\n" 
        "- Deduplicate within a scope; rank by importance (most important first).\n\n" 
        "FORMAT (STRICT JSON):\n" 
        f"{schema_json}\n\n" 
        "SCOPES:\n" 
        f"{scopes_json}\n" 
    ) 
 
    # ----- Call LLM ----- 
    resp = client.chat.completions.create( 
        model=MODEL, 
        messages=[{"role": "user", "content": prompt}], 
        temperature=0 
    ) 
 

    # DEBUG: Print LLM response for keywords 
    llm_response = resp.choices[0].message.content 
    print(f"\n[DEBUG] Keywords Extraction LLM Response:") 
    print(f"Raw response (first 500 chars):") 
    try: 
        print(llm_response[:500]) 
    except UnicodeEncodeError: 
        print(llm_response[:500].encode('utf-8', 
errors='replace').decode('utf-8')) 
 
    # Strip code block markers if present
    if llm_response.startswith('```json'):
        llm_response = llm_response[7:]  # Remove ```json
    if llm_response.startswith('```'):
        llm_response = llm_response[3:]   # Remove ```
    if llm_response.endswith('```'):
        llm_response = llm_response[:-3]  # Remove ending ```
    llm_response = llm_response.strip()

    # ----- Parse & validate ----- 
    try: 
        raw = json.loads(llm_response) 
        print(f"[DEBUG] Parsed keywords JSON successfully. Keywords found:") 
        if isinstance(raw, list): 
            for item in raw: 
                if isinstance(item, dict): 
                    kw_count = len(item.get('keywords', [])) 
                    print(f"  - {item.get('scope_id', 'unknown')}: {kw_count} keywords") 
    except Exception as e: 
        print(f"[DEBUG] Failed to parse keywords JSON: {e}") 
        return [] 
 
    if not isinstance(raw, list): 
        return [] 
 
    # Minimal sanitation to ensure we return exactly the expected structure 
    out = [] 
    for item in raw: 
        sid = (item or {}).get("scope_id") 
        kws = (item or {}).get("keywords") or [] 
        if not sid or not isinstance(kws, list): 
            continue 
 
        # Normalize & clean 
        seen = set() 
        cleaned = [] 
        for kw in kws: 
            k = (kw or "").strip().lower() 
            if not k: 
                continue 
            # remove trailing punctuation 
            while k and k[-1] in ",.;:!?" : 
                k = k[:-1].strip() 
            # allow standardized acronyms, otherwise enforce 2-6 tokens 
            token_len = len(k.split()) 
 
            allowed_acronyms = set(acronyms) if acronyms else {"eft", "qio", "rrb", "cms", "msp", "moop", "hmo", "ppo", "dme", "esrd"}
            if token_len == 1 and k not in allowed_acronyms: 
                continue 
            if token_len > 6: 
                continue 
            if k not in seen: 
                seen.add(k) 
                cleaned.append(k) 
 
        # Limit to top_k (already ranked by the model, we just cap) 
        cleaned = cleaned[:top_k] 
        out.append({"scope_id": sid, "keywords": cleaned}) 
 
    return out 
 
def normalize_level(raw: str, ontology_config=None) -> str: 
    """ 
    Normalize the level string returned by the model to one of the levels from ontology,
    or legacy: 'Org', 'Plan', 'Member'.
    """ 
    if not raw: 
        return "Plan" 
    s = raw.strip().lower() 
    
    if ontology_config and ontology_config.entity_levels:
        for lvl in ontology_config.entity_levels:
            lvl_lower = lvl.lower()
            if lvl_lower in s or s in lvl_lower:
                return lvl
        return ontology_config.entity_levels[0]

    if "org" in s or "organization" in s or "organisation" in s: 
        return "Org" 
    if "member" in s: 
        return "Member" 
    return "Plan" 
 
############################################################### 
# 9) MAIN PIPELINE (Level -> Entity Name -> {scope_id: value}) 
############################################################### 
def save_keywords_json(keywords_list: List[Dict], out_path: str): 
    """ 
    Consolidate into { "keywords": { scope_id: [ ... ] } } and write JSON. 
    """ 
    out = {"keywords": {}} 
    for item in keywords_list: 
        sid = (item or {}).get("scope_id") 
        kws = (item or {}).get("keywords") or [] 
        if not sid or not isinstance(kws, list): 
            continue 
        # Dedup & clean 
        seen = set() 
        cleaned = [] 
        for kw in kws: 

            k = (kw or "").strip().lower() 
            if k and k not in seen: 
                seen.add(k) 
                cleaned.append(k) 
        out["keywords"][sid] = cleaned 
    with open(out_path, "w", encoding="utf-8") as f: 
        json.dump(out, f, indent=2, ensure_ascii=False) 
 
def _merge_keywords_lists(base_list: List[Dict], extra_list: List[Dict]) -> List[Dict]: 
    """ 
    Merge two lists like [{"scope_id": "...", "keywords": [...]}, ...] by scope_id. 
    De-duplicates keywords and preserves order. 
    """ 
    from collections import OrderedDict 
 
    by_id: Dict[str, List[str]] = OrderedDict() 
    def _accumulate(lst): 
        for item in lst: 
            sid = (item or {}).get("scope_id") 
            kws = (item or {}).get("keywords") or [] 
            if not sid or not isinstance(kws, list): 
                continue 
            if sid not in by_id: 
                by_id[sid] = [] 
            seen = set(by_id[sid]) 
            for k in kws: 
                if k and k not in seen: 
                    seen.add(k) 
    _accumulate(base_list) 
    _accumulate(extra_list) 
 
    return [{"scope_id": sid, "keywords": kws} for sid, kws in by_id.items()] 
 
def main(leveller_json_path=None, catalogue_path=None, output_json_path=None, 
keywords_json_path=None, ontology_config=None): 
    """ 
    Main NER extraction pipeline. 
     
    Args: 
        leveller_json_path: Path to custom JSON from data_extraction (default: 
        DEFAULT_LEVELLER_JSON_PATH) 
        catalogue_path: Path to entity catalogue Excel (default: 
        DEFAULT_CATALOGUE_PATH) 
        output_json_path: Path for entity output JSON (default: 
        DEFAULT_OUTPUT_JSON) 
        keywords_json_path: Path for keywords output JSON (default: 
        DEFAULT_KEYWORDS_JSON) 
        ontology_config: Optional OntologyConfig object
    """ 
    # Set global paths from parameters or defaults 
    global LEVELLER_JSON_PATH, CATALOGUE_PATH, OUTPUT_JSON, KEYWORDS_JSON 
     
    LEVELLER_JSON_PATH = leveller_json_path or DEFAULT_LEVELLER_JSON_PATH 
    OUTPUT_JSON = output_json_path or DEFAULT_OUTPUT_JSON 
    KEYWORDS_JSON = keywords_json_path or DEFAULT_KEYWORDS_JSON 

    if ontology_config is None:
        try:
            from domain_aware_rag.ontology.registry import OntologyRegistry
            ontology_config = OntologyRegistry().get_active_ontology()
        except Exception:
            ontology_config = None

    if ontology_config and ontology_config.has_inline_catalogue():
        catalogue_rows = ontology_config.entities
        CATALOGUE_PATH = "inline"
    else:
        effective_path = (ontology_config.entity_catalogue_path if ontology_config else None) or catalogue_path or DEFAULT_CATALOGUE_PATH
        CATALOGUE_PATH = effective_path
        catalogue_rows = load_catalogue(CATALOGUE_PATH)
     
    print(f"[NER] Using inputs:") 
    print(f"  - Leveller JSON: {LEVELLER_JSON_PATH}") 
    print(f"  - Catalogue: {CATALOGUE_PATH}") 
    print(f"[NER] Outputs:") 
    print(f"  - Entities: {OUTPUT_JSON}") 
    print(f"  - Keywords: {KEYWORDS_JSON}") 
     
    # --- Load inputs 
    leveller = load_leveller(LEVELLER_JSON_PATH) 
    catalogue_json = json.dumps(catalogue_rows, ensure_ascii=False) 
 
    # Handle empty catalogue 
    if not catalogue_rows: 
        print("[WARNING] No catalogue entities available. Creating empty entity output.") 
        # Create empty output structure 
        output = { 
            "levels": {}, 
            "metadata": { 
                "catalogue_used": CATALOGUE_PATH, 
                "catalogue_size": 0, 
                "total_scopes": 0, 
                "processed_sections": 0 
            } 
        } 
        # Save empty outputs 
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f: 
            json.dump(output, f, indent=2, ensure_ascii=False) 
        with open(KEYWORDS_JSON, "w", encoding="utf-8") as f: 
            json.dump({"keywords": []}, f, indent=2, ensure_ascii=False) 
        print(f"[NER] Saved empty outputs due to missing catalogue.") 
        return 
 
    # Build lookups from Excel/Ontology (source of truth for Level) 
    NAME_TO_HIGH = {} 
    for r in catalogue_rows: 
        nm = r["entity_name"] 
        if nm not in NAME_TO_HIGH: 
            NAME_TO_HIGH[nm] = r["entity_category_high"] 
    VALID_NAMES = set(NAME_TO_HIGH.keys()) 
 
    # Build Level buckets dynamically (keeps output aligned with the catalogue) 
    LEVEL_ORDER = sorted({r["entity_category_high"] for r in catalogue_rows}) 
    output: Dict[str, Any] = {"levels": {lvl: defaultdict(dict) for lvl in LEVEL_ORDER}} 
 
    # Extract leaf scopes and group for batching 
    scopes = extract_scopes(leveller) 
    grouped = group_scopes(scopes) 
     
    print(f"\n[DEBUG] Total scopes extracted: {len(scopes)}") 
    print(f"[DEBUG] Grouped into {len(grouped)} sections") 
 
    # ------------------------- 
    # (1) ENTITY EXTRACTION 
    # ------------------------- 
    for parent_section, batch in grouped.items(): 
        response = call_gpt_batched(parent_section, batch, catalogue_json, ontology_config=ontology_config) 
 
        # Track duplicates per (level, entity_name, scope_key) 
        section_counter: Dict[Tuple[str, str, str], int] = defaultdict(int) 
 
        if not isinstance(response, list): 
            continue 
 
        for item in response: 
            scope_id = (item or {}).get("scope_id") 
            if not scope_id: 
                continue 
 
            entities = (item or {}).get("entities", []) or [] 
            if not isinstance(entities, list): 
                continue 
 
            for e in entities: 
                name = ((e or {}).get("entity_name") or "").strip() 
                val  = ((e or {}).get("entity_value") or "").strip() 
 
                # Require valid entity_name + value and membership in the catalogue 
                if not name or not val or name not in VALID_NAMES: 
                    continue 
 
                # Level comes ONLY from catalogue/ontology (Entity Category (High Level)) 
                lvl = NAME_TO_HIGH.get(name) 
                if not lvl: 
                    continue  # defensive: should not happen for valid names 
 
                # Base section key is the scope id (may get disambiguation suffix) 
                section_key = scope_id 
                section_counter[(lvl, name, section_key)] += 1 
                if section_counter[(lvl, name, section_key)] > 1: 
                    section_key = f"{section_key} ({section_counter[(lvl, name, section_key)]})" 
 
                output["levels"][lvl][name][section_key] = val 
 
    # Reorder levels to requested order & convert defaultdicts to dicts for clean JSON 
    clean_levels = {} 
    for lvl in LEVEL_ORDER: 
        ent_map = output["levels"][lvl] 
        clean_levels[lvl] = {ent_name: dict(entries) for ent_name, entries in sorted(ent_map.items(), key=lambda x: x[0].lower())} 
 
    final_output = {"levels": clean_levels} 
 
    # --- Write the entity extraction output 
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f: 
        json.dump(final_output, f, indent=2, ensure_ascii=False) 
 
    # ------------------------- 
    # (2) KEYWORDS FOR **ALL** SCOPES 
    # ------------------------- 
    all_keywords_accum = [] 
    for parent_section, batch in grouped.items(): 
        clean_batch = [s for s in batch if (s.get("text", "") or "").strip()] 
        if not clean_batch: 
            continue 
        # Use a slightly richer top_k for better coverage (you can tune 10-12) 
        kw_resp_all = call_gpt_keywords(parent_section, clean_batch, top_k=12, ontology_config=ontology_config) 
        if isinstance(kw_resp_all, list): 
            all_keywords_accum.extend(kw_resp_all) 
 
    # ------------------------- 
    # (3) (Optional) KEYWORDS FOR MISSING-ENTITY SCOPES (kept as safety net) 
    # ------------------------- 
    missing_scopes = scopes_without_entities(scopes, final_output.get("levels", {})) 
    missing_grouped = defaultdict(list) 
    for s in missing_scopes: 
        missing_grouped[s["parent_section"]].append(s) 
 
    missing_keywords_accum = [] 
    for parent_section, batch in missing_grouped.items(): 
        clean_batch = [s for s in batch if (s.get("text", "") or "").strip()] 
        if not clean_batch: 
            continue 
        kw_resp_missing = call_gpt_keywords(parent_section, clean_batch, top_k=12, ontology_config=ontology_config) 
        if isinstance(kw_resp_missing, list): 
            missing_keywords_accum.extend(kw_resp_missing) 
 
    # ------------------------- 
    # (4) MERGE keyword lists (ALL + MISSING) and WRITE once 
    # ------------------------- 
    merged_keywords = _merge_keywords_lists(all_keywords_accum, missing_keywords_accum) 
    save_keywords_json(merged_keywords, KEYWORDS_JSON) 
 
    print(f"DONE. Entities -> {OUTPUT_JSON}") 
    print(f"      Keywords (ALL scopes + missing-entity safety net) -> {KEYWORDS_JSON}") 
 
if __name__ == "__main__": 
    main()
