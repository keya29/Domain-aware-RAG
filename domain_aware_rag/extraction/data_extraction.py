import collections 
import collections.abc 
collections.MutableMapping = collections.abc.MutableMapping 
collections.Mapping = collections.abc.Mapping 
 
from pathlib import Path 
import json 
import pandas as pd 
from io import BytesIO 
 
from docling.document_converter import DocumentConverter 
from docling.datamodel.document import SectionHeaderItem, TableItem, TextItem 
from docling.datamodel.base_models import DocumentStream  # <-- IMPORTANT 
 
from hierarchical.postprocessor import ResultPostprocessor 
 
# --- Page extractor (unchanged) --- 
def get_page_number(item): 
    """ 
    Safely extract page number from Docling items. 
    Tries multiple locations because versions differ. 
    """ 
    # Direct attribute (preferred) 
    if hasattr(item, "page_no") and item.page_no is not None: 
        return item.page_no 
 
    # Provenance list (older results) 
    if hasattr(item, "prov") and item.prov: 
        prov0 = item.prov[0] 
        if hasattr(prov0, "page_no"): 
            return prov0.page_no 
 
    # Metadata (alternative) 
    if hasattr(item, "metadata") and hasattr(item.metadata, "page_no"): 
        return item.metadata.page_no 
 
    return None 
 

# --- Helpers to extract bbox robustly (headers often lack direct .bbox) --- 
def _to_plain_bbox(bb): 
    """Convert a bbox-like object to a plain dict {l,t,r,b,coord_origin} if 
present.""" 
    if bb is None: 
        return None 
    l = getattr(bb, "l", None) 
    t = getattr(bb, "t", None) 
    r = getattr(bb, "r", None) 
    b = getattr(bb, "b", None) 
    if l is None and t is None and r is None and b is None: 
        return None 
    coord_origin = getattr(bb, "coord_origin", None) 
    if coord_origin is not None and hasattr(coord_origin, "value"): 
        coord_origin = coord_origin.value 
    elif coord_origin is not None: 
        coord_origin = str(coord_origin) 
    return { 
        "l": float(l) if l is not None else None, 
        "t": float(t) if t is not None else None, 
        "r": float(r) if r is not None else None, 
        "b": float(b) if b is not None else None, 
        "coord_origin": coord_origin, 
    } 
 
def _union_plain_bboxes(bboxes, origin=None): 
    """Union a list of plain bbox dicts; ignores None entries.""" 
    xs_l, xs_r, ys_t, ys_b = [], [], [], [] 
    c_origin = origin 
    for bb in bboxes: 
        if not bb: 
            continue 
        if c_origin is None and bb.get("coord_origin") is not None: 
            c_origin = bb.get("coord_origin") 
        if bb.get("l") is not None: 
            xs_l.append(bb["l"]) 
        if bb.get("r") is not None: 
            xs_r.append(bb["r"]) 
        if bb.get("t") is not None: 
            ys_t.append(bb["t"]) 
        if bb.get("b") is not None: 
            ys_b.append(bb["b"]) 
    if not xs_l and not xs_r and not ys_t and not ys_b: 
        return None 
    # Envelope works regardless of origin convention 
    return { 
        "l": min(xs_l) if xs_l else None, 
        "r": max(xs_r) if xs_r else None, 

        "t": min(ys_t) if ys_t else None, 
        "b": max(ys_b) if ys_b else None, 
        "coord_origin": c_origin, 
    } 
 
def get_bbox(item): 
    """ 
    Return a plain dict with the item's bounding box if available: 
    {l, t, r, b, coord_origin} 
 
    Tries multiple locations because versions differ: 
      1) item.bbox (direct) 
      2) provenance: prov[i].bbox or prov[i].region.bbox 
      3) provenance lists: prov[i].bboxes, prov[i].regions[*].bbox, 
prov[i].lines[*].bbox, spans[*].bbox 
    """ 
    # 1) Direct attribute (preferred if present) 
    bb = getattr(item, "bbox", None) 
    plain = _to_plain_bbox(bb) 
    if plain and any(plain[k] is not None for k in ("l", "t", "r", "b")): 
        return plain 
 
    # 2) Provenance fallbacks (common for SectionHeaderItem) 
    prov = getattr(item, "prov", None) 
    if prov: 
        collected = [] 
 
        for p in prov: 
            # p.bbox 
            pbb = _to_plain_bbox(getattr(p, "bbox", None)) 
            if pbb: 
                collected.append(pbb) 
 
            # p.region.bbox 
            pregion = getattr(p, "region", None) 
            if pregion is not None: 
                p_reg_bb = _to_plain_bbox(getattr(pregion, "bbox", None)) 
                if p_reg_bb: 
                    collected.append(p_reg_bb) 
 
            # p.bboxes (list) 
            p_bboxes = getattr(p, "bboxes", None) 
            if isinstance(p_bboxes, (list, tuple)): 
                for bbx in p_bboxes: 
                    pbb2 = _to_plain_bbox(bbx) 
                    if pbb2: 
                        collected.append(pbb2) 
 

            # p.regions (list) -> region.bbox 
            p_regions = getattr(p, "regions", None) 
            if isinstance(p_regions, (list, tuple)): 
                for rg in p_regions: 
                    pbb3 = _to_plain_bbox(getattr(rg, "bbox", None)) 
                    if pbb3: 
                        collected.append(pbb3) 
 
            # p.lines[*].bbox and nested spans[*].bbox 
            p_lines = getattr(p, "lines", None) 
            if isinstance(p_lines, (list, tuple)): 
                for ln in p_lines: 
                    pbb4 = _to_plain_bbox(getattr(ln, "bbox", None)) 
                    if pbb4: 
                        collected.append(pbb4) 
                    ln_spans = getattr(ln, "spans", None) 
                    if isinstance(ln_spans, (list, tuple)): 
                        for sp in ln_spans: 
                            pbb5 = _to_plain_bbox(getattr(sp, "bbox", None)) 
                            if pbb5: 
                                collected.append(pbb5) 
 
        u = _union_plain_bboxes(collected) 
        if u and any(u[k] is not None for k in ("l", "t", "r", "b")): 
            return u 
 
    # 3) Nothing found 
    return None 
 
def convert_and_save(source: str, out_dir: str = "outputs") -> None: 
    out_path = Path(out_dir) 
    out_path.mkdir(parents=True, exist_ok=True) 
 
    # Resolve & log the incoming path 
    p = Path(str(source).strip().strip('"').strip("'")).expanduser().resolve() 
    print(f"[DEBUG] CWD: {Path.cwd()}") 
    print(f"[DEBUG] Input path (original): {source}") 
    print(f"[DEBUG] Input path (resolved): {p}") 
    print(f"[DEBUG] Exists? {p.exists()}  Size: {p.stat().st_size if p.exists() else 'N/A'}") 
 
    if not p.exists(): 
        raise FileNotFoundError(f"Input PDF not found at: {p}") 
 
    # --- KEY FIX: wrap bytes in a DocumentStream, not a raw file handle --- 
    # convert() accepts Path | str | DocumentStream. Passing BufferedReader fails validation. 
    # We use DocumentStream(name=..., stream=BytesIO(...)). 

    with p.open("rb") as f: 
        pdf_bytes = f.read() 
    ds = DocumentStream(name=p.name, stream=BytesIO(pdf_bytes))  # <- correct type for convert() 
 
    converter = DocumentConverter() 
    result = converter.convert(ds)  # <- pass DocumentStream, not a file handle 
 
    # Applies hierarchy reconstruction and reorders document items in-place 
    ResultPostprocessor(result).process() 
 
    doc = result.document 
    stem = p.stem 
 
    # Build & save custom JSON (header-only bbox) 
    custom_json = build_custom_hierarchical_json(doc) 
    out_file = out_path / f"{stem}.custom.json" 
    with out_file.open("w", encoding="utf-8") as f: 
        json.dump(custom_json, f, ensure_ascii=False, indent=2) 
 
    print("Saved files:") 
    print(f"- {out_file}") 
 
def extract_pdf_to_json(pdf_path: str, out_dir: str = "outputs") -> str: 
    """ 
    Extract PDF to JSON and return the path to the generated JSON file. 
     
    Args: 
        pdf_path: Path to the PDF file (local or downloaded) 
        out_dir: Output directory for JSON file 
         
    Returns: 
        Path to the generated JSON file 
         
    Raises: 
        FileNotFoundError: If PDF file not found 
        Exception: If extraction fails 
    """ 
    out_path = Path(out_dir) 
    out_path.mkdir(parents=True, exist_ok=True) 
 
    # Resolve the input path 
    p = Path(str(pdf_path).strip().strip('"').strip("'")).expanduser().resolve() 
     
    if not p.exists(): 
        raise FileNotFoundError(f"Input PDF not found at: {p}") 

 
    # Extract using DocumentConverter 
    with p.open("rb") as f: 
        pdf_bytes = f.read() 
    ds = DocumentStream(name=p.name, stream=BytesIO(pdf_bytes)) 
 
    converter = DocumentConverter() 
    result = converter.convert(ds) 
 
    # Apply hierarchy reconstruction 
    ResultPostprocessor(result).process() 
 
    doc = result.document 
    stem = p.stem 
 
    # Build & save custom JSON 
    custom_json = build_custom_hierarchical_json(doc) 
    out_file = out_path / f"{stem}.custom.json" 
    with out_file.open("w", encoding="utf-8") as f: 
        json.dump(custom_json, f, ensure_ascii=False, indent=2) 
 
    print(f"✓ Extracted PDF to JSON: {out_file}") 
    return str(out_file) 
 
 
def build_custom_hierarchical_json(doc): 
    """ 
    Build custom JSON structure using DocTag levels from iterate_items(): 
    { 
        "Level_1": { 
            "heading_name": { 
                "page": 2, 
                "bbox": {...},  # Only section header bbox 
                "content": {"text": "...", "table": [...]}, 
                "Level_2": { ... } 
            } 
        } 
    } 
    """ 
    hierarchy_stack = []  # list of (level, name, node_reference) 
    root = {} 
 
    iterator = None 
    if hasattr(doc, "iterate_items"): 
        iterator = doc.iterate_items() 
    else: 
        print("Warning: Could not find iterate_items method.") 
        return {} 

 
    for item, level in iterator: 
        actual_item = item[0] if isinstance(item, tuple) else item 
        current_level = level if isinstance(level, int) else 1 
 
        # ---------------------------- 
        # Section Headers (capture page+bbox here only) 
        # ---------------------------- 
        if isinstance(actual_item, SectionHeaderItem): 
            heading_text = actual_item.text.strip() 
 
            # Capture header page + bbox ONLY (no child backfill) 
            page_no = get_page_number(actual_item) 
            bbox = get_bbox(actual_item) 
 
            new_node = { 
                "page": page_no, 
                "bbox": bbox,  # header bbox only 
                "content": {"text": "", "table": []}, 
            } 
 
            # Maintain proper parent level 
            while hierarchy_stack and hierarchy_stack[-1][0] >= current_level: 
                hierarchy_stack.pop() 
 
            level_key = f"Level_{current_level}" 
 
            if not hierarchy_stack: 
                if level_key not in root: 
                    root[level_key] = {} 
                root[level_key][heading_text] = new_node 
            else: 
                parent_level, parent_name, parent_node = hierarchy_stack[-1] 
                if level_key not in parent_node: 
                    parent_node[level_key] = {} 
                parent_node[level_key][heading_text] = new_node 
 
            hierarchy_stack.append((current_level, heading_text, new_node)) 
 
            # Optional: warn if header has no bbox (helps debugging) 
            if bbox is None: 
                print(f"[WARN] Header has no bbox: '{heading_text[:80]}' (page={page_no})") 
 
        # ---------------------------- 
        # Tables (no bbox/page backfill) 
        # ---------------------------- 
        elif isinstance(actual_item, TableItem): 

            df = actual_item.export_to_dataframe(doc=doc) 
 
            table_content = [] 
            headers = [str(col).strip() for col in df.columns] 
 
            for _, row in df.iterrows(): 
                row_dict = {} 
                for idx, cell in enumerate(row): 
                    header_name = headers[idx] if idx < len(headers) else f"col_{idx}" 
                    val = str(cell).strip() if pd.notna(cell) else "" 
                    row_dict[header_name] = val 
                table_content.append(row_dict) 
 
            if hierarchy_stack: 
                current_node = hierarchy_stack[-1][2] 
                current_node["content"]["table"].append(table_content) 
 
        # ---------------------------- 
        # Text (no bbox/page backfill) 
        # ---------------------------- 
        elif isinstance(actual_item, TextItem): 
            text = actual_item.text.strip() 
            if text and hierarchy_stack: 
                current_node = hierarchy_stack[-1][2] 
                if current_node["content"]["text"]: 
                    current_node["content"]["text"] += " " + text 
                else: 
                    current_node["content"]["text"] = text 
 
    return root 
 
if __name__ == "__main__": 
    convert_and_save(r"C:\Users\KeyaG\Domain_aware_RAG\domain_aware_rag\data\2026-pc65-medical-only-eoc.pdf") 
