""" 
HTML Entity Mapper - Direct mapping from NER entities to database nodes. 
 
For HTML documents, we don't have hierarchical structure like PDFs. 
Instead, we create flat entities with section-based grouping. 
""" 
 
import json 
import logging 
from pathlib import Path 
from typing import Dict, List, Any, Optional 
 
logger = logging.getLogger(__name__) 
 
def create_html_entity_hierarchy( 
    entities_json_path: str, 
    normalized_json_path: str, 
    output_path: Optional[str] = None 
) -> str: 
    """ 
    Create a hierarchy.json from HTML NER entities for database ingestion. 
     

    Transforms NER's "levels" format into hierarchy.json that ingest_optimized 
can traverse. 
    The ingestion code traverses Level_1 → Level_2 etc and calls 
load_entities_map() 
    to look up entities for each path. 
     
    Args: 
        entities_json_path: Path to entities.json from NER (format: {"levels": 
{...}}) 
        normalized_json_path: Path to html_normalized_for_ner.json (for 
section titles) 
        output_path: Path to save hierarchy.json (if None, overwrites input) 
     
    Returns: 
        Path to the created hierarchy.json 
    """ 
    # Safety checks for None paths
    if not entities_json_path:
        raise ValueError("entities_json_path cannot be None")
    if not normalized_json_path:
        raise ValueError("normalized_json_path cannot be None")
        
    entities_json_path = Path(entities_json_path) 
    normalized_json_path = Path(normalized_json_path) 
     
    if not entities_json_path.exists(): 
        raise FileNotFoundError(f"Entities JSON not found: {entities_json_path}")
     
    if not normalized_json_path.exists(): 
        raise FileNotFoundError(f"Normalized JSON not found: {normalized_json_path}")
     
    # Load entities 
    logger.info(f"Loading entities from: {entities_json_path}") 
    with open(entities_json_path, 'r', encoding='utf-8') as f: 
        entities_data = json.load(f) 
     
    # Load normalized (to get section structure) 
    logger.info(f"Loading normalized HTML from: {normalized_json_path}") 
    with open(normalized_json_path, 'r', encoding='utf-8') as f: 
        normalized_data = json.load(f) 
     
    # Build section_id -> title mapping 
    section_titles = {} 
    for node in normalized_data.get("nodes", []): 
        section_id = node.get("id") 
        title = node.get("text", "") 
        if section_id and section_id != "0":  # Skip root 
            section_titles[section_id] = title 
     
    logger.info(f"Found {len(section_titles)} sections") 
     
    # Build hierarchy structure: Level_1 (sections only, no entities) 

    # Entities will be looked up by ingest_optimized via load_entities_map() 
    hierarchy = { 
        "source_type": "html", 
        "title": normalized_data.get("title", "HTML Document"), 
        "Level_1": {} 
    } 
     
    # Create Level_1 sections (just content, no entities here) 
    for section_id, section_title in section_titles.items(): 
        hierarchy["Level_1"][section_id] = { 
            "content": { 
                "text": section_title 
            } 
        } 
     
    logger.info(f"Created {len(hierarchy['Level_1'])} Level_1 sections in hierarchy")
     
    # The entities.json (in "levels" format) will be processed by load_entities_map()
    # No need to copy entities here - they stay in the entities.json file 
     
    # If output path not specified, use input directory 
    if output_path is None: 
        output_path = str(entities_json_path.parent / "hierarchy.json") 
    else: 
        output_path = str(output_path) 
     
    # Save hierarchy 
    output_path_obj = Path(output_path) 
    output_path_obj.parent.mkdir(parents=True, exist_ok=True) 
     
    with open(output_path, 'w', encoding='utf-8') as f: 
        json.dump(hierarchy, f, indent=2, ensure_ascii=False) 
     
    logger.info(f"Saved HTML hierarchy to: {output_path}") 
    logger.info(f"Structure: {len(hierarchy['Level_1'])} Level_1 sections") 
    logger.info(f"Note: Entities are in {entities_json_path} (levels format)") 
    logger.info("      ingest_optimized will load entities via load_entities_map()")
     
    return output_path 
 
if __name__ == "__main__": 
    import sys 
     
    if len(sys.argv) < 3: 

        print("Usage: python html_entity_mapper.py <entities.json> <normalized.json> [output.json]")
        sys.exit(1) 
     
    entities_file = sys.argv[1] 
    normalized_file = sys.argv[2] 
    output_file = sys.argv[3] if len(sys.argv) > 3 else None 
     
    logging.basicConfig(level=logging.INFO) 
     
    try: 
        result = create_html_entity_hierarchy(entities_file, normalized_file, output_file)
        print(f"✓ Created: {result}") 
    except Exception as e: 
        print(f"✗ Error: {e}") 
        import traceback 
        traceback.print_exc() 
        sys.exit(1) 
 
