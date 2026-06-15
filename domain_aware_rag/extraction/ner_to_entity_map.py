""" 
Convert NER output (list format) to entity map format (levels-based dict). 
 
NER outputs: 
  [ 
    {"scope_id": "Section_0", "entities": [ 
      {"entity_name": "...", "entity_category": "...", "entity_value": "...", 
"level": "..."} 
    ]}, 
    ... 
  ] 
 
But load_entities_map() expects: 
  {"levels": {"Level": {"entity_name": {"scope_id": "value"}}}} 
""" 
 
import json 
import logging 
from pathlib import Path 
from typing import Dict, List, Any 
 

logger = logging.getLogger(__name__) 
 
def convert_ner_to_entity_map( 
    entities_json_path: str, 
    output_path: str = None 
) -> str: 
    """ 
    Convert NER entities list format to entity_map dict format. 
     
    Args: 
        entities_json_path: Path to entities.json from NER (list format) 
        output_path: Path to save converted file (if None, overwrites input) 
     
    Returns: 
        Path to the converted file 
    """ 
    entities_json_path = Path(entities_json_path) 
     
    if not entities_json_path.exists(): 
        raise FileNotFoundError(f"Entities JSON not found:  {entities_json_path}") 
     
    logger.info(f"Loading NER entities from: {entities_json_path}") 
    with open(entities_json_path, 'r', encoding='utf-8') as f: 
        ner_data = json.load(f) 
     
    # NER outputs a list of scope results 
    if not isinstance(ner_data, list): 
        logger.warning(f"Expected list format, got {type(ner_data)}") 
        return str(entities_json_path) 
     
    # Convert to entity_map format: {"levels": {"Level": {"entity_name":  {"scope_id": "value"}}}} 
    entity_map = {"levels": {}} 
     
    total_entities = 0 
     
    for scope_result in ner_data: 
        if not isinstance(scope_result, dict): 
            continue 
         
        scope_id = scope_result.get("scope_id") 
        entities_list = scope_result.get("entities", []) 
         
        if not scope_id or not entities_list: 
            continue 
         
        for entity in entities_list: 

            if not isinstance(entity, dict): 
                continue 
             
            # Extract fields 
            entity_name = entity.get("entity_name", "") 
            entity_category = entity.get("entity_category", "") 
            entity_value = entity.get("entity_value", "") 
            level = entity.get("level", "Other") 
             
            if not entity_name or not entity_value: 
                continue 
             
            # Build nested structure: levels → level → entity_name → scope_id  → value 
            if level not in entity_map["levels"]: 
                entity_map["levels"][level] = {} 
             
            if entity_name not in entity_map["levels"][level]: 
                entity_map["levels"][level][entity_name] = {} 
             
            # Use scope_id as key (handle duplicates with counter) 
            entry_key = scope_id 
            if entry_key in entity_map["levels"][level][entity_name]: 
                # Duplicate found, add counter 
                counter = 2 
                while f"{scope_id} ({counter})" in  entity_map["levels"][level][entity_name]: 
                    counter += 1 
                entry_key = f"{scope_id} ({counter})" 
             
            entity_map["levels"][level][entity_name][entry_key] = entity_value 
            total_entities += 1 
     
    logger.info(f"Converted {total_entities} entities into  {len(entity_map['levels'])} levels") 
     
    # Determine output path 
    if output_path is None: 
        output_path = str(entities_json_path) 
    else: 
        output_path = str(output_path) 
     
    # Save 
    output_path_obj = Path(output_path) 
    output_path_obj.parent.mkdir(parents=True, exist_ok=True) 
     
    with open(output_path, 'w', encoding='utf-8') as f: 
        json.dump(entity_map, f, indent=2, ensure_ascii=False) 

     
    logger.info(f"Saved converted entity_map to: {output_path}") 
    return output_path 
 
if __name__ == "__main__": 
    import sys 
     
    if len(sys.argv) < 2: 
        print("Usage: python ner_to_entity_map.py <entities.json>  [output.json]") 
        sys.exit(1) 
     
    input_file = sys.argv[1] 
    output_file = sys.argv[2] if len(sys.argv) > 2 else None 
     
    logging.basicConfig(level=logging.INFO) 
     
    try: 
        result = convert_ner_to_entity_map(input_file, output_file) 
        print(f"✓ Converted: {result}") 
    except Exception as e: 
        print(f"✗ Error: {e}") 
        import traceback 
        traceback.print_exc() 
        sys.exit(1) 
