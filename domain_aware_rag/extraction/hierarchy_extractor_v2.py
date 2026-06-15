import json 
from pathlib import Path 
from typing import List, Dict, Any, Optional 
 
class HierarchyExtractorV2: 
    """ 
    Extract complete hierarchy paths from hierarchical JSON document. 
     
    Input format (Levels-based): 

    { 
        "levels": { 
            "Org": { 
                "Issuing Organization": { 
                    "Evidence of Coverage for 2026:": "QCC Insurance Company" 
                }, 
                "Medicaid - PA OMAP": { 
                    "Section 3.1 ...": "Medicaid card", 
                    ... 
                } 
            }, 
            "Plan": { ... }, 
            "Member": { ... } 
        } 
    } 
    """ 
     
    def __init__(self, json_file_path: str): 
        """ 
        Initialize with hierarchical JSON file. 
         
        Args: 
            json_file_path: Path to the hierarchical JSON file 
        """ 
        self.json_file_path = Path(json_file_path) 
        with open(self.json_file_path, 'r', encoding='utf-8') as f: 
            self.hierarchy_data = json.load(f) 
     
    def find_section_hierarchy(self, section_name: str) -> Optional[List[str]]:
        """ 
            Find the complete hierarchy path for a given section name. 
            
            Args: 
                section_name: The section name to search for (e.g., "Section 4.1 
    Plan premium") 
            
            Returns: 
                List of keys representing the hierarchy path from root to the 
    section, 
                or None if section not found 
            """ 
        hierarchy_path = [] 
         
        def search_recursive(node: Any, current_path: List[str]) -> bool: 
            """ 
            Recursively search through the hierarchy tree. 
             

            Args: 
                node: Current node being searched 
                current_path: Current path of keys 
             
            Returns: 
                True if section found, False otherwise 
            """ 
            if isinstance(node, dict): 
                for key, value in node.items(): 
                    current_path.append(key) 
                     
                    # Check if this key matches the section name 
                    if key == section_name or key.strip() == section_name.strip():
                        hierarchy_path.extend(current_path) 
                        return True 
                     
                    # Recursively search in nested dictionaries 
                    if isinstance(value, dict): 
                        if search_recursive(value, current_path): 
                            return True 
                     
                    current_path.pop() 
             
            return False 
         
        search_recursive(self.hierarchy_data, []) 
        return hierarchy_path if hierarchy_path else None 
     
    def get_tail_content(self, section_name: str) -> Optional[Dict[str, Any]]: 
        """ 
        Get the content (text and table) of the deepest/tail section. 
         
        Args: 
            section_name: The section name to search for 
         
        Returns: 
            Dictionary with 'text' and 'table' content, or None if not found 
        """ 
        def search_recursive(node: Any) -> Optional[Dict[str, Any]]: 
            """ 
            Recursively search for the section and return its content. 
            """ 
            if isinstance(node, dict): 
                for key, value in node.items(): 
                    # Check if this key matches the section name 
                    if key == section_name or key.strip() == section_name.strip():

                        # Return the content if it exists 
                        if isinstance(value, dict) and 'content' in value: 
                            return value.get('content', {}) 
                        return None 
                     
                    # Recursively search in nested dictionaries 
                    if isinstance(value, dict): 
                        result = search_recursive(value) 
                        if result is not None: 
                            return result 
             
            return None 
         
        return search_recursive(self.hierarchy_data) 
     
    def flatten_to_nested_hierarchy(self, flat_hierarchy: List[str]) -> Dict[str, Any]:
        """ 
        Convert flat hierarchy path list to nested JSON structure for better 
understanding. 
         
        Args: 
            flat_hierarchy: List of keys forming the hierarchy path 
         
        Returns: 
            Nested dictionary structure showing the hierarchy levels 
        """ 
        if not flat_hierarchy: 
            return {} 
         
        nested = {} 
        current = nested 
         
        for i, key in enumerate(flat_hierarchy): 
            # Check if this is a level key (e.g., "Level_1", "Level_2") 
            is_level_key = key.startswith("Level_") 
             
            if is_level_key: 
                # Create level structure 
                if key not in current: 
                    current[key] = {} 
                current = current[key] 
            else: 
                # Create content structure with the key 
                current[key] = {} 
                current = current[key] 
         
        return nested 

     
    def process_levels_input(self, entity_data: Dict[str, Dict[str, Dict[str, 
Dict[str, str]]]],  
                            nested_format: bool = True,  
                            include_content: bool = True) -> List[Dict[str, 
Any]]: 
        """ 
        Process the levels-based input format where data is organized by level 
(Org/Plan/Member). 
         
        Input format: 
        { 
            "levels": { 
                "Org": { 
                    "Issuing Organization": { 
                        "Evidence of Coverage for 2026:": "QCC Insurance 
Company" 
                    }, 
                    "Medicaid - PA OMAP": { 
                        "Section 3.1 ...": "Medicaid card", 
                        "SECTION 7 ...": "Medicaid", 
                        ... 
                    } 
                }, 
                "Plan": { ... }, 
                "Member": { ... } 
            } 
        } 
         
        Args: 
            entity_data: Dictionary with levels structure 
            nested_format: If True, returns hierarchy as nested JSON 
            include_content: If True, includes tail content 
         
        Returns: 
            List of results with level, category, hierarchy and optional 
content 
        """ 
        results = [] 
         
        # Get the levels dict 
        if "levels" not in entity_data: 
            return results 
         
        levels_dict = entity_data["levels"] 
         
        # Process each level (Org, Plan, Member, etc.) 
        for level_name, level_categories in levels_dict.items(): 

            if not isinstance(level_categories, dict): 
                continue 
             
            # Process each category within the level 
            for category_name, sections_dict in level_categories.items(): 
                if not isinstance(sections_dict, dict): 
                    continue 
                 
                # Process each section in this category 
                for section_name, section_value in sections_dict.items(): 
                    result = { 
                        "level": level_name, 
                        "entity_category": category_name, 
                        "entity_value": section_value, 
                        "section_name": section_name 
                    } 
                     
                    # Find hierarchy 
                    hierarchy = self.find_section_hierarchy(section_name) 
                     
                    # Generate rich metadata for retrieval context 
                    metadata = { 
                        "source": "hierarchy_extractor_v2", 
                        "entity_category": category_name,  
                        "entity_value": section_value, 
                        "section_name": section_name, 
                        "hierarchy_context": "", 
                        "hierarchy_depth": 0, 
                        "is_leaf": False, 
                        "has_content": False, 
                        "has_table": False 
                    } 
 
                    if hierarchy: 
                        if nested_format:
                            result['hierarchy_path'] = self.flatten_to_nested_hierarchy(hierarchy)
                        else: 
                            result['hierarchy_path'] = hierarchy 
                         
                        # Populate hierarchy metadata 
                        metadata["hierarchy_context"] = " > ".join(hierarchy) 
                        metadata["hierarchy_depth"] = len(hierarchy) 
                         
                        # Add tail content if requested 
                        if include_content: 
                            tail_content = self.get_tail_content(section_name) 
                            if tail_content: 

                                result['tail_content'] = tail_content 
                                metadata["has_content"] = True 
                                metadata["is_leaf"] = True 
                                metadata["has_text"] = bool(tail_content.get("text"))
                                 
                                tables = tail_content.get("table", []) 
                                if tables: 
                                    metadata["has_table"] = True 
                                    metadata["table_count"] = len(tables) 
                                    # Add basics about the table dimensions 
                                    if tables and isinstance(tables[0], list):
                                        metadata["table_rows"] = len(tables)
                                        metadata["table_cols"] = len(tables[0])
                            else: 
                                result['tail_content'] = None 
                    else: 
                        result['hierarchy_path'] = None 
                        if include_content: 
                            result['tail_content'] = None 
                     
                    result['metadata'] = metadata 
                    results.append(result) 
         
        return results 
     
    def get_hierarchy_for_entities(
        self,
        entities: Dict[str, Dict[str, Dict[str, Dict[str, str]]]],
        nested_format: bool = True,
        include_content: bool = True
    ) -> List[Dict[str, Any]]:
        """ 
        Get hierarchy paths for entities with levels-based input format. 
         
        Input format: 
        { 
            "levels": { 
                "Org": { 
                    "Category Name": { 
                        "Section 1.1 ...": "value1" 
                    } 
                }, 
                "Plan": { ... }, 
                "Member": { ... } 
            } 
        } 

         
        Args: 
            entities: Dictionary with levels structure 
            nested_format: If True, returns hierarchy as nested JSON (default 
True) 
            include_content: If True, includes tail_content field (default 
True) 
         
        Returns: 
            List of entity dictionaries with level, category, hierarchy, and 
optional content 
        """ 
        return self.process_levels_input(entities, nested_format, include_content)
 
def main(custom_json_path=None, entities_json_path=None, output_path=None): 
    """ 
    Process levels-based hierarchy input. 
     
    Args: 
        custom_json_path: Path to custom.json from data extraction (default: 
outputs/2026-pc65-medical-only-eoc.custom.json) 
        entities_json_path: Path to entities JSON from NER (default: 
outputs/entity_output_updated_103.json) 
        output_path: Path for output hierarchy results (default: 
hierarchy_extraction_results.json) 
    """ 
    # Set defaults 
    custom_json_path = custom_json_path or "outputs/2026-pc65-medical-only-eoc.custom.json"
    entities_json_path = entities_json_path or "outputs/entity_output_updated_103.json"
    output_path = output_path or "hierarchy_extraction_results.json" 
     
    print(f"[Hierarchy] Using inputs:") 
    print(f"  - Custom JSON: {custom_json_path}") 
    print(f"  - Entities JSON: {entities_json_path}") 
    print(f"[Hierarchy] Output: {output_path}") 
     
    # Initialize the extractor 
    extractor = HierarchyExtractorV2(custom_json_path) 
     
    # Check if entities-based input file exists 
    if Path(entities_json_path).exists():         
        with open(entities_json_path, "r", encoding="utf-8") as f: 
            levels_entities = json.load(f) 
         
        results = extractor.get_hierarchy_for_entities( 

            levels_entities, 
            nested_format=True, 
            include_content=True 
        ) 
         
        print(f"\n✓ Processed {len(results)} entity-section combinations") 
         
        # Save results 
        with open(output_path, 'w', encoding='utf-8') as f: 
            json.dump(results, f, ensure_ascii=False, indent=2) 
        print(f"✓ Results saved to: {output_path}") 
         
        return output_path 
    else: 
        print(f"✗ Entities JSON file '{entities_json_path}' not found.") 
        print("Run NER extraction first to generate entities.") 
        return None 
 
if __name__ == "__main__": 
    main() 
