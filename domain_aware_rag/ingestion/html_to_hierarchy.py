""" 
HTML to Hierarchical Structure Converter 
========================================= 
Converts flat HTML extraction (with sections array) to hierarchical structure 
compatible with NER pipeline (Level_1, Level_2 format). 
 
This enables HTML documents to use the full NER + hierarchy extraction 
pipeline 

without modifying updated_ner_llm.py. 
 
Input Structure (HTML): 
    { 
        "source_type": "html", 
        "title": "...", 
        "source": "URL/path", 
        "sections": [ 
            {"heading": "...", "content": "..."}, 
            ... 
        ] 
    } 
 
Output Structure (Hierarchical): 
    { 
        "source_type": "html", 
        "title": "...", 
        "source": "URL/path", 
        "Level_1": { 
            "section_1": {...}, 
            "section_2": {...}, 
            ... 
        }, 
        "Level_2": { 
            "section_1_subsection_1": {...}, 
            ... 
        } 
    } 
""" 
 
import json 
import logging 
from pathlib import Path 
from typing import Dict, Any, List, Optional 
 
logger = logging.getLogger(__name__) 
 
def convert_html_to_hierarchical( 
    html_json_path: str,  
    output_path: Optional[str] = None 
) -> str: 
    """ 
    Convert flat HTML JSON to hierarchical format for NER pipeline. 
     
    Args: 
        html_json_path: Path to HTML extraction JSON with flat sections 
        output_path: Path to save converted hierarchical JSON (if None, 
overwrites input) 

     
    Returns: 
        Path to the converted hierarchical JSON file 
         
    Raises: 
        FileNotFoundError: If input file doesn't exist 
        ValueError: If JSON structure is invalid 
    """ 
    html_json_path = Path(html_json_path) 
     
    if not html_json_path.exists(): 
        raise FileNotFoundError(f"HTML JSON file not found: {html_json_path}") 
     
    # Load the flat HTML extraction 
    logger.info(f"Loading HTML extraction from: {html_json_path}") 
    with open(html_json_path, 'r', encoding='utf-8') as f: 
        html_data = json.load(f) 
     
    # Validate structure 
    if not isinstance(html_data, dict): 
        raise ValueError("HTML JSON must be a dictionary") 
     
    if "sections" not in html_data: 
        raise ValueError("HTML JSON must contain 'sections' array") 
     
    sections = html_data.get("sections", []) 
    if not isinstance(sections, list): 
        raise ValueError("'sections' must be an array") 
     
    logger.info(f"Converting {len(sections)} sections to hierarchical structure")
     
    # Build hierarchical structure 
    hierarchical_data = { 
        "source_type": html_data.get("source_type", "html"), 
        "title": html_data.get("title", "Untitled HTML Document"), 
        "source": html_data.get("source", "unknown"), 
    } 
     
    # Group sections into hierarchy 
    level_1_sections = {} 
    level_2_sections = {} 
     
    for idx, section in enumerate(sections): 
        if not isinstance(section, dict): 
            logger.warning(f"Skipping non-dict section at index {idx}") 
            continue 
         

        heading = section.get("heading", "").strip() 
        content = section.get("content", "").strip() 
         
        # Skip empty sections 
        if not heading and not content: 
            logger.warning(f"Skipping empty section at index {idx}") 
            continue 
         
        # Use section index for unique key 
        section_key = f"section_{idx + 1}" 
         
        # Create section node in NER-compatible format 
        # NER's extract_scopes() requires all dict values to also be dicts 
        # Only content node should have text, others should be empty dicts 
        section_node = { 
            "content": { 
                "text": content if content else heading 
            } 
        } 
         
        # Classify as Level_1 or Level_2 based on heading length 
        # Level_1: Main sections (shorter headings, typically primary topics) 
        # Level_2: Subsections (longer headings or subsection-like patterns) 
        heading_length = len(heading) 
        content_length = len(content) 
         
        # Heuristic: Level_1 = shorter headings (< 50 chars), Level_2 = longer or more detailed
        if heading_length < 50 and content_length > 200: 
            # Main section 
            level_1_sections[section_key] = section_node 
        else: 
            # Treat as potential subsection or nested content 
            level_2_sections[section_key] = section_node 
         
        # If we only have a few sections, put them all in Level_1 for structure
        if len(sections) <= 3: 
            level_1_sections[section_key] = section_node 
            level_2_sections.pop(section_key, None) 
     
    # If no clear Level_1 detected, treat all sections as Level_1 
    if not level_1_sections: 
        logger.warning("No primary sections detected, treating all sections as Level_1")
        level_1_sections = level_2_sections 
        level_2_sections = {} 
     

    # DEBUG: Show what we're about to add to the output 
    logger.info(f"[DEBUG] Level_1 structure being created:") 
    for key, node in list(level_1_sections.items())[:2]: 
        logger.info(f"[DEBUG]   {key}: {node}") 
 
    # Add hierarchy to output 
    hierarchical_data["Level_1"] = level_1_sections 
    if level_2_sections: 
        hierarchical_data["Level_2"] = level_2_sections
    logger.info(f"Created {len(level_1_sections)} Level_1 sections, {len(level_2_sections)} Level_2 sections")
     
    # Determine output path 
    if output_path is None: 
        output_path = str(html_json_path) 
    else: 
        output_path = str(output_path) 
     
    # Save hierarchical structure 
    output_path_obj = Path(output_path) 
    output_path_obj.parent.mkdir(parents=True, exist_ok=True) 
     
    with open(output_path, 'w', encoding='utf-8') as f: 
        json.dump(hierarchical_data, f, indent=2, ensure_ascii=False) 
     
    logger.info(f"Saved hierarchical structure to: {output_path}") 
    return output_path 
 
def _detect_hierarchy_type(heading: str, content: str) -> str: 
    """ 
    Detect hierarchy level based on heading and content characteristics. 
    Returns "Level_1" or "Level_2". 
     
    Args: 
        heading: Section heading text 
        content: Section content text 
         
    Returns: 
        Hierarchy level ("Level_1" or "Level_2") 
    """ 
    # Heuristics for classification 
    # Level_1: Main topics, shorter headings 
    # Level_2: Subtopics, longer headings with more detail 
     
    heading_length = len(heading) 
    content_length = len(content) 
     
    # Short heading + substantive content = likely main section 

    if heading_length < 50 and content_length > 100: 
        return "Level_1" 
     
    # Longer heading or minimal content = likely subsection 
    if heading_length >= 50 or content_length < 100: 
        return "Level_2" 
     
    # Default to Level_1 
    return "Level_1" 
 
if __name__ == "__main__": 
    # Test conversion 
    import sys 
     
    if len(sys.argv) < 2: 
        print("Usage: python html_to_hierarchy.py <html_json_path> [output_path]")
        sys.exit(1) 
     
    input_path = sys.argv[1] 
    output_path = sys.argv[2] if len(sys.argv) > 2 else None 
     
    logging.basicConfig(level=logging.INFO) 
     
    try: 
        result = convert_html_to_hierarchical(input_path, output_path) 
        print(f"✓ Conversion successful: {result}") 
    except Exception as e: 
        print(f"✗ Conversion failed: {e}") 
        sys.exit(1) 
