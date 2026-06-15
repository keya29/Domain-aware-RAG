#!/usr/bin/env python
"""
Debug script to test NER extraction with detailed LLM response logging.
"""

import sys
import json
import os
from pathlib import Path

# Change to domain_aware_rag directory for relative paths
os.chdir(Path(__file__).parent / "domain_aware_rag")
sys.path.insert(0, str(Path(__file__).parent / "domain_aware_rag"))

from extraction.updated_ner_llm import main as ner_main

if __name__ == "__main__":
    # Path to the hierarchical JSON (converted from HTML)
    test_json = Path("outputs") / "test_html_hierarchical.json"
    output_json = Path("outputs") / "test_entities_debug.json"
    keywords_json = Path("outputs") / "test_keywords_debug.json"

    # Delete old test file to force regeneration with new format
    if test_json.exists():
        test_json.unlink()

    if not test_json.exists():
        print(f"Creating test HTML hierarchical JSON at {test_json}")

        # Create test data in NER-expected format
        test_data = {
            "source_type": "html",
            "title": "Benefits Overview",
            "source": "https://example.com/benefits.html",
            "Level_1": {
                "section_1": {
                    "content": {
                        "text": "The plan covers preventive services, emergency room visits, and inpatient hospital stays. Medicare provides comprehensive coverage for essential health services."
                    }
                },
                "section_2": {
                    "content": {
                        "text": "Members pay a $150 deductible annually. Copayments are $20 for office visits and $50 for emergency room visits. After you meet your out-of-pocket maximum of $5,500, the plan covers 100% of covered services."
                    }
                },
            },
        }

        test_json.parent.mkdir(parents=True, exist_ok=True)

        with open(test_json, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)

        print(f"Created test JSON with {len(test_data.get('Level_1', {}))} sections")

    print(f"\n{'=' * 70}")
    print("Running NER extraction with debug logging")
    print(f"Input: {test_json}")
    print(f"{'=' * 70}\n")

    try:
        ner_main(
            leveller_json_path=str(test_json),
            output_json_path=str(output_json),
            keywords_json_path=str(keywords_json),
        )

        print(f"\n{'=' * 70}")
        print("NER extraction complete!")
        print("Output files:")
        print(f"  - Entities: {output_json}")
        print(f"  - Keywords: {keywords_json}")

        if output_json.exists():
            with open(output_json, "r", encoding="utf-8") as f:
                entities = json.load(f)
            print(f"\nEntities found: {len(entities) if isinstance(entities, list) else 'N/A'}")

        if keywords_json.exists():
            with open(keywords_json, "r", encoding="utf-8") as f:
                keywords = json.load(f)
            print(f"Keywords found: {len(keywords) if isinstance(keywords, list) else 'N/A'}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()