""" 
Document ID Generator 
===================== 
Generates unique, human-readable document IDs for multi-document support. 
 
Formats 
------- 
PDF: 

  {pdf_stem}_{timestamp}_{content_hash} 
  Example: MyPlan_2026_20260310_143022_a7f3e9c4 
 
HTML: 
  {stem}_{timestamp}_{hash} 
 
This ensures: 
- Uniqueness across uploads 
- Chronological sortability 
- Human readability 
- Content-based deduplication 
  - PDFs: based on file bytes (first 1MB by default) 
  - HTML: based on seed string 
""" 
 
import os 
import re 
import hashlib 
import logging 
from datetime import datetime, timezone 
from pathlib import Path 
from typing import Any, Dict, Optional 
from urllib.parse import urlparse 
 
logger = logging.getLogger(__name__) 
 
# Timestamp format constant 
TIMESTAMP_FMT = "%Y%m%d_%H%M%S" 
 
# --------------------------------------------------------------------------- 
# HTML-safe document ID generator 
# --------------------------------------------------------------------------- 
 
def generate_html_id( 
    seed: str, 
    content_hash_size: int = 8, 
    *, 
    use_utc: bool = True, 
    normalize_lower: bool = True, 
) -> str: 
    """ 
    Generate a document ID for HTML sources (URL or local .html file). 
    Does NOT touch the filesystem. 
 
    Args: 
        seed: HTML URL (http/https) or local .html path 
        content_hash_size: Length of hex hash suffix 
        use_utc: Use UTC timestamp (default True) 

        normalize_lower: Lowercase stem (default True) 
 
    Returns: 
        "{stem}_{timestamp}_{hash}" 
    """ 
    try: 
        # Derive stem 
        try: 
            if seed.lower().startswith(("http://", "https://")): 
                parsed = urlparse(seed) 
                last = parsed.path.rsplit("/", 1)[-1] if parsed.path else "" 
                stem = Path(last).stem or "index" 
            else: 
                stem = Path(seed).stem or "index" 
        except Exception: 
            stem = "index" 
 
        # Normalize 
        stem = re.sub(r"[ \-]+", "_", stem) 
        if normalize_lower: 
            stem = stem.lower() 
 
        # Timestamp 
        now = datetime.now(timezone.utc) if use_utc else datetime.now() 
        timestamp = now.strftime(TIMESTAMP_FMT) 
 
        # Stable hash from seed string 
        content_hash = hashlib.sha256(seed.encode("utf- 8")).hexdigest()[:content_hash_size] 
 
        doc_id = f"{stem}_{timestamp}_{content_hash}" 
        logger.info("Generated HTML doc_id: %s", doc_id) 
        return doc_id 
 
    except Exception as e: 
        logger.error("Failed to generate HTML doc_id for %s: %s", seed, 
str(e)) 
        raise 
 
# --------------------------------------------------------------------------- 
# PDF document ID generator (local files only) 
# --------------------------------------------------------------------------- 
 
def generate_doc_id( 
    pdf_path: str, 
    content_hash_size: int = 8, 
    *, 
    use_utc: bool = True, 

    hash_full_file: bool = False, 
    normalize_lower: bool = True, 
) -> str: 
    """ 
    Generate a unique document ID from a local PDF file. 
 
    Args: 
        pdf_path: Path to a local PDF file 
        content_hash_size: Length of hash suffix 
        use_utc: Use UTC timestamp (default True) 
        hash_full_file: Hash entire file instead of first 1MB 
        normalize_lower: Lowercase stem (default True) 
 
    Returns: 
        "{stem}_{timestamp}_{content_hash}" 
    """ 
    try: 
        p = Path(pdf_path) 
        if not p.exists(): 
            raise FileNotFoundError(f"PDF not found: {pdf_path}") 
        if p.is_dir(): 
            raise IsADirectoryError(f"Expected a file, got directory:  {pdf_path}") 
 
        # Stem normalization 
        stem = p.stem.replace(" ", "_").replace("-", "_") 
        if normalize_lower: 
            stem = stem.lower() 
 
        # Timestamp 
        now = datetime.now(timezone.utc) if use_utc else datetime.now() 
        timestamp = now.strftime(TIMESTAMP_FMT) 
 
        # Content hash 
        content_hash = _compute_content_hash( 
            str(p), 
            hash_size=content_hash_size, 
            full_file=hash_full_file, 
        ) 
 
        doc_id = f"{stem}_{timestamp}_{content_hash}" 
        logger.info("Generated PDF doc_id: %s", doc_id) 
        return doc_id 
 
    except Exception as e: 
        logger.error("Failed to generate doc_id for %s: %s", pdf_path, str(e)) 
        raise 
 

def _compute_content_hash( 
    file_path: str, 
    hash_size: int = 8, 
    *, 
    full_file: bool = False, 
) -> str: 
    """ 
    Compute SHA256 hash of file contents. 
 
    Default: first 1MB for speed. 
    """ 
    try: 
        sha256 = hashlib.sha256() 
        with open(file_path, "rb") as f: 
            if full_file: 
                for chunk in iter(lambda: f.read(1024 * 1024), b""): 
                    sha256.update(chunk) 
            else: 
                sha256.update(f.read(1024 * 1024)) 
 
        return sha256.hexdigest()[:hash_size] 
 
    except Exception as e: 
        logger.warning("Could not compute content hash for %s: %s", file_path, 
str(e)) 
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") 
        fallback = f"fb_{ts}" 
        return fallback[:hash_size].ljust(hash_size, "0") 
 
# --------------------------------------------------------------------------- 
# Metadata helpers 
# --------------------------------------------------------------------------- 
 
def extract_doc_metadata_from_id(doc_id: str) -> Dict[str, Optional[Any]]: 
    """ 
    Extract metadata from a document ID. 
 
    Returns: 
        { 
            'stem': str, 
            'timestamp': datetime | None, 
            'hash': str | None, 
            'upload_date': str | None, 
            'upload_time': str | None 
        } 
    """ 
    try: 
        parts = doc_id.rsplit("_", 2) 

        if len(parts) < 3: 
            return { 
                "stem": doc_id, 
                "timestamp": None, 
                "hash": None, 
                "upload_date": None, 
                "upload_time": None, 
            } 
 
        stem, timestamp_str, content_hash = parts 
 
        try: 
            timestamp = datetime.strptime(timestamp_str, TIMESTAMP_FMT) 
            upload_date = timestamp.strftime("%Y-%m-%d") 
            upload_time = timestamp.strftime("%H:%M:%S") 
        except ValueError: 
            timestamp = upload_date = upload_time = None 
 
        return { 
            "stem": stem, 
            "timestamp": timestamp, 
            "hash": content_hash, 
            "upload_date": upload_date, 
            "upload_time": upload_time, 
        } 
 
    except Exception as e: 
        logger.warning("Could not extract metadata from doc_id '%s': %s", 
doc_id, str(e)) 
        return { 
            "stem": doc_id, 
            "timestamp": None, 
            "hash": None, 
            "upload_date": None, 
            "upload_time": None, 
        } 
 
# --------------------------------------------------------------------------- 
# Output directory helpers 
# --------------------------------------------------------------------------- 
 
def create_document_output_dir(doc_id: str, outputs_dir: str = "outputs") ->  str: 
    """ 
    Create document-specific output directory. 
    """ 
    doc_output_dir = os.path.join(outputs_dir, doc_id) 
    os.makedirs(doc_output_dir, exist_ok=True) 

    logger.info("Created document output directory: %s", doc_output_dir) 
    return doc_output_dir 
 
def get_document_files(doc_id: str, outputs_dir: str = "outputs") -> Dict[str, 
str]: 
    """ 
    Get paths to document-specific output files. 
    """ 
    doc_dir = os.path.join(outputs_dir, doc_id) 
    return { 
        "custom_json": os.path.join(doc_dir, "document.custom.json"), 
        "entities_json": os.path.join(doc_dir, "entities.json"), 
        "keywords_json": os.path.join(doc_dir, "keywords.json"), 
        "hierarchy_json": os.path.join(doc_dir, "hierarchy.json"), 
        "directory": doc_dir, 
    } 
 
