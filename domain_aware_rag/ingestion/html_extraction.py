# extraction/html_extraction.py 
""" 
HTML → JSON Extractor 
--------------------- 
Parses an HTML URL or local .html file, removes boilerplate, extracts 
title/sections/text, and writes a 'custom_json'-compatible artifact so your 
NER + Hierarchy steps can run unchanged. 
 
If your downstream expects a strict Docline schema, share a sample and we will 
align the keys/structure accordingly. 
""" 
 
import os 
import re 
import json 
import time 
import logging 
from pathlib import Path 
from typing import Optional, Dict, Any, List 
 
import requests 
from bs4 import BeautifulSoup 
 
logger = logging.getLogger(__name__) 
 
USER_AGENT = ( 

    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' 
    '(KHTML, like Gecko) Chrome/121.0 Safari/537.36' 
) 
REQUEST_TIMEOUT = 30 
 
# ----------------------------- 
# Helpers for parsing/cleanup 
# ----------------------------- 
 
def _strip_noise(soup: BeautifulSoup) -> None: 
    """Remove scripts, styles, and common boilerplate to reduce noise.""" 
    for tag in soup(['script', 'style', 'noscript', 'template']): 
        tag.decompose() 
 
    # Remove common layout containers if they look like navigation/footers/ads 
    for sel in ['header', 'footer', 'nav', 'aside']: 
        for tag in soup.select(sel): 
            tag.decompose() 
 
    # Heuristic cleanup by id/class names 
    noisy_patterns = re.compile( 
        r'(cookie|consent|banner|ads?|advert|subscribe|newsletter|breadcrumbs| ' 
        r'share|social|sidebar|promo|modal|popup|announcement|skip-link)', 
        re.IGNORECASE 
    ) 
    for tag in list(soup.find_all(True)): 
        attrs = " ".join( 
            [tag.get('id', '')] + tag.get('class', []) 
            if isinstance(tag.get('class', []), list) else [] 
        ) 
        if attrs and noisy_patterns.search(attrs): 
            tag.decompose() 
 
def _extract_title(soup: BeautifulSoup) -> str: 
    if soup.title and soup.title.string: 
        t = soup.title.string.strip() 
        if t: 
            return t 
    h1 = soup.find('h1') 
    if h1: 
        t = h1.get_text(" ", strip=True) 
        if t: 
            return t 
    return "Untitled HTML Document" 
 

def _extract_sections(soup: BeautifulSoup) -> List[Dict[str, str]]: 
    """ 
    Split content by headings (h1-h6). For each heading, gather textual 
siblings 
    until the next heading. If no headings, return a single 'Content' section. 
    """ 
    body = soup.body or soup 
    heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] 
    headings = body.find_all(heading_tags) 
 
    if not headings: 
        text = body.get_text(separator='\n', strip=True) 
        text = _normalize_whitespace(text) 
        return [{"heading": "Content", "content": text} if text else 
{"heading": "Content", "content": ""}] 
 
    sections: List[Dict[str, str]] = [] 
    for idx, h in enumerate(headings): 
        heading_text = _normalize_whitespace(h.get_text(" ", strip=True)) 
        if not heading_text: 
            # Create a generic heading label if missing text 
            heading_text = f"Section {idx+1}" 
 
        collected: List[str] = [] 
        for sib in h.next_siblings: 
            # Stop at next heading 
            if getattr(sib, 'name', None) in heading_tags: 
                break 
            if hasattr(sib, 'get_text'): 
                chunk = _normalize_whitespace(sib.get_text(" ", strip=True)) 
                if chunk: 
                    collected.append(chunk) 
 
        section_text = "\n".join([c for c in collected if c]) 
        sections.append({"heading": heading_text, "content": section_text}) 
 
    # Post-filter: remove empty sections if there are non-empty ones 
    if any(s["content"] for s in sections): 
        sections = [s for s in sections if s["content"]] 
 
    return sections 
 
def _normalize_whitespace(text: str) -> str: 
    text = re.sub(r'[ \t]+', ' ', text) 
    text = re.sub(r'\n\s*\n+', '\n\n', text) 
    return text.strip() 
 

def _to_custom_json(title: str, source_ref: str, sections: List[Dict[str, 
str]], raw_text: str) -> Dict[str, Any]: 
    """ 
    Produce a JSON structure compatible with your downstream steps. 
    Adjust the keys if your NER/Hierarchy requires an exact schema. 
    """ 
    return { 
        "source_type": "html", 
        "source": source_ref, 
        "title": title, 
        "sections": sections,   # [{ "heading": "...", "content": "..." }] 
        "raw_text": raw_text 
    } 
 
def _write_output_json(data: Dict[str, Any], output_dir: str, prefix: str = 
"html") -> str: 
    os.makedirs(output_dir, exist_ok=True) 
    ts = int(time.time() * 1000) 
    out_path = os.path.join(output_dir, f"{prefix}_{ts}.custom.json") 
    with open(out_path, 'w', encoding='utf-8') as f: 
        json.dump(data, f, ensure_ascii=False, indent=2) 
    return out_path 
 
# ----------------------------- 
# Public API 
# ----------------------------- 
 
def extract_html_to_json_from_url(url: str, output_dir: str) -> str: 
    """ 
    Fetch an HTML page and write a JSON file in output_dir. 
    Returns the full JSON file path. 
    """ 
    headers = {'User-Agent': USER_AGENT} 
    logger.info(f"[HTML] Fetching URL: {url}") 
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT) 
    resp.raise_for_status() 
 
    content_type = (resp.headers.get('content-type') or '').lower() 
    if 'html' not in content_type and 'xml' not in content_type: 
        # Some CMSs return text/plain or application/xhtml+xml; we allow xml 
        raise ValueError(f"[HTML] URL does not appear to be HTML (content- type: {content_type})") 
 
    soup = BeautifulSoup(resp.text, 'html.parser') 
    _strip_noise(soup) 
 
    title = _extract_title(soup) 

    sections = _extract_sections(soup) 
    raw_text = _normalize_whitespace((soup.body or 
soup).get_text(separator='\n', strip=True)) 
 
    data = _to_custom_json(title, url, sections, raw_text) 
    out_path = _write_output_json(data, output_dir, prefix="html") 
    logger.info(f"[HTML] Wrote JSON: {out_path}") 
    return out_path 
 
def extract_html_to_json_from_file(html_path: str, output_dir: str) -> str: 
    """ 
    Read a local HTML file and write a JSON file in output_dir. 
    Returns the full JSON file path. 
    """ 
    if not os.path.exists(html_path): 
        raise FileNotFoundError(f"[HTML] File not found: {html_path}") 
 
    logger.info(f"[HTML] Reading local HTML: {html_path}") 
    # errors='ignore' to be resilient to irregular encodings 
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f: 
        html = f.read() 
 
    soup = BeautifulSoup(html, 'html.parser') 
    _strip_noise(soup) 
 
    title = _extract_title(soup) 
    sections = _extract_sections(soup) 
    raw_text = _normalize_whitespace((soup.body or 
soup).get_text(separator='\n', strip=True)) 
 
    data = _to_custom_json(title, str(Path(html_path).resolve()), sections, 
raw_text) 
    out_path = _write_output_json(data, output_dir, prefix="html") 
    logger.info(f"[HTML] Wrote JSON: {out_path}") 
    return out_path 
