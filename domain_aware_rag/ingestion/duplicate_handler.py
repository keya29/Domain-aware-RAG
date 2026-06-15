"""
Duplicate Detection & Merge Logic
==================================
Handles detection of duplicate documents and provides merge options.
Prevents duplicate re-ingestion by consolidating documents with same doc_id.
"""

import logging
from typing import Dict, Optional, Tuple
import psycopg2

logger = logging.getLogger(__name__)


def check_document_exists(conn, doc_id: str) -> Tuple[bool, Optional[Dict]]:
    """
    Check if document with same filename (stem) already exists in database.
    
    This enables merging documents by name: if you upload "myplan.pdf" twice,
    the second upload will be detected as a duplicate of the first, regardless
    of different timestamps or content hashes.
    
    Args:
        conn: Database connection
        doc_id: External document ID to check
               Format: {stem}{timestamp}{hash}
               Example: myplan_20260319_145622_a7f3e9c4
                        my_doc_name_20260319_145622_a7f3e9c4
        
    Returns:
        Tuple of (exists: bool, doc_info: dict or None)
        doc_info contains: {id, external_id, title, version, created_at, node_count}
    """
    try:
        # Extract the stem (filename) from doc_id using rsplit
        # Format: {stem}{YYYYMMDD_HHMMSS}{hash}
        # rsplit("_", 2) splits from right: [stem, timestamp, hash]
        parts = doc_id.rsplit("_", 2)
        
        if len(parts) == 3:
            stem = parts[0]  # Everything before the last two underscores
            logger.info(f"[Duplicate Check] Extracted stem: '{stem}' from doc_id: '{doc_id}'")
        else:
            # Fallback for edge cases (shouldn't happen with normal doc_ids)
            stem = doc_id
            logger.warning(f"[Duplicate Check] Could not parse doc_id format, using as-is: '{stem}'")
        
        logger.info(f"[Duplicate Check] Looking for documents with stem: '{stem}'")
        
        with conn.cursor() as cur:
            # Search for documents where external_id starts with the same stem
            # This matches any version of the same document
            cur.execute("""
                SELECT 
                    id,
                    external_id,
                    title,
                    version,
                    created_at,
                    (SELECT COUNT(*) FROM rag.nodes WHERE document_id = d.id) as node_count
                FROM rag.documents d
                WHERE external_id LIKE %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (f"{stem}_%",))
            
            result = cur.fetchone()
            if result:
                existing_doc_id = result[1]
                logger.info(f"[Duplicate Check] Found existing document: {existing_doc_id}")
                return True, {
                    'id': result[0],
                    'external_id': existing_doc_id,
                    'title': result[2],
                    'version': result[3],
                    'created_at': result[4],
                    'node_count': result[5] or 0
                }
            logger.info(f"[Duplicate Check] No existing documents found with stem: '{stem}'")
            return False, None
    except Exception as e:
        logger.error(f"Error checking document existence: {e}")
        raise


def merge_documents(conn, old_doc_id: str, new_doc_id: str, 
                   keep_strategy: str = "new") -> Dict:
    """
    Merge two documents with the same content.
    
    Args:
        conn: Database connection
        old_doc_id: UUID of existing document
        new_doc_id: UUID of document being re-ingested
        keep_strategy: "old" (keep existing), "new" (replace with new), or "both" (consolidate)
        
    Returns:
        Merge result dict: {success, message, old_node_count, new_node_count, strategy_used}
    """
    try:
        with conn.cursor() as cur:
            # Get counts before merge
            cur.execute("SELECT COUNT(*) FROM rag.nodes WHERE document_id = %s", (old_doc_id,))
            old_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM rag.nodes WHERE document_id = %s", (new_doc_id,))
            new_count = cur.fetchone()[0]
            
            if keep_strategy == "new":
                # Delete old document and all its nodes
                logger.info(f"[Merge] Removing old document {old_doc_id} (had {old_count} nodes)")
                cur.execute("DELETE FROM rag.nodes WHERE document_id = %s", (old_doc_id,))
                cur.execute("DELETE FROM rag.entities WHERE document_id = %s", (old_doc_id,))
                cur.execute("DELETE FROM rag.doc_tables WHERE document_id = %s", (old_doc_id,))
                cur.execute("DELETE FROM rag.documents WHERE id = %s", (old_doc_id,))
                conn.commit()
                
                return {
                    'success': True,
                    'message': f"Document merged: replaced old version (had {old_count} nodes) with new version ({new_count} nodes)",
                    'old_node_count': old_count,
                    'new_node_count': new_count,
                    'strategy_used': 'new',
                    'action': 'replaced'
                }
                
            elif keep_strategy == "old":
                # Delete new document, keep old
                logger.info(f"[Merge] Keeping old document {old_doc_id}, removing new version")
                cur.execute("DELETE FROM rag.nodes WHERE document_id = %s", (new_doc_id,))
                cur.execute("DELETE FROM rag.entities WHERE document_id = %s", (new_doc_id,))
                cur.execute("DELETE FROM rag.doc_tables WHERE document_id = %s", (new_doc_id,))
                cur.execute("DELETE FROM rag.documents WHERE id = %s", (new_doc_id,))
                conn.commit()
                
                return {
                    'success': True,
                    'message': f"Document merge skipped: keeping existing version ({old_count} nodes)",
                    'old_node_count': old_count,
                    'new_node_count': new_count,
                    'strategy_used': 'old',
                    'action': 'skipped'
                }
                
            elif keep_strategy == "both":
                # Mark new document as "merged" version in metadata
                logger.info(f"[Merge] Consolidating both versions: {old_doc_id} + {new_doc_id}")
                cur.execute("""
                    UPDATE rag.documents 
                    SET version = 'merged'
                    WHERE id = %s
                """, (new_doc_id,))
                conn.commit()
                
                return {
                    'success': True,
                    'message': f"Document consolidated: old version ({old_count} nodes) + new version ({new_count} nodes)",
                    'old_node_count': old_count,
                    'new_node_count': new_count,
                    'strategy_used': 'both',
                    'action': 'consolidated'
                }
            else:
                raise ValueError(f"Unknown merge strategy: {keep_strategy}")
                
    except Exception as e:
        logger.error(f"Error during document merge: {e}")
        return {
            'success': False,
            'message': f"Merge failed: {str(e)}",
            'old_node_count': 0,
            'new_node_count': 0,
            'strategy_used': None,
            'action': 'failed'
        }


def get_duplicate_info(conn, external_id: str) -> Optional[Dict]:
    """
    Get detailed information about existing document if duplicate found.
    
    Args:
        conn: Database connection
        external_id: Document external_id (doc_id)
        
    Returns:
        Document info dict or None if not found
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    id,
                    external_id,
                    title,
                    version,
                    created_at,
                    (SELECT COUNT(*) FROM rag.nodes WHERE document_id = d.id) as node_count,
                    (SELECT COUNT(*) FROM rag.entities WHERE document_id = d.id) as entity_count
                FROM rag.documents d
                WHERE external_id = %s
            """, (external_id,))
            
            result = cur.fetchone()
            if result:
                return {
                    'id': result[0],
                    'external_id': result[1],
                    'title': result[2],
                    'version': result[3],
                    'created_at': result[4],
                    'node_count': result[5] or 0,
                    'entity_count': result[6] or 0
                }
            return None
    except Exception as e:
        logger.error(f"Error getting duplicate info: {e}")
        return None