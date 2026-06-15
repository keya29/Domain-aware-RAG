import psycopg2
import logging
import re
import os
from typing import List, Dict, Any, Optional, Tuple, Set
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
import numpy as np
from collections import deque
import threading

# Configure logging
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "log.txt")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
    force=True
)

logger = logging.getLogger(__name__)

# ---------------------------
# Progress Tracker (FIXED ONLY SYNTAX)
# ---------------------------

class IngestionProgressTracker:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    # Initialize attributes on the newly created instance
                    cls._initialized = False
        return cls._instance

    def __init__(self):
        # Ensure initialization only happens once
        if not hasattr(self, "_initialized") or not self._initialized:
            with self._lock:
                if not hasattr(self, "_initialized") or not self._initialized:
                    self._events = deque(maxlen=100)
                    self._doc_events = {}
                    self._current_doc_id = None
                    self._initialized = True

    def start_ingestion(self, doc_id: str):
        """Mark the start of ingestion for a document."""
        self._current_doc_id = doc_id
        if doc_id not in self._doc_events:
            self._doc_events[doc_id] = []

    def log_event(self, message: str):
        """Log a simple message event."""
        if self._current_doc_id:
            self._events.append(message)
            if self._current_doc_id in self._doc_events:
                self._doc_events[self._current_doc_id].append(message)

    def record_event(self, doc_id: str, stage: str, status: str, message: str, duration_ms: int = 0):
        """
        Record a structured pipeline event. 
        Matches the expected signature in pipeline_log_handler.py.
        """
        event_entry = {
            "stage": stage,
            "status": status,
            "message": message,
            "duration_ms": duration_ms
        }
        
        self._events.append(f"[{stage}] {status}: {message}")
        
        if doc_id not in self._doc_events:
            self._doc_events[doc_id] = []
        self._doc_events[doc_id].append(event_entry)
        
        # Also update current doc id if it's different
        self._current_doc_id = doc_id

    def get_events(self):
        """Return all global events."""
        return list(self._events)

    def get_document_events(self, doc_id: str):
        """Return events for a specific document."""
        return self._doc_events.get(doc_id, [])
# ---------------------------------------------------------------------------
# Global singleton instance (REQUIRED for imports)
# ---------------------------------------------------------------------------
progress_tracker: IngestionProgressTracker = IngestionProgressTracker()

# ---------------------------
# STOP WORDS (unchanged)
# ---------------------------

STOP_WORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",
    'your','yours','yourself','yourselves','he','him','his','himself','she',"she's",'her','hers',
    'herself','it',"it's",'its','itself','they','them','their','theirs','themselves','what','which',
    'who','whom','this','that',"that'll",'these','those','am','is','are','was','were','be','been',
    'being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if',
    'or','because','as','until','while','of','at','by','for','with','about','against','between',
    'into','through','during','before','after','above','below','to','from','up','down','in','out',
    'on','off','over','under','again','further','then','once','here','there','when','where','why',
    'how','all','any','both','each','few','more','most','other','some','such','no','nor','not',
    'only','own','same','so','than','too','very','s','t','can','will','just','don',"don't",
    'should',"should've",'now','d','ll','m','o','re','ve','y','ain','aren',"aren't",'couldn',
    "couldn't",'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",
    'isn',"isn't",'ma','mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",
    'shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",'wouldn',"wouldn't"
}

# ---------------------------
# DEBUG PRINT FIXED
# ---------------------------

def debug_print(function_name: str, title: str, data: Any):
    separator = "=" * 30
    header = f"\n{separator} STEP start: [{function_name}] {title} {separator}"
    footer = f"{separator} STEP end: [{function_name}] {title} {separator}\n"

    msg_parts = [header]

    if data is None:
        msg_parts.append("Data: None")

    elif isinstance(data, (list, set)):
        items = list(data)
        msg_parts.append(f"Type: {type(data).__name__}, Count: {len(items)}")

        if items:
            if isinstance(items[0], tuple):
                msg_parts.append("Data (Rows):")
                for i, row in enumerate(items[:20]):
                    msg_parts.append(f"  [{i}] {row}")
            else:
                msg_parts.append(f"Data: {items}")

    elif isinstance(data, dict):
        msg_parts.append(f"Type: dict, Keys Count: {len(data)}")
        msg_parts.append(f"Data: {data}")

    else:
        s = str(data)
        if len(s) > 1000:
            msg_parts.append(f"Data (Truncated): {s[:1000]}...")
        else:
            msg_parts.append(f"Data: {s}")

    msg_parts.append(footer)
    logger.info("\n".join(msg_parts))


# ---------------------------
# DB CONFIG
# ---------------------------

DB_HOST = "localhost"
DB_NAME = "rag_system"
DB_USER = "postgres"
DB_PASS = "root"
DB_PORT = 5432


@dataclass
class RetrievalConfig:
    top_k_semantic: int = 80
    similarity_threshold: float = 0.0
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    top_k_final: int = 3
    include_parents: bool = True
    max_parent_levels: int = 4


class HybridRetriever:

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self.conn = self._get_db_connection()

    def _get_db_connection(self):
        return psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT
        )


def retrieve_for_query(query: str):
    retriever = HybridRetriever()
    try:
        return retriever.retrieve(query=query)
    finally:
        retriever.close()


if __name__ == "__main__":
    logger.info("Testing hybrid retrieval system...")