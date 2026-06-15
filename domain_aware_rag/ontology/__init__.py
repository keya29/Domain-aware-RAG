"""
EKIP Ontology Module
====================
Provides pluggable, domain-aware ontology management for the
Enterprise Knowledge Intelligence Platform-RAG.
"""

from .config import OntologyConfig
from .loader import OntologyLoader
from .registry import OntologyRegistry

__all__ = ["OntologyConfig", "OntologyLoader", "OntologyRegistry"]
