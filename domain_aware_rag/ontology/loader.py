"""
OntologyLoader — loads, validates, and caches domain ontology JSON files.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    AnswerGenerationConfig,
    OntologyConfig,
    QueryExpansionConfig,
    RetrievalHints,
)

logger = logging.getLogger(__name__)

# Default path to ontologies directory (relative to repository root)
_DEFAULT_ONTOLOGIES_DIR = Path(__file__).parent.parent.parent / "ontologies"


class OntologyLoader:
    """
    Loads domain ontology JSON files from a directory.

    Usage:
        loader = OntologyLoader()
        ontologies = loader.load_all()
        insurance = loader.load("insurance")
    """

    def __init__(self, ontologies_dir: Optional[str] = None):
        self.ontologies_dir = Path(ontologies_dir) if ontologies_dir else _DEFAULT_ONTOLOGIES_DIR
        self._cache: Dict[str, OntologyConfig] = {}
        logger.info(f"[OntologyLoader] Using ontologies dir: {self.ontologies_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_domains(self) -> List[str]:
        """Return sorted list of available domain IDs."""
        domain_ids = []
        if not self.ontologies_dir.exists():
            logger.warning(f"[OntologyLoader] Ontologies directory not found: {self.ontologies_dir}")
            return domain_ids

        for f in sorted(self.ontologies_dir.glob("*.json")):
            if f.stem == "ontology_schema":
                continue
            domain_ids.append(f.stem)
        return domain_ids

    def load(self, domain_id: str, force_reload: bool = False) -> Optional[OntologyConfig]:
        """
        Load a single domain ontology by domain_id.

        Returns OntologyConfig, or None if not found / invalid.
        Uses an in-memory cache unless force_reload=True.
        """
        if not force_reload and domain_id in self._cache:
            return self._cache[domain_id]

        path = self.ontologies_dir / f"{domain_id}.json"
        if not path.exists():
            logger.error(f"[OntologyLoader] Ontology file not found: {path}")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            config = self._parse(data)
            self._cache[domain_id] = config
            logger.info(f"[OntologyLoader] Loaded domain '{domain_id}': {len(config.entities)} inline entities, "
                        f"{len(config.anchor_taxonomy)} anchor buckets")
            return config
        except Exception as e:
            logger.error(f"[OntologyLoader] Failed to load '{domain_id}': {e}")
            return None

    def load_all(self) -> Dict[str, OntologyConfig]:
        """Load all available domain ontologies. Returns dict keyed by domain_id."""
        result = {}
        for domain_id in self.list_domains():
            cfg = self.load(domain_id)
            if cfg:
                result[domain_id] = cfg
        return result

    def save(self, config: OntologyConfig, overwrite: bool = False) -> bool:
        """
        Save an OntologyConfig back to its JSON file.
        Returns True on success.
        """
        path = self.ontologies_dir / f"{config.domain_id}.json"
        if path.exists() and not overwrite:
            logger.warning(f"[OntologyLoader] File exists and overwrite=False: {path}")
            return False

        try:
            self.ontologies_dir.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
            # Invalidate cache
            self._cache.pop(config.domain_id, None)
            logger.info(f"[OntologyLoader] Saved domain '{config.domain_id}' to {path}")
            return True
        except Exception as e:
            logger.error(f"[OntologyLoader] Failed to save '{config.domain_id}': {e}")
            return False

    def save_from_dict(self, data: dict, overwrite: bool = True) -> tuple:
        """
        Validate and save an ontology from a raw dict (e.g., uploaded via UI).
        Returns (success: bool, message: str, config: Optional[OntologyConfig])
        """
        # Basic validation
        required = ["domain_id", "display_name", "description", "embedding_strategy",
                    "query_expansion", "answer_generation"]
        for key in required:
            if key not in data:
                return False, f"Missing required field: '{key}'", None

        if not data["domain_id"].replace("_", "").isalnum():
            return False, "domain_id must be alphanumeric with underscores", None

        try:
            config = self._parse(data)
            success = self.save(config, overwrite=overwrite)
            if success:
                return True, f"Domain '{config.domain_id}' saved successfully", config
            return False, "Save failed (file may already exist)", config
        except Exception as e:
            return False, f"Parse/save error: {e}", None

    def invalidate_cache(self, domain_id: Optional[str] = None):
        """Clear the cache for a specific domain or all domains."""
        if domain_id:
            self._cache.pop(domain_id, None)
        else:
            self._cache.clear()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self, data: dict) -> OntologyConfig:
        """Parse a raw JSON dict into a typed OntologyConfig."""

        qe_raw = data.get("query_expansion", {})
        query_expansion = QueryExpansionConfig(
            system_role=qe_raw.get("system_role", "You translate questions into formal search queries."),
            domain_context=qe_raw.get("domain_context", "enterprise documents"),
            validation_scope=qe_raw.get("validation_scope", "enterprise knowledge"),
        )

        ag_raw = data.get("answer_generation", {})
        answer_generation = AnswerGenerationConfig(
            system_role=ag_raw.get("system_role", "You are a knowledge assistant. Answer using only the provided context."),
            greeting_response=ag_raw.get("greeting_response", "Hello! How can I help you today?"),
        )

        rh_raw = data.get("retrieval_hints", {})
        retrieval_hints = RetrievalHints(
            service_indicators=rh_raw.get("service_indicators", []),
            exclusion_indicators=rh_raw.get("exclusion_indicators", []),
            process_indicators=rh_raw.get("process_indicators", []),
            hierarchy_boost_paths=rh_raw.get("hierarchy_boost_paths", []),
        )

        return OntologyConfig(
            domain_id=data["domain_id"],
            display_name=data["display_name"],
            description=data["description"],
            version=data.get("version", "1.0"),
            embedding_strategy=data.get("embedding_strategy", "embed_all"),
            entity_threshold=int(data.get("entity_threshold", 1)),
            query_expansion=query_expansion,
            answer_generation=answer_generation,
            retrieval_hints=retrieval_hints,
            entity_levels=data.get("entity_levels", []),
            keyword_acronyms=data.get("keyword_acronyms", []),
            entity_catalogue_path=data.get("entity_catalogue_path"),
            entities=data.get("entities", []),
            entity_examples=data.get("entity_examples", []),
            anchor_taxonomy=data.get("anchor_taxonomy", {}),
            example_queries=data.get("example_queries", []),
        )
