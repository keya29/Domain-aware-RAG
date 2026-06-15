"""
OntologyRegistry — singleton manager that tracks the active domain and persists the choice.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .config import OntologyConfig
from .loader import OntologyLoader

logger = logging.getLogger(__name__)

# Paths relative to this file
_DEFAULT_ACTIVE_DOMAIN_PATH = Path(__file__).parent.parent / "data" / "active_domain.json"


class OntologyRegistry:
    """
    Singleton registry to manage available domain ontologies and track/persist the active domain.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OntologyRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, ontologies_dir: Optional[str] = None, active_domain_path: Optional[str] = None):
        if self._initialized:
            return

        self.loader = OntologyLoader(ontologies_dir)
        self.active_domain_path = Path(active_domain_path) if active_domain_path else _DEFAULT_ACTIVE_DOMAIN_PATH
        self._active_domain_id = "insurance"  # Default fallback
        self._initialized = True

        # Load persisted choice if available
        self._load_active_domain_id()

    def _load_active_domain_id(self):
        """Loads the active domain ID from persistence or defaults to insurance."""
        should_save = False

        if self.active_domain_path.exists():
            try:
                with open(self.active_domain_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                domain_id = data.get("active_domain_id")
                if domain_id in self.loader.list_domains():
                    self._active_domain_id = domain_id
                    logger.info(f"[OntologyRegistry] Loaded active domain: {self._active_domain_id}")
                    return
                should_save = True
            except Exception as e:
                logger.warning(f"[OntologyRegistry] Failed to load active domain file: {e}")
                should_save = True

        # Fallback if file doesn't exist or refers to an invalid domain
        domains = self.loader.list_domains()
        if "insurance" not in domains and domains:
            self._active_domain_id = domains[0]
        else:
            self._active_domain_id = "insurance"
        logger.info(f"[OntologyRegistry] Default active domain: {self._active_domain_id}")

        if should_save or not self.active_domain_path.exists():
            self._save_active_domain_id()

    def _save_active_domain_id(self):
        """Persists the active domain ID to a JSON file."""
        try:
            self.active_domain_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.active_domain_path, "w", encoding="utf-8") as f:
                json.dump({"active_domain_id": self._active_domain_id}, f, indent=2)
            logger.info(f"[OntologyRegistry] Persisted active domain: {self._active_domain_id}")
        except Exception as e:
            logger.error(f"[OntologyRegistry] Failed to persist active domain: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_active_domain_id(self) -> str:
        """Returns the ID of the currently active domain."""
        return self._active_domain_id

    def get_active_ontology(self) -> OntologyConfig:
        """
        Returns the OntologyConfig of the currently active domain.
        Falls back to a default config if loading fails.
        """
        config = self.loader.load(self._active_domain_id)
        if config is None:
            logger.warning(f"[OntologyRegistry] Registry failed to load active domain '{self._active_domain_id}'. Returning default.")
            # Create a basic default config
            return OntologyConfig(
                domain_id=self._active_domain_id,
                display_name="Default Domain",
                description="Default domain config because active domain failed to load."
            )
        return config

    def get_ontology(self, domain_id: str) -> Optional[OntologyConfig]:
        """Loads and returns config for a specific domain."""
        return self.loader.load(domain_id)

    def switch_domain(self, domain_id: str) -> bool:
        """
        Switch the active domain. Validates that the domain exists.
        Returns True if successful, False otherwise.
        """
        available_domains = self.loader.list_domains()
        if domain_id not in available_domains:
            logger.error(f"[OntologyRegistry] Cannot switch to non-existent domain: {domain_id}")
            return False

        self._active_domain_id = domain_id
        self._save_active_domain_id()
        return True

    def list_domains(self) -> List[str]:
        """Returns sorted list of available domain IDs."""
        return self.loader.list_domains()

    def get_loader(self) -> OntologyLoader:
        """Returns the OntologyLoader instance."""
        return self.loader
