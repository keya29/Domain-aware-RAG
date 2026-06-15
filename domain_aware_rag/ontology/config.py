"""
OntologyConfig — typed dataclass for a loaded domain ontology.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class QueryExpansionConfig:
    system_role: str
    domain_context: str
    validation_scope: str


@dataclass
class AnswerGenerationConfig:
    system_role: str
    greeting_response: str


@dataclass
class RetrievalHints:
    service_indicators: List[str] = field(default_factory=list)
    exclusion_indicators: List[str] = field(default_factory=list)
    process_indicators: List[str] = field(default_factory=list)
    hierarchy_boost_paths: List[str] = field(default_factory=list)


@dataclass
class OntologyConfig:
    """
    Fully-typed representation of a domain ontology JSON file.
    All fields map 1-to-1 to the ontology_schema.json specification.
    """

    # Identity
    domain_id: str
    display_name: str
    description: str
    version: str = "1.0"

    # Embedding
    embedding_strategy: str = "embed_all"   # embed_all | embed_entity_rich | embed_above_threshold
    entity_threshold: int = 1

    # Sub-configs
    query_expansion: QueryExpansionConfig = field(default_factory=lambda: QueryExpansionConfig(
        system_role="You translate questions into formal document search queries.",
        domain_context="enterprise documents",
        validation_scope="enterprise knowledge"
    ))
    answer_generation: AnswerGenerationConfig = field(default_factory=lambda: AnswerGenerationConfig(
        system_role="You are a knowledge assistant. Answer using only the provided context.",
        greeting_response="Hello! I'm a knowledge assistant. How can I help you today?"
    ))
    retrieval_hints: RetrievalHints = field(default_factory=RetrievalHints)

    # Entity catalogue
    entity_levels: List[str] = field(default_factory=list)
    keyword_acronyms: List[str] = field(default_factory=list)
    entity_catalogue_path: Optional[str] = None
    entities: List[Dict[str, str]] = field(default_factory=list)
    entity_examples: List[Dict[str, Any]] = field(default_factory=list)

    # Query anchors
    anchor_taxonomy: Dict[str, Any] = field(default_factory=dict)

    # UI
    example_queries: List[str] = field(default_factory=list)

    # -----------------------------------------------------------------------
    # Derived helpers
    # -----------------------------------------------------------------------

    def has_inline_catalogue(self) -> bool:
        """Returns True if the ontology provides an inline entity list."""
        return bool(self.entities)

    def has_external_catalogue(self) -> bool:
        """Returns True if the ontology points to an Excel catalogue file."""
        return bool(self.entity_catalogue_path)

    def get_effective_catalogue(self) -> Optional[str]:
        """
        Returns the path to the external catalogue, or None if inline entities
        should be used instead (inline takes precedence).
        """
        if self.has_inline_catalogue():
            return None  # use inline entities
        return self.entity_catalogue_path

    def get_effective_embedding_strategy(self) -> str:
        """
        Returns embed_all if the domain has no catalogue at all
        (no inline entities and no external catalogue path), to ensure
        nodes still get embeddings for retrieval.
        """
        if not self.has_inline_catalogue() and not self.has_external_catalogue():
            return "embed_all"
        return self.embedding_strategy

    def to_dict(self) -> Dict[str, Any]:
        """Serialize back to a JSON-compatible dict."""
        return {
            "domain_id": self.domain_id,
            "display_name": self.display_name,
            "description": self.description,
            "version": self.version,
            "embedding_strategy": self.embedding_strategy,
            "entity_threshold": self.entity_threshold,
            "query_expansion": {
                "system_role": self.query_expansion.system_role,
                "domain_context": self.query_expansion.domain_context,
                "validation_scope": self.query_expansion.validation_scope,
            },
            "answer_generation": {
                "system_role": self.answer_generation.system_role,
                "greeting_response": self.answer_generation.greeting_response,
            },
            "retrieval_hints": {
                "service_indicators": self.retrieval_hints.service_indicators,
                "exclusion_indicators": self.retrieval_hints.exclusion_indicators,
                "process_indicators": self.retrieval_hints.process_indicators,
                "hierarchy_boost_paths": self.retrieval_hints.hierarchy_boost_paths,
            },
            "entity_levels": self.entity_levels,
            "keyword_acronyms": self.keyword_acronyms,
            "entity_catalogue_path": self.entity_catalogue_path,
            "entities": self.entities,
            "entity_examples": self.entity_examples,
            "anchor_taxonomy": self.anchor_taxonomy,
            "example_queries": self.example_queries,
        }
