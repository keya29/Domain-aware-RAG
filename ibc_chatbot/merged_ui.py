"""
IBC Medicare Chatbot - Streamlit UI
=====================================
Two-panel layout:
Left  -> Chat interface
Right -> Subgraph visualization + Document sources
"""

import streamlit as st
import os
import sys
import time
import re
import html
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from sentence_transformers import SentenceTransformer
import pandas as pd

# Import ontology modules
sys.path.insert(0, os.path.dirname(__file__))
from ontology.registry import OntologyRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Knowledge Intelligence Platform-RAG",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "chat_history": [],
        "raw_turns": [],
        "active_turn_idx": -1,
        "use_validation": False,
        "right_panel_view": "graph",
        "pending_query": None,
        "pdf_upload_status": None,
        "pdf_upload_message": "",
        "current_pdf_name": "",
        "current_doc_id": "",
        "pdf_upload_url": "",
        "pdf_upload_file": None,
        "last_upload_result": {},
        "active_tab": "chat",
        "active_domain_id": "insurance",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

if st.session_state.right_panel_view not in ("graph", "docs"):
    st.session_state.right_panel_view = "graph"

# Initialize Ontology Registry
registry = OntologyRegistry()

# Sync session state with registry
if st.session_state.active_domain_id != registry.get_active_domain_id():
    st.session_state.active_domain_id = registry.get_active_domain_id()

# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_retriever(_version: str = "v3"):
    sys.path.insert(0, os.path.dirname(__file__))

    # Load .env from parent directory
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    from dotenv import load_dotenv
    load_dotenv(env_path)

    from retrieval.hybrid_retrieval import HybridRetriever, RetrievalConfig

    config = RetrievalConfig(
        top_k_semantic=80,
        similarity_threshold=0.0,
        vector_weight=0.7,
        bm25_weight=0.3,
        top_k_final=5,
        include_parents=True,
        max_parent_levels=4,
    )

    return HybridRetriever(config)


@st.cache_resource(show_spinner=False)
def load_embedmodel():
    return SentenceTransformer("all-MiniLM-L6-v2")


def _expand_query(query: str, use_validation: bool) -> Dict[str, Any]:
    # Load .env from parent directory
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    from dotenv import load_dotenv
    load_dotenv(env_path)

    if use_validation:
        from retrieval.query_validation import expand_and_embed
    else:
        from retrieval.query_selector import expand_and_embed

    return expand_and_embed(query)


# ---------------------------------------------------------------------------
# Upload processing
# ---------------------------------------------------------------------------
def _process_pdf_upload(
    pdf_input: Optional[str],
    uploaded_file,
) -> Tuple[bool, str, str, dict]:

    import tempfile

    try:
        input_source = None
        display_name = ""

        if uploaded_file:
            original_name = uploaded_file.name or "uploaded_file"
            temp_path = os.path.join(tempfile.gettempdir(), original_name)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            input_source = temp_path
            display_name = original_name

        elif pdf_input:
            input_source = pdf_input.strip()
            display_name = os.path.basename(input_source)

        else:
            return False, "No input provided", "", {}

        from ingestion.multi_doc_orchestrator import ingest_any_from_url_or_file

        success, message, result = ingest_any_from_url_or_file(
            input_path_or_url=input_source,
            doc_id=None,
            doc_title=None,
            doc_version="1.0",
            outputs_dir="outputs",
        )

        if success:
            return True, message, display_name, result

        return False, message, "", {}

    except Exception as e:
        logger.exception("Upload error")
        return False, str(e), "", {}


# ---------------------------------------------------------------------------
# Retrieval wrapper
# ---------------------------------------------------------------------------
def _retrieve_with_nodes(
    retriever,
    query: str,
    expansion_result: Dict[str, Any],
    model: Optional[SentenceTransformer] = None,
):

    expanded_queries = expansion_result.get("queries", [])
    anchors = expansion_result.get("anchors", [])

    result = retriever.retrieve(
        query=query,
        query_expansions=expanded_queries[1:] if len(expanded_queries) > 1 else None,
        anchor_entities=anchors or None,
        model=model,
    )

    # Detect query type from retriever's internal logic
    query_type = retriever._detect_query_type(query)
    
    # Get retrieval configuration weights
    vector_weight = getattr(retriever.config, 'vector_weight', 0.7)
    bm25_weight = getattr(retriever.config, 'bm25_weight', 0.3)
    
    # Determine if entity filtering was applied
    entity_filtering_applied = bool(anchors)
    retrieval_mode = "Entity-first retrieval" if entity_filtering_applied else "Hybrid retrieval (BM25 + Vector)"

    return {
        "context": getattr(result, "context_text", ""),
        "nodes": getattr(result, "enriched_nodes", []),
        "top_table": getattr(result, "top_results", []),
        "query_understanding": {
            "intent": query_type.replace('_', ' ').title(),
            "entities": anchors,
            "expanded_queries": expanded_queries
        },
        "retrieval_strategy": {
            "mode": retrieval_mode,
            "vector_weight": vector_weight,
            "bm25_weight": bm25_weight,
            "entity_filtering_applied": entity_filtering_applied
        },
        "subgraph_edges": getattr(result, "subgraph_edges", []),
        "source_documents": getattr(result, "source_documents", [])
    }


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
def _generate_answer(query: str, context: str, chat_history: List) -> str:
    from generation.answer_generator import generate_answer

    return generate_answer(
        user_query=query,
        retrieved_context=context,
        chat_history=chat_history,
        temperature=0.2,
        max_tokens=1024,
    )


# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------
def _render_chat():
    if not st.session_state.chat_history:
        st.info("Ask a question to get started.")
        return

    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = html.escape(msg["content"])

        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**Bot:** {content}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")

    st.session_state.use_validation = st.toggle(
        "Enable Query Guardrails",
        value=st.session_state.use_validation,
    )

    st.markdown("---")
    _render_domain_selector()
    st.markdown("---")

    turns = st.session_state.raw_turns

    if turns:
        for i, t in enumerate(reversed(turns)):
            idx = len(turns) - 1 - i
            label = t["query"][:40]

            if st.button(label, key=f"hist_{idx}"):
                st.session_state.active_turn_idx = idx
                st.rerun()

        if st.button("Clear Conversation"):
            st.session_state.chat_history = []
            st.session_state.raw_turns = []
            st.rerun()
    else:
        st.caption("No history yet.")


# ---------------------------------------------------------------------------
# New UI Components
# ---------------------------------------------------------------------------

def _render_query_understanding_panel(query_understanding: Dict[str, Any]):
    """Render the Query Understanding Panel"""
    with st.expander("🔍 Query Understanding", expanded=True):
        if not query_understanding:
            st.info("No query understanding data available.")
            return
            
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Intent:**")
            st.markdown(f"{query_understanding.get('intent', 'Unknown')}")
            
        with col2:
            st.markdown("**Entities:**")
            entities = query_understanding.get('entities', [])
            if entities:
                for entity in entities:
                    st.markdown(f"• {entity}")
            else:
                st.markdown("No entities detected")
                
        if query_understanding.get('expanded_queries') and len(query_understanding['expanded_queries']) > 1:
            st.markdown("**Expanded Queries:**")
            for i, q in enumerate(query_understanding['expanded_queries'][1:], 1):
                st.markdown(f"{i}. {q}")

def _render_retrieval_strategy_panel(retrieval_strategy: Dict[str, Any]):
    """Render the Retrieval Strategy Panel"""
    with st.expander("⚙️ Retrieval Strategy", expanded=True):
        if not retrieval_strategy:
            st.info("No retrieval strategy data available.")
            return
            
        st.markdown(f"**Retrieval Mode:** {retrieval_strategy.get('mode', 'Unknown')}")
        
        if 'Hybrid' in retrieval_strategy.get('mode', ''):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Vector Weight:** {retrieval_strategy.get('vector_weight', 0.7)}")
            with col2:
                st.markdown(f"**BM25 Weight:** {retrieval_strategy.get('bm25_weight', 0.3)}")
                
        st.markdown(f"**Entity Filtering Applied:** {'Yes' if retrieval_strategy.get('entity_filtering_applied', False) else 'No'}")

def _render_top_retrieved_nodes_panel(top_table: List[Dict[str, Any]]):
    """Render the Top Retrieved Nodes Panel"""
    with st.expander("📚 Retrieved Context", expanded=False):
        if not top_table:
            st.info("No retrieved nodes available.")
            return
            
        for i, node in enumerate(top_table[:5], 1):
            st.markdown(f"**{i}. {node.get('title', 'Untitled')}**")
            
            # Hierarchy Path
            hierarchy_path = node.get('hierarchy_path', [])
            if hierarchy_path:
                st.markdown(f"*Hierarchy:* {' > '.join(hierarchy_path)}")
            else:
                st.markdown(f"*Level:* {node.get('level', 'Unknown')}")
                
            # Scores
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"*Final Score:* {node.get('final_score', 0):.3f}")
            with col2:
                st.markdown(f"*Vector Score:* {node.get('vector_score', 0):.3f}")
            with col3:
                st.markdown(f"*BM25 Score:* {node.get('bm25_score', 0):.3f}")
                
            # Content snippet
            content = node.get('content', node.get('text', ''))
            if content:
                snippet = content[:200] + "..." if len(content) > 200 else content
                st.markdown(f"*Snippet:* {snippet}")
                
            st.markdown("---")

def _render_enhanced_subgraph(turn_data: Optional[Dict]):
    """Render enhanced subgraph visualization"""
    if not turn_data:
        st.info("No graph data yet. Ask a question.")
        return
        
    nodes = turn_data.get("nodes", [])
    edges = turn_data.get("subgraph_edges", [])
    
    if not nodes:
        st.warning("No graph data available.")
        return
        
    # Legend
    st.markdown("**Legend:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🟄 **Document Node**")
    with col2:
        st.markdown("🔵 **Section Node**")
    with col3:
        st.markdown("🟢 **Answer Node**")
    st.markdown("---")
    
    # Display nodes with enhanced information
    st.markdown("**Retrieved Nodes:**")
    for node in nodes[:10]:  # Limit to 10 for readability
        level = node.get('level', 0)
        if level == 0:
            icon = "🟄"
        elif level <= 2:
            icon = "🔵"
        else:
            icon = "🟢"
            
        title = node.get('title', 'Untitled')
        hierarchy_path = node.get('hierarchy_path', [])
        path_str = ' > '.join(hierarchy_path) if hierarchy_path else f"Level {level}"
        
        st.markdown(f"{icon} **{title}**")
        st.markdown(f"*Path:* {path_str}")
        st.markdown(f"*Node ID:* {node.get('node_id', 'Unknown')}")
        st.markdown("---")
        
    # Display edges
    if edges:
        st.markdown("**Relationships:**")
        for edge in edges[:10]:  # Limit for readability
            source_title = edge.get('source_title', 'Unknown')
            target_title = edge.get('target_title', 'Unknown')
            st.markdown(f"📗 {source_title} → {target_title}")

def _render_answer_grounding_panel(turn_data: Optional[Dict]):
    """Render the Answer Grounding Panel"""
    if not turn_data:
        st.info("No grounding data available.")
        return
        
    answer = turn_data.get('answer', '')
    nodes = turn_data.get('nodes', [])
    
    if not answer or not nodes:
        st.info("No grounding data available.")
        return
        
    with st.expander("🧠 Answer Grounding", expanded=False):
        st.markdown("**Sections used in answer:**")
        
        for i, node in enumerate(nodes[:5], 1):
            title = node.get('title', 'Untitled')
            content = node.get('content', node.get('text', ''))
            
            # Simple text overlap for grounding
            if content:
                content_words = set(content.lower().split())
                answer_words = set(answer.lower().split())
                overlap = content_words & answer_words
                
                if overlap:
                    overlap_score = len(overlap) / len(answer_words) * 100
                    st.markdown(f"**{i}. {title}** (Overlap: {overlap_score:.1f}%)")
                    
                    # Show key overlapping phrases
                    key_phrases = list(overlap)[:5]  # Show up to 5 overlapping words
                    if key_phrases:
                        st.markdown(f"*Key terms:* {', '.join(key_phrases)}")
                        
                    # Show snippet
                    snippet = content[:150] + "..." if len(content) > 150 else content
                    st.markdown(f"*Content:* {snippet}")
                else:
                    st.markdown(f"**{i}. {title}** (No direct overlap)")
                    
            st.markdown("---")

def _render_domain_selector():
    """Render active ontology domain selector and persistence status."""
    active_domain = st.session_state.active_domain_id
    domains = registry.list_domains()

    if not domains:
        st.warning("No ontology domains available. Check `Domain_aware_RAG/ontologies/`.")
        return

    current_index = domains.index(active_domain) if active_domain in domains else 0
    selected_domain = st.selectbox(
        "Active Ontology Domain",
        options=domains,
        index=current_index,
        format_func=lambda x: x.replace("_", " ").title(),
        key="active_domain_selector",
    )

    if selected_domain != active_domain:
        if registry.switch_domain(selected_domain):
            st.session_state.active_domain_id = selected_domain
            st.success(f"Active domain switched to '{selected_domain}'.")
            st.experimental_rerun()
        else:
            st.error(f"Unable to switch to domain '{selected_domain}'.")

    st.caption(f"Persisted active domain at: {registry.active_domain_path}")


def _render_example_queries():
    """Render clickable example queries from the active ontology."""
    st.markdown("**Example Queries:**")
    ontology = registry.get_active_ontology()
    examples = ontology.example_queries or [
        "What is the copay for ambulance services?",
        "Which services are not covered?",
        "Do I need prior authorization for emergency care?"
    ]

    if not examples:
        st.info("This domain has no example queries configured.")
        return

    for example in examples:
        if st.button(example, key=f"example_{hash(example)}"):
            st.session_state.pending_query = example
            st.rerun()


def _render_ontology_manager():
    """Render the Ontology Manager tab."""
    ontology = registry.get_active_ontology()
    loader = registry.get_loader()

    st.subheader("Ontology Manager")
    st.markdown(f"**Domain ID:** `{ontology.domain_id}`")
    st.markdown(f"**Display Name:** {ontology.display_name}")
    st.markdown(f"**Version:** {ontology.version}")
    st.markdown(f"**Embedding Strategy:** {ontology.embedding_strategy}")
    st.markdown(f"**Entity Threshold:** {ontology.entity_threshold}")
    st.markdown("---")

    st.markdown("**Description:**")
    st.write(ontology.description)

    with st.expander("Entity Browser", expanded=True):
        if ontology.entities:
            entity_filter = st.text_input("Filter entities", key="entity_browser_filter")
            df = pd.DataFrame(ontology.entities)
            if entity_filter:
                mask = df.apply(lambda row: row.astype(str).str.contains(entity_filter, case=False, na=False).any(), axis=1)
                filtered = df[mask]
            else:
                filtered = df
            st.dataframe(filtered)
        else:
            st.info("No inline entities available for this domain.")

    with st.expander("Anchor Taxonomy", expanded=False):
        if ontology.anchor_taxonomy:
            st.json(ontology.anchor_taxonomy)
        else:
            st.info("No anchor taxonomy configured.")
        anchor_text = st.text_area(
            "Anchor taxonomy JSON",
            value=json.dumps(ontology.anchor_taxonomy or {}, indent=2),
            height=240,
            key="anchor_taxonomy_editor",
        )

    with st.expander("Example Queries", expanded=False):
        example_text = st.text_area(
            "Example queries (one per line)",
            value="\n".join(ontology.example_queries or []),
            height=180,
            key="ontology_example_queries",
        )

    with st.expander("Embedding Strategy", expanded=False):
        strategy_options = [
            "embed_all",
            "embed_entity_rich",
            "embed_above_threshold",
        ]
        selected_strategy = st.selectbox(
            "Embedding strategy",
            options=strategy_options,
            index=strategy_options.index(ontology.embedding_strategy) if ontology.embedding_strategy in strategy_options else 1,
            key="ontology_embedding_strategy",
        )
        threshold = st.number_input(
            "Entity threshold",
            min_value=0,
            value=ontology.entity_threshold or 1,
            step=1,
            key="ontology_entity_threshold",
        )

    with st.expander("Ontology JSON / Upload", expanded=False):
        uploaded = st.file_uploader("Upload or replace ontology JSON", type=["json"], key="ontology_upload")
        if uploaded:
            try:
                payload = json.loads(uploaded.read().decode("utf-8"))
                success, message, saved_config = loader.save_from_dict(payload, overwrite=True)
                if success:
                    st.success(message)
                    loader.invalidate_cache(saved_config.domain_id if saved_config else None)
                else:
                    st.error(message)
            except Exception as e:
                st.error(f"Upload failed: {e}")

    if st.button("Save Ontology Settings", key="save_ontology_settings"):
        try:
            anchor_payload = json.loads(anchor_text or "{}")
        except json.JSONDecodeError as e:
            st.error(f"Anchor taxonomy JSON is invalid: {e}")
            return

        ontology.example_queries = [q.strip() for q in example_text.splitlines() if q.strip()]
        ontology.anchor_taxonomy = anchor_payload
        ontology.embedding_strategy = selected_strategy
        ontology.entity_threshold = threshold

        success = loader.save(ontology, overwrite=True)
        if success:
            st.success("Ontology saved successfully.")
            loader.invalidate_cache(ontology.domain_id)
        else:
            st.error("Failed to save ontology. Please check the ontology contents.")


# ---------------------------------------------------------------------------
# Upload UI
# ---------------------------------------------------------------------------
st.markdown("### Load Document")

col_url, col_file, col_btn = st.columns([2, 2, 1])

with col_url:
    pdf_url = st.text_input("URL or Path", key="pdf_url_input")

with col_file:
    uploaded_file = st.file_uploader("Upload File", type=["pdf", "html"])

with col_btn:
    if st.button("Load"):
        st.session_state.pdf_upload_status = "downloading"
        st.session_state.pdf_upload_url = pdf_url
        st.session_state.pdf_upload_file = uploaded_file
        st.rerun()


# ---------------------------------------------------------------------------
# Upload processing trigger
# ---------------------------------------------------------------------------
if st.session_state.pdf_upload_status == "downloading":
    with st.spinner("Processing document..."):
        success, msg, name, result = _process_pdf_upload(
            st.session_state.pdf_upload_url,
            st.session_state.pdf_upload_file,
        )

    if success:
        st.success(msg)
        st.session_state.current_pdf_name = name
        st.session_state.current_doc_id = result.get("doc_id", "")
        st.session_state.last_upload_result = result
        st.session_state.pdf_upload_status = "success"
    else:
        st.error(msg)
        st.session_state.pdf_upload_status = "error"

    st.rerun()


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
st.title("Enterprise Knowledge Intelligence Platform-RAG")

# Tab layout
tab1, tab2 = st.tabs(["Chat", "Ontology Manager"])

with tab1:

    # Get current turn data for panels
    idx = st.session_state.active_turn_idx
if idx >= 0 and idx < len(st.session_state.raw_turns):
    current_turn = st.session_state.raw_turns[idx]
else:
    current_turn = None

# Top panels (full width)
if current_turn:
    _render_query_understanding_panel(current_turn.get('query_understanding', {}))
    _render_retrieval_strategy_panel(current_turn.get('retrieval_strategy', {}))
    _render_top_retrieved_nodes_panel(current_turn.get('top_table', []))

# 3-column layout
col_chat, col_graph, col_sources = st.columns([1.2, 1, 1])

# LEFT COLUMN - Chat + Answer
with col_chat:
    st.subheader("Chat & Answer")
    _render_chat()
    
    # Show current answer if available
    if current_turn:
        st.markdown("---")
        st.markdown("**Latest Answer:**")
        st.markdown(current_turn.get('answer', 'No answer available.'))
        
        # Answer grounding
        _render_answer_grounding_panel(current_turn)
    
    # Input form with examples
    st.markdown("---")
    _render_example_queries()
    
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question")
        submitted = st.form_submit_button("Send")

# MIDDLE COLUMN - Enhanced Subgraph
with col_graph:
    st.subheader("Knowledge Graph")
    _render_enhanced_subgraph(current_turn)

# RIGHT COLUMN - Sources + Document Info
with col_sources:
    st.subheader("Sources")
    
    if current_turn:
        sources = current_turn.get('doc_sources', [])
        if sources:
            for source in sources:
                st.markdown(f"**📄 {source.get('document_name', 'Unknown Document')}**")
                st.markdown(f"*Document ID:* {source.get('document_id', 'Unknown')}")
                
                node_ids = source.get('node_ids', [])
                if node_ids:
                    st.markdown(f"*Retrieved sections:* {len(node_ids)}")
                st.markdown("---")
        else:
            st.info("No sources available.")
    else:
        st.info("Ask a question to see sources.")


# ---------------------------------------------------------------------------
# Handle chat submission
# ---------------------------------------------------------------------------
if submitted and user_input.strip():
    query = user_input.strip()

    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": query,
    })

    st.session_state.pending_query = query
    st.rerun()

with tab2:
    _render_ontology_manager()


# ---------------------------------------------------------------------------
# Process pending query
# ---------------------------------------------------------------------------
if st.session_state.get("pending_query"):
    query = st.session_state.pending_query
    st.session_state.pending_query = None

    with st.spinner("Searching..."):
        try:
            t0 = time.time()

            retriever = _get_retriever()
            model = load_embedmodel()

            expansion = _expand_query(query, st.session_state.use_validation)

            result = _retrieve_with_nodes(
                retriever,
                query,
                expansion,
                model,
            )

            context = result.get("context", "")

            answer = _generate_answer(
                query,
                context,
                st.session_state.chat_history,
            )

            elapsed = time.time() - t0

            # Store assistant response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
            })

            # Store full turn with enhanced data
            st.session_state.raw_turns.append({
                "query": query,
                "answer": answer,
                "context": context,
                "nodes": result.get("nodes", []),
                "top_table": result.get("top_table", []),
                "doc_sources": result.get("source_documents", []),
                "elapsed": elapsed,
                "query_understanding": result.get("query_understanding", {}),
                "retrieval_strategy": result.get("retrieval_strategy", {}),
                "subgraph_edges": result.get("subgraph_edges", []),
            })

            st.session_state.active_turn_idx = len(st.session_state.raw_turns) - 1

        except Exception as e:
            logger.exception("Query error")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Error: {str(e)}",
            })

    st.rerun()