"""
Answer Generation module.
Takes retrieved context + user query + chat history → produces final LLM
answer.
Includes post-generation attribution to identify which source sections were
used.
"""

import os
import json
import logging
from datetime import datetime
from pyexpat.errors import messages
from typing import List, Dict, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
)

MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT")

SYSTEM_PROMPT = """
You are a Medicare insurance assistant.
Answer the member's question using ONLY the provided context.

Process:
1. Identify the specific service or procedure the member is asking about
(e.g., outpatient surgery at ASC, hospital outpatient facility, physician's
office, etc.)
2. Locate the corresponding "Covered Service" row or section in the context
that explicitly mentions this service or facility type.
3. Extract ALL relevant cost information (copays, coinsurance, deductibles)
for that specific service and facility type.
4. If the question asks for comparisons between different settings (e.g., ASC
vs hospital), provide costs for EACH facility type clearly labeled.
5. Always include both in-network and out-of-network costs if available and
relevant to the question.

Rules:
- Always provide answers in very concise bullet points or a short list format,
not long paragraphs.
- Extract exact dollar amounts and percentages as they appear in the context.
Do not infer, calculate, or round amounts.
- Include the payment structure exactly as written (e.g., "$200 per date of
service", "50% coinsurance").
- For facility-based services, explicitly distinguish between different
facility types when the context provides different costs (e.g., ASC vs
hospital outpatient vs physician's office).
- Treat "copay" and "copayment" as equivalent terms.
- Do not provide general explanations or definitions-focus only on the
specific costs and coverage details requested.
- If the exact service or facility type mentioned in the question is not found
in the context, state: "This information was not found in the plan documents."
- Return only the final answer without preamble, additional context, or
explanations.
"""


def build_messages(
    user_query: str,
    retrieved_context: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """
    Builds the message array for the LLM call.
    """

    messages: List[Dict[str, str]] = []

    if chat_history:
        for msg in chat_history:
            messages.append(
                {
                    "role": msg["role"],
                    "content": msg["content"],
                }
            )

    user_message = (
        f"CONTEXT FROM PLAN DOCUMENTS:\n"
        f"---\n"
        f"{retrieved_context}\n"
        f"---\n\n"
        f"MEMBER QUESTION:\n{user_query}"
    )

    messages.append({"role": "user", "content": user_message})

    return messages


def save_llm_context_to_file(
    user_query: str,
    retrieved_context: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    output_dir: str = "logs",
) -> str:

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"llm_context_{timestamp}.txt")

    try:
        with open(filename, "w", encoding="utf-8") as f:

            f.write("=" * 80 + "\n")
            f.write("LLM CONTEXT LOG\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("-" * 80 + "\n")
            f.write("SYSTEM PROMPT\n")
            f.write("-" * 80 + "\n")
            f.write(SYSTEM_PROMPT + "\n\n")

            f.write("-" * 80 + "\n")
            f.write("CHAT HISTORY\n")
            f.write("-" * 80 + "\n")

            if chat_history:
                for i, msg in enumerate(chat_history):
                    f.write(f"\n[Turn {i+1}] {msg['role'].upper()}:\n")
                    f.write(msg["content"] + "\n")
            else:
                f.write("(No previous chat history)\n")

            f.write("\n")

            f.write("-" * 80 + "\n")
            f.write("RETRIEVED CONTEXT\n")
            f.write("-" * 80 + "\n")
            f.write(retrieved_context if retrieved_context else "(No context retrieved)\n")

            f.write("\n\n")

            f.write("-" * 80 + "\n")
            f.write("USER QUERY\n")
            f.write("-" * 80 + "\n")
            f.write(user_query + "\n\n")

            messages = build_messages(user_query, retrieved_context, chat_history)

            f.write("-" * 80 + "\n")
            f.write("COMPLETE MESSAGE ARRAY (as sent to LLM)\n")
            f.write("-" * 80 + "\n")

            f.write(json.dumps(messages, indent=2, ensure_ascii=False) + "\n")

        logger.info(f"LLM context saved to {filename}")
        return filename

    except Exception as e:
        logger.error(f"Failed to save LLM context to file: {e}")
        return ""


def _is_greeting_message(user_query: str) -> bool:

    greeting_patterns = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "good night", "howdy", "greetings", "what's up", "whats up", "sup",
        "how are you", "how r u", "how do you do", "thanks", "thank you",
        "thx", "bye", "goodbye", "good bye", "see you", "see ya", "later",
        "take care", "have a good day", "have a nice day", "yo", "hola",
        "welcome", "hii", "hiii", "helloo", "hellooo", "hiiii",
    ]

    normalized = user_query.strip().lower().rstrip("!.,?;:")

    if normalized in greeting_patterns:
        return True

    if len(normalized) < 30:
        for pattern in greeting_patterns:
            if normalized.startswith(pattern):
                return True

    return False


def generate_answer(
    user_query: str,
    retrieved_context: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.3,
    max_tokens: int = 1000,
    ontology_config=None,
) -> str:
    """
    Generates an answer using the LLM based on retrieved context.
    
    Args:
        user_query: The user's question
        retrieved_context: Context retrieved from the document store
        chat_history: Optional list of previous chat messages
        temperature: LLM temperature (default 0.3)
        max_tokens: Maximum tokens to generate (default 1000)
        ontology_config: Optional OntologyConfig object
        
    Returns:
        The generated answer string
    """
    if ontology_config is None:
        try:
            from domain_aware_rag.ontology.registry import OntologyRegistry
            ontology_config = OntologyRegistry().get_active_ontology()
        except Exception:
            ontology_config = None

    sys_prompt = ontology_config.answer_generation.system_role if (ontology_config and ontology_config.answer_generation) else SYSTEM_PROMPT
    greeting_resp = ontology_config.answer_generation.greeting_response if (ontology_config and ontology_config.answer_generation) else "Hello! I'm a Medicare insurance assistant. How can I help you today?"

    # Check for greeting messages
    if _is_greeting_message(user_query):
        return greeting_resp
    
    # Build messages for the LLM
    messages = build_messages(user_query, retrieved_context, chat_history)
    
    # Save context for debugging
    try:
        # We temporarily inject sys_prompt to save_llm_context_to_file's global SYSTEM_PROMPT or similar
        # but since it's just for debugging logs, we can just save it.
        save_llm_context_to_file(user_query, retrieved_context, chat_history)
    except Exception as e:
        logger.warning(f"Could not save LLM context: {e}")
    
    # Call Azure OpenAI
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                *messages
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        answer = response.choices[0].message.content
        return answer
        
    except Exception as e:
        logger.error(f"Error calling Azure OpenAI: {e}")
        return "I apologize, but I'm having trouble generating a response. Please try again."
