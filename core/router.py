"""
router.py — LLM-as-Classifier Query Router
Replaces keyword matching with Haiku classification.
Handles typos, varied phrasing, and complex intents.
"""

from anthropic import Anthropic

client = Anthropic()

CLASSIFICATION_TOOL = {
    "name": "classify_query",
    "description": "Classify a user query to route it to the correct handler",
    "input_schema": {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "enum": [
                    "date_extraction",    # questions about dates, expiry, timelines
                    "semantic_search",    # questions about content, meaning, compliance
                    "listing",            # list documents, show all, what's indexed
                    "summarization",      # summarize a document or section
                    "comparison"          # compare two documents or versions
                ],
                "description": "The type of query the user is asking"
            },
            "filter_doc": {
                "type": ["string", "null"],
                "description": "Specific document name if query targets one doc, else null"
            },
            "reasoning": {
                "type": "string",
                "description": "Brief reason for this classification"
            }
        },
        "required": ["query_type", "filter_doc", "reasoning"]
    }
}

SYSTEM_PROMPT = """You are a query classifier for a pharmaceutical document intelligence system.

Classify user queries into one of these types:

date_extraction: User asks about dates, expiry dates, timelines, deadlines, 
  effective dates, submission dates, flagged dates, oldest/newest dates.
  Examples: "what are the expiry dates?", "show flagged dates", 
  "when was this submitted?", "latest date across docs"

semantic_search: User asks about content, meaning, compliance requirements,
  what FDA said, regulations, findings, violations, guidance.
  Examples: "what does FDA say about...", "what are the compliance requirements",
  "what violations were found", "explain part 11"

listing: User wants to list or enumerate documents, files, or indexed items.
  Examples: "what documents are indexed?", "list all docs", "which files do you have"

summarization: User wants a summary of a document or section.
  Examples: "summarize the OPQ report", "give me an overview of...", 
  "what is this document about"

comparison: User wants to compare documents or versions.
  Examples: "compare doc A with doc B", "what changed between versions",
  "differences between these two"

Also identify if the query targets a specific document by name."""


def classify_query(query: str) -> dict:
    """
    Use Haiku to classify the query type.
    Returns dict with query_type, filter_doc, reasoning.
    Falls back to semantic_search on any error.
    """
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=SYSTEM_PROMPT,
            tools=[CLASSIFICATION_TOOL],
            tool_choice={"type": "tool", "name": "classify_query"},
            messages=[{"role": "user", "content": query}]
        )

        result = response.content[0].input
        return {
            "query_type": result["query_type"],
            "filter_doc":  result.get("filter_doc"),
            "reasoning":   result.get("reasoning", "")
        }

    except Exception as e:
        print(f"  Classifier error: {e} — defaulting to semantic_search")
        return {
            "query_type": "semantic_search",
            "filter_doc":  None,
            "reasoning":   "fallback"
        }


def is_structured_query(query: str) -> bool:
    """
    Backwards-compatible wrapper for chatbot.py.
    Returns True if query should go to date index.
    """
    result = classify_query(query)
    return result["query_type"] in ("date_extraction", "listing")