"""
agent_tools.py — Tool layer for the Agentic RAG pipeline.

Each tool is:
  1. Defined as a Gemini-compatible function declaration (TOOL_DECLARATIONS)
  2. Backed by an executor method on ToolExecutor that does real Pinecone retrieval

The agent loop in agent.py calls execute(tool_name, args, executor) to run a tool
and get back a list of text chunks the LLM can reason over.

Tool catalogue
--------------
search_policy          General semantic search across the full index (or one policy)
lookup_exclusions      Targeted search scoped to exclusion-related chunks
check_waiting_period   Targeted search scoped to waiting period clauses
get_definitions        Fetch definition/glossary chunks for a specific term
compare_policies       Run a query against every distinct document in the index
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .query_processor import QueryProcessor


# ── Gemini function-calling declarations ─────────────────────────────────────
# These are passed verbatim to genai.GenerativeModel(tools=TOOL_DECLARATIONS).
# Each maps exactly to one executor method below.

TOOL_DECLARATIONS = [
    {
        "name": "search_policy",
        "description": (
            "Search the insurance policy index for chunks relevant to a question. "
            "Use this as the first tool for any coverage, claim, or policy question. "
            "Optionally filter to a specific policy file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The sub-question or topic to search for.",
                },
                "policy_name": {
                    "type": "string",
                    "description": (
                        "Optional. Exact PDF filename to restrict search to "
                        "(e.g. 'policy.pdf'). Omit to search all loaded policies."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to return. Defaults to 15.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "lookup_exclusions",
        "description": (
            "Search specifically for exclusion clauses that might disqualify a claim. "
            "Always call this after search_policy when the initial results mention "
            "limitations, exceptions, or 'subject to' language."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "procedure_or_condition": {
                    "type": "string",
                    "description": (
                        "The procedure, condition, or benefit to check for exclusions "
                        "(e.g. 'dental surgery', 'pre-existing diabetes', 'cosmetic treatment')."
                    ),
                },
            },
            "required": ["procedure_or_condition"],
        },
    },
    {
        "name": "check_waiting_period",
        "description": (
            "Search for waiting period clauses for a specific benefit type. "
            "Call this whenever the query mentions policy duration, "
            "new policies, or time-sensitive coverage."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "benefit_type": {
                    "type": "string",
                    "description": (
                        "The benefit to check waiting periods for "
                        "(e.g. 'maternity', 'dental', 'pre-existing conditions', 'cataract')."
                    ),
                },
            },
            "required": ["benefit_type"],
        },
    },
    {
        "name": "get_definitions",
        "description": (
            "Fetch the policy's definition or glossary entry for a specific term. "
            "Use this when the query uses domain terms like 'hospitalization', "
            "'day care procedure', 'sum insured', or 'network hospital'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "The policy term to look up (e.g. 'day care procedure', 'network hospital').",
                },
            },
            "required": ["term"],
        },
    },
    {
        "name": "compare_policies",
        "description": (
            "Run the same query against every distinct policy document in the index "
            "and return results grouped by document. Use this when the user asks "
            "'which policy covers X' or when multiple PDFs are loaded."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question to compare across all loaded policies.",
                },
            },
            "required": ["query"],
        },
    },
]


# ── Keyword sets for targeted searches ───────────────────────────────────────

_EXCLUSION_KEYWORDS = [
    "exclusion", "excluded", "not covered", "not payable", "exception",
    "limitation", "shall not", "does not cover", "specifically excluded",
    "not admissible", "not applicable",
    # conditional-coverage terms — benefits that apply only in certain scenarios
    "applicable to", "excluding usa", "outside area of cover", "six weeks",
    "provided that", "subject to", "only if", "only when", "not available",
]

_WAITING_PERIOD_KEYWORDS = [
    "waiting period", "waiting", "initial waiting", "specific waiting",
    "pre-existing waiting", "days from", "months from", "years from",
    "30 days", "60 days", "90 days", "2 years", "3 years", "4 years",
]

_DEFINITION_KEYWORDS = [
    "means", "defined as", "definition", "refers to", "shall mean",
    "interpreted as", "includes", "glossary",
]


def _format_chunks(chunks: List[Dict], label: str = "") -> List[Dict]:
    """Normalise chunk dicts to a consistent shape for the agent loop."""
    out = []
    for c in chunks:
        out.append({
            "text": c.get("text", c.get("content", "")),
            "document": c.get("document_name", c.get("source_document", "unknown")),
            "page": c.get("page_number", c.get("page", None)),
            "score": round(c.get("score", c.get("vector_score", 0.0)), 4),
            "content_type": c.get("content_type", "text"),
            "tool": label,
        })
    return out


# ── ToolExecutor ─────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Executes tool calls against a live QueryProcessor instance.

    Usage:
        executor = ToolExecutor(processor)
        results = executor.execute("search_policy", {"query": "dental surgery"})
    """

    def __init__(self, processor: "QueryProcessor"):
        self.processor = processor

    # ── public dispatcher ────────────────────────────────────────────────────

    def execute(self, tool_name: str, args: Dict[str, Any]) -> List[Dict]:
        """
        Dispatch a tool call by name and return a list of chunk dicts.
        Returns [] on unknown tool name (agent loop should handle gracefully).
        """
        dispatch = {
            "search_policy": self._search_policy,
            "lookup_exclusions": self._lookup_exclusions,
            "check_waiting_period": self._check_waiting_period,
            "get_definitions": self._get_definitions,
            "compare_policies": self._compare_policies,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            print(f"⚠️ Unknown tool: {tool_name}")
            return []
        try:
            return fn(**args)
        except TypeError as e:
            print(f"⚠️ Tool {tool_name} called with bad args {args}: {e}")
            return []

    # ── tool implementations ─────────────────────────────────────────────────

    # Map common colloquial phrases to exact policy terminology used in the document
    _QUERY_REPHRASE = [
        (re.compile(r'\boutside\s+usa\b', re.I),          'outside area of cover'),
        (re.compile(r'\bexcluding\s+usa\b', re.I),         'outside area of cover'),
        (re.compile(r'\bmedical\s+emergency\s+abroad\b', re.I), 'emergency treatment outside area of cover'),
        (re.compile(r'\bgeograph\w+\s+coverage\b', re.I),  'area of cover'),
        (re.compile(r'\bcoverage\s+limit\b', re.I),        'sum insured'),
        (re.compile(r'\bhospital\s+stay\b', re.I),         'hospitalisation'),
        (re.compile(r'\bprior\s+condition\b', re.I),       'pre-existing disease'),
    ]

    def _normalise_query(self, query: str) -> str:
        """Replace colloquial terms with exact policy-document terminology."""
        for pattern, replacement in self._QUERY_REPHRASE:
            query = pattern.sub(replacement, query)
        return query

    def _search_policy(
        self,
        query: str,
        policy_name: Optional[str] = None,
        top_k: int = 15,
    ) -> List[Dict]:
        """General semantic search, optionally filtered to one document."""
        if not self.processor.index:
            return []

        query = self._normalise_query(query)
        embedding = self.processor._encode_query(query)
        # Only use policy_name filter when it looks like an actual filename
        if policy_name and policy_name.endswith(".pdf"):
            filter_dict = {"document_name": {"$eq": policy_name}}
        else:
            filter_dict = None

        try:
            response = self.processor.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict,
            )
            if response.matches:
                scores = [m.score for m in response.matches[:5]]
                print(f"[search_policy] raw scores: {[round(s,4) for s in scores]}")
                m0 = response.matches[0]
                print(f"[search_policy] top match metadata keys: {list((m0.metadata or {}).keys())}, has_text: {bool((m0.metadata or {}).get('text'))}")
            else:
                print(f"[search_policy] Pinecone returned 0 matches (empty response)")
            chunks = self.processor._process_search_results(response, min_score=0.01)
            # If still empty, return top matches regardless of score — the agent needs something
            if not chunks and response.matches:
                print(f"[search_policy] score filter dropped all matches, returning top-5 unfiltered")
                chunks = []
                for m in response.matches[:5]:
                    md = m.metadata or {}
                    content = md.get("text", md.get("content", ""))
                    if content:
                        chunks.append({
                            "score": m.score, "content": content, "text": content,
                            "document_name": md.get("document_name", ""),
                            "page_number": md.get("page_number", 1),
                            "chunk_index": md.get("chunk_index", 0),
                        })
            print(f"🔧 search_policy('{query[:50]}') → {len(chunks)} chunks")
            return _format_chunks(chunks, label="search_policy")
        except Exception as e:
            print(f"❌ search_policy error: {e}")
            return []

    def _lookup_exclusions(self, procedure_or_condition: str) -> List[Dict]:
        """
        Search for exclusion clauses.
        Strategy: embed an exclusion-focused query, then post-filter chunks
        that contain exclusion language.
        """
        if not self.processor.index:
            return []

        search_query = self._normalise_query(f"exclusion not covered {procedure_or_condition}")
        embedding = self.processor._encode_query(search_query)

        try:
            response = self.processor.index.query(
                vector=embedding,
                top_k=20,
                include_metadata=True,
            )
            all_chunks = self.processor._process_search_results(response, min_score=0.01)

            # Post-filter: keep chunks that contain exclusion language
            exclusion_chunks = [
                c for c in all_chunks
                if any(
                    kw in (c.get("content", "") + c.get("text", "")).lower()
                    for kw in _EXCLUSION_KEYWORDS
                )
            ]

            # Fall back to top-8 by score if nothing matched the keyword filter
            result = exclusion_chunks[:8] if exclusion_chunks else all_chunks[:5]
            print(f"🔧 lookup_exclusions('{procedure_or_condition}') → {len(result)} chunks")
            return _format_chunks(result, label="lookup_exclusions")
        except Exception as e:
            print(f"❌ lookup_exclusions error: {e}")
            return []

    def _check_waiting_period(self, benefit_type: str) -> List[Dict]:
        """
        Search for waiting period clauses for a specific benefit.
        Strategy: embed a waiting-period-focused query, post-filter for
        chunks containing waiting period language.
        """
        if not self.processor.index:
            return []

        search_query = f"waiting period {benefit_type} days months years"
        embedding = self.processor._encode_query(search_query)

        try:
            response = self.processor.index.query(
                vector=embedding,
                top_k=20,
                include_metadata=True,
            )
            all_chunks = self.processor._process_search_results(response, min_score=0.01)

            waiting_chunks = [
                c for c in all_chunks
                if any(
                    kw in (c.get("content", "") + c.get("text", "")).lower()
                    for kw in _WAITING_PERIOD_KEYWORDS
                )
            ]

            result = waiting_chunks[:5] if waiting_chunks else all_chunks[:3]
            print(f"🔧 check_waiting_period('{benefit_type}') → {len(result)} chunks")
            return _format_chunks(result, label="check_waiting_period")
        except Exception as e:
            print(f"❌ check_waiting_period error: {e}")
            return []

    def _get_definitions(self, term: str) -> List[Dict]:
        """
        Fetch definition/glossary chunks for a policy term.
        Strategy: embed a definition-focused query, post-filter for
        chunks that look like definitions.
        """
        if not self.processor.index:
            return []

        search_query = f"definition meaning {term} means defined as"
        embedding = self.processor._encode_query(search_query)

        try:
            response = self.processor.index.query(
                vector=embedding,
                top_k=15,
                include_metadata=True,
            )
            all_chunks = self.processor._process_search_results(response, min_score=0.01)

            definition_chunks = [
                c for c in all_chunks
                if any(
                    kw in (c.get("content", "") + c.get("text", "")).lower()
                    for kw in _DEFINITION_KEYWORDS
                )
            ]

            result = definition_chunks[:3] if definition_chunks else all_chunks[:2]
            print(f"🔧 get_definitions('{term}') → {len(result)} chunks")
            return _format_chunks(result, label="get_definitions")
        except Exception as e:
            print(f"❌ get_definitions error: {e}")
            return []

    def _compare_policies(self, query: str) -> List[Dict]:
        """
        Run the query against each distinct document in the index separately
        and return results grouped by document name.
        """
        if not self.processor.index:
            return []

        embedding = self.processor._encode_query(query)

        # Discover distinct document names from a broad search
        try:
            probe = self.processor.index.query(
                vector=embedding,
                top_k=50,
                include_metadata=True,
            )
            doc_names = list({
                m.metadata.get("document_name", "")
                for m in probe.matches
                if m.metadata and m.metadata.get("document_name")
            })
        except Exception as e:
            print(f"❌ compare_policies probe error: {e}")
            return []

        if len(doc_names) <= 1:
            # Only one doc loaded — fall back to regular search
            return self._search_policy(query, top_k=5)

        all_results = []
        for doc in doc_names:
            chunks = self._search_policy(query, policy_name=doc, top_k=3)
            for c in chunks:
                c["tool"] = "compare_policies"
            all_results.extend(chunks)

        print(f"🔧 compare_policies('{query[:50]}') → {len(all_results)} chunks across {len(doc_names)} docs")
        return all_results
