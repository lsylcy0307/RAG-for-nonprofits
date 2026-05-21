import json
import logging
import os
from typing import Any

import anthropic
import vertexai
from pinecone import Pinecone
from vertexai.language_models import TextEmbeddingModel

log = logging.getLogger(__name__)

PINECONE_NAMESPACE = "pf_grants"
VALID_GRANT_SIZES  = {"small", "medium", "large", "major"}

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "ai-agent-platform-496418")
REGION     = os.environ.get("GCP_REGION", "us-central1")


def decompose_query(client: anthropic.Anthropic, query: str) -> dict[str, Any]:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        system="Return only valid JSON. No markdown. No explanation.",
        messages=[{
            "role": "user",
            "content": f"""
Extract structured filters from this grant funder search query.

Query: "{query}"

Return a JSON object with:
- semantic_query: core topic for vector search (remove geographic/eligibility language)
- states: list of US state codes, or []
- grant_size: "small", "medium", "large", "major", or null
- open_only: true or null

Grant sizes: small=<$5K, medium=$5K-$25K, large=$25K-$100K, major=$100K+
""",
        }],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].replace("json", "", 1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Retry once with stricter system prompt
        log.warning("Query decomposition failed — retrying...")
        retry = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            system="You are a JSON API. Return only raw JSON with no explanation, no markdown, no code fences.",
            messages=[{
                "role": "user",
                "content": (
                    f'Return a JSON object for this grant search query: "{query}"\n\n'
                    'Keys: semantic_query (string), states (array of 2-letter codes), '
                    'grant_size ("small"/"medium"/"large"/"major" or null), '
                    'open_only (true or null).'
                )
            }]
        )
        try:
            return json.loads(retry.content[0].text.strip())
        except json.JSONDecodeError:
            log.warning("Decomposition retry failed — using raw query, no filters")
            return {
                "semantic_query": query,
                "states":     [],
                "grant_size": None,
                "open_only":  None,
            }


def build_pinecone_filter(
    states:     list[str] | None = None,
    grant_size: str | None       = None,
    open_only:  bool | None      = None,
) -> dict[str, Any] | None:
    filters: dict[str, Any] = {}
    states = states or []

    # Filter on grantee_state (where funded org is located) not filer_state
    # (where foundation is headquartered). This finds any foundation that has
    # actually funded organizations in the target state — which is what a grant
    # writer needs, not just foundations headquartered there.
    if len(states) == 1:
        filters["grantee_state_filter"] = {"$eq": states[0].upper()}
    elif len(states) > 1:
        filters["grantee_state_filter"] = {"$in": [s.upper() for s in states]}

    if grant_size:
        if grant_size not in VALID_GRANT_SIZES:
            raise ValueError(f"grant_size must be one of {VALID_GRANT_SIZES}")
        filters["grant_size"] = {"$eq": grant_size}

    if open_only:
        filters["open_to_apply"] = {"$eq": True}

    return filters or None


def rerank(
    client:  anthropic.Anthropic,
    query:   str,
    matches: list[dict[str, Any]],
    top_k:   int,
) -> list[dict[str, Any]]:
    if len(matches) <= top_k:
        return matches

    candidates = "\n".join(
        f"{i + 1}. {m.get('metadata', {}).get('embed_text', '')}"
        for i, m in enumerate(matches)
    )

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=100,
        system="Return only a comma-separated list of numbers.",
        messages=[{
            "role": "user",
            "content": (
                f"Query: {query}\n\n"
                f"You are helping a nonprofit fundraiser find funders. "
                f"Rank these foundation grant records by relevance to the query "
                f"(most relevant first). Prioritize records where: "
                f"(1) the grantee's work closely matches the query topic, "
                f"(2) the location matches if specified, "
                f"(3) the foundation accepts unsolicited proposals if possible.\n\n"
                f"Return ONLY a comma-separated list of numbers, nothing else.\n\n"
                f"{candidates}"
            ),
        }],
    )

    try:
        order = [
            int(x.strip()) - 1
            for x in response.content[0].text.strip().split(",")
            if x.strip().isdigit()
        ]
        reranked = [matches[i] for i in order if 0 <= i < len(matches)]
        used = set(order)
        reranked.extend([m for i, m in enumerate(matches) if i not in used])
        return reranked[:top_k]
    except Exception:
        return matches[:top_k]


def format_amount(raw: Any) -> str:
    try:
        return f"${int(raw):,}"
    except (TypeError, ValueError):
        return "amount not reported"


def format_assets(raw: Any) -> str:
    try:
        amount = int(raw)
        if amount >= 1_000_000:
            return f"${amount / 1_000_000:.1f}M assets"
        return f"${amount:,} assets"
    except (TypeError, ValueError):
        return "assets not reported"


def format_match(match: dict[str, Any]) -> dict[str, Any]:
    metadata = match.get("metadata", {})
    open_app  = metadata.get("accepts_unsolicited_apps")

    return {
        "score":             match.get("score"),
        "foundation_name":   metadata.get("filer_name"),
        "foundation_state":  metadata.get("filer_state"),
        "foundation_ein":    metadata.get("filer_ein"),
        "foundation_assets": format_assets(metadata.get("fmv_assets_raw")),
        "filing_year":       metadata.get("filing_year"),
        "accepts_unsolicited_apps": open_app,
        "eligibility": (
            "open"    if open_app is True
            else "invitation_only" if open_app is False
            else "unknown"
        ),
        "grantee_name":  metadata.get("grantee_name"),
        "grantee_city":  metadata.get("grantee_city"),
        "grantee_state": metadata.get("grantee_state"),
        "grant_amount":  format_amount(metadata.get("grant_amount_raw")),
        "grant_purpose": metadata.get("grant_purpose"),
        "embed_text":    metadata.get("embed_text"),
    }


def search_grants(
    query:             str,
    top_k:             int       = 5,
    state:             str | None = None,
    grant_size:        str | None = None,
    open_only:         bool       = False,
    use_decomposition: bool       = True,
    use_reranking:     bool       = True,
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("query cannot be empty")

    anthropic_client = anthropic.Anthropic()

    # ── Query decomposition ───────────────────────────────────────────────────
    if use_decomposition:
        decomposed         = decompose_query(anthropic_client, query)
        semantic_query     = decomposed.get("semantic_query") or query
        states             = decomposed.get("states") or []
        inferred_size      = decomposed.get("grant_size")
        inferred_open_only = decomposed.get("open_only")
    else:
        semantic_query     = query
        states             = []
        inferred_size      = None
        inferred_open_only = None

    # Explicit args override LLM-extracted values
    if state:       states        = [state.upper()]
    if grant_size:  inferred_size = grant_size
    if open_only:   inferred_open_only = True

    # ── Embed + search ────────────────────────────────────────────────────────
    vertexai.init(project=PROJECT_ID, location=REGION)
    model        = TextEmbeddingModel.from_pretrained("text-embedding-004")
    query_vector = model.get_embeddings([semantic_query])[0].values

    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(
        os.environ["PINECONE_INDEX_NAME"]
    )

    pinecone_filter = build_pinecone_filter(
        states=states,
        grant_size=inferred_size,
        open_only=inferred_open_only,
    )

    results = index.query(
        vector=query_vector,
        top_k=top_k * 4 if use_reranking else top_k,
        namespace=PINECONE_NAMESPACE,
        include_metadata=True,
        filter=pinecone_filter,
    )

    matches = results.get("matches", [])

    # ── Rerank ────────────────────────────────────────────────────────────────
    # Only rerank when there are enough candidates to meaningfully reorder.
    # With fewer than RERANK_THRESHOLD results the LLM adds latency and cost
    # without improving quality — vector similarity is already doing all the work.
    RERANK_THRESHOLD = 10

    should_rerank = (
        use_reranking
        and len(matches) > RERANK_THRESHOLD
        and top_k < len(matches)
    )

    if should_rerank:
        log.info("Reranking %d candidates → top %d", len(matches), top_k)
        matches = rerank(anthropic_client, query, matches, top_k)
    else:
        if use_reranking and len(matches) <= RERANK_THRESHOLD:
            log.info(
                "Skipping rerank — only %d candidates (threshold=%d)",
                len(matches), RERANK_THRESHOLD,
            )

    return {
        "query":          query,
        "semantic_query": semantic_query,
        "filters": {
            "states":     states,
            "grant_size": inferred_size,
            "open_only":  inferred_open_only,
        },
        "results": [format_match(m) for m in matches[:top_k]],
    }