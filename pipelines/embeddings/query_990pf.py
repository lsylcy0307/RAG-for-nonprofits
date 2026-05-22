"""
Semantic search over 990-PF grant data.

Usage
-----
export PINECONE_API_KEY=...
export PINECONE_INDEX_NAME=...

python query_990pf.py \
    --project_id ai-agent-platform-496418 \
    --dataset_id query_dataset \
    --query "foundations that fund youth mental health programs in Minnesota" \
    [--top_k 5] \
    [--state MN] \
    [--open_only] \
    [--grant_size medium] \
    [--region us-central1]

Query decomposition
-------------------
The script uses an LLM to extract filters from natural language query before
searching, so explicit flags are optional. For example:

  --query "open foundations funding food banks in Texas under $10K"

automatically sets state=TX, grant_size=small, open_only=True.
Explicit flags override the LLM-extracted values if both are provided.
"""

import argparse
import json
import logging
import os
import sys

import anthropic
import vertexai
from google.cloud import bigquery
from pinecone import Pinecone
from vertexai.language_models import TextEmbeddingModel

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

PINECONE_NAMESPACE = "pf_grants"

VALID_SIZES = ["small","medium","large","major"]


def decompose_query(client: anthropic.Anthropic, query: str) -> dict:
    """
    Use an LLM to extract structured filters from a natural language query.

    Returns a dict with keys:
      semantic_query  cleaned query text for vector search
      states          list of 2-letter state codes (may be empty)
      grant_size      "small"/"medium"/"large"/"major" or null
      open_only       true/false/null
    """
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"""Extract structured filters from this grant funder search query.

Query: "{query}"

Return ONLY a JSON object with these keys:
- semantic_query: the core topic to search (remove geographic/eligibility language)
- states: list of US state codes mentioned or implied (e.g. ["MN"] for Minnesota,
  ["FL","GA","SC","NC","VA","TN","AL","MS"] for Southeast). Empty list if none.
- grant_size: one of "small"/"medium"/"large"/"major" if mentioned, else null.
  (small=<$5K, medium=$5K-$25K, large=$25K-$100K, major=$100K+)
- open_only: true if query asks for foundations accepting proposals, else null.

Return only valid JSON, no explanation."""
        }]
    )

    text = response.content[0].text.strip()

    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning("First decomposition attempt returned non-JSON — retrying...")
        retry = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            system="You are a JSON API. You return only raw JSON with no explanation, no markdown, no code fences.",
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
            log.warning("Decomposition retry also failed — using raw query, no filters")
            return {
                "semantic_query": query,
                "states":     [],
                "grant_size": None,
                "open_only":  None,
            }


def rerank(
    client: anthropic.Anthropic,
    query: str,
    matches: list[dict],
    top_n: int,
) -> list[dict]:
    """
    Use an LLM to re-score Pinecone results by true relevance.
    Fetches top_k*4 from Pinecone, reranks, returns top_n.
    """
    if len(matches) <= top_n:
        return matches

    candidates = "\n".join(
        f"{i+1}. {m['metadata'].get('embed_text', '')}"
        for i, m in enumerate(matches)
    )

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=100,
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
            )
        }]
    )

    try:
        order = [
            int(x.strip()) - 1
            for x in response.content[0].text.strip().split(",")
            if x.strip().isdigit()
        ]
        reranked = [matches[i] for i in order if i < len(matches)]
        # Append any matches not mentioned by LLM at the end
        mentioned = set(order)
        reranked += [m for i, m in enumerate(matches) if i not in mentioned]
        return reranked[:top_n]
    except Exception:
        return matches[:top_n]


def format_amount(raw: str | None) -> str:
    try:
        return f"${int(raw):,}"
    except (TypeError, ValueError):
        return "amount not reported"

def format_assets(raw: str | None) -> str:
    try:
        amt = int(raw)
        if amt >= 1_000_000:
            return f"${amt/1_000_000:.1f}M assets"
        return f"${amt:,} assets"
    except (TypeError, ValueError):
        return "assets not reported"

def main() -> None:
    parser = argparse.ArgumentParser(description="Search 990-PF grant data")
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--query",      required=True)
    parser.add_argument("--top_k",      type=int, default=5)
    parser.add_argument("--region",     default="us-central1")
    # Optional explicit filters — override LLM-extracted values
    parser.add_argument("--state",      help="2-letter state code e.g. MN")
    parser.add_argument("--grant_size", choices=VALID_SIZES)
    parser.add_argument("--open_only",  action="store_true",
                        help="Only return foundations that accept unsolicited proposals")
    args = parser.parse_args()

    anthropic_client = anthropic.Anthropic()
    vertexai.init(project=args.project_id, location=args.region)
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(
        os.environ["PINECONE_INDEX_NAME"]
    )

    log.info("Decomposing query...")
    decomposed = decompose_query(anthropic_client, args.query)

    semantic_query = decomposed.get("semantic_query") or args.query
    states         = decomposed.get("states") or []
    grant_size     = decomposed.get("grant_size")
    open_only      = decomposed.get("open_only")

    if args.state:
        states = [args.state.upper()]
    if args.grant_size:
        grant_size = args.grant_size
    if args.open_only:
        open_only = True

    log.info(
        "Searching: '%s'  states=%s  size=%s  open_only=%s",
        semantic_query, states or "any", grant_size or "any", open_only,
    )

    pinecone_top_k = args.top_k * 4   # fetch more for reranker
    pinecone_filter: dict = {}

    if len(states) == 1:
        pinecone_filter["grantee_state_filter"] = {"$eq": states[0]}
    elif len(states) > 1:
        pinecone_filter["grantee_state_filter"] = {"$in": states}

    if grant_size:
        pinecone_filter["grant_size"] = {"$eq": grant_size}

    if open_only:
        pinecone_filter["open_to_apply"] = {"$eq": True}

    query_vector = model.get_embeddings([semantic_query])[0].values

    results = index.query(
        vector=query_vector,
        top_k=pinecone_top_k,
        namespace=PINECONE_NAMESPACE,
        include_metadata=True,
        filter=pinecone_filter if pinecone_filter else None,
    )

    matches = results.get("matches", [])

    if not matches:
        print("\nNo results found. Try broadening your query or removing filters.")
        return

    # Only rerank when there are enough candidates to meaningfully reorder.
    RERANK_THRESHOLD = 10

    should_rerank = (
        len(matches) > RERANK_THRESHOLD
        and args.top_k < len(matches)
    )

    if should_rerank:
        log.info("Reranking %d candidates → top %d...", len(matches), args.top_k)
        matches = rerank(anthropic_client, args.query, matches, args.top_k)
    else:
        log.info(
            "Skipping rerank — only %d candidates (threshold=%d)",
            len(matches), RERANK_THRESHOLD,
        )
        matches = matches[:args.top_k]

    print(f'\nQuery: "{args.query}"')
    if states or grant_size or open_only:
        active = []
        if states:     active.append(f"state={','.join(states)}")
        if grant_size: active.append(f"size={grant_size}")
        if open_only:  active.append("open to proposals")
        print(f"Filters: {' | '.join(active)}")
    print("─" * 60)

    for rank, match in enumerate(matches, start=1):
        m      = match["metadata"]
        score  = match["score"]

        filer    = m.get("filer_name", "Unknown foundation")
        f_state  = m.get("filer_state", "")
        grantee  = m.get("grantee_name", "Unknown grantee")
        g_state  = m.get("grantee_state", "")
        purpose  = m.get("grant_purpose", "")
        amount   = format_amount(m.get("grant_amount_raw"))
        assets   = format_assets(m.get("fmv_assets_raw"))
        year     = m.get("filing_year", "")
        open_app = m.get("accepts_unsolicited_apps")

        eligibility = (
            "✓ Accepts proposals"  if open_app is True
            else "✗ Invitation only" if open_app is False
            else "? Eligibility unknown"
        )

        print(f"\n#{rank}  {filer} ({f_state})  [{score:.3f}]")
        print(f"    {assets}  ·  {year}  ·  {eligibility}")
        print(f"    Funded: {grantee} ({g_state})  {amount}")
        if purpose:
            print(f"    Purpose: {purpose}")

    print()


if __name__ == "__main__":
    main()