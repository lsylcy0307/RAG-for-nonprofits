"""
Enrich pf_grantees with LLM-generated descriptions (parallel).

Runs Claude calls concurrently using asyncio — far faster than sequential
when your API rate limit allows it. Use benchmark_enrich.py first to find
the right --max_concurrent for your tier.

Schema required
---------------
pf_grantees must have: description STRING, description_source STRING, updated_at TIMESTAMP

Usage
-----
python enrich_990pf.py \
    --project_id ai-agent-platform-496418 \
    --dataset_id query_dataset \
    [--limit 500] \
    [--max_concurrent 5] \
    [--dry_run] \
    [--dry_run_limit 5]
"""

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime, timezone

import anthropic
from google.cloud import bigquery

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL          = "claude-haiku-4-5"
BQ_CONCURRENCY = 10

SYSTEM_PROMPT = """You are helping build a grant search tool for nonprofit
fundraisers. Given information about a grantee organization, write one
specific sentence describing what they do — the type of work, population
served, and location if clear.

Rules:
- Be specific, not generic. "provides cancer research and patient care"
  is good. "provides support to the community" is not.
- If the grantee name makes the work obvious, use that signal even if
  the purpose is vague.
- If the purpose is specific, use it. If it is generic like
  "PROVIDE OPERATING FUNDS", ignore it and infer from the name.
- One sentence only. No preamble. No "This organization..."."""

USER_TEMPLATE = """Grantee: {grantee_name} ({grantee_state})
Grant purpose: {purpose}

Describe what the grantee does:"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_prompt(row: dict) -> str:
    return USER_TEMPLATE.format(
        grantee_name=row.get("grantee_name")   or "Unknown grantee",
        grantee_state=row.get("grantee_state") or "unknown state",
        purpose=row.get("sample_purpose")      or "not specified",
    )


# ── Async Claude call ─────────────────────────────────────────────────────────

async def enrich_one(
    client:    anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    row:       dict,
    retries:   int = 2,
) -> dict | None:
    async with semaphore:
        for attempt in range(retries + 1):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=120,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": build_prompt(row)}]
                )
                return {
                    "grantee_id":  row["grantee_id"],
                    "description": response.content[0].text.strip(),
                }
            except anthropic.RateLimitError:
                if attempt < retries:
                    wait = 10 * (2 ** attempt)
                    log.warning("Rate limited — waiting %ds (attempt %d/%d)...",
                                wait, attempt + 1, retries)
                    await asyncio.sleep(wait)
                else:
                    log.warning("Retries exhausted for %s",
                                (row.get("grantee_name") or "?")[:30])
                    return None
            except Exception as exc:
                log.warning("Failed for %s: %s",
                            (row.get("grantee_name") or "?")[:30], exc)
                return None


# ── Async BigQuery write ──────────────────────────────────────────────────────

def _bq_write_one(bq: bigquery.Client, table: str, u: dict, now: str) -> None:
    bq.query(
        f"UPDATE `{table}` "
        "SET description = @description, "
        "    description_source = 'llm', "
        "    updated_at = @updated_at "
        "WHERE grantee_id = @grantee_id",
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("description", "STRING",    u["description"]),
            bigquery.ScalarQueryParameter("updated_at",  "TIMESTAMP", now),
            bigquery.ScalarQueryParameter("grantee_id",  "STRING",    u["grantee_id"]),
        ])
    ).result()


async def write_all(bq: bigquery.Client, table: str, updates: list[dict]) -> None:
    now       = datetime.now(timezone.utc).isoformat()
    semaphore = asyncio.Semaphore(BQ_CONCURRENCY)

    async def write_one(u: dict) -> None:
        async with semaphore:
            await asyncio.to_thread(_bq_write_one, bq, table, u, now)

    await asyncio.gather(*[write_one(u) for u in updates])


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> None:
    bq = bigquery.Client(project=args.project_id)

    grantees_table  = f"{args.project_id}.{args.dataset_id}.pf_grantees"
    grants_table    = f"{args.project_id}.{args.dataset_id}.pf_grants"
    effective_limit = args.dry_run_limit if args.dry_run else args.limit

    # ── Fetch unenriched grantees ─────────────────────────────────────────────
    rows = [
        dict(r) for r in bq.query(f"""
            SELECT
                gr.grantee_id,
                gr.grantee_name,
                gr.grantee_state,
                g.grant_purpose AS sample_purpose
            FROM `{grantees_table}` gr
            LEFT JOIN (
                SELECT grantee_id, grant_purpose,
                       ROW_NUMBER() OVER (
                           PARTITION BY grantee_id ORDER BY created_at DESC
                       ) AS rn
                FROM `{grants_table}`
                WHERE grant_purpose IS NOT NULL
            ) g ON g.grantee_id = gr.grantee_id AND g.rn = 1
            WHERE gr.description IS NULL
              AND gr.grantee_name IS NOT NULL
            ORDER BY gr.grantee_id
            LIMIT {effective_limit}
        """).result()
    ]

    if not rows:
        log.info("All grantees already have descriptions.")
        return

    log.info(
        "Enriching %d grantees  model=%s  max_concurrent=%d",
        len(rows), MODEL, args.max_concurrent,
    )

    # ── Run all Claude calls in parallel ──────────────────────────────────────
    client    = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(args.max_concurrent)

    start   = time.perf_counter()
    results = await asyncio.gather(*[enrich_one(client, semaphore, row) for row in rows])
    elapsed = time.perf_counter() - start

    updates = [r for r in results if r is not None]
    failed  = len(results) - len(updates)

    log.info(
        "Done in %.1fs — %d enriched, %d failed",
        elapsed, len(updates), failed,
    )

    if not updates:
        return

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        for u in updates[:3]:
            log.info("[dry_run] %s → %s", u["grantee_id"][:12], u["description"][:100])
        if len(updates) > 3:
            log.info("[dry_run] ...and %d more", len(updates) - 3)
        return

    # ── Write to BigQuery ─────────────────────────────────────────────────────
    log.info("Writing %d descriptions to BigQuery...", len(updates))
    bq_start = time.perf_counter()
    await write_all(bq, grantees_table, updates)
    log.info("Wrote in %.1fs", time.perf_counter() - bq_start)

    if failed:
        log.info("%d grantees failed — re-run to retry them", failed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich pf_grantees with parallel Claude descriptions"
    )
    parser.add_argument("--project_id",     required=True)
    parser.add_argument("--dataset_id",     required=True)
    parser.add_argument("--limit",          type=int,   default=500)
    parser.add_argument("--max_concurrent", type=int,   default=5,
                        help="Max simultaneous Claude calls — match to your RPM limit")
    parser.add_argument("--dry_run",        action="store_true")
    parser.add_argument("--dry_run_limit",  type=int,   default=5)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()