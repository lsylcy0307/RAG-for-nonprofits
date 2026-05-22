"""
Benchmark sequential vs parallel enrichment on a small sample.

Usage
-----
python benchmark_enrich.py \
    --project_id ai-agent-platform-496418 \
    --dataset_id query_dataset \
    [--sample_size 10] \
    [--max_concurrent 5]
"""

import argparse
import asyncio
import logging
import statistics
import sys
import time

import anthropic
from google.cloud import bigquery

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = """You are helping build a grant search tool for nonprofit
fundraisers. Given information about a grantee organization, write one
specific sentence describing what they do — the type of work, population
served, and location if clear.
- Be specific, not generic.
- One sentence only. No preamble. No "This organization..."."""

USER_TEMPLATE = """Grantee: {grantee_name} ({grantee_state})
Grant purpose: {purpose}

Describe what the grantee does:"""


def build_prompt(row: dict) -> str:
    return USER_TEMPLATE.format(
        grantee_name=row.get("grantee_name")   or "Unknown",
        grantee_state=row.get("grantee_state") or "unknown state",
        purpose=row.get("sample_purpose")      or "not specified",
    )


async def call_claude(
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
                return {"grantee_id": row["grantee_id"], "description": response.content[0].text.strip()}
            except anthropic.RateLimitError:
                if attempt < retries:
                    wait = 10 * (2 ** attempt)
                    log.warning("Rate limited — waiting %ds...", wait)
                    await asyncio.sleep(wait)
                else:
                    return None
            except Exception as exc:
                log.warning("Failed for %s: %s", row.get("grantee_name", "?")[:30], exc)
                return None


async def run_sequential(rows: list[dict]) -> dict:
    client    = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(1)
    times     = []
    for row in rows:
        start = time.perf_counter()
        await call_claude(client, semaphore, row)
        times.append(time.perf_counter() - start)
    return {
        "total":  sum(times),
        "mean":   statistics.mean(times),
        "median": statistics.median(times),
    }


async def run_parallel(rows: list[dict], max_concurrent: int) -> dict:
    client    = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(max_concurrent)
    start     = time.perf_counter()
    results   = await asyncio.gather(*[call_claude(client, semaphore, row) for row in rows])
    total     = time.perf_counter() - start
    return {"total": total, "succeeded": sum(1 for r in results if r)}


async def main_async(args: argparse.Namespace) -> None:
    bq = bigquery.Client(project=args.project_id)

    rows = [
        dict(r) for r in bq.query(f"""
            SELECT gr.grantee_id, gr.grantee_name, gr.grantee_state,
                   g.grant_purpose AS sample_purpose
            FROM `{args.project_id}.{args.dataset_id}.pf_grantees` gr
            LEFT JOIN (
                SELECT grantee_id, grant_purpose,
                       ROW_NUMBER() OVER (PARTITION BY grantee_id ORDER BY created_at DESC) AS rn
                FROM `{args.project_id}.{args.dataset_id}.pf_grants`
                WHERE grant_purpose IS NOT NULL
            ) g ON g.grantee_id = gr.grantee_id AND g.rn = 1
            WHERE gr.grantee_name IS NOT NULL
            ORDER BY gr.grantee_id
            LIMIT {args.sample_size}
        """).result()
    ]

    if not rows:
        log.info("No grantees found.")
        return

    log.info("Benchmarking on %d grantees  model=%s", len(rows), MODEL)
    log.info("─" * 50)

    log.info("Sequential (max_concurrent=1)...")
    seq = await run_sequential(rows)
    log.info("  Total: %.2fs  Mean: %.2fs  Median: %.2fs",
             seq["total"], seq["mean"], seq["median"])

    log.info("Waiting 60s for rate limit to reset...")
    await asyncio.sleep(60)

    log.info("Parallel (max_concurrent=%d)...", args.max_concurrent)
    par = await run_parallel(rows, args.max_concurrent)
    log.info("  Total: %.2fs  Succeeded: %d/%d",
             par["total"], par["succeeded"], len(rows))

    log.info("─" * 50)
    speedup       = seq["total"] / par["total"] if par["total"] > 0 else 0
    seq_projected = seq["mean"] * 1000
    par_projected = par["total"] / len(rows) * 1000

    log.info("Speedup: %.1fx", speedup)
    log.info("Projected for 1,000 grantees:")
    log.info("  Sequential: %.0fs (%.1f min)", seq_projected, seq_projected / 60)
    log.info("  Parallel:   %.0fs (%.1f min)", par_projected, par_projected / 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id",    required=True)
    parser.add_argument("--dataset_id",    required=True)
    parser.add_argument("--sample_size",   type=int, default=10)
    parser.add_argument("--max_concurrent", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()