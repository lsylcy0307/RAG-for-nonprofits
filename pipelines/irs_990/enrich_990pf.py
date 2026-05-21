"""
Enrich pf_grantees with LLM-generated descriptions.

Queries pf_grantees WHERE description IS NULL, calls Claude once per row,
and writes the result back. No deduplication logic needed — each row in
pf_grantees IS already a unique grantee.

When Phase 2 adds standard 990 program descriptions, update
description and set description_source = '990_program'. Then run:

  UPDATE pf_grants SET embedding_status = 'PENDING'
  WHERE grantee_id IN (
    SELECT grantee_id FROM pf_grantees
    WHERE description_source = '990_program'
  )

to trigger re-embedding with the richer text.

Usage
-----
python enrich_990pf.py \
    --project_id ai-agent-platform-496418 \
    --dataset_id query_dataset \
    [--limit 500] \
    [--batch_size 20] \
    [--call_delay 0.5] \
    [--dry_run] \
    [--dry_run_limit 5]
"""

import argparse
import logging
import sys
import time
from itertools import islice

import anthropic
from google.cloud import bigquery

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are helping build a grant search tool for nonprofit
fundraisers. Given information about a grantee organization, write one
specific sentence describing what they do — the type of work, population
served, and location if clear.

Rules:
- Be specific, not generic. "provides cancer research and patient care"
  is good. "provides support to the community" is not.
- If the grantee name makes the work obvious, use that signal even if
  the purpose is vague.
- If the purpose is specific, use it. If it's generic like
  "PROVIDE OPERATING FUNDS", ignore it and infer from the name.
- One sentence only. No preamble. No "This organization..."."""

USER_TEMPLATE = """Grantee: {grantee_name} ({grantee_state})
Grant purpose: {purpose}

Describe what the grantee does:"""


def batched(items: list, size: int):
    it = iter(items)
    while chunk := list(islice(it, size)):
        yield chunk


def call_claude(client: anthropic.Anthropic, row: dict) -> str | None:
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=120,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": USER_TEMPLATE.format(
                    grantee_name=row.get("grantee_name")  or "Unknown grantee",
                    grantee_state=row.get("grantee_state") or "unknown state",
                    purpose=row.get("sample_purpose")     or "not specified",
                )
            }]
        )
        return response.content[0].text.strip()
    except anthropic.RateLimitError:
        log.warning("Rate limited — sleeping 10s...")
        time.sleep(10)
        return None
    except Exception as exc:
        log.warning("Claude call failed for grantee %s: %s", row.get("grantee_id"), exc)
        return None


def save_batch(
    bq: bigquery.Client,
    table: str,
    updates: list[dict],
    dry_run: bool,
) -> None:
    if not updates:
        return
    if dry_run:
        for u in updates[:3]:
            log.info("[dry_run] %s → %s", u["grantee_id"][:12], u["description"][:100])
        if len(updates) > 3:
            log.info("[dry_run] ...and %d more", len(updates) - 3)
        return

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    for u in updates:
        bq.query(
            f"UPDATE `{table}` "
            "SET description = @description, "
            "    description_source = 'llm', "
            "    updated_at = @updated_at "
            "WHERE grantee_id = @grantee_id",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("description", "STRING", u["description"]),
                    bigquery.ScalarQueryParameter("updated_at",  "STRING", now),
                    bigquery.ScalarQueryParameter("grantee_id",  "STRING", u["grantee_id"]),
                ]
            )
        ).result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id",    required=True)
    parser.add_argument("--dataset_id",    required=True)
    parser.add_argument("--limit",         type=int,   default=500)
    parser.add_argument("--batch_size",    type=int,   default=20)
    parser.add_argument("--call_delay",    type=float, default=0.5)
    parser.add_argument("--dry_run",       action="store_true")
    parser.add_argument("--dry_run_limit", type=int,   default=5)
    args = parser.parse_args()

    bq     = bigquery.Client(project=args.project_id)
    client = anthropic.Anthropic()

    grantees_table = f"{args.project_id}.{args.dataset_id}.pf_grantees"
    grants_table   = f"{args.project_id}.{args.dataset_id}.pf_grants"
    effective_limit = args.dry_run_limit if args.dry_run else args.limit

    # Fetch unenriched grantees, joining to grants to get a sample purpose
    # for context (most recent grant purpose for that grantee)
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
                           PARTITION BY grantee_id
                           ORDER BY created_at DESC
                       ) AS rn
                FROM `{grants_table}`
                WHERE grant_purpose IS NOT NULL
            ) g ON g.grantee_id = gr.grantee_id AND g.rn = 1
            WHERE gr.description IS NULL
            ORDER BY gr.grantee_id
            LIMIT {effective_limit}
        """).result()
    ]

    if not rows:
        log.info("All grantees already have descriptions.")
        return

    log.info("Enriching %d grantees...", len(rows))
    total_enriched = total_failed = 0

    for batch_num, batch in enumerate(batched(rows, args.batch_size), start=1):
        log.info("Batch %d — %d grantees...", batch_num, len(batch))
        updates = []

        for row in batch:
            description = call_claude(client, row)
            time.sleep(args.call_delay)

            if not description:
                total_failed += 1
                continue

            log.info(
                "  %s → %s",
                (row.get("grantee_name") or "")[:35],
                description[:90],
            )
            updates.append({
                "grantee_id":  row["grantee_id"],
                "description": description,
            })

        save_batch(bq, grantees_table, updates, args.dry_run)
        total_enriched += len(updates)
        log.info("  Batch %d done — %d enriched, %d failed", batch_num, total_enriched, total_failed)

        if batch_num * args.batch_size < len(rows):
            time.sleep(1)

    log.info("Done — %d enriched, %d failed", total_enriched, total_failed)


if __name__ == "__main__":
    main()