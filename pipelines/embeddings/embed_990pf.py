"""
Embed pf_grants and upsert vectors to Pinecone.

Joins pf_grants to pf_grantees to get the grantee description,
builds enriched embed text on the fly, and upserts to Pinecone.

Falls back to embed_text (built at parse time) when grantee has
no description yet — so this can be run before enrichment is complete.

Usage
-----
export PINECONE_API_KEY=...
export PINECONE_INDEX_NAME=...

python embed_990pf.py \
    --project_id ai-agent-platform-496418 \
    --dataset_id query_dataset \
    [--region us-central1] \
    [--limit 1000]
"""

import argparse
import logging
import os
import sys
from itertools import islice

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

VERTEX_BATCH_LIMIT   = 250
PINECONE_BATCH_LIMIT = 100
PINECONE_NAMESPACE   = "pf_grants"


def batched(items: list, size: int):
    it = iter(items)
    while chunk := list(islice(it, size)):
        yield chunk


def build_embed_text(
    filer_name:    str | None,
    filer_state:   str | None,
    grantee_name:  str | None,
    grantee_state: str | None,
    description:   str | None,
    fallback:      str | None,
) -> str:
    """
    Build text to embed for this grant.
    If grantee has a description, use: filer | grantee | description
    Otherwise fall back to the base embed_text from parse time.
    """
    if description:
        filer   = f"{filer_name} ({filer_state})"    if filer_state   else filer_name or ""
        grantee = f"{grantee_name} ({grantee_state})" if grantee_state else grantee_name or ""
        return " | ".join(p for p in [filer, grantee, description] if p)
    return fallback or ""


def grant_size_bucket(amount_raw: str | None) -> str:
    try:
        amt = int(amount_raw or "")
        if amt < 5_000:   return "small"
        if amt < 25_000:  return "medium"
        if amt < 100_000: return "large"
        return "major"
    except (ValueError, TypeError):
        return "unknown"


def mark_completed(bq: bigquery.Client, table: str, grant_ids: list[str]) -> None:
    ids_sql = ", ".join(f"'{gid}'" for gid in grant_ids)
    bq.query(f"""
        UPDATE `{table}`
        SET embedding_status = 'COMPLETED'
        WHERE grant_id IN ({ids_sql})
    """).result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id",          required=True)
    parser.add_argument("--dataset_id",          required=True)
    parser.add_argument("--region",              default="us-central1")
    parser.add_argument("--limit",               type=int, default=1_000)
    parser.add_argument("--embed_batch_size",    type=int, default=VERTEX_BATCH_LIMIT)
    parser.add_argument("--pinecone_batch_size", type=int, default=PINECONE_BATCH_LIMIT)
    args = parser.parse_args()

    if args.embed_batch_size > VERTEX_BATCH_LIMIT:
        parser.error(f"--embed_batch_size cannot exceed {VERTEX_BATCH_LIMIT}")

    grants_table      = f"{args.project_id}.{args.dataset_id}.pf_grants"
    foundations_table = f"{args.project_id}.{args.dataset_id}.pf_foundations"
    grantees_table    = f"{args.project_id}.{args.dataset_id}.pf_grantees"

    bq    = bigquery.Client(project=args.project_id)
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(
        os.environ["PINECONE_INDEX_NAME"]
    )
    vertexai.init(project=args.project_id, location=args.region)
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    # ── Fetch pending grants joined to grantee descriptions ───────────────────
    rows = [
        dict(r) for r in bq.query(f"""
            SELECT
                g.grant_id,
                g.embed_text,
                g.filer_ein,
                g.filer_name,
                g.filer_state,
                g.grantee_name,
                g.grantee_city,
                g.grantee_state,
                g.grant_amount_raw,
                g.grant_purpose,
                f.filing_year,
                f.accepts_unsolicited_apps,
                f.fmv_assets_raw,
                gr.description,
                gr.description_source
            FROM `{grants_table}` g
            JOIN `{foundations_table}` f  ON g.foundation_id = f.foundation_id
            JOIN `{grantees_table}`    gr ON g.grantee_id    = gr.grantee_id
            WHERE g.embedding_status = 'PENDING'
              AND g.embed_text IS NOT NULL
            LIMIT {args.limit}
        """).result()
    ]

    if not rows:
        log.info("No PENDING grants found — nothing to do.")
        return

    enriched_count = sum(1 for r in rows if r.get("description"))
    log.info(
        "Found %d PENDING grants (%d with grantee description, %d using base embed_text)",
        len(rows), enriched_count, len(rows) - enriched_count,
    )

    # ── Build embed text per row ──────────────────────────────────────────────
    for row in rows:
        row["_text_to_embed"] = build_embed_text(
            filer_name=row.get("filer_name"),
            filer_state=row.get("filer_state"),
            grantee_name=row.get("grantee_name"),
            grantee_state=row.get("grantee_state"),
            description=row.get("description"),
            fallback=row.get("embed_text"),
        )

    # ── Embed ─────────────────────────────────────────────────────────────────
    vectors: list[list[float]] = []
    embed_batches = list(batched(rows, args.embed_batch_size))

    for i, batch in enumerate(embed_batches):
        log.info("  Embedding batch %d/%d (%d texts)...", i + 1, len(embed_batches), len(batch))
        embeddings = model.get_embeddings([r["_text_to_embed"] for r in batch])
        vectors.extend(e.values for e in embeddings)

    # ── Build Pinecone records ────────────────────────────────────────────────
    records = []
    for row, vector in zip(rows, vectors):
        records.append({
            "id":     row["grant_id"],
            "values": vector,
            "metadata": {
                "filer_name":              row["filer_name"]    or "",
                "filer_ein":               row["filer_ein"]     or "",
                "filer_state":             row["filer_state"]   or "",
                "grantee_name":            row["grantee_name"]  or "",
                "grantee_city":            row["grantee_city"]  or "",
                "grantee_state":           row["grantee_state"] or "",
                "grant_purpose":           (row["grant_purpose"] or "")[:300],
                "grant_amount_raw":        row["grant_amount_raw"] or "",
                "filing_year":             row["filing_year"]   or "",
                "fmv_assets_raw":          row["fmv_assets_raw"] or "",
                "accepts_unsolicited_apps": (
                    row["accepts_unsolicited_apps"]
                    if row["accepts_unsolicited_apps"] is not None
                    else "unknown"
                ),
                "description_source":      row["description_source"] or "none",
                "embed_text":              row["_text_to_embed"][:400],
                "grant_size":              grant_size_bucket(row.get("grant_amount_raw")),
                "state":                   row["filer_state"]   or "",
                "grantee_state_filter":    row["grantee_state"] or "",
                "open_to_apply":           row["accepts_unsolicited_apps"] is True,
            },
        })

    # ── Upsert + mark completed per batch ─────────────────────────────────────
    total_upserted = 0
    for batch in batched(records, args.pinecone_batch_size):
        index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)
        mark_completed(bq, grants_table, [r["id"] for r in batch])
        total_upserted += len(batch)
        log.info("  Upserted + marked %d / %d", total_upserted, len(records))

    log.info("Done — %d vectors in namespace '%s'", total_upserted, PINECONE_NAMESPACE)


if __name__ == "__main__":
    main()