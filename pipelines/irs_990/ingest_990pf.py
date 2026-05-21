"""
IRS 990-PF parser — ingests private foundation filings into BigQuery.

Writes to three tables:
  pf_foundations  one row per filing
  pf_grantees     one row per unique grantee (deduped by name+state)
  pf_grants       one row per grant, linked to both tables via FKs

Usage
-----
python ingest_990pf.py \
    --project_id ai-agent-platform-496418 \
    --dataset_id query_dataset \
    --bucket_name ai-agent-platform-496418-ai-documents \
    --prefix raw/irs_990_xml/2025_990PF_TEST/ \
    [--limit 100] \
    [--dry_run]
"""

import argparse
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from google.cloud import bigquery, storage

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

_NS_URI = "http://www.irs.gov/efile"
_NS     = {"ns": _NS_URI}
_Q      = f"{{{_NS_URI}}}"

_GENERIC_PURPOSES = {
    "provide operating funds", "provide operating support",
    "provide operatings funds", "general operating support",
    "general support", "operating support", "support",
    "charitable contribution", "charitable purposes",
    "charitable support", "n/a", "none",
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()

def _first(node, *xpaths):
    if node is None:
        return None
    for xpath in xpaths:
        el = node.find(xpath, _NS)
        if el is not None and el.text and el.text.strip():
            return el.text.strip()
    return None

def _iter_first(node, *local_tags):
    if node is None:
        return None
    for tag in local_tags:
        for el in node.iter(f"{_Q}{tag}"):
            if el.text and el.text.strip():
                return el.text.strip()
    return None

def _grantee_id(name: str | None, state: str | None) -> str:
    """Stable ID for a grantee — normalized so minor formatting differences match."""
    key = f"{(name or '').strip().upper()}-{(state or '').strip().upper()}"
    return _sha(key)

def _is_generic(purpose: str | None) -> bool:
    return (purpose or "").strip().lower() in _GENERIC_PURPOSES

def _build_embed_text(filer_name, filer_state, grantee_name, grantee_state, purpose):
    filer   = f"{filer_name} ({filer_state})"   if filer_state   else filer_name
    grantee = f"{grantee_name} ({grantee_state})" if grantee_state else grantee_name
    parts   = [filer, grantee]
    if purpose and not _is_generic(purpose):
        parts.append(purpose)
    return " | ".join(p for p in parts if p)


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_990pf(
    xml_bytes: bytes,
    filename: str,
    gcs_path: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Parse one 990-PF XML file.

    Returns
    -------
    (foundation_row, grantee_rows, grant_rows)
    """
    root   = ET.fromstring(xml_bytes)
    header = root.find("ns:ReturnHeader", _NS)
    rd     = root.find(f"{_Q}ReturnData")

    ein = _first(header, "ns:Filer/ns:EIN")
    org_name = _first(
        header,
        "ns:Filer/ns:BusinessName/ns:BusinessNameLine1Txt",
        "ns:Filer/ns:BusinessName/ns:BusinessNameLine1",
    )
    tax_period_end   = _first(header, "ns:TaxPeriodEndDt")
    tax_period_begin = _first(header, "ns:TaxPeriodBeginDt")
    tax_year_str     = _first(header, "ns:TaxYr")
    tax_period       = tax_period_end or tax_period_begin or "UNKNOWN"

    filing_year: int | None = None
    if tax_year_str and tax_year_str.isdigit():
        filing_year = int(tax_year_str)
    elif tax_period and len(tax_period) >= 4:
        filing_year = int(tax_period[:4])

    foundation_id = _sha(f"{ein}-{tax_period}-{filename}")

    pf   = rd.find(f"{_Q}IRS990PF") if rd is not None else None
    supp = pf.find(f"{_Q}SupplementaryInformationGrp") if pf is not None else None

    only_preselected = supp.find(f"{_Q}OnlyContriToPreselectedInd") if supp is not None else None
    accepts_unsolicited: bool | None = None
    if only_preselected is not None:
        accepts_unsolicited = (only_preselected.text or "").strip() != "X"

    foundation_row: dict[str, Any] = {
        "foundation_id":            foundation_id,
        "ein":                      ein or "UNKNOWN",
        "organization_name":        org_name,
        "filing_year":              filing_year,
        "tax_period":               tax_period,
        "state":                    _iter_first(pf, "OrgReportOrRegisterStateCd"),
        "fmv_assets_raw":           _iter_first(pf, "FMVAssetsEOYAmt", "TotalAssetsEOYFMVAmt"),
        "total_revenue_raw":        _iter_first(pf, "TotalRevAndExpnssAmt"),
        "total_expenses_raw":       _iter_first(pf, "TotalExpensesRevAndExpnssAmt"),
        "total_grants_paid_raw":    _first(supp, "ns:TotalGrantOrContriPdDurYrAmt"),
        "accepts_unsolicited_apps": accepts_unsolicited,
        "xml_filename":             filename,
        "gcs_path":                 gcs_path,
        "created_at":               now_iso(),
    }

    grantee_rows: list[dict[str, Any]] = []
    grant_rows:   list[dict[str, Any]] = []
    seen_grantees: set[str]            = set()

    if supp is not None:
        for i, grp in enumerate(supp.findall(f"{_Q}GrantOrContributionPdDurYrGrp")):
            grantee_name = _first(
                grp,
                "ns:RecipientBusinessName/ns:BusinessNameLine1Txt",
                "ns:RecipientBusinessName/ns:BusinessNameLine1",
            )
            grantee_city  = _first(grp, "ns:RecipientUSAddress/ns:CityNm")
            grantee_state = _first(grp, "ns:RecipientUSAddress/ns:StateAbbreviationCd")
            purpose       = _first(grp, "ns:GrantOrContributionPurposeTxt")
            amount        = _first(grp, "ns:Amt")

            if not grantee_name:
                continue

            gid = _grantee_id(grantee_name, grantee_state)

            # Collect unique grantees encountered in this file
            if gid not in seen_grantees:
                seen_grantees.add(gid)
                grantee_rows.append({
                    "grantee_id":         gid,
                    "grantee_name":       grantee_name,
                    "grantee_state":      grantee_state,
                    "grantee_city":       grantee_city,
                    "description":        None,   # filled by enrich_990pf.py
                    "description_source": None,
                    "created_at":         now_iso(),
                    "updated_at":         None,
                })

            embed_text = _build_embed_text(
                filer_name=org_name,
                filer_state=foundation_row["state"],
                grantee_name=grantee_name,
                grantee_state=grantee_state,
                purpose=purpose,
            )

            grant_rows.append({
                "grant_id":         _sha(f"{foundation_id}-grant-{i}"),
                "foundation_id":    foundation_id,
                "grantee_id":       gid,
                "filer_ein":        ein or "UNKNOWN",
                "filer_name":       org_name,
                "filer_state":      foundation_row["state"],
                "grantee_name":     grantee_name,
                "grantee_city":     grantee_city,
                "grantee_state":    grantee_state,
                "grant_amount_raw": amount,
                "grant_purpose":    purpose,
                "embed_text":       embed_text,
                "embedding_status": "PENDING",
                "created_at":       now_iso(),
            })

    return foundation_row, grantee_rows, grant_rows


# ── BigQuery helpers ──────────────────────────────────────────────────────────

def insert_rows(
    client: bigquery.Client,
    table_id: str,
    rows: list[dict[str, Any]],
    dry_run: bool = False,
) -> None:
    if not rows:
        return
    if dry_run:
        log.info("[dry_run] Would insert %d rows into %s", len(rows), table_id)
        return
    errors = client.insert_rows_json(table_id, rows)
    if errors:
        raise RuntimeError(f"BigQuery insert failed for {table_id}: {errors}")


def insert_new_grantees(
    client: bigquery.Client,
    table_id: str,
    grantees: list[dict[str, Any]],
    dry_run: bool = False,
) -> None:
    """
    Insert only grantees that don't already exist in BigQuery.
    Checks by grantee_id so re-running ingest never creates duplicates.
    """
    if not grantees or dry_run:
        if dry_run and grantees:
            log.info("[dry_run] Would upsert %d grantees into %s", len(grantees), table_id)
        return

    # Deduplicate within this batch first
    seen:   set[str]            = set()
    unique: list[dict[str, Any]] = []
    for g in grantees:
        if g["grantee_id"] not in seen:
            seen.add(g["grantee_id"])
            unique.append(g)

    # Check which grantee_ids already exist in BigQuery
    ids_sql  = ", ".join(f"'{g['grantee_id']}'" for g in unique)
    existing = {
        row["grantee_id"]
        for row in client.query(f"""
            SELECT grantee_id
            FROM `{table_id}`
            WHERE grantee_id IN ({ids_sql})
        """).result()
    }

    new_grantees = [g for g in unique if g["grantee_id"] not in existing]

    if new_grantees:
        # Use DML INSERT (not streaming) so rows are immediately mutable.
        # Streaming inserts go into a buffer that BigQuery won't let you
        # UPDATE/DELETE for up to 90 minutes — enrich_990pf.py needs to
        # UPDATE these rows right after ingest.
        #
        # Use parameterized queries to handle ALL special characters
        # (apostrophes, quotes, backslashes) without manual escaping.
        for g in new_grantees:
            client.query(
                f"""
                INSERT INTO `{table_id}`
                    (grantee_id, grantee_name, grantee_state, grantee_city,
                     description, description_source, created_at, updated_at)
                VALUES
                    (@grantee_id, @grantee_name, @grantee_state, @grantee_city,
                     NULL, NULL, @created_at, NULL)
                """,
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("grantee_id",    "STRING", g["grantee_id"]),
                        bigquery.ScalarQueryParameter("grantee_name",  "STRING", g.get("grantee_name")  or ""),
                        bigquery.ScalarQueryParameter("grantee_state", "STRING", g.get("grantee_state") or ""),
                        bigquery.ScalarQueryParameter("grantee_city",  "STRING", g.get("grantee_city")  or ""),
                        bigquery.ScalarQueryParameter("created_at",    "STRING", g["created_at"]),
                    ]
                )
            ).result()
        log.info("  Inserted %d new grantees (%d already existed)", len(new_grantees), len(existing))


# ── Entry point ───────────────────────────────────────────────────────────────

def ingest_gcs_prefix(
    project_id: str,
    dataset_id: str,
    bucket_name: str,
    prefix: str,
    limit: int | None = None,
    dry_run: bool = False,
) -> None:
    storage_client = storage.Client(project=project_id)
    bq_client      = bigquery.Client(project=project_id)

    foundations_table = f"{project_id}.{dataset_id}.pf_foundations"
    grantees_table    = f"{project_id}.{dataset_id}.pf_grantees"
    grants_table      = f"{project_id}.{dataset_id}.pf_grants"

    xml_blobs = [
        b for b in storage_client.list_blobs(bucket_name, prefix=prefix)
        if b.name.endswith(".xml")
    ]
    if limit:
        xml_blobs = xml_blobs[:limit]

    log.info("Found %d XML files under gs://%s/%s", len(xml_blobs), bucket_name, prefix)

    ok = failed = 0

    for blob in xml_blobs:
        log.info("Ingesting: %s", blob.name)
        try:
            xml_bytes = blob.download_as_bytes()
            foundation_row, grantee_rows, grant_rows = parse_990pf(
                xml_bytes=xml_bytes,
                filename=Path(blob.name).name,
                gcs_path=f"gs://{bucket_name}/{blob.name}",
            )
            insert_rows(bq_client, foundations_table, [foundation_row], dry_run)
            insert_new_grantees(bq_client, grantees_table, grantee_rows, dry_run)
            insert_rows(bq_client, grants_table, grant_rows, dry_run)

            log.info(
                "  → %s (%s) | %d grants | %d grantees | unsolicited=%s",
                foundation_row["organization_name"],
                foundation_row["state"],
                len(grant_rows),
                len(grantee_rows),
                foundation_row["accepts_unsolicited_apps"],
            )
            ok += 1
        except Exception as exc:
            log.warning("  ✗ Skipped %s: %s", blob.name, exc)
            failed += 1

    log.info("Done — %d succeeded, %d failed", ok, failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest 990-PF filings into BigQuery")
    parser.add_argument("--project_id",  required=True)
    parser.add_argument("--dataset_id",  required=True)
    parser.add_argument("--bucket_name", required=True)
    parser.add_argument("--prefix",      required=True)
    parser.add_argument("--limit",       type=int, default=None)
    parser.add_argument("--dry_run",     action="store_true")
    args = parser.parse_args()

    ingest_gcs_prefix(
        project_id=args.project_id,
        dataset_id=args.dataset_id,
        bucket_name=args.bucket_name,
        prefix=args.prefix,
        limit=args.limit,
        dry_run=args.dry_run,
    )