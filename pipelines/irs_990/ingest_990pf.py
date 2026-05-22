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


def _p(name: str, bq_type: str, val: Any) -> "bigquery.ScalarQueryParameter":
    return bigquery.ScalarQueryParameter(name, bq_type, val)


def insert_foundation(
    client:   bigquery.Client,
    table_id: str,
    row:      dict[str, Any],
) -> None:
    """Insert one pf_foundations row via DML."""
    client.query(
        f"INSERT INTO `{table_id}` "
        "(foundation_id, ein, organization_name, filing_year, tax_period, state, "
        " fmv_assets_raw, total_revenue_raw, total_expenses_raw, total_grants_paid_raw, "
        " accepts_unsolicited_apps, xml_filename, gcs_path, created_at) "
        "VALUES (@foundation_id, @ein, @organization_name, @filing_year, @tax_period, @state, "
        " @fmv_assets_raw, @total_revenue_raw, @total_expenses_raw, @total_grants_paid_raw, "
        " @accepts_unsolicited_apps, @xml_filename, @gcs_path, @created_at)",
        job_config=bigquery.QueryJobConfig(query_parameters=[
            _p("foundation_id",         "STRING",    row["foundation_id"]),
            _p("ein",                   "STRING",    row["ein"]),
            _p("organization_name",     "STRING",    row.get("organization_name") or ""),
            _p("filing_year",           "INT64",     row.get("filing_year")),
            _p("tax_period",            "STRING",    row.get("tax_period") or ""),
            _p("state",                 "STRING",    row.get("state") or ""),
            _p("fmv_assets_raw",        "STRING",    row.get("fmv_assets_raw") or ""),
            _p("total_revenue_raw",     "STRING",    row.get("total_revenue_raw") or ""),
            _p("total_expenses_raw",    "STRING",    row.get("total_expenses_raw") or ""),
            _p("total_grants_paid_raw", "STRING",    row.get("total_grants_paid_raw") or ""),
            _p("accepts_unsolicited_apps", "BOOL",   row.get("accepts_unsolicited_apps")),
            _p("xml_filename",          "STRING",    row.get("xml_filename") or ""),
            _p("gcs_path",              "STRING",    row.get("gcs_path") or ""),
            _p("created_at",            "TIMESTAMP", row["created_at"]),
        ])
    ).result()


def insert_grants(
    client:   bigquery.Client,
    table_id: str,
    rows:     list[dict[str, Any]],
) -> None:
    """Insert pf_grants rows via batched DML (one query for all rows)."""
    if not rows:
        return

    select_parts: list[str] = []
    params:       list      = []

    for i, row in enumerate(rows):
        select_parts.append(
            f"SELECT @grant_id_{i}, @foundation_id_{i}, @grantee_id_{i}, "
            f"       @filer_ein_{i}, @filer_name_{i}, @filer_state_{i}, "
            f"       @grantee_name_{i}, @grantee_city_{i}, @grantee_state_{i}, "
            f"       @grant_amount_raw_{i}, @grant_purpose_{i}, "
            f"       @embed_text_{i}, @embedding_status_{i}, @created_at_{i}"
        )
        params.extend([
            _p(f"grant_id_{i}",         "STRING",    row["grant_id"]),
            _p(f"foundation_id_{i}",    "STRING",    row["foundation_id"]),
            _p(f"grantee_id_{i}",       "STRING",    row["grantee_id"]),
            _p(f"filer_ein_{i}",        "STRING",    row.get("filer_ein") or ""),
            _p(f"filer_name_{i}",       "STRING",    row.get("filer_name") or ""),
            _p(f"filer_state_{i}",      "STRING",    row.get("filer_state") or ""),
            _p(f"grantee_name_{i}",     "STRING",    row.get("grantee_name") or ""),
            _p(f"grantee_city_{i}",     "STRING",    row.get("grantee_city") or ""),
            _p(f"grantee_state_{i}",    "STRING",    row.get("grantee_state") or ""),
            _p(f"grant_amount_raw_{i}", "STRING",    row.get("grant_amount_raw") or ""),
            _p(f"grant_purpose_{i}",    "STRING",    row.get("grant_purpose") or ""),
            _p(f"embed_text_{i}",       "STRING",    row.get("embed_text") or ""),
            _p(f"embedding_status_{i}", "STRING",    row.get("embedding_status") or "PENDING"),
            _p(f"created_at_{i}",       "TIMESTAMP", row["created_at"]),
        ])

    union_sql = " UNION ALL ".join(select_parts)
    client.query(
        f"INSERT INTO `{table_id}` "
        "(grant_id, foundation_id, grantee_id, filer_ein, filer_name, filer_state, "
        " grantee_name, grantee_city, grantee_state, grant_amount_raw, grant_purpose, "
        " embed_text, embedding_status, created_at) "
        f"{union_sql}",
        job_config=bigquery.QueryJobConfig(query_parameters=params)
    ).result()


def insert_rows(
    client:   bigquery.Client,
    table_id: str,
    rows:     list[dict[str, Any]],
    dry_run:  bool = False,
) -> None:
    """Dispatcher — routes to the correct explicit insert function."""
    if not rows:
        return
    if dry_run:
        log.info("[dry_run] Would insert %d rows into %s", len(rows), table_id)
        return
    if "pf_foundations" in table_id:
        for row in rows:
            insert_foundation(client, table_id, row)
    elif "pf_grants" in table_id:
        insert_grants(client, table_id, rows)
    else:
        raise ValueError(f"No explicit insert defined for {table_id}")


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

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for g in grantees:
        if g["grantee_id"] not in seen:
            seen.add(g["grantee_id"])
            unique.append(g)

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
        # Batch all inserts into one query using UNION ALL.
        # One round trip regardless of how many grantees — Molina Healthcare
        # had 137 grantees which caused 137 sequential BQ calls (very slow).
        # BigQuery parameterized queries don't support multi-row VALUES, so
        # we build a SELECT ... UNION ALL ... query from struct literals.
        # Each row uses individual named parameters to stay safe from injection.
        select_parts = []
        params       = []

        for i, g in enumerate(new_grantees):
            select_parts.append(
                f"SELECT @grantee_id_{i}, @grantee_name_{i}, @grantee_state_{i}, "
                f"       @grantee_city_{i}, CAST(NULL AS STRING), CAST(NULL AS STRING), "
                f"       @created_at_{i}, CAST(NULL AS TIMESTAMP)"
            )
            params.extend([
                _p(f"grantee_id_{i}",    "STRING",    g["grantee_id"]),
                _p(f"grantee_name_{i}",  "STRING",    g.get("grantee_name")  or ""),
                _p(f"grantee_state_{i}", "STRING",    g.get("grantee_state") or ""),
                _p(f"grantee_city_{i}",  "STRING",    g.get("grantee_city")  or ""),
                _p(f"created_at_{i}",    "TIMESTAMP", g["created_at"]),
            ])

        union_sql = " UNION ALL ".join(select_parts)
        client.query(
            f"""
            INSERT INTO `{table_id}`
                (grantee_id, grantee_name, grantee_state, grantee_city,
                 description, description_source, created_at, updated_at)
            {union_sql}
            """,
            job_config=bigquery.QueryJobConfig(query_parameters=params)
        ).result()
        log.info("  Inserted %d new grantees (%d already existed)", len(new_grantees), len(existing))

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

    already_ingested: set[str] = set()
    if not dry_run:
        already_ingested = {
            row["xml_filename"]
            for row in bq_client.query(f"""
                SELECT xml_filename
                FROM `{foundations_table}`
                WHERE xml_filename IS NOT NULL
            """).result()
        }
        if already_ingested:
            log.info("Skipping %d already-ingested files", len(already_ingested))

    ok = failed = skipped = 0

    for blob in xml_blobs:
        filename = Path(blob.name).name
        if filename in already_ingested:
            skipped += 1
            log.debug("  Skipping %s — already ingested", filename)
            continue

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

    log.info("Done — %d succeeded, %d failed, %d skipped (already ingested)", ok, failed, skipped)


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