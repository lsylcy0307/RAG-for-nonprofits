# ─────────────────────────────────────────────────────────────────────────────
# Three tables:
#   pf_foundations   one row per 990-PF filing
#   pf_grantees      one row per unique grantee — description lives here
#   pf_grants        one row per grant — links foundation to grantee
#
# Key design: grantee descriptions live in pf_grantees, not pf_grants.
# The same grantee (e.g. Boys and Girls Club) appears across many foundation
# filings — storing the description once and joining is cleaner than
# duplicating it on every grant row and managing cache invalidation.
#
# When Phase 2 adds standard 990 program descriptions, update
# pf_grantees.description and set description_source = '990_program'.
# Then reset embedding_status = 'PENDING' on all grants for that grantee
# to trigger re-embedding with the richer text.
# ─────────────────────────────────────────────────────────────────────────────

resource "google_bigquery_table" "pf_foundations" {
  dataset_id          = google_bigquery_dataset.query_dataset.dataset_id
  table_id            = "pf_foundations"
  deletion_protection = false

  schema = jsonencode([
    { name = "foundation_id",     type = "STRING",    mode = "REQUIRED",
      description = "SHA-256 of ein+tax_period+filename" },
    { name = "ein",               type = "STRING",    mode = "REQUIRED"  },
    { name = "organization_name", type = "STRING",    mode = "NULLABLE"  },
    { name = "filing_year",       type = "INTEGER",   mode = "NULLABLE"  },
    { name = "tax_period",        type = "STRING",    mode = "NULLABLE"  },
    { name = "state",             type = "STRING",    mode = "NULLABLE"  },
    { name = "fmv_assets_raw",        type = "STRING", mode = "NULLABLE",
      description = "Fair market value of assets EOY" },
    { name = "total_revenue_raw",     type = "STRING", mode = "NULLABLE" },
    { name = "total_expenses_raw",    type = "STRING", mode = "NULLABLE" },
    { name = "total_grants_paid_raw", type = "STRING", mode = "NULLABLE" },
    { name = "accepts_unsolicited_apps", type = "BOOLEAN", mode = "NULLABLE",
      description = "false=invitation-only, true=open, null=not specified" },
    { name = "xml_filename",      type = "STRING",    mode = "NULLABLE"  },
    { name = "gcs_path",          type = "STRING",    mode = "NULLABLE"  },
    { name = "created_at",        type = "TIMESTAMP", mode = "REQUIRED"  },
  ])
}

resource "google_bigquery_table" "pf_grantees" {
  dataset_id          = google_bigquery_dataset.query_dataset.dataset_id
  table_id            = "pf_grantees"
  deletion_protection = false

  schema = jsonencode([
    { name = "grantee_id",   type = "STRING",    mode = "REQUIRED",
      description = "SHA-256 of normalized (grantee_name + grantee_state) — stable across filings" },
    { name = "grantee_name",  type = "STRING",   mode = "NULLABLE" },
    { name = "grantee_state", type = "STRING",   mode = "NULLABLE" },
    { name = "grantee_city",  type = "STRING",   mode = "NULLABLE" },

    # Description of what the grantee does — used to build embed text.
    # Source priority: 990_program (real filing) > llm (Claude inference)
    { name = "description",        type = "STRING", mode = "NULLABLE",
      description = "What this grantee does — one sentence used in embed text" },
    { name = "description_source", type = "STRING", mode = "NULLABLE",
      description = "'llm' = Claude-generated, '990_program' = from grantee's own 990 filing" },

    { name = "created_at",  type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "updated_at",  type = "TIMESTAMP", mode = "NULLABLE",
      description = "Set when description is upgraded from llm to 990_program" },
  ])
}

resource "google_bigquery_table" "pf_grants" {
  dataset_id          = google_bigquery_dataset.query_dataset.dataset_id
  table_id            = "pf_grants"
  deletion_protection = false

  schema = jsonencode([
    { name = "grant_id",      type = "STRING", mode = "REQUIRED",
      description = "SHA-256 of foundation_id+index — also used as Pinecone vector ID" },
    { name = "foundation_id", type = "STRING", mode = "REQUIRED",
      description = "FK to pf_foundations.foundation_id" },
    { name = "grantee_id",    type = "STRING", mode = "NULLABLE",
      description = "FK to pf_grantees.grantee_id" },

    { name = "filer_ein",   type = "STRING", mode = "NULLABLE" },
    { name = "filer_name",  type = "STRING", mode = "NULLABLE" },
    { name = "filer_state", type = "STRING", mode = "NULLABLE" },

    # Kept for display without needing a join
    { name = "grantee_name",  type = "STRING", mode = "NULLABLE" },
    { name = "grantee_city",  type = "STRING", mode = "NULLABLE" },
    { name = "grantee_state", type = "STRING", mode = "NULLABLE" },

    { name = "grant_amount_raw", type = "STRING", mode = "NULLABLE" },
    { name = "grant_purpose",    type = "STRING", mode = "NULLABLE",
      description = "Raw purpose text — may be generic boilerplate" },

    { name = "embed_text", type = "STRING", mode = "NULLABLE",
      description = "Base embed text built at parse time (generic purposes stripped). Fallback when grantee has no description yet." },

    { name = "embedding_status", type = "STRING", mode = "NULLABLE",
      description = "PENDING → COMPLETED. Reset to PENDING when grantee description is upgraded." },

    { name = "created_at", type = "TIMESTAMP", mode = "REQUIRED" },
  ])
}
