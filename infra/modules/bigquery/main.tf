# The vector store, the audit trail and the run history. One dataset, three
# tables, and no other database in the system.
resource "google_bigquery_dataset" "this" {
  dataset_id  = var.dataset_id
  location    = var.location
  description = "Alert triage: ingested alerts with embeddings, agent verdicts, run history."

  delete_contents_on_destroy = !var.deletion_protection
}

# One row per alert per run. Both the audit archive of what we ingested and the
# vector index the next run searches -- the embedding travels with the
# projection rather than living in a table of its own, because every query that
# wants one wants the other.
#
# NOTE ON `embedding`: its width is fixed by the embedding model
# (text-embedding-004 -> 768). Changing the model changes the width, and
# VECTOR_SEARCH cannot compare vectors of different widths -- the query fails
# outright rather than returning bad matches, which is the good failure mode.
# Re-embedding the history is the migration; there is no in-place fix.
resource "google_bigquery_table" "alerts" {
  dataset_id          = google_bigquery_dataset.this.dataset_id
  table_id            = "alerts"
  deletion_protection = var.deletion_protection

  time_partitioning {
    type  = "DAY"
    field = "run_date"
  }
  # Every query in the pipeline filters or joins on alert_id.
  clustering = ["alert_id"]

  schema = file("${path.module}/schemas/alerts.json")
}

# One row per alert that reached the top-issues agent -- selected or not.
# "Why was this NOT in today's report" is the question the audit trail most
# often has to answer, and it is unanswerable if only the winners are stored.
resource "google_bigquery_table" "triage" {
  dataset_id          = google_bigquery_dataset.this.dataset_id
  table_id            = "triage"
  deletion_protection = var.deletion_protection

  time_partitioning {
    type  = "DAY"
    field = "run_date"
  }
  clustering = ["alert_id", "priority"]

  schema = file("${path.module}/schemas/triage.json")
}

# One row per run. `stats` is a JSON string rather than a RECORD on purpose:
# RunStats gains a field every time the pipeline learns to count something new,
# and a schema migration per counter is not a trade worth making for an MVP.
resource "google_bigquery_table" "runs" {
  dataset_id          = google_bigquery_dataset.this.dataset_id
  table_id            = "runs"
  deletion_protection = var.deletion_protection

  schema = file("${path.module}/schemas/runs.json")
}

# Dataset-scoped, not project-scoped: the job reads and writes its own three
# tables and has no business touching any other dataset in the project.
resource "google_bigquery_dataset_iam_member" "writer" {
  dataset_id = google_bigquery_dataset.this.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = var.writer_member
}
