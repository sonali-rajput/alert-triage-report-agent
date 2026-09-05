# The vector store, the audit trail and the run history. One dataset, three
# tables, and no other database in the system.
resource "google_bigquery_dataset" "this" {
  dataset_id  = var.dataset_id
  location    = var.location
  description = "Alert triage: ingested alerts with embeddings, agent verdicts, run history."

  delete_contents_on_destroy = !var.deletion_protection
}

# One row per alert per run.
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
# RunStats gains a field every time the pipeline learns to count something new.
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
