variable "project_id" {
  description = "GCP project ID for the dev environment. Must already be bootstrapped (infra/bootstrap)."
  type        = string
}

variable "region" {
  description = "Region for Cloud Run, BigQuery and GCS."
  type        = string
  default     = "europe-west2"
}

variable "vertex_location" {
  description = <<-EOT
    Region for Vertex AI (Gemini and gemini-embedding-001). Kept separate from
    `region` because model availability is per-region and lags behind the
    general Cloud Run/BigQuery footprint -- pinning them together means a model
    that is not served in your data region becomes a silent deploy blocker.
  EOT
  type        = string
  default     = "europe-west2"
}

variable "name_prefix" {
  description = "Must match the `name_prefix` used in infra/bootstrap: this is how the service accounts are found."
  type        = string
  default     = "triage"
}

variable "image" {
  description = <<-EOT
    Container image for the job, e.g.
    europe-west2-docker.pkg.dev/PROJECT/triage/pipeline:a1b2c3d.
    Built and pushed by CI, not by Terraform: Terraform owns the
    infrastructure, the pipeline owns the artifact, and they meet here.
    Prefer an immutable tag over `latest` -- it is what makes "which code
    produced this report" answerable after the fact.
  EOT
  type        = string
}

variable "bigquery_dataset" {
  description = "Dataset holding the vector store, the audit trail and the run history."
  type        = string
  default     = "prodtools_triage"
}

variable "schedule" {
  description = "Cron for the daily run, in `schedule_timezone`."
  type        = string
  default     = "0 8 * * *"
}

variable "schedule_timezone" {
  type    = string
  default = "Europe/London"
}

variable "sentry_org" {
  description = "Sentry organisation slug."
  type        = string
}

variable "sentry_team" {
  description = <<-EOT
    Sentry team slug. Every issue query is scoped to the projects this team
    owns, resolved from the slug on each run. Leave empty only if you mean
    "whatever the token can see" -- which is a property of the token, not a
    decision, and changes silently if the token is ever swapped.
  EOT
  type        = string
  default     = ""
}

variable "sentry_environments" {
  description = <<-EOT
    CANONICAL environment names to fetch, comma-separated. Each is expanded at
    runtime against the environments the org actually reports, so "production"
    also fetches `live`, `@prod` and `<service>@production`. Measured on the
    real org, the literal spelling reached 14 issues and the expanded list
    reached 18. A name matching nothing is a hard error, not a silent empty
    run. Empty means one unfiltered query.
  EOT
  type        = string
  default     = "production"
}

variable "top_n" {
  description = "How many issues the top-issues agent selects for triage and the report."
  type        = number
  default     = 10
}

variable "gemini_model" {
  description = <<-EOT
    Must match `PipelineSettings.gemini_model`, and there is a test that fails
    if it does not. Not pedantry: this variable sets GEMINI_MODEL on the job,
    so it silently overrides the code default, and the deployed model would
    then be one no evaluation covers. Every real-model result this project has
    was measured against gemini-3.5-flash.
  EOT
  type        = string
  default     = "gemini-3.5-flash"
}

variable "embedding_dimensions" {
  description = <<-EOT
    Matryoshka output width. This lives here, next to the BigQuery table that
    defines the `embedding` column, because the two must agree: VECTOR_SEARCH
    cannot compare vectors of different widths and fails the query outright.
    Changing it invalidates every stored vector -- re-embedding the history is
    the migration.
  EOT
  type        = number
  default     = 768
}

variable "embedding_model" {
  description = "Changing this changes the vector width, which invalidates every stored embedding -- see the note in modules/bigquery."
  type        = string
  default     = "gemini-embedding-001"
}

variable "job_timeout" {
  type    = string
  default = "1800s"
}

variable "report_retention_days" {
  type    = number
  default = 90
}
