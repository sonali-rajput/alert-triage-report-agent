# Applied ONCE per project, by a human with project-admin rights. Everything
# here outlives any single environment, which is the whole reason it is a
# separate stack: `terraform destroy` on a disposable dev environment must not
# be able to take an API, a service account or the image repo with it.
#
#   terraform init -backend-config=backend/dev.hcl
#   terraform apply -var-file=envs/dev.tfvars

# --- APIs --------------------------------------------------------------------

# Enabling an API is a project-wide act. `disable_on_destroy = false` because
# something else in the project may depend on one of these.
resource "google_project_service" "required" {
  for_each = toset([
    "run.googleapis.com",
    "cloudscheduler.googleapis.com",
    "bigquery.googleapis.com",
    "aiplatform.googleapis.com",
    "storage.googleapis.com",
    "secretmanager.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "iamcredentials.googleapis.com", # signBlob, for the report's signed URLs
  ])

  service            = each.value
  disable_on_destroy = false
}

# --- Artifact Registry -------------------------------------------------------

# The image repo outlives any environment: two environments in one project
# should pull the same immutable tag rather than each building their own.
resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = var.name_prefix
  description   = "Alert triage agent images"
  format        = "DOCKER"

  depends_on = [google_project_service.required]
}

# --- Identities --------------------------------------------------------------

# One service account for the job. Nothing else in the project should use it,
# so its permissions describe exactly what the pipeline can do.
resource "google_service_account" "job" {
  account_id   = "${var.name_prefix}-agent-sa"
  display_name = "Alert triage agent (Cloud Run Job)"
  description  = "Runs the daily triage pipeline: Vertex AI, BigQuery, GCS, Secret Manager."

  depends_on = [google_project_service.required]
}

# Project-level roles only. Everything that CAN be scoped to a resource --
# the BigQuery dataset, the reports bucket, the two secrets -- is granted by
# the environment that owns that resource, so a destroyed environment takes
# its own grants with it and leaves nothing behind.
resource "google_project_iam_member" "job" {
  for_each = toset([
    "roles/aiplatform.user",  # Gemini + text-embedding-004
    "roles/bigquery.jobUser", # run queries (the vector search)
    "roles/logging.logWriter",
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.job.email}"
}

# Signing a V4 URL from Cloud Run needs the SA to be able to sign AS ITSELF via
# the IAM signBlob API -- there is no private key on the instance. Without this
# binding the report still uploads and the run still succeeds; the Chat card
# just carries an unusable gs:// URI instead of a link, which is a confusing
# failure to diagnose after the fact.
resource "google_service_account_iam_member" "job_self_sign" {
  service_account_id = google_service_account.job.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:${google_service_account.job.email}"
}

# Cloud Scheduler needs its own identity to start the job. It gets exactly one
# permission, and the environment grants it: run.invoker on that job alone.
resource "google_service_account" "scheduler" {
  account_id   = "${var.name_prefix}-scheduler-sa"
  display_name = "Alert triage scheduler"
  description  = "Starts the triage Cloud Run Job on a cron schedule. Can do nothing else."

  depends_on = [google_project_service.required]
}
