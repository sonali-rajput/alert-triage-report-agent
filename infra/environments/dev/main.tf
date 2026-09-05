# The service accounts come from infra/bootstrap.
data "google_service_account" "job" {
  account_id = "${var.name_prefix}-agent-sa"
  project    = var.project_id
}

data "google_service_account" "scheduler" {
  account_id = "${var.name_prefix}-scheduler-sa"
  project    = var.project_id
}

locals {
  job_member = "serviceAccount:${data.google_service_account.job.email}"
}

module "bigquery" {
  source = "../../modules/bigquery"

  dataset_id    = var.bigquery_dataset
  location      = var.region
  writer_member = local.job_member

  # Dev: `terraform destroy` should actually destroy it. In production this is
  # the opposite of what you want.
  deletion_protection = false
}

module "storage" {
  source = "../../modules/storage"

  bucket_name    = "${var.project_id}-${var.name_prefix}-reports"
  location       = var.region
  writer_member  = local.job_member
  retention_days = var.report_retention_days
  force_destroy  = true # dev
}

module "secrets" {
  source = "../../modules/secrets"

  secrets = {
    sentry_token = "${var.name_prefix}-sentry-token"
    chat_webhook = "${var.name_prefix}-chat-webhook"
  }
  accessor_member = local.job_member
}

module "job" {
  source = "../../modules/cloud_run_job"

  name            = "${var.name_prefix}-pipeline"
  location        = var.region
  image           = var.image
  service_account = data.google_service_account.job.email
  timeout         = var.job_timeout

  env = {
    GCP_PROJECT          = var.project_id
    GCP_LOCATION         = var.vertex_location
    STORE_BACKEND        = "bigquery"
    BIGQUERY_DATASET     = module.bigquery.dataset_id
    GCS_BUCKET           = module.storage.bucket_name
    LLM_PROVIDER         = "vertex"
    GEMINI_MODEL         = var.gemini_model
    EMBEDDING_PROVIDER   = "vertex"
    EMBEDDING_MODEL      = var.embedding_model
    EMBEDDING_DIMENSIONS = tostring(var.embedding_dimensions)
    SENTRY_SOURCE        = "api"
    SENTRY_ORG           = var.sentry_org
    SENTRY_TEAM          = var.sentry_team
    SENTRY_ENVIRONMENTS  = var.sentry_environments
    TOP_N                = tostring(var.top_n)
  }

  secret_env = {
    SENTRY_TOKEN     = module.secrets.secret_ids["sentry_token"]
    CHAT_WEBHOOK_URL = module.secrets.secret_ids["chat_webhook"]
  }
}

module "scheduler" {
  source = "../../modules/scheduler"

  name            = "${var.name_prefix}-daily"
  project_id      = var.project_id
  region          = var.region
  job_name        = module.job.name
  service_account = data.google_service_account.scheduler.email
  schedule        = var.schedule
  timezone        = var.schedule_timezone
}
