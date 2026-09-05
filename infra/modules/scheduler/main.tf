# run.invoker on this one job, nothing else. The scheduler's identity should be
# able to start this pipeline and do absolutely nothing else in the project.
resource "google_cloud_run_v2_job_iam_member" "invoker" {
  project  = var.project_id
  location = var.region
  name     = var.job_name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${var.service_account}"
}

resource "google_cloud_scheduler_job" "this" {
  name        = var.name
  description = "Triggers the daily alert triage run."
  schedule    = var.schedule
  time_zone   = var.timezone
  region      = var.region

  # 30 minutes of retries. Beyond that the run has missed the morning it was
  # written for, and the right answer is to look at why rather than keep trying.
  retry_config {
    retry_count          = 3
    min_backoff_duration = "60s"
    max_backoff_duration = "600s"
    max_retry_duration   = "1800s"
  }

  http_target {
    http_method = "POST"
    uri         = "https://${var.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${var.project_id}/jobs/${var.job_name}:run"

    oauth_token {
      service_account_email = var.service_account
    }
  }

  # The invoker binding is not a dependency Terraform can infer, and a schedule
  # that exists before its permission does will fail its first firing.
  depends_on = [google_cloud_run_v2_job_iam_member.invoker]
}
