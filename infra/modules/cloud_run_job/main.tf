# A Cloud Run JOB, not a Service. This is a scheduled batch task: there is no
# request to keep alive, no request timeout to fit a few hundred Sentry detail
# calls into, and no HTTP endpoint anyone could reach by accident. The job runs
# to completion, the container stops, the billing stops.
resource "google_cloud_run_v2_job" "this" {
  name     = var.name
  location = var.location

  deletion_protection = false

  template {
    template {
      service_account = var.service_account
      timeout         = var.timeout

      # No task-level retries. Cloud Scheduler retries the whole job, and the
      # pipeline's own idempotency guard makes that safe; a task retry would
      # re-run stages that already wrote to BigQuery without passing through
      # that guard.
      max_retries = 0

      containers {
        image = var.image

        resources {
          limits = {
            cpu    = var.cpu
            memory = var.memory
          }
        }

        dynamic "env" {
          for_each = var.env
          content {
            name  = env.key
            value = env.value
          }
        }

        # Mounted from Secret Manager, so the values never appear in the job
        # definition, in Terraform state, or in `gcloud run jobs describe`.
        dynamic "env" {
          for_each = var.secret_env
          content {
            name = env.key
            value_source {
              secret_key_ref {
                secret  = env.value
                version = "latest"
              }
            }
          }
        }
      }
    }
  }
}
