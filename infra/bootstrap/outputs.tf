output "job_service_account" {
  description = "Environments look this up by name; it is printed here to make the contract visible."
  value       = google_service_account.job.email
}

output "scheduler_service_account" {
  value = google_service_account.scheduler.email
}

output "image_repository" {
  description = "Push images here, e.g. <this>/pipeline:<git-sha>"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}
