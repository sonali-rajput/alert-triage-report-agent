output "job_name" {
  description = "Run it by hand with: gcloud run jobs execute <job_name> --region <region> --wait"
  value       = module.job.name
}

output "bigquery_dataset" {
  description = "Query the audit trail: SELECT * FROM `<project>.<dataset>.triage` WHERE priority = 'critical'"
  value       = module.bigquery.dataset_id
}

output "reports_bucket" {
  value = module.storage.bucket_name
}

output "secrets_to_populate" {
  description = "Terraform creates these empty. Add a version to each before the first run."
  value       = values(module.secrets.secret_ids)
}
