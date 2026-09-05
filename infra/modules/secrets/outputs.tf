output "secret_ids" {
  description = "logical name => secret_id, for wiring into the job's env."
  value       = { for k, v in google_secret_manager_secret.this : k => v.secret_id }
}
