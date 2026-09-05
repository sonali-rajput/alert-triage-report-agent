variable "secrets" {
  description = "Map of logical name => secret_id. A container is created for each; values are added out of band."
  type        = map(string)
}

variable "accessor_member" {
  description = "IAM member granted secretAccessor on these secrets (the job's service account)."
  type        = string
}
