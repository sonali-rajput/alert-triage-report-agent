variable "project_id" {
  description = "GCP project to bootstrap."
  type        = string
}

variable "region" {
  description = "Region for the Artifact Registry repo. Environments may use others."
  type        = string
  default     = "europe-west2"
}

variable "name_prefix" {
  description = <<-EOT
    Prefix for the service accounts and the image repo. The environment stacks
    look the service accounts up by the email this produces, so changing it
    here means changing `name_prefix` in every environment too.
  EOT
  type        = string
  default     = "triage"
}
