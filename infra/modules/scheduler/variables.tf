variable "name" {
  type = string
}

variable "project_id" {
  type = string
}

variable "region" {
  description = "Must match the job's region: the Cloud Run Admin API endpoint is regional."
  type        = string
}

variable "job_name" {
  description = "Name of the Cloud Run Job to start."
  type        = string
}

variable "service_account" {
  description = "Email of the scheduler's identity. Granted run.invoker on that job alone."
  type        = string
}

variable "schedule" {
  type    = string
  default = "0 8 * * *"
}

variable "timezone" {
  description = "The report says 'the last 24 hours', so this decides what a day means."
  type        = string
  default     = "Europe/London"
}
