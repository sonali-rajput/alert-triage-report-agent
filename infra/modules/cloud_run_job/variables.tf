variable "name" {
  type = string
}

variable "location" {
  type = string
}

variable "image" {
  description = "Fully qualified image, ideally an immutable tag: .../pipeline:<git-sha>."
  type        = string
}

variable "service_account" {
  description = "Email of the identity the job runs as."
  type        = string
}

variable "env" {
  description = "Plain environment variables."
  type        = map(string)
  default     = {}
}

variable "secret_env" {
  description = "Environment variables sourced from Secret Manager: env var name => secret_id."
  type        = map(string)
  default     = {}
}

variable "timeout" {
  description = "Task timeout. A few hundred issues means a few hundred Sentry detail calls plus a handful of LLM calls."
  type        = string
  default     = "1800s"
}

variable "cpu" {
  type    = string
  default = "1"
}

variable "memory" {
  description = "WeasyPrint holds the whole rendered report in memory."
  type        = string
  default     = "2Gi"
}
