variable "dataset_id" {
  description = "Dataset holding the vector store, the audit trail and the run history."
  type        = string
}

variable "location" {
  type = string
}

variable "writer_member" {
  description = "IAM member granted dataEditor on this dataset (the job's service account)."
  type        = string
}

variable "deletion_protection" {
  description = "Set true anywhere the data matters. False makes a dev environment genuinely disposable."
  type        = bool
  default     = false
}
