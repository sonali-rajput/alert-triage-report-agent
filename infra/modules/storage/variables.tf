variable "bucket_name" {
  description = "Globally unique bucket name for the PDF reports."
  type        = string
}

variable "location" {
  type = string
}

variable "writer_member" {
  description = "IAM member granted objectAdmin on this bucket (the job's service account)."
  type        = string
}

variable "retention_days" {
  description = "How long reports are kept. Signed URLs last 7 days; the object outliving the link is deliberate."
  type        = number
  default     = 90
}

variable "force_destroy" {
  description = "True lets `terraform destroy` delete a bucket that still has reports in it. Dev only."
  type        = bool
  default     = false
}
