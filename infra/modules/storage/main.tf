# Private bucket for the PDF reports. Everyone reads them through a V4 signed
# URL in the Chat card; nothing is ever readable by URL alone.
resource "google_storage_bucket" "reports" {
  name                        = var.bucket_name
  location                    = var.location
  force_destroy               = var.force_destroy
  uniform_bucket_level_access = true

  # Belt and braces, and the belt is the important one: this makes it
  # impossible to grant allUsers access to this bucket even by mistake, which
  # uniform access alone does not.
  public_access_prevention = "enforced"

  lifecycle_rule {
    condition {
      age = var.retention_days
    }
    action {
      type = "Delete"
    }
  }
}

# Bucket-scoped. objectAdmin rather than objectCreator because re-running a
# date overwrites that date's report.
resource "google_storage_bucket_iam_member" "writer" {
  bucket = google_storage_bucket.reports.name
  role   = "roles/storage.objectAdmin"
  member = var.writer_member
}
