# Secret *containers* only. Terraform creates them empty and the values are
# added out of band:
#
#   echo -n "$TOKEN" | gcloud secrets versions add triage-sentry-token --data-file=-
#
# A value passed as a Terraform variable ends up in the state file, and the
# state file is shared. The empty container is the thing Terraform should own;
# the value is not.
resource "google_secret_manager_secret" "this" {
  for_each = var.secrets

  secret_id = each.value

  replication {
    auto {}
  }
}

# Scoped to these secrets rather than granted project-wide, so a third secret
# added to this project later is not automatically readable by the job.
resource "google_secret_manager_secret_iam_member" "accessor" {
  for_each = google_secret_manager_secret.this

  secret_id = each.value.id
  role      = "roles/secretmanager.secretAccessor"
  member    = var.accessor_member
}
