# terraform init -backend-config=backend/dev.hcl
#
# Credentials are NOT in this file. The HTTP backend reads them from the
# environment:
#   export TF_HTTP_USERNAME='your-gitlab-username'
#   export TF_HTTP_PASSWORD='glpat-...'   # PAT with the `api` scope
#
# Replace GITLAB_PROJECT_ID with your numeric GitLab project ID (Settings >
# General). The state name is what keeps this stack's state separate, so an
# apply in one can never lock or overwrite the other.

address        = "https://gitlab.com/api/v4/projects/GITLAB_PROJECT_ID/terraform/state/alert-triage-bootstrap"
lock_address   = "https://gitlab.com/api/v4/projects/GITLAB_PROJECT_ID/terraform/state/alert-triage-bootstrap/lock"
unlock_address = "https://gitlab.com/api/v4/projects/GITLAB_PROJECT_ID/terraform/state/alert-triage-bootstrap/lock"
lock_method    = "POST"
unlock_method  = "DELETE"
retry_wait_min = 5
