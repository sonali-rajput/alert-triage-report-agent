# terraform init -backend-config=backend/dev.hcl
#
# Credentials are NOT in this file. The HTTP backend reads them from the
# environment:
#   export TF_HTTP_USERNAME='your-gitlab-username'
#   export TF_HTTP_PASSWORD='glpat-...'   # PAT with the `api` scope
#

address        = "https://gitlab.com/api/v4/projects/85296736/terraform/state/alert-triage-dev"
lock_address   = "https://gitlab.com/api/v4/projects/85296736/terraform/state/alert-triage-dev/lock"
unlock_address = "https://gitlab.com/api/v4/projects/85296736/terraform/state/alert-triage-dev/lock"
lock_method    = "POST"
unlock_method  = "DELETE"
retry_wait_min = 5
