# Infrastructure

This is AI triage Agent Infrastructure walkthrough.

```
infra/
  bootstrap/         run once per GCP project.Enabled APIs, service accounts, project-level
                     IAM, and the Artifact Registry repo the image lives in.
    main.tf            APIs + Artifact Registry + identities, in one file
    backend/dev.hcl    Terraform state location
    envs/dev.tfvars    your values
  modules/           reusable building blocks, no environment knowledge in them
    bigquery/          dataset + the three tables (alerts, triage, runs)
    storage/           the private reports bucket
    secrets/           Secret Manager containers (values added out of band)
    cloud_run_job/     the pipeline job
    scheduler/         the daily Cloud Scheduler trigger
  environments/
    dev/             the actual dev environment: composes the modules
      backend/dev.hcl  state location (tf state, placeholder project ID)
      envs/dev.tfvars  your values
```

## Why bootstrap is separate

Bootstrap and the environment change at completely different rates and carry
completely different blast radii.

* **Bootstrap is project-scoped and near-permanent.** Enabling an API, creating
  a service account, granting `roles/aiplatform.user`. Applied once, then
  touched perhaps twice a year. Destroying it is disruptive: an API disabled
  project-wide affects everything in the project, not just this pipeline.
* **The environment is disposable by design.**  `terraform destroy` costs you nothing and
  `terraform apply` brings it back in minutes. That is only true if destroying
  it does not also delete the identities and API enablements everything else
  depends on.

Keeping them in one state makes the second property impossible without
endangering the first. It also creates a genuine ordering problem: a service
account cannot be granted a role on an API that is not enabled yet, and
Terraform's dependency graph does not know that.

**The environment does not read bootstrap's state.** It looks the service
accounts up by name with `data "google_service_account"`.

`name_prefix` (default `triage`) is what ties the two together: bootstrap
creates `${name_prefix}-agent-sa`, the environment looks it up by that name.
Change it in one place and you must change it in the other.

## State: GitLab-managed Terraform state

Both stacks keep state in GitLab's HTTP backend, with locking. Nothing is
stored locally. Each stack has its own state name, so a `terraform apply` in
one cannot lock or corrupt the other:

| Stack | State name |
|---|---|
| `infra/bootstrap` | `alert-triage-bootstrap` |
| `infra/environments/dev` | `alert-triage-dev` |


```bash
export TF_HTTP_USERNAME='your-gitlab-username'
export TF_HTTP_PASSWORD='glpat-xxxxxxxxxxxxxxxx'   # PAT with `api` scope

cd infra/environments/dev
terraform init -backend-config=backend/dev.hcl
```

## First run, in order

```bash
# 1. Apply the bootstrap layer first
cd infra/bootstrap
terraform init -backend-config=backend/dev.hcl
terraform apply -var-file=envs/dev.tfvars

# 2. Build and push the image
gcloud builds submit --project <GCP-PROJECT-ID> \
  --tag europe-west2-docker.pkg.dev/<GCP-PROJECT-ID>/triage/pipeline:$(git rev-parse --short HEAD) .

# 3. The environment (set `image` in envs/dev.tfvars to the tag you just pushed)
cd ../environments/dev
terraform init -backend-config=backend/dev.hcl
terraform apply -var-file=envs/dev.tfvars

# 4. Populate the secrets Terraform created empty
echo -n "$SENTRY_TOKEN"     | gcloud secrets versions add triage-sentry-token --data-file=-
echo -n "$CHAT_WEBHOOK_URL" | gcloud secrets versions add triage-chat-webhook --data-file=-

# 5. Run it once to check
gcloud run jobs execute triage-pipeline --region europe-west2 --wait
```

