# Alert Triage Agent

Daily AI alert-triage pipeline running on Google Cloud Platform.

Every morning one **Cloud Run Job**, started by Cloud Scheduler, pulls the last
24h of unresolved Sentry issues with their stack traces, masks PII and secrets,
drops known noise, and embeds each alert into BigQuery. Two Gemini agents then
work on the result: a **top-issues agent** decides what is a repeat and ranks
the day's ten most important issues, and a **triage agent** assigns each one a
priority and explains why. The reasoning goes to BigQuery, the PDF report to a
private GCS bucket, and a card with a signed link to Google Chat.

```
Cloud Scheduler --> start --> Cloud Run Job
   08:00 daily               ├─ fetch Sentry issues + stack traces
                             ├─ mask PII/secrets, drop known noise
                             ├─ embed → BigQuery, VECTOR_SEARCH for similar history
                             ├─ top-issues agent: dedup, rank, pick 10
                             ├─ triage agent: priority + summary
                             ├─ audit trail → BigQuery (every alert, selected or not)
                             ├─ PDF → private GCS bucket → signed URL
                             └─ Google Chat card
```

## Quick start


```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest

cp .env.local.example .env
python -m pipeline.main --run-date 2026-08-11
```

```bash
docker compose build triage         # production image FIRST
docker compose build test           # local image is built FROM it
docker compose run --rm triage      # one offline run -> ./artifacts
docker compose run --rm test        # the suite, against the local image
```

Build order matters: `Dockerfile.local` starts `FROM alert-triage:runtime`, so
a bare `docker compose build` tries to pull that image from Docker Hub and
fails.

The run writes the report to `artifacts/` and its state to `.local_state/`.
Re-running a date is a no-op (idempotency guard); `--force` replays it.

Before committing:

```bash
./scripts/check.sh                  # lint, types, tests, leak guard
```

## Deploying

Terraform, in two stacks: **bootstrap** (APIs, service accounts, project IAM,
Artifact Registry — once per project) and **environments/dev** (BigQuery, the
reports bucket, secrets, the job, the schedule). State lives in GitLab's HTTP
backend with locking. The image build and `terraform apply` are run manually.

```bash
cd infra/bootstrap
terraform init -backend-config=backend/dev.hcl
terraform apply -var-file=envs/dev.tfvars
```

## Repository layout

```
shared/     Pydantic schemas, config loader, logging
pipeline/   The job: sentry_client, masking, prefilter, embeddings, bq,
            agents/, pdf_report, chat_notify, storage, orchestrator, main
config/     priority_matrix.yaml (prose rules), noise_filters.yaml, masking_patterns.yaml
infra/      Terraform: bootstrap/, modules/, environments/dev/
deploy/     Dockerfile (production) and Dockerfile.local (production + pytest)
fixtures/   sample raw Sentry payloads for offline runs
eval/       label-free agent checks (invariants, groundedness, stability)
scripts/    triage_fixture.py, check.sh
tests/      unit tests — no GCP, no network
```
