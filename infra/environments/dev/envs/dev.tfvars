project_id = "prodigious-macaque-791341-r8"
region     = "europe-west2"

sentry_org  = "framestore"
sentry_team = "prodtools"

# Read the real list off the org first -- this is an allowlist, and a name that
# does not match is silently never ingested:
#   python -c "from pipeline.sentry_client import SentryApiClient as C; \
#       print(C('https://sentry.io', '<token>', '<org>').list_environments())"
sentry_environments = "production,prod,@prod"

# Prefer an immutable tag over `latest`: it is what makes "which code produced
# this report" answerable after the fact.
image = "europe-west2-docker.pkg.dev/prodigious-macaque-791341-r8/triage/pipeline:1ddd4fc"
