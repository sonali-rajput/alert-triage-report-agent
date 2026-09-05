#!/usr/bin/env bash
# Run before every commit.
#
#   ./scripts/check.sh          lint, types, tests, and the leak guard
#   ./scripts/check.sh --fast   skip the offline end-to-end run
#
# This is deliberately a script you run, not a CI pipeline that runs you. For a
# one-person MVP, a hook that fires on every commit is friction; a command you
# type when you are about to push is enough, and it works the same whether or
# not you have a CI runner today.
#
# The last check is the one that matters most. Everything above it protects the
# code; the leak guard protects the things that cannot be undone -- a token or a
# colleague's email in a commit that reaches a remote is in every clone forever.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=./.venv/bin/python
[ -x "$PY" ] || PY=python3
FAST=${1:-}
FAILED=0

step() { printf '\n\033[1m▸ %s\033[0m\n' "$1"; }
fail() { printf '\033[31m  FAILED: %s\033[0m\n' "$1"; FAILED=1; }

step "ruff"
$PY -m ruff check . || fail "lint"

step "mypy"
$PY -m mypy pipeline shared eval || fail "types"

step "pytest"
$PY -m pytest -q || fail "tests"

if [ "$FAST" != "--fast" ]; then
  step "offline end-to-end (fixture -> agents -> report)"
  TMP=$(mktemp -d)
  if SENTRY_SOURCE=fixture \
     SENTRY_FIXTURE_PATH=fixtures/sentry_org_issues_last_24h_anonymized.json \
     SENTRY_ENVIRONMENTS=production LLM_PROVIDER=mock EMBEDDING_PROVIDER=hash \
     STORE_BACKEND=local GCS_BUCKET= CHAT_WEBHOOK_URL= ARTIFACTS_DIR="$TMP" \
     $PY -m pipeline.main --run-date 2026-01-01 --force >/dev/null 2>&1; then
    find "$TMP" -name 'triage-report-*' -print -quit | grep -q . \
      && echo "  report rendered" || fail "the run produced no report"
  else
    fail "the pipeline did not complete"
  fi
  rm -rf "$TMP" .local_state
fi

# --------------------------------------------------------------------------
# The leak guard
# --------------------------------------------------------------------------
step "staged-for-commit leak check"

STAGED=$(git diff --cached --name-only 2>/dev/null || true)
if [ -z "$STAGED" ]; then
  echo "  nothing staged — run 'git add' first if you meant to check a commit"
else
  # Whole paths that must never be committed, whatever they contain.
  BANNED=$(echo "$STAGED" | grep -E '(^|/)\.env$|^\.sentry-capture/|^\.real-run-state/|^reports-real/|^artifacts/|(^|/)\.local_state|(^|/)backend\.hcl$|\.tfstate|(^|/)[^/]*\.tfvars$' || true)
  if [ -n "$BANNED" ]; then
    fail "these are staged and must not be committed:"
    echo "$BANNED" | sed 's/^/      /'
  fi

  # Secret-shaped strings in staged content. Placeholders (xxx, ..., <token>)
  # are what the .example files are made of, so they are not findings.
  SECRETS=$(git diff --cached -U0 2>/dev/null \
    | grep -E '^\+' \
    | grep -oE 'AIzaSy[A-Za-z0-9_-]{20,}|sntrys_[A-Za-z0-9]{20,}|glpat-[A-Za-z0-9_-]{15,}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----' \
    | grep -viE 'xxx|\.\.\.|example|placeholder|redacted|masked|AbCdEf|ABC123|XY12' || true)
  if [ -n "$SECRETS" ]; then
    fail "staged content contains something secret-shaped:"
    echo "$SECRETS" | cut -c1-14 | sed 's/$/…/' | sort -u | sed 's/^/      /'
  fi

  # A fixture derived from real Sentry data needs the human read (todo.md).
  if echo "$STAGED" | grep -q '^fixtures/sentry_event_latest'; then
    echo "  NOTE: fixtures/sentry_event_latest.sample.json is staged."
    echo "        It came from a live capture. Automated checks cannot judge a show"
    echo "        name, an internal hostname or a person's name in a module path:"
    echo "          $PY scripts/capture_sentry_payloads.py --audit-fixture fixtures/sentry_event_latest.sample.json"
  fi
  [ "$FAILED" -eq 0 ] && echo "  clean"
fi

echo
if [ "$FAILED" -eq 0 ]; then
  printf '\033[32mall checks passed\033[0m\n'
else
  printf '\033[31msomething failed — see above\033[0m\n'
fi
exit "$FAILED"
