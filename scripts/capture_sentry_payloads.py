#!/usr/bin/env python
"""Capture real Sentry payloads and test this pipeline's assumptions against them.

Several design decisions rest on claims about what Sentry actually returns, and
some of those claims have never been checked against a real response:

  Q1  Does `GET /issues/{id}/events/latest/` carry tag VALUES, including
      `environment`?  (The ISSUE payload does not -- confirmed twice -- which is
      why the client queries per environment and stamps the value on. If the
      EVENT carries it, SENTRY_ENVIRONMENTS could stop being the only source.)
  Q2  Does the latest event's environment ever DIFFER from the environment the
      issue was queried under?  (An issue firing in prod and dev would then be
      labelled by whichever fired most recently -- a downgrade.)
  Q3  How many issues does an unfiltered fetch return versus the allowlist?
      (This is the cost of dropping SENTRY_ENVIRONMENTS: every extra issue is
      one extra detail call before you can discard it.)
  Q4  Do the configured environment names exist in the org at all?
      (An allowlist miss ingests NOTHING, silently. See FINDINGS session 2 B2.)
  Q5  Does `event_to_body_text` extract real frames and breadcrumbs?
      (The detail path has only ever seen hand-built events in tests.)
  Q6  Does the masker fire on real event text, where credentials actually hide?
  Q9  Which fetch STRATEGY actually reaches every issue?  Compare a literal
      allowlist, an allowlist expanded to every production-ish name the org
      reports, and no environment filter at all. The gap between them is
      issues we are silently not triaging today.
  Q10 If the environment came from the EVENT's tags instead of the query,
      how many issues would be labelled differently -- and how many would be
      dropped as dev/staging noise despite also firing in production?
  Q7  How much of the org is NOT ours?  The pipeline sends no `project` filter
      today, so it triages every project the token can see and relies on a
      two-entry denylist. If the org holds other teams' projects, we are
      ranking their errors against ours -- and paying a detail call for each.

Usage
-----
    export SENTRY_TOKEN='sntrys_...'          # read-only auth token
    export SENTRY_ORG='your-org-slug'
    export SENTRY_TEAM='prodtools'            # optional: scope to one team

    # capture + probe (read-only against Sentry; writes to .sentry-capture/)
    python scripts/capture_sentry_payloads.py --team prodtools

    # re-run the probe later with no network
    python scripts/capture_sentry_payloads.py --check-only .sentry-capture

    # produce anonymized copies you can commit as fixtures
    python scripts/capture_sentry_payloads.py --anonymize

SAFETY
------
* Every call is a GET. This script never writes to Sentry.
* Raw output goes to `.sentry-capture/`, which is gitignored. **Raw payloads
  carry colleague names, emails and gravatar hashes (which are hashes OF
  emails) -- never commit them.**
* `--anonymize` is BEST EFFORT. It replaces known PII-bearing keys, rewrites
  identifiers, and runs the pipeline's own masker over every string -- then
  prints what it could not classify. Read that report before committing
  anything.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.event_text import event_to_body_text  # noqa: E402
from pipeline.masking import Masker  # noqa: E402
from pipeline.sentry_client import (  # noqa: E402
    MAX_PAGES,
    PAGE_SIZE,
    PII_BEARING_FIELDS,
    SentryApiClient,
    _next_cursor,
)
from shared.environments import canonical_environment  # noqa: E402

DEFAULT_OUT = Path(".sentry-capture")
DEFAULT_EVENTS = 8


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------


# Which scope a 403 is most likely complaining about, by endpoint. Sentry does
# not say in the response body, and "403" on its own sends people to re-read
# their token settings without knowing what to look for.
_SCOPE_HINT = {
    "environments": "org:read (or project:read)",
    "teams": "org:read, and project:read for the team's project list",
    "projects": "project:read",
    "issues": "event:read",
    "events": "event:read",
}


def _try(label: str, fn, default):
    """Run one capture step; on failure record why and carry on.

    Every step here is independent evidence, and the events are the most
    valuable of them -- so a token that cannot list teams must not cost us the
    event payloads. Before this, a 403 anywhere aborted the whole capture,
    and the team step ran BEFORE the events.
    """
    try:
        return fn()
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        hint = next((h for key, h in _SCOPE_HINT.items() if key in str(exc.request.url)), "")
        print(f"  ! {label}: HTTP {code}"
              + (f" — this endpoint wants {hint}" if code in (401, 403) and hint else ""))
        print("    skipping it; the rest of the capture continues")
        return default
    except Exception as exc:
        print(f"  ! {label}: {type(exc).__name__}: {exc} — skipping")
        return default


def _paginate(client: SentryApiClient, path: str, params: dict, label: str) -> list[dict]:
    """Follow the Link header to the end, not just the first page.

    Sentry pages at 100 and signals exhaustion with results="false" rather than
    by omitting rel="next", so following it blindly loops forever. Capturing
    one page would have made every count in this report a floor presented as a
    total -- which is exactly the kind of quiet wrongness this script exists to
    stop.
    """
    collected: list[dict] = []
    cursor: str | None = None
    for page in range(MAX_PAGES):
        query = {**params, "limit": PAGE_SIZE}
        if cursor:
            query["cursor"] = cursor
        response = client._request(path, query)
        batch = response.json()
        if not batch:
            break
        collected.extend(batch)
        cursor = _next_cursor(response.headers.get("Link", ""))
        if not cursor:
            break
    else:
        print(f"  ! {label}: hit the {MAX_PAGES}-page cap; this is a floor, not a total")
    print(f"  {label}: {len(collected)} issues")
    return collected


def _team_projects(client: SentryApiClient, out: Path, team: str) -> list[dict]:
    """The projects a team owns.

    Ownership, not assignment. Sentry's `assigned:#team` search token filters
    issues someone has explicitly assigned to a team, which on most orgs is
    almost nothing. "The team's issues" in practice means "issues in the
    projects the team owns", and that is a `project` filter on the query.
    """
    print(f"\n→ GET /organizations/{client._org}/teams/")
    teams = client._get(f"/organizations/{client._org}/teams/")
    teams = teams if isinstance(teams, list) else []
    (out / "teams.json").write_text(
        json.dumps([{"slug": t.get("slug"), "name": t.get("name"),
                     "projects": [p.get("slug") for p in (t.get("projects") or [])]}
                    for t in teams], indent=1), encoding="utf-8")
    print(f"  {len(teams)} team(s): {[t.get('slug') for t in teams]}")

    wanted = team.lstrip("#").lower()
    match = next((t for t in teams if str(t.get("slug", "")).lower() == wanted), None)
    if match is None:
        print(f"  ** no team with slug '{wanted}' — check the list above **")
        return []

    projects = match.get("projects") or []
    if not projects:  # some deployments omit projects from the teams listing
        print(f"→ GET /teams/{client._org}/{wanted}/projects/")
        fetched = _try("team projects",
                       lambda: client._get(f"/teams/{client._org}/{wanted}/projects/"), [])
        projects = fetched if isinstance(fetched, list) else []

    (out / "team_projects.json").write_text(
        json.dumps([{"id": p.get("id"), "slug": p.get("slug")} for p in projects], indent=1),
        encoding="utf-8")
    print(f"  team '{wanted}' owns {len(projects)} project(s): {[p.get('slug') for p in projects]}")
    return projects


def capture(client: SentryApiClient, out: Path, event_count: int, window: str,
            team: str = "") -> None:
    out.mkdir(parents=True, exist_ok=True)

    print(f"→ GET /organizations/{client._org}/environments/")
    environments = _try("environments", client.list_environments, [])
    (out / "environments.json").write_text(json.dumps(environments, indent=1), encoding="utf-8")
    print(f"  {len(environments)} environment(s): {environments}")

    issues_path = f"/organizations/{client._org}/issues/"
    base = {"statsPeriod": window, "query": "is:unresolved", "sort": "freq"}

    # Resolve the team FIRST, so every query below is scoped to it. Measuring
    # the proposed configuration means measuring it, not measuring the old one
    # and extrapolating.
    team_ids: list[str] = []
    if team:
        projects = _try(f"team '{team}'", lambda: _team_projects(client, out, team), [])
        team_ids = [str(p["id"]) for p in projects if isinstance(p, dict) and p.get("id")]
        if team_ids:
            base["project"] = team_ids
            print(f"  scoping every query below to {len(team_ids)} project(s)")

    # THE BASELINE: no project filter, no environment filter, everything the
    # token can see. This is the same query the Issues feed runs behind
    # "My Projects · All Environments · 24h · Unresolved".
    print(f"\n→ GET issues, statsPeriod={window}, no project or environment filter")
    unfiltered = _try("issues (token default)",
                      lambda: _paginate(client, issues_path, base, "token default"), [])
    (out / "issues_all_environments.json").write_text(json.dumps(unfiltered, indent=1), encoding="utf-8")

    # Explicitly ALL projects. The API's default project scope depends on the
    # token: a user token tends to mean "projects of teams I am in" (what the
    # dashboard shows you), an org token can mean everything. project=-1 asks
    # for everything unambiguously, so comparing the two says which one you have.
    (out / "issues_team_scoped.json").write_text(json.dumps(unfiltered, indent=1), encoding="utf-8")

    # project=-1 means the WHOLE ORG, including teams whose issues we have no
    # business holding. It exists only to answer Q8 ("is the token narrower
    # than the org?"), so it runs only when there is no team to scope to --
    # and a stale copy from an earlier unscoped run is deleted, because a
    # capture directory should contain what you asked for and nothing else.
    stale = out / "issues_all_projects.json"
    if team_ids:
        print("→ skipping project=-1: team scoping is on, so the whole org is out of scope")
        if stale.exists():
            stale.unlink()
            print("  (removed a stale issues_all_projects.json from an earlier unscoped run)")
    else:
        print("→ GET issues, project=-1 (the whole org, for comparison)")
        all_projects = _try("issues (project=-1)",
                            lambda: _paginate(client, issues_path, {**base, "project": "-1"},
                                              "all projects"), [])
        stale.write_text(json.dumps(all_projects, indent=1), encoding="utf-8")

    # Per environment: what an allowlist would ingest, and which issues overlap
    # more than one environment (Q2/Q3).
    print("\n→ GET issues, one query per environment")
    per_env: dict[str, list[str]] = {}
    for environment in environments:
        issues = _try(f"issues (environment={environment})",
                      lambda env=environment: _paginate(
                          client, issues_path, {**base, "environment": env}, env), [])
        per_env[environment] = [str(i["id"]) for i in issues]
    (out / "issue_ids_per_environment.json").write_text(json.dumps(per_env, indent=1), encoding="utf-8")

    # The unknown: real event payloads (Q1, Q5, Q6).
    if not unfiltered:
        print("\n! no issues captured, so there is nothing to pull events for.")
        print("  Check the token has event:read and that the window is not empty.")

    wanted = unfiltered if event_count <= 0 else unfiltered[:event_count]
    print(f"\n→ GET events/latest/ for {len(wanted)} issue(s)")
    events: dict[str, Any] = {}
    for issue in wanted:
        issue_id = str(issue["id"])
        try:
            events[issue_id] = client._fetch_latest_event(issue_id)
            print(f"  {issue_id}  ok")
        except Exception as exc:
            events[issue_id] = None
            print(f"  {issue_id}  no event ({type(exc).__name__}) — routine, retention outlives events")
    (out / "events_latest.json").write_text(json.dumps(events, indent=1), encoding="utf-8")

    print(f"\nRaw capture written to {out}/ (gitignored — do not commit)")


# --------------------------------------------------------------------------
# Probe: what the real data says about each assumption
# --------------------------------------------------------------------------


def _tag_values(payload: dict) -> dict[str, str]:
    """Tags as key -> value, when the payload carries values at all."""
    tags = payload.get("tags")
    if not isinstance(tags, list):
        return {}
    return {
        str(t.get("key")): str(t.get("value"))
        for t in tags
        if isinstance(t, dict) and t.get("value") is not None
    }


def probe(out: Path, configured: list[str]) -> None:
    def load(name: str, default):
        path = out / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    environments = load("environments.json", [])
    unfiltered = load("issues_all_environments.json", [])
    per_env = load("issue_ids_per_environment.json", {})
    events = load("events_latest.json", {})
    real_events = {k: v for k, v in events.items() if isinstance(v, dict)}

    print("\n" + "=" * 72)
    print("WHAT THE REAL DATA SAYS")
    print("=" * 72)

    # -- Q1 -----------------------------------------------------------------
    print("\nQ1. Does events/latest/ carry tag VALUES, including environment?")
    if not real_events:
        print("    UNKNOWN — no event payload was captured (all 404s, or none requested).")
    else:
        with_values = {k: _tag_values(v) for k, v in real_events.items()}
        any_values = sum(1 for t in with_values.values() if t)
        with_env = {k: t.get("environment") for k, t in with_values.items() if t.get("environment")}
        print(f"    tag values present : {any_values}/{len(real_events)} events")
        print(f"    environment tag    : {len(with_env)}/{len(real_events)} events")
        if with_env:
            print(f"    values seen        : {sorted(set(with_env.values()))}")
        print("    also check a top-level 'environment' key:")
        top_level = {k: v.get("environment") for k, v in real_events.items() if v.get("environment")}
        print(f"    top-level present  : {len(top_level)}/{len(real_events)} — {sorted(set(top_level.values()))}")
        verdict = "YES" if len(with_env) == len(real_events) else "PARTIALLY" if with_env else "NO"
        print(f"    => {verdict}. Anything less than every event means the event tag cannot be")
        print("       the only source of environment.")

    # -- Q2 -----------------------------------------------------------------
    print("\nQ2. Does an issue appear under more than one environment?")
    appearances = Counter()
    for env, ids in per_env.items():
        for issue_id in ids:
            appearances[issue_id] += 1
    multi = {i: c for i, c in appearances.items() if c > 1}
    print(f"    issues in >1 environment: {len(multi)} of {len(appearances)}")
    if multi:
        print("    => The per-environment query is a STRONGER claim than the latest event's tag:")
        print("       it says 'this occurs in production', not 'the last one happened to be'.")
        for issue_id, count in list(multi.items())[:5]:
            envs = [e for e, ids in per_env.items() if issue_id in ids]
            event_env = _tag_values(real_events.get(issue_id) or {}).get("environment", "n/a")
            print(f"       issue {issue_id}: queried in {envs}, latest event says '{event_env}'")

    # -- Q3 -----------------------------------------------------------------
    print("\nQ3. What does dropping the allowlist cost?")
    allowlisted = {i for env, ids in per_env.items()
                   if canonical_environment(env) in {canonical_environment(c) for c in configured}
                   for i in ids}
    print(f"    unfiltered fetch      : {len(unfiltered)} issues (first page)")
    print(f"    configured allowlist  : {len(allowlisted)} issues  ({configured})")
    extra = len(unfiltered) - len(allowlisted)
    print(f"    => {extra} extra issues, i.e. {extra} extra events/latest/ calls per run,")
    print("       paid BEFORE you can tell they are not in scope.")

    # -- Q4 -----------------------------------------------------------------
    print("\nQ4. Do the configured names exist in the org?")
    known = {canonical_environment(e): e for e in environments}
    for name in configured:
        canon = canonical_environment(name)
        if name in environments:
            print(f"    '{name}' — exact match")
        elif canon in known:
            print(f"    '{name}' — NOT an exact match; org spells it '{known[canon]}' "
                  f"(canonicalises the same, so the prefilter copes, but the FETCH does not)")
        else:
            print(f"    '{name}' — ** NOT IN THE ORG. This fetches NOTHING, silently. **")
    unconfigured = [e for e in environments
                    if canonical_environment(e) not in {canonical_environment(c) for c in configured}]
    if unconfigured:
        print(f"    never ingested at all: {unconfigured}")

    # -- Q5 -----------------------------------------------------------------
    print("\nQ5. Does event_to_body_text extract real frames and breadcrumbs?")
    if not real_events:
        print("    UNKNOWN — no event payload captured.")
    else:
        extracted = {k: event_to_body_text(v) for k, v in real_events.items()}
        with_text = {k: t for k, t in extracted.items() if t}
        frames = sum(1 for t in with_text.values() if "Exception:" in t)
        crumbs = sum(1 for t in with_text.values() if "Breadcrumbs" in t)
        print(f"    produced text : {len(with_text)}/{len(real_events)} events")
        print(f"    stack traces  : {frames}")
        print(f"    breadcrumbs   : {crumbs}")
        if with_text:
            sample = next(iter(with_text.values()))
            print("    sample (first 5 lines):")
            for line in sample.splitlines()[:5]:
                print(f"      {line}")
        if len(with_text) < len(real_events):
            print("    => Events yielding NOTHING are worth opening by hand: either that platform")
            print("       uses a shape event_text.py does not read, or the event genuinely has none.")

    # -- Q6 -----------------------------------------------------------------
    print("\nQ6. Does the masker fire on real event text?")
    masker = Masker()
    hits: Counter = Counter()
    for text in (event_to_body_text(v) for v in real_events.values()):
        if text:
            _masked, fired = masker.mask_text(text)
            hits.update(fired)
    if hits:
        print(f"    rules fired: {dict(hits)}")
        print("    => Each one is an application logging something it should not.")
    else:
        print("    no masking rules fired on this sample.")
        print("    => Not proof of clean logs. Read one event body by hand and look for anything")
        print("       token-shaped the rules missed; that is how masking_patterns.yaml grows.")

    # -- Q9 -----------------------------------------------------------------
    print("\nQ9. Which fetch strategy actually reaches every issue?")
    all_env_ids = {str(i.get("id")) for i in unfiltered if isinstance(i, dict)}
    literal = {i for name in configured for i in per_env.get(name, [])}
    prod_names = [e for e in environments if canonical_environment(e) == "production"]
    expanded = {i for name in prod_names for i in per_env.get(name, [])}

    print(f"    (a) literal allowlist {configured}")
    print(f"        reaches {len(literal)} issue(s)")
    print(f"    (b) every production-ish name the org reports ({len(prod_names)}):")
    print(f"        {prod_names}")
    print(f"        reaches {len(expanded)} issue(s)")
    print("    (c) no environment filter at all")
    print(f"        reaches {len(all_env_ids)} issue(s)")
    missed_by_literal = expanded - literal
    if missed_by_literal:
        print(f"    => (a) MISSES {len(missed_by_literal)} production issue(s) that (b) reaches.")
        print("       Those are production errors nobody is being told about.")
        for issue_id in sorted(missed_by_literal)[:5]:
            envs = [e for e in prod_names if issue_id in per_env.get(e, [])]
            print(f"       issue {issue_id}: only reachable via {envs}")
    elif prod_names != configured:
        print("    => (a) and (b) reach the same issues today, but (b) is the safer")
        print(f"       spelling: {sorted(set(prod_names) - set(configured))} exist and would")
        print("       start mattering the moment something errors there.")
    non_prod = all_env_ids - expanded
    print(f"    (c) additionally reaches {len(non_prod)} issue(s) in non-production environments,")
    print("        which the prefilter would then have to drop.")

    # -- Q10 ----------------------------------------------------------------
    print("\nQ10. Could the EVENT's tag replace the query as the source of environment?")
    if not real_events:
        print("    UNKNOWN — no event payload captured.")
    else:
        queried: dict[str, list[str]] = {}
        for env, ids in per_env.items():
            for issue_id in ids:
                queried.setdefault(issue_id, []).append(env)

        agree = disagree = unknown = 0
        dangerous: list[str] = []
        for issue_id, event in real_events.items():
            event_env = _tag_values(event).get("environment", "")
            envs = queried.get(issue_id, [])
            if not event_env or not envs:
                unknown += 1
                continue
            canon_event = canonical_environment(event_env)
            canon_queried = {canonical_environment(e) for e in envs}
            if canon_event in canon_queried and len(canon_queried) == 1:
                agree += 1
            else:
                disagree += 1
                # The dangerous direction: fires in production, but the latest
                # event came from somewhere the prefilter would drop.
                if "production" in canon_queried and canon_event != "production":
                    dangerous.append(f"{issue_id}: queried {sorted(canon_queried)}, event '{event_env}'")

        print(f"    agree with the queried environment : {agree}")
        print(f"    disagree                           : {disagree}")
        print(f"    no tag or not in a per-env result  : {unknown}")
        if dangerous:
            print(f"    ** {len(dangerous)} issue(s) fire in PRODUCTION but their latest event does not: **")
            for line in dangerous[:5]:
                print(f"       {line}")
            print("       => Trusting the event tag alone would label these non-production, and")
            print("          drop_environments would then SUPPRESS a production error.")
        else:
            print("    => No case where an issue fires in production but its latest event says")
            print("       otherwise. The event tag looks safe as the primary source on this")
            print("       sample -- re-check it on a bigger one before relying on it.")

    # -- Q8 -----------------------------------------------------------------
    print("\nQ8. Does the API see what your dashboard sees?")
    all_projects = load("issues_all_projects.json", [])
    if not all_projects and load("team_projects.json", []):
        print("    skipped — the capture was scoped to a team, so the whole org was never")
        print("    fetched. Re-run without --team if you want this comparison.")
        all_projects = []
    default_ids = {str(i.get("id")) for i in unfiltered if isinstance(i, dict)}
    all_ids = {str(i.get("id")) for i in all_projects if isinstance(i, dict)}
    print(f"    token default (no project param) : {len(default_ids)} issues")
    print(f"    project=-1 (all projects)        : {len(all_ids)} issues")
    if not all_ids:
        print("    (project=-1 returned nothing — was it captured?)")
    elif default_ids == all_ids:
        print("    => IDENTICAL. The token sees the whole org, so the dashboard's")
        print("       'My Projects' filter is NOT being applied to our fetch. Any team")
        print("       scoping has to be explicit, in the query.")
    else:
        only_all = all_ids - default_ids
        print(f"    => DIFFERENT: {len(only_all)} issue(s) only visible with project=-1.")
        print("       The token's default scope is narrower than the org — most likely")
        print("       'projects of teams this user belongs to', i.e. the same thing your")
        print("       dashboard shows. That is a property of the TOKEN, so it silently")
        print("       changes if the token is ever swapped for an org-level one.")

    # -- Q7 -----------------------------------------------------------------
    print("\nQ7. How much of the org is not ours?")
    team_projects = load("team_projects.json", [])
    scoped = load("issues_team_scoped.json", [])
    by_project = Counter(
        (i.get("project") or {}).get("slug", "?") for i in unfiltered if isinstance(i, dict)
    )
    print(f"    projects in an unfiltered fetch : {len(by_project)}")
    print(f"    {dict(by_project.most_common(12))}")
    if team_projects:
        owned = {p.get("slug") for p in team_projects}
        ours = sum(c for slug, c in by_project.items() if slug in owned)
        theirs = sum(c for slug, c in by_project.items() if slug not in owned)
        print(f"    team owns                       : {len(owned)} project(s)")
        print(f"    team-scoped fetch returned      : {len(scoped)} issues")
        print(f"    of the unfiltered page          : {ours} ours, {theirs} someone else's")
        strangers = sorted(slug for slug in by_project if slug not in owned)
        if strangers:
            print(f"    NOT ours: {strangers}")
            print("    => Today the pipeline sends no `project` filter, so these are being")
            print("       ranked against ours and each costs one events/latest/ call. A")
            print("       `project` allowlist at query time is cheaper and safer than the")
            print("       two-entry drop_projects denylist in config/noise_filters.yaml.")
        else:
            print("    => Every issue on this page is already the team's. Project scoping")
            print("       would change nothing today, but it would keep it that way when")
            print("       another team joins the org.")
    else:
        print("    (no team captured — pass --team <slug> to measure this)")

    print("\n" + "=" * 72)


# --------------------------------------------------------------------------
# Anonymize
# --------------------------------------------------------------------------

# Keys whose values are replaced wholesale. The first group is the projection's
# own PII list; the rest is what event payloads add.
_DROP_KEYS = set(PII_BEARING_FIELDS) | {
    "user", "seenBy", "assignedTo", "owners", "sentry_app", "avatar", "avatarUrl",
    "emails", "email", "username", "ip_address", "userReport",
}
_ID_KEYS = {"id", "eventID", "groupID", "projectID", "machine_id", "epic_account_id"}
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_URL_HOST = re.compile(r"https?://([\w.-]+)")


def anonymize(value: Any, org: str, counters: Counter, masker: Masker) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in _DROP_KEYS:
                counters["dropped_keys"] += 1
                out[key] = None if not isinstance(item, list) else []
                continue
            if key in _ID_KEYS and isinstance(item, (str, int)):
                counters["rewritten_ids"] += 1
                out[key] = f"anon-{abs(hash(str(item))) % 10**8}"
                continue
            out[key] = anonymize(item, org, counters, masker)
        return out
    if isinstance(value, list):
        return [anonymize(v, org, counters, masker) for v in value]
    if isinstance(value, str):
        text = _EMAIL.sub("someone@example.invalid", value)
        text = text.replace(org, "example-org")
        text = _URL_HOST.sub(lambda m: "https://example.invalid", text)
        masked, fired = masker.mask_text(text)
        counters.update(f"masked:{f}" for f in fired)
        return masked
    return value


def anonymize_capture(out: Path, org: str) -> None:
    masker = Masker()
    counters: Counter = Counter()
    target = out / "anonymized"
    target.mkdir(parents=True, exist_ok=True)

    for source in sorted(out.glob("*.json")):
        data = json.loads(source.read_text(encoding="utf-8"))
        cleaned = anonymize(data, org, counters, masker)
        (target / source.name).write_text(json.dumps(cleaned, indent=1), encoding="utf-8")
        print(f"  {source.name} -> anonymized/{source.name}")

    print(f"\n  keys dropped: {counters['dropped_keys']}, ids rewritten: {counters['rewritten_ids']}")
    masked = {k: v for k, v in counters.items() if k.startswith("masked:")}
    print(f"  masker fired: {masked or 'nothing'}")
    print("\n  ⚠ BEST EFFORT. Read these files before committing any of them. Grep for your")
    print("    colleagues' names, internal hostnames and project code names — this script")
    print("    only knows about patterns it was told about.")


# --------------------------------------------------------------------------
# Build a committable event fixture
#
# An ALLOWLIST projection, not a denylist -- the same discipline as
# `issue_to_alert`, and for the same reason: a denylist protects you from the
# PII you thought of. Event payloads carry far more than issue payloads do
# (request headers and cookies, frame-local variables, user contexts, device
# ids, internal hostnames), so the fixture keeps only the fields this pipeline
# actually reads, plus enough shape to stay realistic.
# --------------------------------------------------------------------------

# Tags worth keeping. `user`, `server_name`, `url`, `machine_id` and friends are
# dropped: an internal hostname is as identifying as a name.
_SAFE_TAGS = {
    "environment", "level", "handled", "mechanism", "release", "transaction",
    "os.name", "browser.name", "runtime.name", "logger", "platform", "dist",
}
# `request` entries hold headers and cookies; `message` can hold anything.
_SAFE_ENTRY_TYPES = {"exception", "breadcrumbs"}
# Frame `vars` and `context` are the local variables and surrounding source --
# exactly where a token ends up. Never kept.
_FRAME_KEYS = {"filename", "module", "function", "lineNo", "colNo", "inApp", "package"}
# Breadcrumb `data` holds URLs, ids and payloads. `message` is masked instead.
_CRUMB_KEYS = {"category", "level", "type", "timestamp"}

_RESIDUAL_PII = {
    "email address": _EMAIL,
    "gravatar hash (a hash OF an email)": re.compile(r"gravatar\.com/avatar/\w+"),
    "IPv4 address": re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    "http host": re.compile(r"https?://[\w.-]+"),
    # A username in a filesystem path. This is how the masker's
    # home-directory-path rule was found in the first place.
    "home directory (a username)": re.compile(r"(?i)(?:/home/|/Users/|[A-Z]:\\Users\\)[^/\\\s\"]+"),
}


def _project_event(event: dict, masker: Masker) -> dict:
    """Keep only what the pipeline reads, masked."""
    def text(value: object) -> str:
        return masker.mask_text(str(value))[0] if value else ""

    entries = []
    for entry in event.get("entries") or []:
        if not isinstance(entry, dict) or entry.get("type") not in _SAFE_ENTRY_TYPES:
            continue
        data = entry.get("data") or {}
        if entry["type"] == "exception":
            values = []
            for value in data.get("values") or []:
                if not isinstance(value, dict):
                    continue
                frames = [
                    {k: v for k, v in frame.items() if k in _FRAME_KEYS}
                    for frame in ((value.get("stacktrace") or {}).get("frames") or [])
                    if isinstance(frame, dict)
                ]
                for frame in frames:
                    if "filename" in frame:
                        frame["filename"] = text(frame["filename"])
                values.append({
                    "type": text(value.get("type")),
                    "value": text(value.get("value")),
                    "stacktrace": {"frames": frames},
                })
            entries.append({"type": "exception", "data": {"values": values}})
        else:
            crumbs = []
            for crumb in data.get("values") or []:
                if not isinstance(crumb, dict):
                    continue
                kept = {k: v for k, v in crumb.items() if k in _CRUMB_KEYS}
                kept["message"] = text(crumb.get("message"))
                crumbs.append(kept)
            entries.append({"type": "breadcrumbs", "data": {"values": crumbs}})

    tags = [
        {"key": t["key"], "value": text(t.get("value"))}
        for t in (event.get("tags") or [])
        if isinstance(t, dict) and t.get("key") in _SAFE_TAGS
    ]
    return {
        "id": "anon-event",
        "eventID": "anon" * 8,
        "groupID": "anon-group",
        "title": text(event.get("title")),
        "platform": event.get("platform"),
        "dateCreated": event.get("dateCreated"),
        "tags": tags,
        "entries": entries,
    }


def make_event_fixture(out: Path, destination: Path) -> int:
    source = out / "events_latest.json"
    if not source.exists():
        print(f"no {source} — run a capture first", file=sys.stderr)
        return 2

    events = {k: v for k, v in json.loads(source.read_text(encoding="utf-8")).items()
              if isinstance(v, dict)}
    if not events:
        print("the capture holds no event payloads (all 404s?)", file=sys.stderr)
        return 2

    # The richest event makes the most useful fixture: it exercises frame
    # truncation and the breadcrumb tail, which a two-frame event does not.
    def richness(event: dict) -> tuple[int, int]:
        text = event_to_body_text(event)
        return text.count("\n"), len(text)

    issue_id, best = max(events.items(), key=lambda kv: richness(kv[1]))
    print(f"picked the richest of {len(events)} event(s) (issue {issue_id})")

    fixture = _project_event(best, Masker())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(fixture, indent=1), encoding="utf-8")

    print(f"\nwrote {destination}")
    print(f"  {len(json.dumps(fixture))} bytes, "
          f"{sum(len(e['data'].get('values', [])) for e in fixture['entries'])} entry value(s)")

    print("\nWhat the pipeline extracts from it:")
    print("-" * 60)
    body = event_to_body_text(fixture)
    print(body or "(nothing — the projection dropped too much, or this event had no trace)")
    print("-" * 60)

    print("\nResidual-PII scan of the fixture:")
    blob = json.dumps(fixture)
    found = False
    for label, pattern in _RESIDUAL_PII.items():
        # A masked value is the rule working, not a leak. Reporting it would
        # train the reader to ignore this section, which is the last thing you
        # want from a PII scan.
        hits = sorted({h for h in pattern.findall(blob) if "[MASKED_" not in h})
        if hits:
            found = True
            print(f"  ** {label}: {hits[:5]}")
    if not found:
        print("  nothing matched the known patterns.")
    print("\n  ⚠ READ THE FILE ANYWAY. This scan knows about emails, gravatars, IPs and")
    print("    hosts. It does not know your colleagues' names, your internal hostnames,")
    print("    or your unreleased project code names. Those are yours to spot.")
    return 0


# --------------------------------------------------------------------------
# Pre-commit audit
# --------------------------------------------------------------------------


def audit_fixture(path: Path) -> int:
    """Everything a machine can check, plus the exact fields it cannot.

    Run this before committing a captured fixture. A commit to a remote is
    permanent -- rewriting history to remove a colleague's name is a bad day,
    and on a shared remote it is a bad day for everyone who has already pulled.

    The automated half checks the patterns the projection knows about. The
    manual half prints the fields where the risk is a JUDGEMENT: a show name, a
    service code name, an internal hostname. No regex can decide whether
    `prodtools-framecast-ui` is fine to publish and `project-nightfall` is not.
    """
    if not path.exists():
        print(f"no such fixture: {path}", file=sys.stderr)
        return 2

    payload = json.loads(path.read_text(encoding="utf-8"))
    blob = json.dumps(payload)

    print(f"AUDIT {path}  ({len(blob)} bytes)\n")
    print("Automated checks (all must be 0):")
    failures = 0
    structural = {
        "request entry (headers, cookies, body)": r'"type":\s*"request"',
        "frame locals or source context": r'"vars"|"context"',
        "user / contexts / sdk objects": r'"(?:user|contexts|sdk|packages)"\s*:\s*[{\[]',
    }
    for label, pattern in {**structural, **{k: v.pattern for k, v in _RESIDUAL_PII.items()}}.items():
        hits = {h for h in re.findall(pattern, blob) if "[MASKED_" not in str(h)}
        failures += bool(hits)
        print(f"  {'FAIL' if hits else '  ok'}  {label:<44}{len(hits)}")

    print("\nRead these yourself — no check can judge them:")
    frames = [
        frame
        for entry in payload.get("entries") or []
        for value in (entry.get("data") or {}).get("values") or []
        for frame in ((value.get("stacktrace") or {}).get("frames") or [])
    ]
    roots = sorted({
        str(frame.get(key, "")).split("/")[0].split("\\")[0]
        for frame in frames for key in ("filename", "module", "package")
        if frame.get(key)
    })
    print(f"  path/module/package roots ({len(roots)}):")
    for root in roots:
        print(f"      {root}")
    crumbs = [
        str(crumb.get("message", ""))
        for entry in payload.get("entries") or []
        if entry.get("type") == "breadcrumbs"
        for crumb in (entry.get("data") or {}).get("values") or []
    ]
    print(f"  breadcrumb messages ({len(crumbs)}):")
    for message in crumbs:
        print(f"      {message[:110]}")
    tags = {t.get("key"): t.get("value") for t in payload.get("tags") or []}
    print(f"  tags: {tags}")
    print(f"  title: {payload.get('title')}")

    print("\n  Looking for: a show or project code name, an internal hostname, a")
    print("  person's name in a module path, an unreleased product. If any of that")
    print("  is here, do not commit -- re-capture from a different issue, or edit")
    print("  the values by hand (the fixture only needs to keep its SHAPE).")
    if failures:
        print(f"\n{failures} automated check(s) FAILED — do not commit this.")
    return 1 if failures else 0


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="where to write the capture")
    parser.add_argument("--check-only", metavar="DIR", default="",
                        help="skip the network; probe an existing capture directory")
    parser.add_argument("--make-event-fixture", nargs="?", const="fixtures/sentry_event_latest.sample.json",
                        default="", metavar="PATH",
                        help="project the richest captured event into a committable fixture "
                             "(default: fixtures/sentry_event_latest.sample.json)")
    parser.add_argument("--audit-fixture", metavar="PATH", default="",
                        help="pre-commit check on a captured fixture: automated PII checks "
                             "plus the fields only a human can judge")
    parser.add_argument("--anonymize", action="store_true",
                        help="also write anonymized copies under <out>/anonymized/")
    parser.add_argument("--events", type=int, default=DEFAULT_EVENTS,
                        help="how many issues to pull an event for; 0 = all of them, "
                             "which is what Q10 needs to be trustworthy")
    parser.add_argument("--window", default="24h", help="statsPeriod for the issue queries")
    parser.add_argument("--team", default=os.environ.get("SENTRY_TEAM", ""),
                        help="team slug to scope by, e.g. prodtools (default: $SENTRY_TEAM)")
    parser.add_argument("--environments", default="",
                        help="comma-separated names to test as the allowlist "
                             "(default: $SENTRY_ENVIRONMENTS or 'production')")
    args = parser.parse_args()

    configured = [e.strip() for e in (
        args.environments or os.environ.get("SENTRY_ENVIRONMENTS", "production")
    ).strip("[]").replace('"', "").split(",") if e.strip()]

    out = Path(args.check_only or args.out)

    if not args.check_only and not args.audit_fixture:
        token = os.environ.get("SENTRY_TOKEN", "")
        org = os.environ.get("SENTRY_ORG", "")
        if not token or not org:
            print("set SENTRY_TOKEN and SENTRY_ORG first", file=sys.stderr)
            return 2
        client = SentryApiClient(os.environ.get("SENTRY_BASE_URL", "https://sentry.io"), token, org)
        try:
            capture(client, out, args.events, args.window, args.team)
        finally:
            client.close()

    if args.audit_fixture:
        return audit_fixture(Path(args.audit_fixture))

    if args.make_event_fixture:
        return make_event_fixture(out, Path(args.make_event_fixture))

    probe(out, configured)

    if args.anonymize:
        print("\nAnonymizing…")
        anonymize_capture(out, os.environ.get("SENTRY_ORG", "example-org"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
