"""Environment-name normalization.

Sentry environment names are free text set by each SDK's own config. A real
org's list is not `["production", "staging"]` -- it contains `prod`, `@prod`,
`@dev`, `live`, `test` and one-offs. Every environment check in the pipeline
used to be an exact case-folded string match against a hand-written config
list, so `@dev` was never dropped as noise and `@prod` never read as
production. The failure is silent: no error, just an alert quietly treated as
if it came from somewhere it did not.
"""

from __future__ import annotations

import pytest

from pipeline.prefilter import Prefilter
from shared.environments import canonical_environment, matches_environment, normalize_environment
from shared.models import Alert, AlertSource


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("production", "production"),
        ("@prod", "prod"),
        ("  Prod  ", "prod"),
        ("pre-prod", "preprod"),
        ("pre_prod", "preprod"),
        ("*staging*", "staging"),
        ("", ""),
    ],
)
def test_normalize_strips_punctuation_and_case(raw, expected):
    assert normalize_environment(raw) == expected


@pytest.mark.parametrize("raw", ["production", "prod", "@prod", "PROD", "live", "prd"])
def test_production_spellings_canonicalize_together(raw):
    assert canonical_environment(raw) == "production"


@pytest.mark.parametrize("raw", ["development", "dev", "@dev", "local", "localhost"])
def test_development_spellings_canonicalize_together(raw):
    assert canonical_environment(raw) == "development"


def test_test_is_not_an_alias_of_development():
    """Plenty of teams run a real, user-facing `test` environment they want
    triaged. Folding it into `development` would silently drop it."""
    assert canonical_environment("test") != canonical_environment("development")


def test_an_unknown_environment_compares_against_itself():
    assert matches_environment("edit-suite", ["edit_suite"])
    assert not matches_environment("edit-suite", ["production"])


def test_empty_environment_matches_nothing():
    """An unstamped environment is unknown, not non-production -- it must not
    accidentally match a drop rule."""
    assert not matches_environment("", ["development", ""])


def make_alert(**kwargs) -> Alert:
    base = dict(
        source=AlertSource.sentry, source_id="1", kind="sentry_issue",
        title="Something broke", project="core-tools", event_count=10,
    )
    base.update(kwargs)
    return Alert(**base)


@pytest.mark.parametrize("env", ["production", "prod", "@prod", "Live"])
def test_production_spellings_survive_the_prefilter(env):
    """The prefilter is the only place a production spelling can now do
    damage: everything downstream of it hands the raw string to a model, which
    is told in the prompt that '@prod', 'live' and 'prod' all mean production."""
    kept, dropped = Prefilter().apply([make_alert(environment=env)])
    assert (len(kept), dropped) == (1, 0), f"'{env}' was dropped as non-production noise"


@pytest.mark.parametrize("env", ["development", "@dev", "Local"])
def test_prefilter_drops_every_development_spelling(env):
    kept, dropped = Prefilter().apply([make_alert(environment=env)])
    assert (kept, dropped) == ([], 1), f"'{env}' was not dropped"


# --------------------------------------------------------------------------
# The <service>@<environment> convention
#
# Not hypothetical. The real org's environment list contains
# flock-cron@production, help-web-api@production, flock-websockets@staging and
# help-web-api-dev@development alongside plain `production` and `staging`.
# None of them normalized anywhere near `production` before this, so
# SENTRY_ENVIRONMENTS=["production"] was silently ingesting NOTHING from five
# production environments.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["flock-cron@production", "help-web-api@production", "flock-web-api@production",
     "flock-websockets@production", "help-cron@production", "None@production"],
)
def test_service_scoped_production_names_are_production(raw):
    assert canonical_environment(raw) == "production"
    assert matches_environment(raw, ["production"])


@pytest.mark.parametrize(
    "raw,expected",
    [("flock-websockets@staging", "staging"),
     ("help-web-api-dev@development", "development"),
     ("help-cron@staging", "staging")],
)
def test_service_scoped_non_production_names_keep_their_environment(raw, expected):
    assert canonical_environment(raw) == expected


def test_a_service_scoped_dev_environment_is_droppable_noise():
    """The other half of the same bug: drop_environments never matched these,
    so a developer's environment was ingested and ranked as if unknown."""
    assert matches_environment("help-web-api-dev@development", ["development"])


def test_a_bare_at_prefix_is_unaffected():
    """`@prod` has an empty left half; the older leading-punctuation behaviour
    must survive the change."""
    assert canonical_environment("@prod") == "production"


def test_an_unknown_service_scoped_name_keeps_its_own_identity():
    """`vlaw` and `edit-suite` are real names in that org. They must compare
    against themselves, not get forced into a known bucket."""
    assert canonical_environment("renderfarm@edit-suite") == "editsuite"
    assert not matches_environment("renderfarm@edit-suite", ["production"])
