"""Normalization for Sentry environment names.

Sentry does not curate environment names. Each SDK reports whatever string its
own config sets, so a real org's environment list is messy: alongside
`production` and `development` you get `prod`, `@prod`, `@dev`, `live`, `test`,
`Staging`, and one-off names left behind by a service that was configured once
and never revisited.

Every environment check in this pipeline used to be an exact, case-folded string
match against a hand-written list:

  * `config/noise_filters.yaml` -> `drop_environments`
  * `PipelineSettings.sentry_environments` -> which environments are fetched AT ALL

An exact match against `["production"]` silently misses `@prod` and `Live`. The
failure is invisible: no error, the alert is just never dropped as dev noise,
or never fetched in the first place.

So comparisons go through `normalize_environment` first, and membership goes
through `matches_environment`, which also understands the common aliases. The
config lists stay authoritative -- this only stops punctuation and casing from
deciding whether a production outage is recognised as one.
"""

from __future__ import annotations

import re

# Leading/trailing punctuation Sentry environment names pick up in the wild:
# "@prod", "prod ", "*staging*".
_EDGE_PUNCTUATION = re.compile(r"^[^a-z0-9]+|[^a-z0-9]+$")
# Internal separators, so "pre-prod", "pre_prod" and "pre prod" compare equal.
_SEPARATORS = re.compile(r"[\s_-]+")

# Names that mean the same thing. Keyed by the canonical name a config list is
# most likely to spell. Only unambiguous synonyms belong here: `test` is NOT an
# alias of `development`, because plenty of teams run a real, user-facing test
# environment they very much want triaged.
_ALIASES: dict[str, set[str]] = {
    "production": {"production", "prod", "prd", "live", "production1", "prodution"},
    "development": {"development", "dev", "devel", "local", "localhost"},
    "staging": {"staging", "stage", "stg", "preprod", "preproduction"},
}


def normalize_environment(value: str) -> str:
    """Casefold, strip edge punctuation, collapse separators.

    `"@Prod"` -> `"prod"`, `"pre-prod "` -> `"preprod"`, `""` -> `""`.
    """
    if not value:
        return ""
    lowered = value.strip().lower()
    lowered = _EDGE_PUNCTUATION.sub("", lowered)
    return _SEPARATORS.sub("", lowered)


def environment_suffix(value: str) -> str:
    """The environment half of a `<service>@<environment>` name.

    Measured on the real org: alongside `production` and `staging` the list
    contains `flock-cron@production`, `help-web-api@production`,
    `flock-websockets@staging` and `help-web-api-dev@development` -- a
    per-service naming convention where the environment is the part after the
    last `@`. Before this, none of those normalized anywhere near `production`
    (`flock-cron@production` -> `flockcron@production`), so an allowlist of
    `["production"]` silently ingested NOTHING from five production
    environments, and `drop_environments` never dropped their staging twins.

    A bare `@prod` has an empty left half and is unaffected, so the existing
    leading-punctuation behaviour is unchanged.
    """
    return value.rsplit("@", 1)[-1] if "@" in value else value


def canonical_environment(value: str) -> str:
    """Normalized name, mapped through the alias table when it is a known
    synonym. `"@prod"` -> `"production"`, `"help-web-api@production"` ->
    `"production"`; an unknown name is returned normalized but otherwise
    untouched, so a team's own `edit-suite` env still compares against
    itself."""
    normalized = normalize_environment(environment_suffix(value))
    for canonical, aliases in _ALIASES.items():
        if normalized in aliases:
            return canonical
    return normalized


def matches_environment(value: str, configured: object) -> bool:
    """True when `value` names one of the `configured` environments, comparing
    canonically so `"@prod"` matches a config that says `"production"`.

    `configured` is whatever the YAML held: normally a list of strings.
    """
    if not value or not isinstance(configured, (list, tuple, set, frozenset)):
        return False
    target = canonical_environment(value)
    return any(canonical_environment(str(c)) == target for c in configured)
