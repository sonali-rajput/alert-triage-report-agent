from pipeline.prefilter import Prefilter
from shared.models import Alert, AlertSource


def make_alert(
    title: str, body: str = "", project: str = "core-tools", count: int = 5,
    environment: str = "",
) -> Alert:
    return Alert(
        source=AlertSource.sentry, source_id=title, title=title, body=body,
        project=project, event_count=count, environment=environment,
    )


def test_drops_known_flaky_tests():
    kept, dropped = Prefilter().apply([make_alert("Known flaky test failing again")])
    assert kept == [] and dropped == 1


def test_drops_dev_environment_alerts():
    """`environment` is a real field now (stamped on by the per-environment
    Sentry query), so this is an exact match rather than a regex over the body."""
    kept, dropped = Prefilter().apply([make_alert("Error", environment="development")])
    assert kept == [] and dropped == 1


def test_keeps_production_alerts_mentioning_dev():
    """The old regex dropped anything whose text said 'environment: dev', which
    caught production alerts that merely mentioned a dev path."""
    alert = make_alert("Error", body="failed to read environment: dev config", environment="production")
    kept, dropped = Prefilter().apply([alert])
    assert len(kept) == 1 and dropped == 0


def test_drops_sandbox_projects():
    kept, dropped = Prefilter().apply([make_alert("Real-looking error", project="sandbox")])
    assert kept == [] and dropped == 1


def test_keeps_real_alerts():
    alerts = [
        make_alert("Database connection pool exhausted"),
        make_alert("Unhandled TypeError in export"),
    ]
    kept, dropped = Prefilter().apply(alerts)
    assert len(kept) == 2 and dropped == 0
