import pytest

from pipeline.masking import Masker
from shared.models import Alert, AlertSource


def make_alert(body: str, title: str = "Some alert") -> Alert:
    return Alert(source=AlertSource.sentry, source_id="1", title=title, body=body)


def test_masks_bearer_tokens_and_passwords():
    masker = Masker()
    alert = make_alert(
        "Authorization: Bearer abcd1234efgh5678\npassword=SuperSecret99\nDB_PASSWORD=hunter2s"
    )
    masked = masker.mask_alert(alert)
    assert "abcd1234efgh5678" not in masked.body
    assert "SuperSecret99" not in masked.body
    assert "hunter2s" not in masked.body
    assert "[MASKED" in masked.body


def test_masks_gitlab_pat_email_and_ip():
    masker = Masker()
    alert = make_alert("token glpat-ABC123def456ghi7 from user.name@framestore.com at 10.1.2.3")
    masked = masker.mask_alert(alert)
    assert "glpat-" not in masked.body
    assert "user.name@framestore.com" not in masked.body
    assert "10.1.2.3" not in masked.body


def test_masks_title_too():
    masker = Masker()
    alert = make_alert("body", title="Login failed for admin@framestore.com")
    masked = masker.mask_alert(alert)
    assert "admin@framestore.com" not in masked.title


def test_truncates_long_bodies():
    masker = Masker()
    alert = make_alert("A" * 3000 + "MIDDLE" + "B" * 3000)
    masked = masker.mask_alert(alert)
    assert len(masked.body) < 6006
    assert "truncated" in masked.body
    assert masked.body.startswith("A")
    assert masked.body.endswith("B")


def test_short_bodies_untouched():
    masker = Masker()
    alert = make_alert("nothing sensitive here")
    assert masker.mask_alert(alert).body == "nothing sensitive here"


def test_records_which_rules_fired():
    """A masking hit is a security finding in its own right: it means an
    application is logging a credential into its error messages."""
    masker = Masker()
    alert = make_alert("Authorization: Bearer abcd1234efgh5678")
    assert masker.mask_alert(alert).masking_hits == ["bearer-token"]


def test_records_hits_from_title_and_body_together():
    masker = Masker()
    alert = make_alert("connecting from 10.1.2.3", title="Login failed for admin@framestore.com")
    hits = masker.mask_alert(alert).masking_hits
    assert set(hits) == {"email-address", "ipv4-address"}


def test_no_hits_when_nothing_matches():
    masker = Masker()
    assert masker.mask_alert(make_alert("nothing sensitive here")).masking_hits == []


def test_hits_accumulate_across_repeat_masking():
    """A body can pass through the masker more than once -- it grows when the
    event detail is folded in. Findings from an earlier pass must survive a
    later pass that happens not to reproduce them."""
    masker = Masker()
    first = masker.mask_alert(make_alert("Bearer abcd1234efgh5678"))
    enriched = first.model_copy(update={"body": first.body + "\nuser=a.person@framestore.com"})
    assert set(masker.mask_alert(enriched).masking_hits) == {"bearer-token", "email-address"}


def test_hits_are_not_duplicated():
    masker = Masker()
    alert = make_alert("a@b.com and c@d.com and e@f.com")
    assert masker.mask_alert(alert).masking_hits == ["email-address"]


def test_ordinary_error_prose_is_not_a_credential_finding():
    """A masking hit feeds the score's highest-weight signal and the report's
    'applications logging secrets' callout, so false positives here are false
    security findings. 'Invalid token: expired' is prose, not a leak."""
    masker = Masker()
    masked = masker.mask_alert(make_alert("Invalid token: expired at gateway"))
    assert masked.body == "Invalid token: expired at gateway"
    assert masked.masking_hits == []


def test_words_merely_containing_key_are_not_credentials():
    """The old case-insensitive substring match flagged monkey=patch as a
    leaked credential."""
    masker = Masker()
    masked = masker.mask_alert(make_alert("monkey=patch failed, TURKEY=roast skipped"))
    assert masked.body == "monkey=patch failed, TURKEY=roast skipped"
    assert masked.masking_hits == []


def test_real_env_var_secrets_are_still_masked():
    masker = Masker()
    masked = masker.mask_alert(make_alert("env: API_KEY=abcd TOKEN=zz SESSION_SECRET_V2=qq"))
    assert "abcd" not in masked.body
    assert "SESSION_SECRET_V2=[MASKED]" in masked.body
    assert "env-var-secretish" in masked.masking_hits


# --------------------------------------------------------------------------
# Home-directory paths
#
# Found by projecting a real captured event into a fixture: a stack frame read
# `/home/<person>/src/app/db.py` and no rule caught it. Every issue now gets a
# stack trace (before the rework it was the top 15 by score), so this is a
# colleague's name going to Vertex AI and into BigQuery on every run.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/home/ada.testperson/src/app/db.py", "/home/[MASKED_USER]/src/app/db.py"),
        ("/Users/ada.testperson/dev/app.js", "/Users/[MASKED_USER]/dev/app.js"),
        ("C:\\Users\\ada.testperson\\AppData\\app.exe", "C:\\Users\\[MASKED_USER]\\AppData\\app.exe"),
        ("D:\\Users\\someone\\build.log", "D:\\Users\\[MASKED_USER]\\build.log"),
    ],
)
def test_a_username_in_a_path_is_masked(path, expected):
    masked, hits = Masker().mask_text(path)
    assert masked == expected
    assert "home-directory-path" in hits


def test_the_path_shape_survives_masking():
    """The frame is in the prompt so an engineer can see where the code lives.
    Masking the whole path would take the finding away with the name."""
    masked, _ = Masker().mask_text("Error in /home/someone/services/render/submit.py:42")
    assert "services/render/submit.py:42" in masked


@pytest.mark.parametrize(
    "path",
    ["/usr/lib/python3.12/socket.py", "./node_modules/flux/lib/Dispatcher.js",
     "../../../src/helpers.ts", "/opt/app/main.py", "/var/log/app.log"],
)
def test_ordinary_paths_are_left_alone(path):
    """Over-masking costs the reader the one part of a frame they actually
    navigate by."""
    masked, hits = Masker().mask_text(path)
    assert masked == path and hits == []


def test_a_home_path_is_not_treated_as_a_leaked_credential():
    """Personal data, not a secret. The credential rules feed the report's
    'applications logging secrets' callout, and a home path in a stack trace is
    an ordinary trace, not a finding about the application."""
    from pipeline.agents.providers import _CREDENTIAL_HITS

    assert "home-directory-path" not in _CREDENTIAL_HITS


@pytest.mark.parametrize(
    "text,rule",
    [
        ("push failed with glpat-AbCdEf1234567890", "gitlab-pat"),
        ("denied for AKIAIOSFODNN7EXAMPLE", "aws-access-key"),
        ("api_key=sk_live_9f8a7b6c5d4e rejected", "api-key-assignment"),
        ("Bearer abcd1234efgh5678 rejected", "bearer-token"),
        ("DB_PASSWORD=hunter2hunter2 not set", "env-var-secretish"),
        ("contact ada@example.com", "email-address"),
        ("host 10.4.2.19 unreachable", "ipv4-address"),
    ],
)
def test_each_rule_reports_itself_by_name(text, rule):
    """The hit NAME is load-bearing, not just the redaction: `_CREDENTIAL_HITS`
    matches on it to decide what counts as a security finding, and the report
    prints it. A rule that redacts correctly but reports under another name
    would silently drop out of the security signal."""
    _masked, hits = Masker().mask_text(text)
    assert rule in hits
