"""Report storage: a private bucket, a signed URL, and a local fallback.

The signing path is the one worth testing, because its failure mode is quiet:
if the service account cannot sign, the report still uploads and the run still
succeeds — the Chat card just carries a `gs://` URI nobody can open. That is a
confusing thing to diagnose weeks later, so the fallback is deliberate,
logged, and asserted here.
"""

from __future__ import annotations

from pipeline.storage import ArtifactStore


class FakeBlob:
    def __init__(self, name: str, sign_error: Exception | None = None):
        self.name = name
        self.data: bytes | None = None
        self.content_type: str | None = None
        self.sign_kwargs: dict | None = None
        self._sign_error = sign_error

    def upload_from_string(self, data, content_type="application/octet-stream"):
        self.data = data
        self.content_type = content_type

    def generate_signed_url(self, **kwargs):
        self.sign_kwargs = kwargs
        if self._sign_error:
            raise self._sign_error
        return f"https://signed.example/{self.name}?exp=1"


class FakeBucket:
    def __init__(self, sign_error: Exception | None = None):
        self.blobs: dict[str, FakeBlob] = {}
        self._sign_error = sign_error

    def blob(self, path: str) -> FakeBlob:
        self.blobs[path] = FakeBlob(path, self._sign_error)
        return self.blobs[path]


def cloud_store(sign_error: Exception | None = None) -> tuple[ArtifactStore, FakeBucket]:
    store = ArtifactStore(local_dir="unused")
    bucket = FakeBucket(sign_error)
    store._bucket = bucket          # what a configured GCS_BUCKET produces
    store._bucket_name = "reports"
    return store, bucket


# --------------------------------------------------------------------------
# Cloud path
# --------------------------------------------------------------------------


def test_uploads_and_returns_a_signed_url():
    store, bucket = cloud_store()
    url = store.save_report("2026-08-11", b"%PDF-1.7", "pdf")

    blob = bucket.blobs["reports/triage-report-2026-08-11.pdf"]
    assert blob.data == b"%PDF-1.7"
    assert blob.content_type == "application/pdf"
    assert url.startswith("https://signed.example/")


def test_the_signed_url_is_v4_and_read_only():
    """A signing method other than GET, or the v2 scheme, would hand out more
    than the ability to read one report."""
    store, bucket = cloud_store()
    store.save_report("2026-08-11", b"x", "pdf")

    kwargs = bucket.blobs["reports/triage-report-2026-08-11.pdf"].sign_kwargs
    assert kwargs["version"] == "v4"
    assert kwargs["method"] == "GET"


def test_the_url_lifetime_follows_the_setting():
    store, bucket = cloud_store()
    store._signed_url_days = 3
    store.save_report("2026-08-11", b"x", "pdf")

    kwargs = bucket.blobs["reports/triage-report-2026-08-11.pdf"].sign_kwargs
    assert kwargs["expiration"].days == 3


def test_a_signing_failure_falls_back_to_the_gs_uri_rather_than_losing_the_run():
    """Signing needs the IAM signBlob permission (the job's SA over itself).
    Without it the report is still safely stored, so the run must continue —
    but the URI is the visible symptom, which is why it is preserved."""
    store, _bucket = cloud_store(sign_error=RuntimeError("no signBlob permission"))
    url = store.save_report("2026-08-11", b"x", "pdf")
    assert url == "gs://reports/reports/triage-report-2026-08-11.pdf"


def test_html_reports_get_the_right_content_type():
    """The fallback format when WeasyPrint's native libs are missing. Served
    as application/pdf it would download instead of rendering."""
    store, bucket = cloud_store()
    store.save_report("2026-08-11", b"<html>", "html")
    assert bucket.blobs["reports/triage-report-2026-08-11.html"].content_type == "text/html"


def test_re_running_a_date_overwrites_rather_than_accumulating():
    """The report path is derived from the date alone, so a forced replay
    replaces the day's report instead of leaving two."""
    store, bucket = cloud_store()
    store.save_report("2026-08-11", b"first", "pdf")
    store.save_report("2026-08-11", b"second", "pdf")
    assert list(bucket.blobs) == ["reports/triage-report-2026-08-11.pdf"]
    assert bucket.blobs["reports/triage-report-2026-08-11.pdf"].data == b"second"


# --------------------------------------------------------------------------
# Local path
# --------------------------------------------------------------------------


def test_writes_locally_when_no_bucket_is_configured(tmp_path):
    store = ArtifactStore(local_dir=str(tmp_path / "artifacts"))
    path = store.save_report("2026-08-11", b"%PDF-1.7", "pdf")

    written = tmp_path / "artifacts" / "reports" / "triage-report-2026-08-11.pdf"
    assert written.read_bytes() == b"%PDF-1.7"
    assert path == str(written.resolve())


def test_creates_missing_directories(tmp_path):
    store = ArtifactStore(local_dir=str(tmp_path / "deep" / "nested"))
    store.save_report("2026-08-11", b"x", "pdf")
    assert (tmp_path / "deep" / "nested" / "reports").is_dir()
