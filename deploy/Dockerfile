# PRODUCTION image — the Cloud Run Job. Build from the repo root:
#   docker build -f deploy/Dockerfile -t alert-triage .
#
# Runtime dependencies and application code only. No tests, no fixtures, no
# dev tooling: a test runner inside a production container is code you did not
# mean to run, in a process holding your service account.
#
# For local work use deploy/Dockerfile.local, which builds FROM this image and
# adds the test tooling on top. That direction matters — it means the thing you
# test locally IS this image plus extras, rather than a parallel build that can
# drift away from it.
#
# A Job, not a Service: the container runs the pipeline once and exits. Exit
# code 0 means the run completed, non-zero is what Cloud Scheduler retries on.
#
# Two stages so the build toolchain never reaches the shipped image. The
# builder resolves dependencies into a venv and strips them; the runtime stage
# copies that venv onto a clean base. See the size notes at the bottom.

FROM python:3.12-slim AS builder

# binutils is here for `strip` only, and dies with this stage. Note that
# python:3.12-slim does NOT ship strip — without this the strip step below
# silently does nothing.
RUN apt-get update && apt-get install -y --no-install-recommends binutils \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip uninstall -y pip setuptools wheel

# Debug symbols in the native extensions (grpc, cryptography, pillow,
# pydantic-core) are ~24 MB that nothing in a Cloud Run Job can use: there is
# no debugger attached to a batch job. Vendored test suites go too.
RUN find /opt/venv -name '*.so' -exec strip --strip-unneeded {} + \
 && find /opt/venv -type d -name tests -prune -exec rm -rf {} + \
 && find /opt/venv -name '*.pyx' -delete

FROM python:3.12-slim AS runtime

# WeasyPrint native dependencies (Pango/Cairo) for PDF rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 libpangocairo-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 \
    shared-mime-info fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY shared/ shared/
COPY pipeline/ pipeline/
COPY config/ config/

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "pipeline.main"]

# Size, uncompressed: 447 MB -> 398 MB.
#
# What is left, and why it stays:
#   ~125 MB  python:3.12-slim base
#   ~36 MB   Pango/Cairo/gdk-pixbuf — WeasyPrint segfaults without them, and
#            the PDF is the deliverable
#   ~132 MB  the venv: fontTools and Pillow (WeasyPrint), grpc and google-*
#            (BigQuery, GCS, Gemini)
#
# Deliberately NOT removed: the ~38 MB of .pyc under __pycache__. A daily Job
# is cold-started every single run, so that bytecode is read on the only start
# there is. Trading ~38 MB of registry storage for CPU on every run is the
# wrong side of that trade.
