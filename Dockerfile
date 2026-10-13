FROM python:3.13-slim-bookworm@sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26
LABEL qfbench2.interface_version="2.0" qfbench2.track="forecasting" qfbench2.verb="forecast"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HOME=/tmp XDG_CACHE_HOME=/tmp/.cache PYTHONPATH=/opt/final-sprint \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
COPY requirements-runtime.txt /opt/requirements-runtime.txt
RUN pip install --no-cache-dir --only-binary=:all: -r /opt/requirements-runtime.txt
COPY runtime_candidate /opt/final-sprint/runtime_candidate
COPY forecast /usr/local/bin/forecast
COPY LICENSE ARTIFACT_PROVENANCE.md /opt/final-sprint/
RUN chmod 0555 /usr/local/bin/forecast
WORKDIR /opt/final-sprint
USER 65534:65534
CMD ["forecast", "--help"]
