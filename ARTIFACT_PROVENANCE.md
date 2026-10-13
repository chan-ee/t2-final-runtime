# Artifact provenance

This image contains only pure forecasting code, dependency packages and their metadata.
It bundles no fitted coefficients, pretrained model weights, numeric research data,
market outcomes, answer keys, private references, credentials or unit-specific forecasts.

The canonical M0 procedure follows the public Track2 M0-BASELINE.md specification.
The official monthly-horizon helper is copied unchanged from the frozen public Track2 source.
The fixed PCE/CPI model uses train120/min60/ridge20/intercept0/OOF60/draws1000/seed1729.
All coefficients and covariance estimates are fitted only from the supplied own-unit prefix.
Eight package source files and the forecast launcher are byte-identical to the reviewed V4
runtime source. No ad-hoc mean/width/precision adjustment is enabled.

Inference is offline and uses no LLM/model API. No pretrained/fitted model artifact is
packaged; model inventory is models=[] and statistical code is disclosed by this provenance.
The optional PCE branch requires explicit verified own-unit metadata; current public corpus
activation is zero and measurement is disabled. Source timestamps are not independent PIT proof.

The local V4 image passed C3 and g0-g3 on raw/staged exemplar with1000 finite draws,
and20 synthetic metadata/core checks. These are format/runtime tests, not performance claims.
This remote recipe has not yet been built or published; package versions are pinned to local
V4 runtime versions. A remote build must still run equivalent container/gate checks.

Included project LICENSE is retained. Bundled Python distributions include their own license metadata.
