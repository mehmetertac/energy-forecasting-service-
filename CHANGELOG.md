# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- README **API reference** table for `/health`, `/plants`, `/metrics`, `/forecast`, and `/schema`.
- This changelog.

## [1.0.1] - 2026-11-05

### Added

- Portfolio polish: API reference, changelog, release notes, dashboard media refresh.

### Fixed

- Documented Docker compose startup fix from v1.0.0 tag (merge #4 on `main`).

## [1.0.0] - 2026-09-25

### Added

- Recruiter-facing README: architecture diagram, dashboard GIF, three-command Docker quickstart, dispatch narrative.
- Streamlit dashboard with P50 and P10–P90 band, pinball/coverage panel.
- FastAPI batch inference: `GET /health`, `GET /plants`, `GET /metrics`, `POST /forecast`, structured errors and `X-Request-ID`.
- SQLite forecast store and scheduled `daily_forecast.py` publish from MLflow Production pyfunc.
- Multi-stage Docker serve image and compose stack (MLflow → batch → API → dashboard).
- GitHub Actions CI: ruff, pytest (no torch), Docker build.

### Changed

- README rewritten as product page; public write-up in `docs/linkedin.md`.

## [0.3.0] - 2026-09-24

### Added

- Batch publish pipeline, SQLite serving store, plants/metrics API endpoints.
- Streamlit dashboard (initial).

## [0.2.0] - 2026-09-23

### Added

- CI workflow, expanded API test contract, Dockerized serve stack with runtime MLflow model loading.

## [0.1.0] - 2026-09-19

### Added

- Service repo scaffold: Week 5 TFT port, MLflow tracking, registry-backed serving, FastAPI batch API.

[Unreleased]: https://github.com/mehmetertac/energy-forecasting-service-/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/mehmetertac/energy-forecasting-service-/releases/tag/v1.0.1
[1.0.0]: https://github.com/mehmetertac/energy-forecasting-service-/releases/tag/v1.0.0
[0.3.0]: https://github.com/mehmetertac/energy-forecasting-service-/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/mehmetertac/energy-forecasting-service-/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mehmetertac/energy-forecasting-service-/releases/tag/v0.1.0
