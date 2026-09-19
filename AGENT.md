# AGENT.md — rules for AI agents

Guidance for agents working in **energy-forecasting-service**. Read this file first, then follow the linked docs.

---

## Documentation map

| Doc | Purpose |
|---|---|
| [README.md](README.md) | Recruiter entry: setup, MLflow, dispatch/market reading of P10/P50/P90 |
| [handover.md](handover.md) | Current status, repo layout, module API, week roadmap |
| [WEEK_09_REFLECTION.md](WEEK_09_REFLECTION.md) | Week 9 build notes |
| [data/README.md](data/README.md) | OPSD / Open-Meteo provenance and schemas |
| [notebooks/README.md](notebooks/README.md) | Notebooks are thin drivers only |
| [docker/README.md](docker/README.md) | Serving image (not wired yet) |
| [dashboard/README.md](dashboard/README.md) | Streamlit demo (not wired yet) |
| [src/energy_forecasting/data/](src/energy_forecasting/data/) | Fetch, disaggregate, calendar/solar/weather features |
| [src/energy_forecasting/model/](src/energy_forecasting/model/) | TFT, metrics, rolling-origin CV, MLflow tracking |
| [src/energy_forecasting/api/](src/energy_forecasting/api/) | Quantile forecast schema + FastAPI stub |
| [tests/](tests/) | Unit tests on synthetic frames (CI does not download OPSD) |

Headline contract: **probabilistic forecasts (P10/P50/P90)** through training metrics, MLflow, API schema, and (later) the dashboard. Do not collapse the product to a point forecast.

Source model: Week 5 [`multi-site-solar-hierarchy`](https://github.com/mehmetertac/multi-site-solar-hierarchy). Do not port MinT / N-HiTS unless a later task asks.

---

## Rules

### 1. File size limit

- **No file should exceed 1,000 lines.**
- If a file approaches or exceeds that limit, **stop and suggest a refactor** before adding more code.
- Pre-commit and CI run [`scripts/check_file_size.py`](scripts/check_file_size.py).

### 2. Documentation before every push

- **Update documentation before every push** to the repository.
- At minimum, check:
  - [README.md](README.md) — run commands, structure, user-facing behavior
  - [handover.md](handover.md) — done/next steps, API table, artifacts

### 2a. Update handover.md on every push (required)

- **[handover.md](handover.md) must be updated before every push**, even for small changes.
- On each push, at minimum refresh:
  - **Last updated** date at the top
  - **What is done** table
  - **Repo layout** if files or directories changed
  - **Core module API** if public symbols, CLI flags, or defaults changed
  - **Suggested next step** if priorities shifted
  - **Key commit** hash and one-line summary for the work being pushed
- If nothing functional changed, still bump **Last updated** and note “no functional change.”
- Do not push without reviewing [handover.md](handover.md).

### 3. Tests — always, at least minimal

- **Always create at least minimal unit tests**, even for small changes.
- Add **integration** tests when wiring modules (loader → features → TFT → metrics).
- Add **functional** tests when the project supports them (CLI smoke, API contract).
- Existing pattern: [tests/](tests/) uses synthetic panels so CI does not depend on OPSD downloads or GPU.
- New forecast or metric logic needs numeric assertions, not only “runs without error.”
- Keep P10/P50/P90 in any new API or dashboard schema tests.

### 4. Run tests before commit or push

```powershell
pytest tests/ -q
```

For training-script changes, also smoke (short):

```powershell
python scripts/train.py --n-splits 2 --max-epochs 1 --max-plants 2 --hidden-size 16 --limit-train-batches 5 --limit-val-batches 2
```

**Git hooks:** After creating the venv:

```powershell
pip install -e ".[dev]"
pre-commit install
```

Hooks run file-size check + pytest (see [`.pre-commit-config.yaml`](.pre-commit-config.yaml)).

### 5. Keep reading in-repo docs

- Do not guess API or roadmap from memory — use [handover.md](handover.md) for status and [README.md](README.md) for how to run.
- Match conventions in [`src/energy_forecasting/model/tft.py`](src/energy_forecasting/model/tft.py) (QuantileLoss, `pred_q10`/`pred_q50`/`pred_q90` columns).
- Tracking goes through [`src/energy_forecasting/model/tracking.py`](src/energy_forecasting/model/tracking.py); MLflow remains the backbone, W&B is optional.

---

## Quick checklist (before push)

- [ ] No file > 1,000 lines (or refactor proposed)
- [ ] [README.md](README.md) updated if behavior or layout changed
- [ ] [handover.md](handover.md) updated (required on every push — date, status, API, key commit)
- [ ] New/changed logic has tests in [tests/](tests/)
- [ ] `pytest tests/ -q` passes
- [ ] Relevant smoke command run if training CLI or TFT changed
