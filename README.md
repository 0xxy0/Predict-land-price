# House Prediction

A reproducible house-price prediction system built as a learning project for
data engineering, machine learning, API development, testing, and Docker.

## Development Setup

Use the repository virtual environment on Windows:

```powershell
Set-Location C:\0xxy0\house_prediction
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Run the Phase 0 checks:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\mypy.exe src
```

The raw dataset in `data/raw/` is immutable. Generated data belongs in
`data/processed/`, and generated model artifacts belong in `models/`.

## Phase 3 Cleaning

Run the deterministic cleaning pipeline with:

```powershell
.\.venv\Scripts\house-clean.exe data\raw\KC_housing_data.csv `
	--output data\processed\cleaned_housing.csv `
	--audit-output data\processed\cleaning_audit.json
```

The pipeline parses dates, normalizes text, splits `statezip` into `state` and
`zip_code`, removes invalid rows, preserves `yr_renovated == 0` as the raw
dataset sentinel for no renovation, and writes an auditable result without
modifying the raw CSV.

## Learning Workflow

Each phase follows a small loop:

1. State the design decision and its reason.
2. Implement one focused slice.
3. Run the narrowest useful check.
4. Explain the result and record the decision.

Phase 3 is complete when the cleaning tests pass and the command above creates
both processed data and a JSON audit report. The next checkpoint is Phase 4:
build a reusable feature pipeline on top of the cleaned data.
