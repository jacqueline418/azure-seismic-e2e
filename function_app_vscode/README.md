# Azure Functions (SourceReplay)

This Functions app provides:
- `replay`: send historical CSV rows to Event Hub
- `score`: read latest raw data from Blob, apply ML model, write scored CSV outputs

## Endpoints

### Replay
`GET /api/replay?container=raw&blob=<file.csv>&start=0&limit=6000&rate=50`

- Reads CSV from storage account
- Publishes JSON to Event Hub `telemetry`

### Score
`GET /api/score?mode=latest&threshold=0.2`

- Finds most recent `stream-output/raw/YYYY/MM/DD/HH/`
- Loads model `models/model.joblib`
- Writes:
  - `predictions/scores/raw/YYYY/MM/DD/HH/preds_<timestamp>.csv`
  - `predictions/scores/latest.csv` (overwrite)

## Required App Settings
See `../infra/app-settings.md`.

## Local settings
Copy:
- `local.settings.json.example` -> `local.settings.json`
Fill secrets locally, do not commit.
