# Azure Seismic E2E Pipeline

## Architecture (High-level)
Event Hub (telemetry) → Stream Analytics (raw + alert outputs) → Blob Storage (raw/) → Azure Function (score) → Blob Storage (predictions/scores/latest.csv + history)

## Azure Resources
- Resource group: rg-seismic-e2e
- Event Hubs namespace: ehseismic
  - Event hub: telemetry
- Stream Analytics job: saseismic
  - Input: TelemetryInput (Event Hub telemetry)
  - Outputs: RawOutput, AlertOutput (Blob Storage)
- Storage account: stseismice2e
  - Containers: raw, predictions, models, curated (optional)
- Function App: SourceReplay (HTTP endpoints: /api/replay, /api/score)

## Storage Layout
- raw/
  - raw/YYYY/MM/DD/HH/*.json (ASA RawOutput)
- predictions/
  - scores/latest.csv (overwritten each score run)
  - scores/raw/YYYY/MM/DD/HH/preds_<timestamp>.csv (history)
- models/
  - model.joblib

## HTTP Endpoints
### Replay
GET /api/replay?container=raw&blob=...&start=0&limit=2000&rate=50

### Score
GET /api/score?mode=latest&threshold=0.2
GET /api/score?prefix=raw/YYYY/MM/DD/HH/&threshold=0.2

## Configuration (Function App Settings)
List required env vars (no secrets in repo).
