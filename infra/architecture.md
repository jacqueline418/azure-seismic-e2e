# Architecture (Azure Seismic E2E)

This project is an end-to-end streaming + ML scoring pipeline on Azure.

## Data flow (end-to-end)

1) **HTTP Replay (Azure Functions)**
- Function App: `SourceReplay`
- Endpoint: `/api/replay`
- Reads a historical CSV from Blob Storage (container `raw`)
- Publishes each row as a JSON event to Event Hub `telemetry` (namespace `ehseismic`)

2) **Stream Processing (Azure Stream Analytics)**
- Job: `saseismic`
- Input: Event Hub `telemetry`
- Outputs: Azure Storage account `stseismice2e`
  - Raw stream output (for training/scoring): container `stream-output`, prefix `raw/YYYY/MM/DD/HH/`
  - Alert stream output (rule-based): container `stream-output`, prefix `alert/YYYY/MM/DD/HH/`

3) **On-demand ML Scoring (Azure Functions)**
- Function App: `SourceReplay`
- Endpoint: `/api/score`
- Loads latest raw hour partition from `stream-output/raw/...`
- Loads model artifact from container `models` (blob `model.joblib`)
- Produces scored outputs into container `predictions`:
  - Historical per-run file: `scores/raw/YYYY/MM/DD/HH/preds_<timestamp>.csv`
  - Stable file for dashboards: `scores/latest.csv` (overwritten every run)

## Storage layout (high level)

- `raw/` : replay input CSV(s)
- `stream-output/` : ASA output (raw + alerts)
- `models/` : ML model artifacts
- `predictions/` : scoring outputs (CSV)
- Platform-managed containers (do not modify):
  - `azure-webjobs-hosts`, `$logs`


