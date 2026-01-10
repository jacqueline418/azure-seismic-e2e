# Azure Resources

Resource Group: `rg-seismic-e2e`  
Subscription: Azure for Students

## Function App
- Name: `SourceReplay`
- Functions:
  - `replay` (HTTP trigger)
  - `score` (HTTP trigger)

## Stream Analytics
- Job: `saseismic`
- Input:
  - Event Hub: `telemetry`
- Outputs:
  - Storage account: `stseismice2e`
  - Container: `stream-output`
    - raw output prefix: `raw/{date}/{time}`
    - alert output prefix: `alert/{date}/{time}`

## Event Hubs
- Namespace: `ehseismic`
- Event Hub: `telemetry`

## Storage
- Storage account: `stseismice2e`

Expected containers:
- `raw` (replay input CSV files)
- `stream-output` (ASA outputs)
- `models` (model artifacts: `model.joblib`)
- `predictions` (scoring outputs: `scores/...`)
System-managed containers:
- `azure-webjobs-hosts`, `$logs`
