# App Settings / Environment Variables

## Azure Functions App: `SourceReplay`

### Required secrets
- `AZURE_STORAGE_CONNECTION_STRING`  
  Used by scoring to read/write blobs and download model.
- `EVENTHUB_CONN_STR`  
  Used by `replay` to publish events.
- `EVENTHUB_NAME`  
  Expected: `telemetry`

> Note: `AzureWebJobsStorage` is required by Functions runtime, but scoring/replay logic uses the variables above.

### Scoring configuration (defaults shown)
- `IN_CONTAINER=stream-output`
- `RAW_ROOT_PREFIX=raw/`
- `MODEL_CONTAINER=models`
- `MODEL_BLOB=model.joblib`
- `OUT_CONTAINER=predictions`
- `OUT_ROOT_PREFIX=scores/`

Optional:
- `PRED_THRESHOLD=0.20`
- `MAX_SCORE_BLOBS=` (unset = use all)

### Local dev
Create `function_app/local.settings.json` (do NOT commit) based on:
`function_app/local.settings.json.example`
