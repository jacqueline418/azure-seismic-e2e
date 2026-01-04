import os, logging
import json
import time
import csv
import io
import azure.functions as func

from azure.storage.blob import BlobClient
from azure.eventhub import EventHubProducerClient, EventData

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

def _get_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise ValueError(f"Missing required env var: {name}")
    return v

def _parse_int(qs: str | None, default: int) -> int:
    try:
        return int(qs) if qs is not None else default
    except Exception:
        return default

def _parse_float(qs: str | None, default: float) -> float:
    try:
        return float(qs) if qs is not None else default
    except Exception:
        return default

def _download_blob(container: str, blob: str) -> bytes:
    # Use either explicit connection string, or the default Azure Functions storage one.
    conn_str = os.environ.get("BLOB_CONN_STR") or os.environ.get("AzureWebJobsStorage")
    if not conn_str:
        raise ValueError("Missing BLOB_CONN_STR (preferred) or AzureWebJobsStorage")

    bc = BlobClient.from_connection_string(
        conn_str=conn_str,
        container_name=container,
        blob_name=blob,
    )
    return bc.download_blob().readall()

def _rows_from_csv_bytes(data: bytes) -> list[dict]:
    # Assumes UTF-8 CSV with header row
    text = data.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    return [row for row in reader]

def _to_number_if_possible(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return x
    s = str(x).strip()
    if s == "":
        return s
    # Try int then float
    try:
        return int(s)
    except Exception:
        pass
    try:
        return float(s)
    except Exception:
        return s

def _clean_row(row: dict) -> dict:
    # Convert obvious numeric strings to numbers (optional but nice)
    out = {}
    for k, v in row.items():
        out[k] = _to_number_if_possible(v)
    return out

@app.route(route="replay", methods=["GET", "POST"])
def replay(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP trigger:
      /api/replay?container=raw&blob=your.csv&start=0&limit=2000&rate=50&partition_key=mykey

    Env vars required:
      EVENTHUB_CONN_STR
      EVENTHUB_NAME
    Blob auth:
      BLOB_CONN_STR (recommended) OR AzureWebJobsStorage
    """
    try:
        container = req.params.get("container", "raw")
        blob = req.params.get("blob")
        if not blob:
            return func.HttpResponse(
                "Missing required query param: blob (e.g., ?container=raw&blob=seismic-bumps.csv)",
                status_code=400,
            )

        start = _parse_int(req.params.get("start"), 0)
        limit = _parse_int(req.params.get("limit"), 2000)
        rate = _parse_float(req.params.get("rate"), 50.0)
        partition_key = req.params.get("partition_key")
        
        eh = os.getenv("EVENTHUB_CONN_STR", "")
        logging.warning("EH conn len=%d head=%s", len(eh), eh[:30])
        logging.warning("EH name=%s", os.getenv("EVENTHUB_NAME"))

        eh_conn = _get_env("EVENTHUB_CONN_STR")
        eh_name = _get_env("EVENTHUB_NAME")

        # Download + parse CSV
        data = _download_blob(container, blob)
        rows = _rows_from_csv_bytes(data)

        start = max(start, 0)
        end = min(start + max(limit, 0), len(rows))
        rows_slice = rows[start:end]

        # Prepare Event Hub producer
        producer = EventHubProducerClient.from_connection_string(
            conn_str=eh_conn,
            eventhub_name=eh_name,
        )

        sleep_s = 0.0 if rate <= 0 else (1.0 / rate)
        sent = 0

        with producer:
            batch = producer.create_batch(partition_key=partition_key)

            for row in rows_slice:
                payload = _clean_row(row)
                event = EventData(json.dumps(payload))

                try:
                    batch.add(event)
                except ValueError:
                    # batch full -> send then start new batch
                    producer.send_batch(batch)
                    batch = producer.create_batch(partition_key=partition_key)
                    batch.add(event)

                sent += 1
                if sleep_s > 0:
                    time.sleep(sleep_s)

            # flush final batch
            if len(batch) > 0:
                producer.send_batch(batch)

        return func.HttpResponse(
            f"OK: sent {sent} events from {container}/{blob} (rows {start}..{end-1}) "
            f"to EventHub={eh_name} at ~{rate}/sec",
            status_code=200,
        )

    except Exception as e:
        return func.HttpResponse(
            f"Error: {type(e).__name__}: {e}",
            status_code=500,
        )