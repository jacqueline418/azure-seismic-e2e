# function_app.py
import os
import logging
import json
import time
import csv
import io
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import azure.functions as func

from azure.storage.blob import BlobClient, BlobServiceClient
from azure.eventhub import EventHubProducerClient, EventData

import pandas as pd
import joblib

# Only needed if you want eval metrics in HTTP response
from sklearn.metrics import classification_report, roc_auc_score

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# ----------------------------
# Replay helpers (existing)
# ----------------------------
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
    try:
        return int(s)
    except Exception:
        pass
    try:
        return float(s)
    except Exception:
        return s

def _clean_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        out[k] = _to_number_if_possible(v)
    return out


# ----------------------------
# Score helpers (new)
# ----------------------------
CATEGORICAL = ["seismic", "seismoacoustic", "shift", "ghazard"]
NUMERIC = [
    "genergy", "gpuls", "gdenergy", "gdpuls",
    "nbumps", "nbumps2", "nbumps3", "nbumps4", "nbumps5", "nbumps6", "nbumps7", "nbumps89",
    "energy", "maxenergy"
]
FEATURES = CATEGORICAL + NUMERIC
LABEL = "class"

MAP_ABCD = {"a": 1, "b": 2, "c": 3, "d": 4}
MAP_SHIFT = {"W": 1, "N": 0}

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _normalize_prefix(prefix: str) -> str:
    prefix = (prefix or "").strip()
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return prefix

def _bsc_from_conn() -> BlobServiceClient:
    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise ValueError("Missing AZURE_STORAGE_CONNECTION_STRING in Function App settings")
    return BlobServiceClient.from_connection_string(conn_str)

def _download_model_to_tmp(
    bsc: BlobServiceClient,
    model_container: str,
    model_blob: str
) -> str:
    os.makedirs("/tmp/models", exist_ok=True)
    local_path = f"/tmp/models/{os.path.basename(model_blob)}"
    bc = bsc.get_blob_client(container=model_container, blob=model_blob)
    data = bc.download_blob().readall()
    with open(local_path, "wb") as f:
        f.write(data)
    return local_path

def _list_latest_hour_prefix(cc, base_prefix: str) -> str:
    """
    Find latest prefix like raw/YYYY/MM/DD/HH/ by scanning blobs under base_prefix.
    Lexicographic max works because YYYY/MM/DD/HH is zero-padded.
    """
    base_prefix = _normalize_prefix(base_prefix)
    prefixes = set()

    for b in cc.list_blobs(name_starts_with=base_prefix):
        parts = b.name.split("/")
        # Expect: raw/YYYY/MM/DD/HH/<file>
        if len(parts) >= 5:
            # Build hour prefix: raw/YYYY/MM/DD/HH/
            hour_prefix = "/".join(parts[:5]) + "/"
            # Only keep ones that match base_prefix root
            if hour_prefix.startswith(base_prefix):
                prefixes.add(hour_prefix)

    if not prefixes:
        raise FileNotFoundError(f"No blobs found under base prefix: {base_prefix}")

    return sorted(prefixes)[-1]

def _load_jsonl_from_prefix(cc, prefix: str, max_blobs: Optional[int] = None) -> pd.DataFrame:
    prefix = _normalize_prefix(prefix)
    blobs = list(cc.list_blobs(name_starts_with=prefix))
    if max_blobs is not None:
        blobs = blobs[:max_blobs]
    if not blobs:
        raise FileNotFoundError(f"No blobs found under prefix: {prefix}")

    rows: List[Dict[str, Any]] = []
    for b in blobs:
        data = cc.get_blob_client(b.name).download_blob().readall().decode("utf-8", errors="ignore")
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows), len(blobs)

def _encode_features_report(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    out = df.copy()
    out["seismic"] = out["seismic"].map(MAP_ABCD)
    out["seismoacoustic"] = out["seismoacoustic"].map(MAP_ABCD)
    out["ghazard"] = out["ghazard"].map(MAP_ABCD)
    out["shift"] = out["shift"].map(MAP_SHIFT)

    if out[CATEGORICAL].isna().any().any():
        raise ValueError("Unexpected categorical values. Expected a/b/c/d and W/N.")

    for c in FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Drop rows with NaNs in any feature
    out = out.dropna(subset=FEATURES)
    return out

def _upload_csv(bsc: BlobServiceClient, container: str, blob_name: str, df: pd.DataFrame) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    data = buf.getvalue().encode("utf-8")
    bsc.get_blob_client(container=container, blob=blob_name).upload_blob(data, overwrite=True)


# ----------------------------
# HTTP: replay (unchanged)
# ----------------------------
@app.function_name(name="replay")
@app.route(route="replay", methods=["GET", "POST"], auth_level=func.AuthLevel.FUNCTION)
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

        eh_conn = _get_env("EVENTHUB_CONN_STR")
        eh_name = _get_env("EVENTHUB_NAME")

        # Download + parse CSV
        data = _download_blob(container, blob)
        rows = _rows_from_csv_bytes(data)

        start = max(start, 0)
        end = min(start + max(limit, 0), len(rows))
        rows_slice = rows[start:end]

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
                    producer.send_batch(batch)
                    batch = producer.create_batch(partition_key=partition_key)
                    batch.add(event)

                sent += 1
                if sleep_s > 0:
                    time.sleep(sleep_s)

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


# ----------------------------
# HTTP: score (new)
# ----------------------------
@app.function_name(name="score")
@app.route(route="score", methods=["GET", "POST"], auth_level=func.AuthLevel.FUNCTION)
def score(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP trigger (on-demand scoring)

    Usage:
      /api/score?mode=latest&threshold=0.2
      /api/score?prefix=raw/2026/01/06/01/&threshold=0.2&max_blobs=50

    Reads:
      AZURE_STORAGE_CONNECTION_STRING
      IN_CONTAINER (default stream-output)
      RAW_ROOT_PREFIX (default raw/)
      MODEL_CONTAINER (default models)
      MODEL_BLOB (default model.joblib)

    Writes:
      OUT_CONTAINER (default predictions)
      OUT_ROOT_PREFIX (default scores/)
    """
    try:
        # Config
        in_container = os.environ.get("IN_CONTAINER", "stream-output")
        raw_root = os.environ.get("RAW_ROOT_PREFIX", "raw/")
        model_container = os.environ.get("MODEL_CONTAINER", "models")
        model_blob = os.environ.get("MODEL_BLOB", "model.joblib")
        out_container = os.environ.get("OUT_CONTAINER", "predictions")
        out_root = os.environ.get("OUT_ROOT_PREFIX", "scores/")

        mode = req.params.get("mode", "latest").strip().lower()
        prefix = req.params.get("prefix", "").strip()

        threshold = float(req.params.get("threshold", os.environ.get("PRED_THRESHOLD", "0.20")))
        max_blobs_q = req.params.get("max_blobs") or os.environ.get("MAX_SCORE_BLOBS", "")
        max_blobs = int(max_blobs_q) if str(max_blobs_q).strip() else None

        bsc = _bsc_from_conn()
        cc = bsc.get_container_client(in_container)

        # Determine input prefix
        if prefix:
            chosen_prefix = _normalize_prefix(prefix)
        else:
            if mode != "latest":
                return func.HttpResponse(
                    "Missing prefix. Use ?prefix=raw/YYYY/MM/DD/HH/ or set mode=latest",
                    status_code=400,
                )
            chosen_prefix = _list_latest_hour_prefix(cc, raw_root)

        # Load data
        t0 = time.time()
        raw_df, blobs_used = _load_jsonl_from_prefix(cc, chosen_prefix, max_blobs=max_blobs)
        rows_loaded = int(len(raw_df))

        # Encode
        encoded = _encode_features_report(raw_df)
        # Align original rows with encoded rows (after dropna)
        out_cols = FEATURES + [LABEL]  # keep report cols + class
        for tcol in ["EventEnqueuedUtcTime", "EventProcessedUtcTime"]:
            if tcol in raw_df.columns:
                out_cols.append(tcol)

        out_rows = raw_df.loc[encoded.index, [c for c in out_cols if c in raw_df.columns]].copy()

        X = encoded[FEATURES].astype(float)

        # Load model from Blob into /tmp and score
        model_path = _download_model_to_tmp(bsc, model_container, model_blob)
        pipe = joblib.load(model_path)
        if not hasattr(pipe, "predict_proba"):
            raise RuntimeError("Model does not support predict_proba. Expected RF pipeline.")

        y_prob = pipe.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        out_rows["y_prob"] = y_prob
        out_rows["y_pred"] = y_pred
        out_rows["threshold"] = threshold
        out_rows["scored_at_utc"] = _now_utc_iso()

        # Write predictions
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_root = _normalize_prefix(out_root)
        chosen_prefix_norm = _normalize_prefix(chosen_prefix)
        out_blob = f"{out_root}{chosen_prefix_norm}preds_{ts}.csv".replace("//", "/")

        _upload_csv(bsc, out_container, out_blob, out_rows)

        # Build report-aligned HTTP response
        resp: Dict[str, Any] = {
            "input": {
                "container": in_container,
                "prefix": chosen_prefix_norm,
                "raw_root_prefix": _normalize_prefix(raw_root),
                "blobs_used": int(blobs_used),
                "rows_loaded": rows_loaded,
                "rows_scored": int(len(out_rows)),
                "max_blobs": max_blobs,
            },
            "model": {
                "container": model_container,
                "blob": model_blob,
                "features": FEATURES,
                "encoding": {"MAP_ABCD": MAP_ABCD, "MAP_SHIFT": MAP_SHIFT},
                "threshold": threshold,
            },
            "predictions": {
                "predicted_positive": int((y_pred == 1).sum()),
                "prob_min": float(y_prob.min()) if len(y_prob) else None,
                "prob_mean": float(y_prob.mean()) if len(y_prob) else None,
                "prob_max": float(y_prob.max()) if len(y_prob) else None,
            },
            "output": {
                "container": out_container,
                "blob": out_blob,
            },
            "timing": {
                "elapsed_sec": round(time.time() - t0, 4),
                "scored_at_utc": out_rows["scored_at_utc"].iloc[0] if len(out_rows) else _now_utc_iso(),
            },
        }

        # Optional: quick eval if class exists and fully valid
        if LABEL in out_rows.columns:
            try:
                y_true = pd.to_numeric(out_rows[LABEL], errors="coerce").astype("Int64")
                if y_true.notna().all():
                    report = classification_report(
                        y_true.astype(int),
                        y_pred,
                        output_dict=True,
                        zero_division=0,
                    )
                    auc = roc_auc_score(y_true.astype(int), y_prob)

                    resp["eval"] = {
                        "roc_auc": float(auc),
                        "accuracy": float(report["accuracy"]),
                        "positive_class_1": {
                            "precision": float(report["1"]["precision"]),
                            "recall": float(report["1"]["recall"]),
                            "f1": float(report["1"]["f1-score"]),
                            "support": int(report["1"]["support"]),
                        },
                        "macro_avg_f1": float(report["macro avg"]["f1-score"]),
                        "weighted_avg_f1": float(report["weighted avg"]["f1-score"]),
                    }
            except Exception as e:
                resp["eval_error"] = str(e)

        return func.HttpResponse(
            json.dumps(resp, indent=2),
            status_code=200,
            mimetype="application/json",
        )

    except Exception as e:
        logging.exception("Score failed")
        return func.HttpResponse(
            json.dumps({"error": f"{type(e).__name__}: {e}"}),
            status_code=500,
            mimetype="application/json",
        )
