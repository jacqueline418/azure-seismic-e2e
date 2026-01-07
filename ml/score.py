# ml/score.py
import os
import io
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import joblib
import pandas as pd
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient


# Must match train.py
CATEGORICAL = ["seismic", "seismoacoustic", "shift", "ghazard"]
NUMERIC = [
    "genergy", "gpuls", "gdenergy", "gdpuls",
    "nbumps", "nbumps2", "nbumps3", "nbumps4", "nbumps5", "nbumps6", "nbumps7", "nbumps89",
    "energy", "maxenergy"
]
LABEL = "class"

MAP_ABCD = {"a": 1, "b": 2, "c": 3, "d": 4}
MAP_SHIFT = {"W": 1, "N": 0}

FEATURES = CATEGORICAL + NUMERIC


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _normalize_prefix(prefix: str) -> str:
    prefix = (prefix or "").strip()
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return prefix


def load_jsonl_from_adls(conn_str: str, container: str, prefix: str, max_blobs: Optional[int] = None) -> pd.DataFrame:
    prefix = _normalize_prefix(prefix)
    log(f"Connecting to ADLS container='{container}', prefix='{prefix}'")

    bsc = BlobServiceClient.from_connection_string(conn_str)
    cc = bsc.get_container_client(container)

    blobs = list(cc.list_blobs(name_starts_with=prefix))
    if max_blobs is not None:
        blobs = blobs[:max_blobs]

    log(f"Found {len(blobs)} blobs under prefix.")
    if not blobs:
        raise FileNotFoundError(f"No blobs found under {container}/{prefix}")

    rows = []
    t0 = time.time()
    for i, b in enumerate(blobs, 1):
        if i == 1 or i % 10 == 0 or i == len(blobs):
            log(f"Downloading blob {i}/{len(blobs)}: {b.name}")

        data = cc.get_blob_client(b.name).download_blob().readall().decode("utf-8", errors="ignore")
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    log(f"Loaded {len(rows):,} JSON rows from {len(blobs)} blobs in {time.time() - t0:.2f}s")
    return pd.DataFrame(rows)


def apply_report_encoding_for_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    out = df.copy()

    out["seismic"] = out["seismic"].map(MAP_ABCD)
    out["seismoacoustic"] = out["seismoacoustic"].map(MAP_ABCD)
    out["ghazard"] = out["ghazard"].map(MAP_ABCD)
    out["shift"] = out["shift"].map(MAP_SHIFT)

    if out[CATEGORICAL].isna().any().any():
        raise ValueError(
            "Found unexpected categorical values (not in a/b/c/d or W/N). "
            "Check incoming stream schema/values."
        )

    for c in FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def upload_csv_to_adls(conn_str: str, container: str, blob_name: str, df: pd.DataFrame) -> None:
    bsc = BlobServiceClient.from_connection_string(conn_str)
    cc = bsc.get_container_client(container)

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    data = buf.getvalue().encode("utf-8")

    cc.upload_blob(name=blob_name, data=data, overwrite=True)


def find_latest_hour_prefix(conn_str: str, container: str, base_prefix: str, lookback_hours: int = 48) -> str:
    """
    Find the most recent hour-partition folder that has at least 1 blob.
    Assumes ASA path pattern: <base_prefix>/{date}/{time}
    => raw/YYYY/MM/DD/HH/
    """
    base_prefix = _normalize_prefix(base_prefix)

    bsc = BlobServiceClient.from_connection_string(conn_str)
    cc = bsc.get_container_client(container)

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    for i in range(0, lookback_hours + 1):
        dt = now - timedelta(hours=i)
        prefix = f"{base_prefix}{dt.strftime('%Y/%m/%d/%H')}/"

        it = cc.list_blobs(name_starts_with=prefix)
        try:
            next(it)
            return prefix
        except StopIteration:
            continue

    raise FileNotFoundError(
        f"Could not find any blobs under {container}/{base_prefix} in the last {lookback_hours} hours."
    )


def main():
    load_dotenv()

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise RuntimeError(
            "Missing AZURE_STORAGE_CONNECTION_STRING. "
            "Set it via export or put it in ml/.env (do NOT commit secrets)."
        )

    # ---- INPUT: Raw ASA output container/prefix ----
    in_container = os.getenv("IN_CONTAINER", "stream-output")
    base_prefix = os.getenv("RAW_ROOT_PREFIX", "raw/")
    lookback_hours = int(os.getenv("ASA_LOOKBACK_HOURS", "48"))

    # optional override (manual)
    in_prefix = os.getenv("RAW_ROOT_PREFIX", "").strip()
    if in_prefix:
        in_prefix = _normalize_prefix(in_prefix)
        log(f"Using explicit RAW_ROOT_PREFIX={in_prefix}")
    else:
        log(f"Auto-detecting latest hour under '{base_prefix}' (lookback {lookback_hours}h)...")
        in_prefix = find_latest_hour_prefix(conn_str, in_container, base_prefix, lookback_hours=lookback_hours)
        log(f"Detected latest raw prefix: {in_prefix}")

    # ---- MODEL ----
    model_path = os.getenv("MODEL_PATH", "ml/model.joblib")

    # ---- OUTPUT: predictions go to their own container ----
    out_container = os.getenv("OUT_CONTAINER", "predictions")

    # make output paths clean (not nested under "predictions/" inside stream-output)
    # We'll mirror the hour folder and write: raw_scored/YYYY/MM/DD/HH/preds_xxx.csv
    pred_base = os.getenv("OUT_ROOT_PREFIX", "scores/")
    pred_base = _normalize_prefix(pred_base)

    # derive hour folder from in_prefix like raw/YYYY/MM/DD/HH/
    # strip base_prefix from in_prefix if possible
    hour_suffix = in_prefix
    norm_base = _normalize_prefix(base_prefix)
    if hour_suffix.startswith(norm_base):
        hour_suffix = hour_suffix[len(norm_base):]  # -> YYYY/MM/DD/HH/
    hour_suffix = _normalize_prefix(hour_suffix)

    out_prefix = _normalize_prefix(os.getenv("PRED_PREFIX", f"{pred_base}{hour_suffix}"))

    threshold = float(os.getenv("PRED_THRESHOLD", "0.20"))

    max_blobs_env = os.getenv("MAX_SCORE_BLOBS")
    max_blobs = int(max_blobs_env) if max_blobs_env else None

    log("=== Scoring started (Report-aligned) ===")
    log(f"INPUT  container = {in_container}")
    log(f"INPUT  prefix    = {in_prefix}")
    log(f"MODEL  path      = {model_path}")
    log(f"OUTPUT container = {out_container}")
    log(f"OUTPUT prefix    = {out_prefix}")
    log(f"THRESHOLD        = {threshold}")
    if max_blobs is not None:
        log(f"MAX_SCORE_BLOBS  = {max_blobs}")

    # 1) Load latest raw rows
    raw_df = load_jsonl_from_adls(conn_str, in_container, in_prefix, max_blobs=max_blobs)
    log(f"DataFrame shape (raw): {raw_df.shape}")

    # 2) Encode features like report
    encoded = apply_report_encoding_for_features(raw_df)
    X = encoded[FEATURES].astype(float)

    # Drop rows with any NaNs after coercion
    before = len(X)
    mask_valid = X.notna().all(axis=1)
    X = X[mask_valid]
    dropped = before - len(X)
    if dropped > 0:
        log(f"Dropped {dropped} rows due to NaNs after coercion/encoding.")

    out_rows = raw_df.loc[X.index].copy()

    # 3) Load model + predict
    pipe = joblib.load(model_path)
    if not hasattr(pipe, "predict_proba"):
        raise RuntimeError("Loaded model does not support predict_proba. Expected RF pipeline.")

    y_prob = pipe.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    out_rows["y_prob"] = y_prob
    out_rows["y_pred"] = y_pred
    out_rows["threshold"] = threshold
    out_rows["scored_at_utc"] = datetime.now(timezone.utc).isoformat()

    pos = int((y_pred == 1).sum())
    log(f"Predicted positives @ threshold={threshold}: {pos}/{len(y_pred)}")
    log(f"Prob stats: min={float(y_prob.min()):.4f}, mean={float(y_prob.mean()):.4f}, max={float(y_prob.max()):.4f}")

    # Optional quick eval if class exists (for replay/demo)
    if LABEL in out_rows.columns:
        try:
            y_true = pd.to_numeric(out_rows[LABEL], errors="coerce").astype("Int64")
            if y_true.notna().all():
                from sklearn.metrics import classification_report, roc_auc_score
                print("\n=== Quick Eval (because 'class' exists in stream) ===")
                print(classification_report(y_true.astype(int), y_pred))
                print("ROC-AUC:", roc_auc_score(y_true.astype(int), y_prob))
        except Exception as e:
            log(f"Skipped quick eval due to error: {e}")

    # 4) Write predictions CSV
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_name = f"{out_prefix}preds_{ts}.csv"

    upload_csv_to_adls(conn_str, out_container, out_name, out_rows)
    log(f"Wrote predictions to: {out_container}/{out_name}")
    log("=== Scoring finished ===")


if __name__ == "__main__":
    main()
