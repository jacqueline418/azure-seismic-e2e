# # ml/score.py
import os
import io
import json
import time
from datetime import datetime, timezone
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
    prefix = prefix.strip()
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
    # Ensure features exist
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


def main():
    load_dotenv()

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise RuntimeError(
            "Missing AZURE_STORAGE_CONNECTION_STRING. "
            "Set it via export or put it in ml/.env (do NOT commit secrets)."
        )

    in_container = os.getenv("ASA_CONTAINER", "stream-output")
    in_prefix = _normalize_prefix(os.getenv("ASA_PREFIX", "raw/2026/01/04/20/"))

    model_path = os.getenv("MODEL_PATH", "ml/model.joblib")

    # Output location
    out_container = os.getenv("PRED_CONTAINER", in_container)

    # default: predictions/<same prefix>
    # e.g. predictions/raw/2026/01/04/20/
    out_prefix = os.getenv("PRED_PREFIX", f"predictions/{in_prefix}")
    out_prefix = _normalize_prefix(out_prefix)

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

    # 1) Load new stream rows
    raw_df = load_jsonl_from_adls(conn_str, in_container, in_prefix, max_blobs=max_blobs)
    log(f"DataFrame shape (raw): {raw_df.shape}")

    # 2) Encode features exactly like report
    encoded = apply_report_encoding_for_features(raw_df)
    X = encoded[FEATURES].astype(float)

    # Drop rows with any NaNs after coercion
    before = len(X)
    mask_valid = X.notna().all(axis=1)
    X = X[mask_valid]
    kept = len(X)
    dropped = before - kept
    if dropped > 0:
        log(f"Dropped {dropped} rows due to NaNs after coercion/encoding.")

    # Keep aligned original rows for output
    out_rows = raw_df.loc[X.index].copy()

    # 3) Load model + predict probabilities
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

    # Optional quick eval if ground-truth exists in the stream (for your demo/replay)
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

    # 4) Write predictions CSV back to ADLS
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_name = f"{out_prefix}preds_{ts}.csv"

    upload_csv_to_adls(conn_str, out_container, out_name, out_rows)
    log(f"Wrote predictions to: {out_container}/{out_name}")
    log("=== Scoring finished ===")


if __name__ == "__main__":
    main()
