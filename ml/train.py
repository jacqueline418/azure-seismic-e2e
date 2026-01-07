# # ml/train.py
import os
import io
import json
import time
from datetime import datetime, timezone
from collections import Counter
from typing import Optional, List
from datetime import datetime, timezone, timedelta

import joblib
import pandas as pd
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ----------------------------
# Report-aligned feature spec
# ----------------------------
CATEGORICAL = ["seismic", "seismoacoustic", "shift", "ghazard"]
NUMERIC = [
    "genergy", "gpuls", "gdenergy", "gdpuls",
    "nbumps", "nbumps2", "nbumps3", "nbumps4", "nbumps5", "nbumps6", "nbumps7", "nbumps89",
    "energy", "maxenergy"
]
LABEL = "class"

# Report-aligned label encoding
MAP_ABCD = {"a": 1, "b": 2, "c": 3, "d": 4}
MAP_SHIFT = {"W": 1, "N": 0}

REQUIRED = CATEGORICAL + NUMERIC + [LABEL]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _normalize_prefix(prefix: str) -> str:
    # ensure it ends with '/'
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

def find_latest_hour_prefix(conn_str: str, container: str, base_prefix: str, lookback_hours: int = 48) -> str:
    """
    Find the most recent hour-partition folder that has at least 1 blob.
    Assumes your ASA path pattern is: <base_prefix>/{date}/{time}
    where date format is YYYY/MM/DD and time format is HH.
    So the generated folder is: raw/YYYY/MM/DD/HH/
    """
    base_prefix = _normalize_prefix(base_prefix)

    bsc = BlobServiceClient.from_connection_string(conn_str)
    cc = bsc.get_container_client(container)

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    for i in range(0, lookback_hours + 1):
        dt = now - timedelta(hours=i)
        prefix = f"{base_prefix}{dt.strftime('%Y/%m/%d/%H')}/"

        # Try to see if there is at least 1 blob under this prefix
        it = cc.list_blobs(name_starts_with=prefix)
        try:
            next(it)
            return prefix
        except StopIteration:
            continue

    raise FileNotFoundError(
        f"Could not find any blobs under {container}/{base_prefix} in the last {lookback_hours} hours."
    )

def apply_report_encoding(df: pd.DataFrame) -> pd.DataFrame:
    # Enforce required columns exist
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED].copy()

    # Label encoding per report
    df["seismic"] = df["seismic"].map(MAP_ABCD)
    df["seismoacoustic"] = df["seismoacoustic"].map(MAP_ABCD)
    df["ghazard"] = df["ghazard"].map(MAP_ABCD)
    df["shift"] = df["shift"].map(MAP_SHIFT)

    if df[CATEGORICAL].isna().any().any():
        bad = {}
        for col in ["seismic", "seismoacoustic", "ghazard"]:
            bad_vals = sorted(set(df.loc[df[col].isna(), col].tolist()))
            # Above line isn't useful after mapping (NaN), so instead inspect original by re-reading
        raise ValueError(
            "Found unexpected categorical values (not in a/b/c/d or W/N). "
            "Check your input data for typos or new categories."
        )

    # Cast numeric + label
    for c in CATEGORICAL + NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[LABEL] = pd.to_numeric(df[LABEL], errors="coerce")

    before = len(df)
    df = df.dropna()
    after = len(df)
    return df


def main():
    load_dotenv()  # reads ml/.env if present

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise RuntimeError(
            "Missing AZURE_STORAGE_CONNECTION_STRING. "
            "Set it via export or put it in ml/.env (do NOT commit secrets)."
        )

    # ASA output container (where your RawOutput writes)
    container = os.getenv("ASA_CONTAINER", "stream-output")

    # Root prefix for full raw output (NOT alerts)
    base_prefix = os.getenv("ASA_BASE_PREFIX", "raw/")

    lookback_hours = int(os.getenv("ASA_LOOKBACK_HOURS", "48"))

    # If you still set RAW_ROOT_PREFIX manually, we respect it; otherwise auto-pick latest.
    prefix = os.getenv("RAW_ROOT_PREFIX", "").strip()
    if prefix:
        prefix = _normalize_prefix(prefix)
        log(f"Using explicit RAW_ROOT_PREFIX={prefix}")
    else:
        log(f"Auto-detecting latest hour under base prefix '{base_prefix}' (lookback {lookback_hours}h)...")
        prefix = find_latest_hour_prefix(conn_str, container, base_prefix, lookback_hours=lookback_hours)
        log(f"Detected latest raw prefix: {prefix}")

    out_model = os.getenv("MODEL_OUT", "ml/model.joblib")
    out_meta = os.getenv("MODEL_META_OUT", "ml/model_meta.json")

    max_blobs_env = os.getenv("MAX_TRAIN_BLOBS")
    max_blobs = int(max_blobs_env) if max_blobs_env else None

    test_size = float(os.getenv("TEST_SIZE", "0.2"))
    seed = int(os.getenv("SEED", "42"))

    log("=== Training started (Report-aligned) ===")
    log(f"IN_CONTAINER   = {container}")
    log(f"RAW_ROOT_PREFIX      = {prefix}")
    log(f"MODEL_OUT       = {out_model}")
    if max_blobs is not None:
        log(f"MAX_TRAIN_BLOBS = {max_blobs}")

    # 1) Load
    t0 = time.time()
    raw_df = load_jsonl_from_adls(conn_str, container, prefix, max_blobs=max_blobs)
    log(f"DataFrame shape (raw): {raw_df.shape} (load took {time.time() - t0:.2f}s)")

    # 2) Encode + clean
    t1 = time.time()
    df = apply_report_encoding(raw_df)
    log(f"After encoding+dropna: {df.shape} (took {time.time() - t1:.2f}s)")

    y_counts = Counter(df[LABEL].astype(int).tolist())
    log(f"Label distribution (class): {dict(y_counts)}")

    X = df[CATEGORICAL + NUMERIC].astype(float)
    y = df[LABEL].astype(int)

    log("Building pipeline: SMOTE -> StandardScaler -> RandomForest(best params)")
    pipe = ImbPipeline(steps=[
        ("smote", SMOTE(k_neighbors=5, random_state=seed)),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1
        ))
    ])

    log("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    log(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    log("Fitting model... (this may take a bit)")
    t_fit = time.time()
    pipe.fit(X_train, y_train)
    log(f"Fit complete in {time.time() - t_fit:.2f}s ✅")

    log("Evaluating on test set...")
    y_pred = pipe.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        y_prob = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        log(f"ROC-AUC: {auc:.6f}")

    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    joblib.dump(pipe, out_model)
    log(f"Saved model pipeline to: {out_model}")

    meta = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "container": container,
        "prefix": prefix,
        "features": CATEGORICAL + NUMERIC,
        "label": LABEL,
        "encoding": {"MAP_ABCD": MAP_ABCD, "MAP_SHIFT": MAP_SHIFT},
        "pipeline_order": ["SMOTE", "StandardScaler", "RandomForest"],
        "rf_params": {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": seed
        }
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"Saved model metadata to: {out_meta}")

    log("=== Training finished ===")


if __name__ == "__main__":
    main()
