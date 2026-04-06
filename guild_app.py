"""
HaulMark Challenge - Fuel Consumption Prediction
Predicts daily fuel consumption (litres) per dumper using telemetry data.
Uses domain-specific features: haul cycles, dump site stops, refueling events,
grade/slope, and operational behavior patterns.
Outputs Kaggle submission CSV with (id, fuel_consumption).
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import spatial_features # Custom geospatial haul cycle extractor

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TELEMETRY_PATTERN = os.path.join(BASE_DIR, "telemetry_*.parquet")
SUMMARY_FILES = [
    os.path.join(BASE_DIR, "smry_jan_train_ordered.csv"),
    os.path.join(BASE_DIR, "smry_feb_train_ordered.csv"),
    os.path.join(BASE_DIR, "smry_mar_train_ordered.csv"),
]
REFUEL_FILE = os.path.join(BASE_DIR, "rfid_refuels_2026-01-01_2026-03-31.parquet")
ID_MAPPING_FILE = os.path.join(BASE_DIR, "id_mapping_new.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "submission.csv")

# Thresholds for domain-specific feature engineering
DUMP_STOP_MIN_SECONDS = 180   # minimum seconds stationary to count as a dump stop
REFUEL_VOLUME_JUMP = 50       # minimum litres jump to count as a refueling event

DUMP_MODEL_FILE = os.path.join(BASE_DIR, "dump_model.joblib")
if os.path.exists(DUMP_MODEL_FILE):
    dump_model = joblib.load(DUMP_MODEL_FILE)
    print("[INFO] Analog Dump Signal ML Model Loaded Successfully")
else:
    dump_model = None


# ── Step 1: Load Summary / Training Data ───────────────────────────────────────
def load_summary_data():
    """Load and concatenate all summary CSV files (train labels)."""
    dfs = []
    for f in SUMMARY_FILES:
        if os.path.exists(f):
            df = pd.read_csv(f)
            dfs.append(df)
    summary = pd.concat(dfs, ignore_index=True)
    summary["date"] = pd.to_datetime(summary["date"])
    summary["acons"] = pd.to_numeric(summary["acons"], errors="coerce")
    print(f"[INFO] Loaded summary data: {summary.shape[0]} rows, "
          f"vehicles: {summary['vehicle'].nunique()}, "
          f"date range: {summary['date'].min()} to {summary['date'].max()}")
    return summary


# ── Step 2: Load ID Mapping (Test Set) ─────────────────────────────────────────
def load_id_mapping():
    """Load id_mapping.csv which defines the test set to predict."""
    id_map = pd.read_csv(ID_MAPPING_FILE)
    id_map["date"] = pd.to_datetime(id_map["date"])
    print(f"[INFO] ID mapping loaded: {id_map.shape[0]} rows, "
          f"vehicles: {id_map['vehicle'].nunique()}, "
          f"date range: {id_map['date'].min()} to {id_map['date'].max()}")
    return id_map


# ── Step 3: Domain-Specific Telemetry Feature Engineering ──────────────────────
def detect_vehicle_col(df):
    """Detect the correct vehicle column name in telemetry."""
    if "vehicle" in df.columns:
        return "vehicle"
    elif "vehicle_anon" in df.columns:
        return "vehicle_anon"
    else:
        raise ValueError(f"Cannot find vehicle column. Columns: {list(df.columns)}")


def _count_dump_stops(group_df):
    """
    Count dump site stops using the trained LightGBM analog signal dump_model (if available)
    or fallback to the default heuristic.
    """
    if "speed" not in group_df.columns or "ts" not in group_df.columns:
        return 0

    speed = group_df["speed"].values
    ts = group_df["ts"].values
    alt = group_df["altitude"].values if "altitude" in group_df.columns else np.zeros(len(group_df))

    stops = 0
    in_stop = False
    stop_start = None
    start_idx = 0

    for i in range(len(speed)):
        if speed[i] == 0:
            if not in_stop:
                in_stop = True
                stop_start = ts[i]
                start_idx = i
        else:
            if in_stop:
                stop_duration = (ts[i] - stop_start) / np.timedelta64(1, 's')
                if stop_duration >= 60:
                    if dump_model is not None:
                        mean_alt = np.mean(alt[start_idx:i])
                        is_dump = dump_model.predict([[stop_duration, mean_alt]])[0]
                        stops += is_dump
                    else:
                        if stop_duration >= DUMP_STOP_MIN_SECONDS:
                            stops += 1
                in_stop = False
                stop_start = None
                start_idx = 0

    # Check if still in a stop at end
    if in_stop and stop_start is not None:
        stop_duration = (ts[-1] - stop_start) / np.timedelta64(1, 's')
        if stop_duration >= 60:
            if dump_model is not None:
                mean_alt = np.mean(alt[start_idx:])
                is_dump = dump_model.predict([[stop_duration, mean_alt]])[0]
                stops += is_dump
            else:
                if stop_duration >= DUMP_STOP_MIN_SECONDS:
                    stops += 1

    return stops


def _count_refueling_events(group_df):
    """
    Count refueling events from telemetry: detect sudden jumps in fuel_volume.
    A refueling event is when fuel_volume increases by >= REFUEL_VOLUME_JUMP litres.
    """
    if "fuel_volume" not in group_df.columns:
        return 0

    fv = group_df["fuel_volume"].dropna().values
    if len(fv) < 2:
        return 0

    diffs = np.diff(fv)
    return int(np.sum(diffs >= REFUEL_VOLUME_JUMP))


def _compute_total_idle_seconds(group_df):
    """Compute total idle time in seconds (ignition on, speed = 0)."""
    if "ignition" not in group_df.columns or "speed" not in group_df.columns:
        return 0

    mask = (group_df["ignition"] == 1) & (group_df["speed"] == 0)
    idle_rows = group_df[mask]
    if len(idle_rows) < 2:
        return 0

    ts = idle_rows["ts"].values
    if len(ts) < 2:
        return 0

    # Approximate: count rows * average interval
    total_seconds = (ts[-1] - ts[0]) / np.timedelta64(1, 's')
    fraction_idle = len(idle_rows) / max(len(group_df), 1)
    return total_seconds * fraction_idle


def _compute_grade(group_df):
    """
    Compute average absolute grade (slope): altitude change / distance.
    This captures how steep the haul routes are - steeper = more fuel.
    """
    if "altitude" not in group_df.columns or "cumdist" not in group_df.columns:
        return 0.0

    alt = group_df["altitude"].values
    dist = group_df["cumdist"].values

    if len(alt) < 2:
        return 0.0

    alt_diff = np.abs(np.diff(alt.astype(float)))
    dist_diff = np.abs(np.diff(dist.astype(float)))

    # Avoid division by zero
    valid = dist_diff > 0
    if not np.any(valid):
        return 0.0

    grades = alt_diff[valid] / dist_diff[valid]
    return float(np.mean(grades))


def _compute_net_elevation_change(group_df):
    """Net elevation change (end altitude - start altitude) for the day."""
    if "altitude" not in group_df.columns:
        return 0.0
    alt = group_df["altitude"].dropna().values
    if len(alt) < 2:
        return 0.0
    return float(alt[-1] - alt[0])


def _compute_total_elevation_gain(group_df):
    """Total positive elevation gain (sum of all uphill segments)."""
    if "altitude" not in group_df.columns:
        return 0.0
    alt = group_df["altitude"].dropna().values
    if len(alt) < 2:
        return 0.0
    diffs = np.diff(alt.astype(float))
    return float(np.sum(diffs[diffs > 0]))


def compute_telemetry_features(telem_df, vehicle_col):
    """
    Aggregate raw telemetry into daily per-vehicle features.
    Includes domain-specific features: dump stops, refueling, idle time, grade.
    """
    df = telem_df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"])

    ts_naive = df["ts"].dt.tz_localize(None)
    is_march = ts_naive >= pd.Timestamp("2026-03-01")
    ts_ist = ts_naive.copy()
    ts_ist.loc[is_march] = ts_ist.loc[is_march] + pd.Timedelta(hours=5, minutes=30)
    
    h = ts_ist.dt.hour
    df["shift"] = "C"
    df.loc[(h >= 6) & (h < 14), "shift"] = "A"
    df.loc[(h >= 14) & (h < 22), "shift"] = "B"
    
    df["date"] = pd.to_datetime((ts_ist + pd.Timedelta(hours=2)).dt.date)

    # 1. Clean the data (Hints)
    # Remove negative speeds and extreme outliers
    if "speed" in df.columns:
        df = df[(df["speed"] >= 0) & (df["speed"] <= 120)]
    if "altitude" in df.columns:
        df = df[df["altitude"] > -500]

    # Only keep dumpers
    df = df[df[vehicle_col].str.startswith("Dump", na=False)].copy()

    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby([vehicle_col, "date", "shift"])

    features = pd.DataFrame()

    # ── Basic speed features ──
    features["speed_mean"] = grouped["speed"].mean()
    features["speed_max"] = grouped["speed"].max()
    features["speed_std"] = grouped["speed"].std().fillna(0)
    features["speed_median"] = grouped["speed"].median()
    # Fraction of time moving (speed > 0)
    features["moving_fraction"] = grouped["speed"].apply(lambda x: (x > 0).mean())

    # ── Altitude / elevation features ──
    features["altitude_mean"] = grouped["altitude"].mean()
    features["altitude_max"] = grouped["altitude"].max()
    features["altitude_min"] = grouped["altitude"].min()
    features["altitude_range"] = features["altitude_max"] - features["altitude_min"]
    features["altitude_std"] = grouped["altitude"].std().fillna(0)

    # ── Domain: Grade/slope (altitude change per unit distance) ──
    features["avg_grade"] = grouped.apply(_compute_grade)
    features["net_elevation_change"] = grouped.apply(_compute_net_elevation_change)
    features["total_elevation_gain"] = grouped.apply(_compute_total_elevation_gain)

    # ── Heading / angle variability ──
    features["angle_std"] = grouped["angle"].std().fillna(0)

    # ── Distance features ──
    features["cumdist_max"] = grouped["cumdist"].max()
    features["cumdist_min"] = grouped["cumdist"].min()
    features["distance_travelled"] = features["cumdist_max"] - features["cumdist_min"]
    
    # ── Stage 3: Spatial Features & Haul Cycles ──
    print("       Computing Geospatial Haul Cycles...")
    mine_id = df["mine_anon"].iloc[0] if "mine_anon" in df.columns else "mine001"
    loading_gdf, dumping_gdf = spatial_features.get_mine_polys(mine_id, BASE_DIR)
    
    spatial_feats = grouped.apply(lambda g: spatial_features.extract_spatial_features(g, loading_gdf, dumping_gdf))
    for col in ["haul_cycles", "avg_load_time", "avg_dump_time", "avg_cycle_dist"]:
        features[col] = spatial_feats[col]

    # ── Fuel volume features ──
    if "fuel_volume" in df.columns:
        features["fuel_vol_start"] = grouped["fuel_volume"].first()
        features["fuel_vol_end"] = grouped["fuel_volume"].last()
        features["fuel_vol_range"] = features["fuel_vol_start"] - features["fuel_vol_end"]
        features["fuel_vol_mean"] = grouped["fuel_volume"].mean()
        features["fuel_vol_std"] = grouped["fuel_volume"].std().fillna(0)
    else:
        for col in ["fuel_vol_start", "fuel_vol_end", "fuel_vol_range",
                     "fuel_vol_mean", "fuel_vol_std"]:
            features[col] = np.nan

    # ── Ignition features ──
    if "ignition" in df.columns:
        features["ignition_rate"] = grouped["ignition"].mean()
        features["ignition_sum"] = grouped["ignition"].sum()

    # ── Record count (proxy for active time) ──
    features["record_count"] = grouped.size()

    # ── Haversine distance features ──
    if "disthav" in df.columns:
        features["disthav_sum"] = grouped["disthav"].sum()
        features["disthav_mean"] = grouped["disthav"].mean()

    # ── GPS quality ──
    if "satellites" in df.columns:
        features["satellites_mean"] = grouped["satellites"].mean()

    # ── Domain: Dump site stops (video hint: ~3-5 min stationary stops) ──
    print("       Computing dump site stops...")
    features["dump_stop_count"] = grouped.apply(_count_dump_stops)

    # ── Domain: Telemetry-based refueling events ──
    print("       Computing refueling events from telemetry...")
    features["telem_refuel_count"] = grouped.apply(_count_refueling_events)

    # ── Domain: Total idle time (ignition on, speed=0) ──
    print("       Computing idle time...")
    features["idle_seconds"] = grouped.apply(_compute_total_idle_seconds)

    features = features.reset_index()
    features = features.rename(columns={vehicle_col: "vehicle"})

    return features


def load_all_telemetry_features():
    """Process all telemetry parquet files and aggregate features."""
    files = sorted(glob.glob(TELEMETRY_PATTERN))
    print(f"[INFO] Found {len(files)} telemetry files")

    all_features = []
    for i, f in enumerate(files):
        fname = os.path.basename(f)
        print(f"[INFO] Processing ({i+1}/{len(files)}): {fname} ...")
        try:
            telem = pd.read_parquet(f)
            vehicle_col = detect_vehicle_col(telem)
            feat = compute_telemetry_features(telem, vehicle_col)
            if not feat.empty:
                all_features.append(feat)
                print(f"       → {feat.shape[0]} vehicle-day records extracted")
            else:
                print(f"       → No dumper records found")
        except Exception as e:
            print(f"       → ERROR: {e}")

    if not all_features:
        print("[WARN] No telemetry features extracted!")
        return pd.DataFrame()

    combined = pd.concat(all_features, ignore_index=True)

    # Deduplicate overlapping date ranges
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    combined = combined.groupby(["vehicle", "date", "shift"], as_index=False)[numeric_cols].mean()

    print(f"[INFO] Total telemetry features: {combined.shape[0]} rows, "
          f"{combined.shape[1]} columns")
    return combined


# ── Step 4: Load Refuelling Data ───────────────────────────────────────────────
def load_refuel_features():
    """Load refuelling data and compute daily refuel stats per vehicle."""
    if not os.path.exists(REFUEL_FILE):
        print("[WARN] Refuel file not found, skipping refuel features")
        return pd.DataFrame()

    try:
        refuel = pd.read_parquet(REFUEL_FILE)
        print(f"[INFO] Refuel data: {refuel.shape[0]} rows, columns: {list(refuel.columns)}")

        ts_col = "ts" if "ts" in refuel.columns else None
        veh_col = "vehicle" if "vehicle" in refuel.columns else None
        litres_col = "litres" if "litres" in refuel.columns else None

        if "date_dpr" in refuel.columns and "shift_dpr" in refuel.columns:
            refuel["date"] = pd.to_datetime(refuel["date_dpr"])
            refuel["shift"] = refuel["shift_dpr"]

            agg_dict = {ts_col: ("size", "size")} if ts_col else {"shift_dpr": ("size", "size")}
            rename_cols = ["refuel_count"]
            
            if litres_col:
                agg_dict["refuel_litres_total"] = (litres_col, "sum")
                agg_dict["refuel_litres_mean"] = (litres_col, "mean")
                rename_cols.extend(["refuel_litres_total", "refuel_litres_mean"])

            # Pass agg_dict directly without kwargs expansion to avoid tuple issues if using Pandas 0.25+ syntax
            refuel_agg = refuel.groupby([veh_col, "date", "shift"]).agg(**agg_dict).reset_index()
            # Rename the first aggregated col to refuel_count
            refuel_agg.rename(columns={list(agg_dict.keys())[0]: "refuel_count"}, inplace=True)
            print(f"[INFO] Refuel features: {refuel_agg.shape[0]} rows")
            return refuel_agg
        else:
            print(f"[WARN] Could not identify date/shift columns in refuel data")
            return pd.DataFrame()
    except Exception as e:
        print(f"[WARN] Error loading refuel data: {e}")
        return pd.DataFrame()


# ── Step 5: Build Feature Matrix ──────────────────────────────────────────────
def build_features(data_df, telem_features, refuel_features):
    """
    Merge data with telemetry and refuel features.
    """
    merged = data_df.merge(telem_features, on=["vehicle", "date", "shift"], how="left")

    if not refuel_features.empty:
        merged = merged.merge(refuel_features, on=["vehicle", "date", "shift"], how="left")
        for col in ["refuel_count", "refuel_litres_total", "refuel_litres_mean"]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)

    # Time features
    merged["day_of_week"] = merged["date"].dt.dayofweek
    merged["day_of_month"] = merged["date"].dt.day
    merged["month"] = merged["date"].dt.month
    merged["is_weekend"] = (merged["day_of_week"] >= 5).astype(int)

    # Shift feature
    if "shift" in merged.columns:
        merged["shift_code"] = merged["shift"].astype("category").cat.codes

    # Encode vehicle as category
    merged["vehicle_code"] = merged["vehicle"].astype("category").cat.codes

    return merged


def get_feature_cols(df):
    """Get feature column names."""
    drop_cols = {"vehicle", "date", "acons", "id", "shift", "mine", "lph", "initlev", "endlev", "arefill", "runhrs", "mine_anon", "fleet_type"}
    return [c for c in df.columns if c not in drop_cols]


# ── Step 6: Train Model ──────────────────────────────────────────────────────
def train_model(X_train, y_train, feature_cols):
    """Train LightGBM model with cross-validation."""

    valid_mask = y_train.notna() & (y_train >= 0)
    X_tr = X_train[valid_mask].copy()
    y_tr = y_train[valid_mask].copy()

    print(f"[INFO] Valid training samples: {X_tr.shape[0]} / {X_train.shape[0]}")

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    fold_scores = []

    print("\n[INFO] Training with 5-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr)):
        X_fold_tr, X_fold_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
        y_fold_tr, y_fold_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]

        train_data = lgb.Dataset(X_fold_tr, label=y_fold_tr)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=0),
            ],
        )

        preds = np.maximum(model.predict(X_fold_val), 0)
        rmse = root_mean_squared_error(y_fold_val, preds)
        fold_scores.append(rmse)
        models.append(model)
        print(f"       Fold {fold+1}: RMSE = {rmse:.4f} (best_iter={model.best_iteration})")

    mean_rmse = np.mean(fold_scores)
    print(f"\n[RESULT] Cross-validation RMSE: {mean_rmse:.4f} "
          f"(± {np.std(fold_scores):.4f})")

    # Feature importances
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": np.mean([m.feature_importance(importance_type="gain")
                               for m in models], axis=0)
    }).sort_values("importance", ascending=False)
    print("\n[INFO] Top 20 Feature Importances (gain):")
    print(importance.head(20).to_string(index=False))

    return models, mean_rmse


# ── Step 7: Predict & Generate Submission ─────────────────────────────────────
def generate_submission(models, test_df, feature_cols, id_map):
    """Generate predictions for test set and write submission CSV."""
    if "refuel_litres_total" not in test_df.columns:
        test_df["refuel_litres_total"] = 0
    test_df["fuel_vol_start"] = test_df["fuel_vol_start"].fillna(1076.92)
    test_df["fuel_vol_end"] = test_df["fuel_vol_end"].fillna(1076.92)
    test_df["fuel_balance"] = np.maximum(test_df["fuel_vol_start"] + test_df["refuel_litres_total"] - test_df["fuel_vol_end"], 0)

    X_test = test_df[feature_cols].fillna(-1)

    # Ensemble: average predictions across folds
    preds = np.mean([m.predict(X_test) for m in models], axis=0)
    preds = np.maximum(preds, 0)

    # Build submission
    submission = pd.DataFrame({
        "id": test_df["id"].astype(int),
        "fuel_consumption": preds,
    })
    submission = submission.sort_values("id").reset_index(drop=True)

    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[INFO] Submission saved to: {OUTPUT_FILE}")
    print(f"[INFO] Submission shape: {submission.shape}")
    print(f"\n[INFO] Sample predictions:")
    print(submission.head(10).to_string(index=False))
    print(f"\n[INFO] Prediction stats:")
    print(f"       Mean: {preds.mean():.2f} litres")
    print(f"       Median: {np.median(preds):.2f} litres")
    print(f"       Min: {preds.min():.2f}, Max: {preds.max():.2f}")

    return submission


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("HaulMark Challenge — Fuel Consumption Prediction")
    print("  (with domain-specific features: dump stops, refueling,")
    print("   grade/slope, idle time, elevation gain)")
    print("=" * 70)

    # Step 1: Load training labels
    print("\n" + "─" * 40)
    print("Step 1: Loading Summary Data")
    print("─" * 40)
    summary_data = load_summary_data()
    print(f"[INFO] Summary data rows: {summary_data.shape[0]}")

    # Step 2: Load test set ID mapping
    print("\n" + "─" * 40)
    print("Step 2: Loading Test Set (ID Mapping)")
    print("─" * 40)
    id_map = load_id_mapping()

    # Step 3: Extract telemetry features (with domain-specific features)
    print("\n" + "─" * 40)
    print("Step 3: Extracting Telemetry Features")
    print("  (includes: dump stops, refueling detection,")
    print("   grade/slope, idle time, elevation gain)")
    print("─" * 40)
    telem_features = load_all_telemetry_features()

    # Step 4: Load refuel data
    print("\n" + "─" * 40)
    print("Step 4: Loading Refuel Features")
    print("─" * 40)
    refuel_features = load_refuel_features()

    # Step 5: Prepare training data
    print("\n" + "─" * 40)
    print("Step 5: Preparing Training Data")
    print("─" * 40)
    train_merged = build_features(summary_data, telem_features, refuel_features)
    # Apply engineering to train_merged
    if "refuel_litres_total" not in train_merged.columns:
        train_merged["refuel_litres_total"] = 0
    train_merged["fuel_vol_start"] = train_merged["fuel_vol_start"].fillna(1076.92)
    train_merged["fuel_vol_end"] = train_merged["fuel_vol_end"].fillna(1076.92)
    train_merged["fuel_balance"] = np.maximum(train_merged["fuel_vol_start"] + train_merged["refuel_litres_total"] - train_merged["fuel_vol_end"], 0)
    
    feature_cols = get_feature_cols(train_merged)
    X_train = train_merged[feature_cols].fillna(-1)
    y_train = train_merged["acons"]

    print(f"[INFO] Training data: X={X_train.shape}, y={y_train.shape}")
    print(f"[INFO] Features ({len(feature_cols)}):")
    for i, f in enumerate(feature_cols):
        print(f"       {i+1:2d}. {f}")

    # Step 6: Train model
    print("\n" + "─" * 40)
    print("Step 6: Training LightGBM Model")
    print("─" * 40)
    models, rmse = train_model(X_train, y_train, feature_cols)

    # Step 7: Prepare test data and generate submission
    print("\n" + "─" * 40)
    print("Step 7: Generating Submission")
    print("─" * 40)

    test_df = id_map.copy()
    test_df = test_df.merge(telem_features, on=["vehicle", "date", "shift"], how="left")

    if not refuel_features.empty:
        test_df = test_df.merge(refuel_features, on=["vehicle", "date", "shift"], how="left")
        for col in ["refuel_count", "refuel_litres_total", "refuel_litres_mean"]:
            if col in test_df.columns:
                test_df[col] = test_df[col].fillna(0)

    # Time and shift features
    test_df["day_of_week"] = test_df["date"].dt.dayofweek
    test_df["day_of_month"] = test_df["date"].dt.day
    test_df["month"] = test_df["date"].dt.month
    test_df["is_weekend"] = (test_df["day_of_week"] >= 5).astype(int)
    test_df["vehicle_code"] = test_df["vehicle"].astype("category").cat.codes
    if "shift" in test_df.columns:
        test_df["shift_code"] = test_df["shift"].astype("category").cat.codes

    # Add missing training-only columns
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0
            print(f"       [NOTE] Added missing feature '{col}' with default 0")

    telem_matched = test_df["speed_mean"].notna().sum() if "speed_mean" in test_df.columns else 0
    print(f"[INFO] Test rows with telemetry: {telem_matched}/{len(test_df)}")

    submission = generate_submission(models, test_df, feature_cols, id_map)

    print("\n" + "=" * 70)
    print(f"DONE! Final CV RMSE: {rmse:.4f}")
    print(f"Submission file: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
