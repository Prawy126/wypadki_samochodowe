"""
Preprocessing pipeline for US Accidents dataset.
Usage:
    python run_preprocessing.py
    python run_preprocessing.py --input ../US_Accidents_1M.csv --output-dir data
"""

import os
import json
import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import MiniBatchKMeans
import joblib

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DROP_COLS = [
    'ID', 'Country', 'Turning_Loop', 'Roundabout', 'Description',
    'End_Lat', 'End_Lng', 'Weather_Timestamp', 'Airport_Code',
    'Street', 'Zipcode',
]

REQUIRED_COLS = ['Severity', 'Start_Lat', 'Start_Lng', 'Start_Time', 'End_Time']

# Numeric columns to impute & scale
NUM_COLS_BASE = [
    'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
    'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
    'Distance(mi)', 'Duration_min', 'Start_Lat', 'Start_Lng',
    'Dist_from_center', 'Hour_sin', 'Hour_cos',
    'DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin', 'Month_cos',
    'Year',
]

LOG_TRANSFORM_COLS = ['Distance(mi)', 'Precipitation(in)', 'Wind_Speed(mph)', 'Duration_min']

HIGH_MISSING = ['Wind_Chill(F)', 'Precipitation(in)', 'Wind_Speed(mph)']

CAT_COLS = [
    'Source', 'State', 'City', 'County', 'Timezone',
    'Wind_Direction', 'Weather_Category', 'Geo_Cluster',
]

BOOL_COLS = [
    'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
    'No_Exit', 'Railway', 'Station', 'Stop', 'Traffic_Calming',
    'Traffic_Signal',
]

TWILIGHT_COLS = [
    'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight',
]

WIND_DIR_MAP = {
    'Calm': 'CALM', 'CALM': 'CALM',
    'Variable': 'VAR', 'VAR': 'VAR',
    'South': 'S', 'S': 'S', 'North': 'N', 'N': 'N',
    'West': 'W', 'W': 'W', 'East': 'E', 'E': 'E',
    'SSW': 'SSW', 'WNW': 'WNW', 'SW': 'SW', 'NW': 'NW',
    'SSE': 'SSE', 'WSW': 'WSW', 'NNW': 'NNW', 'SE': 'SE',
    'ESE': 'ESE', 'NE': 'NE', 'ENE': 'ENE', 'NNE': 'NNE',
}

WEATHER_MAP = {
    'Fair': 'Clear', 'Clear': 'Clear', 'Fair / Windy': 'Clear',
    'Mostly Cloudy': 'Cloudy', 'Cloudy': 'Cloudy', 'Partly Cloudy': 'Cloudy',
    'Overcast': 'Cloudy', 'Scattered Clouds': 'Cloudy',
    'Mostly Cloudy / Windy': 'Cloudy', 'Cloudy / Windy': 'Cloudy',
    'Partly Cloudy / Windy': 'Cloudy',
    'Light Rain': 'Rain', 'Rain': 'Rain', 'Heavy Rain': 'Rain',
    'Light Drizzle': 'Rain', 'Drizzle': 'Rain', 'Heavy Rain / Windy': 'Rain',
    'Light Rain / Windy': 'Rain', 'Rain / Windy': 'Rain',
    'Light Snow': 'Snow', 'Snow': 'Snow', 'Heavy Snow': 'Snow',
    'Light Snow / Windy': 'Snow', 'Snow / Windy': 'Snow',
    'Fog': 'Fog', 'Haze': 'Fog', 'Mist': 'Fog', 'Smoke': 'Fog',
    'Patches of Fog': 'Fog', 'Shallow Fog': 'Fog',
    'T-Storm': 'Storm', 'Thunder in the Vicinity': 'Storm', 'Thunder': 'Storm',
    'Heavy T-Storm': 'Storm', 'Light Thunderstorms and Rain': 'Storm',
    'Heavy T-Storm / Windy': 'Storm', 'T-Storm / Windy': 'Storm',
}

CAPPING_RULES = {
    'Temperature(F)':    (-40.0, 120.0),
    'Wind_Speed(mph)':   (0.0,   60.0),
    'Precipitation(in)': (0.0,    3.0),
    'Visibility(mi)':    (0.0,   10.0),
    'Pressure(in)':      (27.0,  32.0),
    'Humidity(%)':        (0.0, 100.0),
    'Distance(mi)':      (0.0,   30.0),
    'Duration_min':      (0.0, 1440.0),  # max 24h
}

US_CENTER_LAT = 39.8283
US_CENTER_LNG = -98.5795


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def haversine_np(lon1, lat1, lon2, lat2):
    """Great-circle distance in miles."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * np.arcsin(np.sqrt(a)) * 3956


def validate_columns(df: pd.DataFrame) -> None:
    """Raise if any required column is absent."""
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")


def save_dataframe_with_fallback(df: pd.DataFrame, parquet_path: str) -> str:
    """Save to Parquet when possible; otherwise fall back to compressed CSV.

    Returns the actual file path used so the caller can log it in metadata or
    diagnostics if needed.
    """
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except ImportError:
        csv_gz_path = os.path.splitext(parquet_path)[0] + ".csv.gz"
        logger.warning(
            "Brak silnika Parquet (`pyarrow`/`fastparquet`) — zapisuję zamiast tego: %s",
            csv_gz_path,
        )
        df.to_csv(csv_gz_path, index=False, compression="gzip")
        return csv_gz_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(input_csv: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1 — Load raw data
    # ------------------------------------------------------------------
    logger.info("--- STEP 1: Loading raw data ---")
    df = pd.read_csv(input_csv)
    N_TOTAL = len(df)
    logger.info(f"Loaded dataset shape: {df.shape}")
    validate_columns(df)

    # ------------------------------------------------------------------
    # STEP 2 — Drop useless columns & rows without target/coords
    # ------------------------------------------------------------------
    logger.info("--- STEP 2: Cleaning and Initial Dropping ---")
    df = df.drop(columns=[col for col in DROP_COLS if col in df.columns])
    df = df.dropna(subset=['Severity', 'Start_Lat', 'Start_Lng'])
    df['Severity'] = df['Severity'].astype(int) - 1  # map 1-4 -> 0-3
    logger.info(f"Shape after initial row and column drops: {df.shape}")

    # ------------------------------------------------------------------
    # STEP 3 — Temporal Feature Engineering
    # ------------------------------------------------------------------
    logger.info("--- STEP 3: Temporal Feature Engineering ---")
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time']   = pd.to_datetime(df['End_Time'],   errors='coerce')

    # Drop rows where start or end time is invalid
    df = df.dropna(subset=['Start_Time', 'End_Time'])

    # Duration in minutes
    df['Duration_min'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60.0
    df = df[df['Duration_min'] >= 0]  # drop negative durations (data noise)

    # Extract temporal components
    df['Hour']      = df['Start_Time'].dt.hour
    df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
    df['Month']     = df['Start_Time'].dt.month
    df['Year']      = df['Start_Time'].dt.year
    df['IsWeekend']  = (df['DayOfWeek'] >= 5).astype(int)
    df['IsRushHour'] = df['Hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    # Cyclic encoding for time features
    df['Hour_sin']      = np.sin(2 * np.pi * df['Hour']      / 24.0)
    df['Hour_cos']      = np.cos(2 * np.pi * df['Hour']      / 24.0)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7.0)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7.0)
    df['Month_sin']     = np.sin(2 * np.pi * df['Month']     / 12.0)
    df['Month_cos']     = np.cos(2 * np.pi * df['Month']     / 12.0)

    # Drop raw temporal + datetime columns (cyclic versions replace them)
    df = df.drop(columns=['Start_Time', 'End_Time', 'Hour', 'DayOfWeek', 'Month'])

    # Data loss check
    rows_lost = N_TOTAL - len(df)
    pct_lost  = rows_lost / N_TOTAL * 100
    logger.info(f"Shape after temporal engineering: {df.shape}")
    logger.info(f"Rows lost so far: {rows_lost:,} ({pct_lost:.2f}%) — limit 20%: {'OK' if pct_lost <= 20 else 'EXCEEDED!'}")

    # ------------------------------------------------------------------
    # STEP 4 — Geographic Distance & Categorical Cleaning
    # ------------------------------------------------------------------
    logger.info("--- STEP 4: Geographic Distance & Clean Categorical Values ---")
    df['Dist_from_center'] = haversine_np(
        df['Start_Lng'], df['Start_Lat'], US_CENTER_LNG, US_CENTER_LAT
    )

    # Standardize Wind Direction
    df['Wind_Direction'] = df['Wind_Direction'].map(WIND_DIR_MAP).fillna('VAR')

    # Log unmapped weather conditions before grouping
    unmapped = df['Weather_Condition'][~df['Weather_Condition'].isin(WEATHER_MAP)].value_counts()
    if not unmapped.empty:
        logger.warning(f"Unmapped Weather_Condition values (-> 'Other'):\n{unmapped.head(15)}")

    df['Weather_Category'] = df['Weather_Condition'].map(WEATHER_MAP).fillna('Other')
    df = df.drop(columns=['Weather_Condition'])

    # Twilight columns -> binary (0/1)
    for col in TWILIGHT_COLS:
        df[col] = df[col].fillna('Day')
        df[col] = (df[col] == 'Night').astype(int)

    logger.info("Categorical cleaning complete.")

    # ------------------------------------------------------------------
    # STEP 5 — Stratified Train / Val / Test Split (70 / 15 / 15)
    #           Split BEFORE any fit-based transforms to prevent leakage
    # ------------------------------------------------------------------
    logger.info("--- STEP 5: Stratified Train / Val / Test Split ---")
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(sss1.split(df, df['Severity']))
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    temp_df  = df.iloc[temp_idx].copy().reset_index(drop=True)

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(sss2.split(temp_df, temp_df['Severity']))
    val_df  = temp_df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)

    logger.info(f"Train size: {train_df.shape}")
    logger.info(f"Val size:   {val_df.shape}")
    logger.info(f"Test size:  {test_df.shape}")

    # ------------------------------------------------------------------
    # STEP 6 — Geoclustering via K-Means (fit on train only)
    # ------------------------------------------------------------------
    logger.info("--- STEP 6: Geoclustering via K-Means ---")
    kmeans = MiniBatchKMeans(n_clusters=30, random_state=42, batch_size=2048, n_init=3)
    kmeans.fit(train_df[['Start_Lat', 'Start_Lng']])
    train_df['Geo_Cluster'] = kmeans.predict(train_df[['Start_Lat', 'Start_Lng']])
    val_df['Geo_Cluster']   = kmeans.predict(val_df[['Start_Lat', 'Start_Lng']])
    test_df['Geo_Cluster']  = kmeans.predict(test_df[['Start_Lat', 'Start_Lng']])

    # ------------------------------------------------------------------
    # STEP 7 — Missingness Indicators (BEFORE imputation)
    # ------------------------------------------------------------------
    logger.info("--- STEP 7: Numeric Missing Indicators ---")
    for col in HIGH_MISSING:
        for split in (train_df, val_df, test_df):
            split[f'{col}_was_missing'] = split[col].isnull().astype(int)

    # ------------------------------------------------------------------
    # STEP 8 — Outlier Capping (BEFORE imputation so medians are clean)
    # ------------------------------------------------------------------
    logger.info("--- STEP 8: Outlier Handling (Capping / Winsorization) ---")
    for col, (lo, hi) in CAPPING_RULES.items():
        for split in (train_df, val_df, test_df):
            split[col] = split[col].clip(lower=lo, upper=hi)
    logger.info("Outlier capping complete.")

    # ------------------------------------------------------------------
    # STEP 9 — Numeric Imputation (AFTER capping, values from train only)
    # ------------------------------------------------------------------
    logger.info("--- STEP 9: Numeric Imputation ---")
    imputation_values = {}
    for col in NUM_COLS_BASE:
        if col == 'Precipitation(in)':
            imputation_values[col] = 0.0  # domain: no precipitation
        elif col == 'Wind_Chill(F)':
            # FIX: Wind chill without wind ~ ambient temperature
            imputation_values[col] = float(train_df['Temperature(F)'].median())
        else:
            imputation_values[col] = float(train_df[col].median())

    for col in NUM_COLS_BASE:
        val = imputation_values[col]
        for split in (train_df, val_df, test_df):
            split[col] = split[col].fillna(val)

    logger.info("Numeric imputation complete.")

    # ------------------------------------------------------------------
    # STEP 10 — Distribution Transformations (log1p)
    # ------------------------------------------------------------------
    logger.info("--- STEP 10: Distribution Transformations (log1p) ---")
    for col in LOG_TRANSFORM_COLS:
        new_name = f'{col}_log'
        for split in (train_df, val_df, test_df):
            split[new_name] = np.log1p(split[col])
            split.drop(columns=[col], inplace=True)

    # Update NUM_COLS list to match renamed columns
    NUM_COLS = [f'{c}_log' if c in LOG_TRANSFORM_COLS else c for c in NUM_COLS_BASE]
    logger.info("Log1p transformations complete.")

    # Feature schema guardrail: Year must remain numeric, never categorical.
    if "Year" not in NUM_COLS:
        raise RuntimeError("Schema error: 'Year' musi być cechą numeryczną.")
    if "Year" in CAT_COLS:
        raise RuntimeError("Schema error: 'Year' nie może być cechą kategoryczną.")
    if set(NUM_COLS).intersection(CAT_COLS):
        overlap = sorted(set(NUM_COLS).intersection(CAT_COLS))
        raise RuntimeError(f"Schema error: kolumny jednocześnie num i cat: {overlap}")

    # ------------------------------------------------------------------
    # STEP 11 — Scaling Continuous Features (fit on train only)
    # ------------------------------------------------------------------
    logger.info("--- STEP 11: Scaling Continuous Features ---")
    scaler = RobustScaler()
    train_df[NUM_COLS] = scaler.fit_transform(train_df[NUM_COLS])
    val_df[NUM_COLS]   = scaler.transform(val_df[NUM_COLS])
    test_df[NUM_COLS]  = scaler.transform(test_df[NUM_COLS])
    logger.info("Continuous scaling complete.")

    # ------------------------------------------------------------------
    # STEP 12 — Categorical Encoding (integer IDs for nn.Embedding)
    # ------------------------------------------------------------------
    logger.info("--- STEP 12: Categorical Encoding (Entity Embedding Mapping) ---")
    cat_mappings = {}
    categorical_dims = {}
    embedding_dims = {}

    for col in CAT_COLS:
        unique_vals = train_df[col].dropna().unique().tolist()

        # Build mapping: 'Unknown' -> 0, others -> 1..K
        mapping = {'Unknown': 0}
        for v in unique_vals:
            if v != 'Unknown':
                mapping[v] = len(mapping)

        cat_mappings[col] = mapping
        vocab_size = len(mapping)
        categorical_dims[col] = vocab_size
        embed_dim = min(50, (vocab_size + 1) // 2)
        embedding_dims[col] = [vocab_size, embed_dim]

        for split in (train_df, val_df, test_df):
            split[col] = split[col].map(mapping).fillna(0).astype(int)

    logger.info("Categorical mapping complete.")

    # ------------------------------------------------------------------
    # STEP 13 — Binary Feature Setup
    # ------------------------------------------------------------------
    logger.info("--- STEP 13: Binary Feature Setup ---")
    BINARY_COLS = (
        BOOL_COLS
        + TWILIGHT_COLS
        + [f'{col}_was_missing' for col in HIGH_MISSING]
        + ['IsWeekend', 'IsRushHour']
    )

    for col in BINARY_COLS:
        for split in (train_df, val_df, test_df):
            split[col] = split[col].fillna(0).astype(int)

    logger.info("Binary columns setup complete.")

    # ------------------------------------------------------------------
    # STEP 14 — Target & Class Weights
    # ------------------------------------------------------------------
    logger.info("--- STEP 14: Target & Class Weights Setup ---")
    y_train = train_df['Severity'].values
    unique_classes, counts = np.unique(y_train, return_counts=True)
    n_classes = len(unique_classes)
    total_samples = len(y_train)

    class_weights = {
        int(cl): float(total_samples / (n_classes * cnt))
        for cl, cnt in zip(unique_classes, counts)
    }

    logger.info(f"Class frequencies on Train: {dict(zip(unique_classes.tolist(), counts.tolist()))}")
    logger.info(f"Computed Class Weights: {class_weights}")

    # ------------------------------------------------------------------
    # STEP 15 — Assemble & Save
    # ------------------------------------------------------------------
    logger.info("--- STEP 15: Save Preprocessed Datasets and Metadata ---")

    final_cols = ['Severity'] + NUM_COLS + CAT_COLS + BINARY_COLS
    train_final = train_df[final_cols]
    val_final   = val_df[final_cols]
    test_final  = test_df[final_cols]

    # Save datasets. Prefer Parquet, but fall back to csv.gz when no engine is installed.
    saved_train_path = save_dataframe_with_fallback(
        train_final, os.path.join(output_dir, "train_preprocessed.parquet")
    )
    saved_val_path = save_dataframe_with_fallback(
        val_final, os.path.join(output_dir, "val_preprocessed.parquet")
    )
    saved_test_path = save_dataframe_with_fallback(
        test_final, os.path.join(output_dir, "test_preprocessed.parquet")
    )

    # Save sklearn artifacts
    joblib.dump(scaler,            os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(kmeans,            os.path.join(output_dir, "kmeans.joblib"))
    joblib.dump(cat_mappings,      os.path.join(output_dir, "categorical_mappings.joblib"))
    joblib.dump(imputation_values, os.path.join(output_dir, "imputation_values.joblib"))

    metadata = {
        "num_cols":             NUM_COLS,
        "cat_cols":             CAT_COLS,
        "bin_cols":             BINARY_COLS,
        "target_col":           "Severity",
        "categorical_dims":     categorical_dims,
        "embedding_dims":       embedding_dims,
        "class_weights":        class_weights,
        "imputation_values":    imputation_values,
        "capping_rules":        {k: list(v) for k, v in CAPPING_RULES.items()},
        "num_features_count":   len(NUM_COLS),
        "cat_features_count":   len(CAT_COLS),
        "bin_features_count":   len(BINARY_COLS),
        "total_features_count": len(NUM_COLS) + len(CAT_COLS) + len(BINARY_COLS),
        "rows_original":        N_TOTAL,
        "rows_after_cleaning":  len(df),
        "rows_lost_pct":        round(pct_lost, 2),
        "saved_train_path":     saved_train_path,
        "saved_val_path":       saved_val_path,
        "saved_test_path":      saved_test_path,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info("==========================================")
    logger.info("Preprocessing completed successfully!")
    logger.info(f"Train dimensions: {train_final.shape}")
    logger.info(f"Val dimensions:   {val_final.shape}")
    logger.info(f"Test dimensions:  {test_final.shape}")
    logger.info(f"Total features:   {metadata['total_features_count']}")
    logger.info(f"Files saved to '{output_dir}/' folder.")
    logger.info("==========================================")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.abspath(os.path.join(script_dir, "..", "US_Accidents_1M.csv"))
    default_output = os.path.abspath(os.path.join(script_dir, "data"))

    parser = argparse.ArgumentParser(description="Preprocess US Accidents dataset.")
    parser.add_argument("--input",      default=default_input, help="Path to raw CSV")
    parser.add_argument("--output-dir", default=default_output, help="Output directory")
    args = parser.parse_args()

    run_preprocessing(input_csv=args.input, output_dir=args.output_dir)