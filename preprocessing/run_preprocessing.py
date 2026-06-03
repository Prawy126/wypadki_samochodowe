import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import MiniBatchKMeans
import joblib

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3956 # Radius of earth in miles
    return c * r

def run_preprocessing():
    input_csv = "../US_Accidents_1M.csv"
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- STEP 1: Loading raw data ---")
    df = pd.read_csv(input_csv)
    print(f"Loaded dataset shape: {df.shape}")
    
    print("\n--- STEP 2: Cleaning and Initial Dropping ---")
    # Dropping columns that are completely useless or have too many missing values
    DROP_COLS = [
        'ID', 'Country', 'Turning_Loop', 'Roundabout', 'Description', 
        'End_Lat', 'End_Lng', 'Weather_Timestamp', 'Airport_Code', 
        'Street', 'Zipcode'
    ]
    df = df.drop(columns=[col for col in DROP_COLS if col in df.columns])
    
    # Drop rows with missing targets or missing coordinates
    df = df.dropna(subset=['Severity', 'Start_Lat', 'Start_Lng'])
    # Map severity to 0-3 for PyTorch
    df['Severity'] = df['Severity'].astype(int) - 1
    
    print(f"Shape after initial row and column drops: {df.shape}")
    
    print("\n--- STEP 3: Temporal Feature Engineering ---")
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    
    # Drop rows where start or end time is invalid
    df = df.dropna(subset=['Start_Time', 'End_Time'])
    
    # Duration in minutes
    df['Duration_min'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60.0
    
    # Drop negative durations (data noise)
    df = df[df['Duration_min'] >= 0]
    
    # Temporal component extraction
    df['Hour'] = df['Start_Time'].dt.hour
    df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
    df['Month'] = df['Start_Time'].dt.month
    df['Year'] = df['Start_Time'].dt.year
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['IsRushHour'] = df['Hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    
    # Cyclic encoding for time features
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7.0)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7.0)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)
    
    # Drop the original datetimes
    df = df.drop(columns=['Start_Time', 'End_Time'])
    print(f"Shape after temporal engineering: {df.shape}")
    
    print("\n--- STEP 4: Geographic Distance & Clean Categorical Values ---")
    # Geographic center of contiguous US (Lebanon, Kansas)
    US_CENTER_LAT = 39.8283
    US_CENTER_LNG = -98.5795
    df['Dist_from_center'] = haversine_np(df['Start_Lng'], df['Start_Lat'], US_CENTER_LNG, US_CENTER_LAT)
    
    # Standardize Wind Direction
    WIND_DIR_MAP = {
        'Calm': 'CALM', 'CALM': 'CALM',
        'Variable': 'VAR', 'VAR': 'VAR',
        'South': 'S', 'S': 'S',
        'North': 'N', 'N': 'N',
        'West': 'W', 'W': 'W',
        'East': 'E', 'E': 'E',
        'SSW': 'SSW', 'WNW': 'WNW', 'SW': 'SW', 'NW': 'NW',
        'SSE': 'SSE', 'WSW': 'WSW', 'NNW': 'NNW', 'SE': 'SE',
        'ESE': 'ESE', 'NE': 'NE', 'ENE': 'ENE', 'NNE': 'NNE',
    }
    df['Wind_Direction'] = df['Wind_Direction'].map(WIND_DIR_MAP).fillna('VAR')
    
    # Group Weather Condition into macro categories
    WEATHER_MAP = {
        'Fair': 'Clear', 'Clear': 'Clear', 'Fair / Windy': 'Clear',
        'Mostly Cloudy': 'Cloudy', 'Cloudy': 'Cloudy', 'Partly Cloudy': 'Cloudy',
        'Overcast': 'Cloudy', 'Scattered Clouds': 'Cloudy', 
        'Mostly Cloudy / Windy': 'Cloudy', 'Cloudy / Windy': 'Cloudy',
        'Light Rain': 'Rain', 'Rain': 'Rain', 'Heavy Rain': 'Rain',
        'Light Drizzle': 'Rain', 'Drizzle': 'Rain', 'Heavy Rain / Windy': 'Rain',
        'Light Snow': 'Snow', 'Snow': 'Snow', 'Heavy Snow': 'Snow',
        'Fog': 'Fog', 'Haze': 'Fog', 'Mist': 'Fog',
        'T-Storm': 'Storm', 'Thunder in the Vicinity': 'Storm', 'Thunder': 'Storm',
        'Heavy T-Storm': 'Storm', 'Light Thunderstorms and Rain': 'Storm'
    }
    df['Weather_Category'] = df['Weather_Condition'].map(WEATHER_MAP).fillna('Other')
    df = df.drop(columns=['Weather_Condition'])
    
    # Process twilight columns to binary (0/1)
    TWILIGHT_COLS = ['Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']
    for col in TWILIGHT_COLS:
        # Fill missing with mode/default Day
        df[col] = df[col].fillna('Day')
        df[col] = (df[col] == 'Night').astype(int)
        
    print("Categorical cleaning complete.")
    
    print("\n--- STEP 5: Stratified Train / Val / Test Split ---")
    # Split: 70% Train, 15% Val, 15% Test
    # We do the split now to prevent data leakage from here onwards
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(sss1.split(df, df['Severity']))
    
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    temp_df = df.iloc[temp_idx].copy().reset_index(drop=True)
    
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(sss2.split(temp_df, temp_df['Severity']))
    
    val_df = temp_df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)
    
    print(f"Train size: {train_df.shape}")
    print(f"Val size:   {val_df.shape}")
    print(f"Test size:  {test_df.shape}")
    
    print("\n--- STEP 6: Geoclustering via K-Means ---")
    # Fit K-Means on train lat/lng and assign cluster classes
    kmeans = MiniBatchKMeans(n_clusters=30, random_state=42, batch_size=2048)
    train_coords = train_df[['Start_Lat', 'Start_Lng']]
    kmeans.fit(train_coords)
    
    train_df['Geo_Cluster'] = kmeans.predict(train_df[['Start_Lat', 'Start_Lng']])
    val_df['Geo_Cluster'] = kmeans.predict(val_df[['Start_Lat', 'Start_Lng']])
    test_df['Geo_Cluster'] = kmeans.predict(test_df[['Start_Lat', 'Start_Lng']])
    
    print("\n--- STEP 7: Numeric Missing Indicators & Imputation ---")
    # Define columns with high missing rates to track indicator
    HIGH_MISSING = ['Wind_Chill(F)', 'Precipitation(in)', 'Wind_Speed(mph)']
    for col in HIGH_MISSING:
        train_df[f'{col}_was_missing'] = train_df[col].isnull().astype(int)
        val_df[f'{col}_was_missing'] = val_df[col].isnull().astype(int)
        test_df[f'{col}_was_missing'] = test_df[col].isnull().astype(int)
        
    # Numeric imputation values based on train split
    NUM_COLS = [
        'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
        'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
        'Distance(mi)', 'Duration_min', 'Start_Lat', 'Start_Lng',
        'Dist_from_center', 'Hour_sin', 'Hour_cos', 'DayOfWeek_sin',
        'DayOfWeek_cos', 'Month_sin', 'Month_cos', 'Year'
    ]
    
    imputation_values = {}
    for col in NUM_COLS:
        if col == 'Precipitation(in)':
            imputation_values[col] = 0.0 # Standard assumption: no precipitation
        else:
            imputation_values[col] = train_df[col].median()
            
    # Apply imputation
    for col in NUM_COLS:
        val = imputation_values[col]
        train_df[col] = train_df[col].fillna(val)
        val_df[col] = val_df[col].fillna(val)
        test_df[col] = test_df[col].fillna(val)
        
    print("Numeric imputation complete.")
    
    print("\n--- STEP 8: Outlier Handling (Capping / Winsorization) ---")
    # Capping bounds (from train set distribution to avoid leakage)
    capping_bounds = {}
    
    # Custom capping bounds based on domain limits or train percentiles
    capping_rules = {
        'Temperature(F)': (-40.0, 120.0),
        'Wind_Speed(mph)': (0.0, 60.0),
        'Precipitation(in)': (0.0, 3.0),
        'Visibility(mi)': (0.0, 10.0),
        'Pressure(in)': (27.0, 32.0),
        'Humidity(%)': (0.0, 100.0),
        'Distance(mi)': (0.0, 30.0),
        'Duration_min': (0.0, 1440.0) # max 24 hours
    }
    
    for col, bounds in capping_rules.items():
        capping_bounds[col] = bounds
        
    # Apply capping
    for col, (low, high) in capping_bounds.items():
        train_df[col] = train_df[col].clip(lower=low, upper=high)
        val_df[col] = val_df[col].clip(lower=low, upper=high)
        test_df[col] = test_df[col].clip(lower=low, upper=high)
        
    print("Outlier capping complete.")
    
    print("\n--- STEP 9: Distribution Transformations (log1p) ---")
    LOG_TRANSFORM_COLS = ['Distance(mi)', 'Precipitation(in)', 'Wind_Speed(mph)', 'Duration_min']
    for col in LOG_TRANSFORM_COLS:
        train_df[col] = np.log1p(train_df[col])
        val_df[col] = np.log1p(val_df[col])
        test_df[col] = np.log1p(test_df[col])
        # Rename column to indicate log
        train_df.rename(columns={col: f'{col}_log'}, inplace=True)
        val_df.rename(columns={col: f'{col}_log'}, inplace=True)
        test_df.rename(columns={col: f'{col}_log'}, inplace=True)
        
    # Update NUM_COLS list to match new names
    NUM_COLS = [f'{col}_log' if col in LOG_TRANSFORM_COLS else col for col in NUM_COLS]
    
    print("Log1p transformations complete.")
    
    print("\n--- STEP 10: Scaling Continuous Features ---")
    scaler = RobustScaler()
    train_df[NUM_COLS] = scaler.fit_transform(train_df[NUM_COLS])
    val_df[NUM_COLS] = scaler.transform(val_df[NUM_COLS])
    test_df[NUM_COLS] = scaler.transform(test_df[NUM_COLS])
    print("Continuous scaling complete.")
    
    print("\n--- STEP 11: Categorical Encoding (Entity Embedding Mapping) ---")
    # Columns to be mapped to integer IDs starting from 0 to K-1 (for nn.Embedding)
    CAT_COLS = ['Source', 'State', 'City', 'County', 'Timezone', 'Wind_Direction', 'Weather_Category', 'Geo_Cluster']
    
    cat_mappings = {}
    categorical_dims = {}
    embedding_dims = {}
    
    for col in CAT_COLS:
        # Get unique values from train set
        unique_vals = train_df[col].dropna().unique().tolist()
        
        # Build mapping: 'Unknown' -> 0, others -> 1..K
        mapping = {'Unknown': 0}
        for idx, val in enumerate(unique_vals):
            if val != 'Unknown':
                mapping[val] = len(mapping)
                
        cat_mappings[col] = mapping
        
        # Vocab size for embedding is the size of the mapping
        vocab_size = len(mapping)
        categorical_dims[col] = vocab_size
        
        # Choose embedding dimension: min(50, (vocab_size + 1) // 2)
        embed_dim = min(50, (vocab_size + 1) // 2)
        embedding_dims[col] = [vocab_size, embed_dim]
        
        # Map values, defaulting to 0 for unseen/missing values
        train_df[col] = train_df[col].map(mapping).fillna(0).astype(int)
        val_df[col] = val_df[col].map(mapping).fillna(0).astype(int)
        test_df[col] = test_df[col].map(mapping).fillna(0).astype(int)
        
    print("Categorical mapping complete.")
    
    print("\n--- STEP 12: Binary Feature Setup ---")
    # Get all binary and boolean columns
    BOOL_COLS = [
        'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 
        'No_Exit', 'Railway', 'Station', 'Stop', 'Traffic_Calming', 
        'Traffic_Signal'
    ]
    
    # Add engineered indicator and twilight binary columns
    BINARY_COLS = BOOL_COLS + TWILIGHT_COLS + [f'{col}_was_missing' for col in HIGH_MISSING] + ['IsWeekend', 'IsRushHour']
    
    # Cast to integer (0/1) and fillna
    for col in BINARY_COLS:
        train_df[col] = train_df[col].fillna(0).astype(int)
        val_df[col] = val_df[col].fillna(0).astype(int)
        test_df[col] = test_df[col].fillna(0).astype(int)
        
    print("Binary columns setup complete.")
    
    print("\n--- STEP 13: Target & Class Weights Setup ---")
    y_train = train_df['Severity'].values
    unique_classes, counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    n_classes = len(unique_classes)
    
    # Compute inverse class frequencies for weighting
    class_weights = {}
    for cl, count in zip(unique_classes, counts):
        class_weights[int(cl)] = float(total_samples / (n_classes * count))
        
    print(f"Class frequencies on Train: {dict(zip(unique_classes, counts))}")
    print(f"Computed Class Weights: {class_weights}")
    
    # Arrange columns: Severity (target), Numeric columns, Categorical columns, Binary columns
    final_cols = ['Severity'] + NUM_COLS + CAT_COLS + BINARY_COLS
    
    train_final = train_df[final_cols]
    val_final = val_df[final_cols]
    test_final = test_df[final_cols]
    
    print("\n--- STEP 14: Save Preprocessed Datasets and Metadata ---")
    # Save datasets to csv.gz for space efficiency
    train_final.to_csv(os.path.join(output_dir, "train_preprocessed.csv.gz"), index=False, compression="gzip")
    val_final.to_csv(os.path.join(output_dir, "val_preprocessed.csv.gz"), index=False, compression="gzip")
    test_final.to_csv(os.path.join(output_dir, "test_preprocessed.csv.gz"), index=False, compression="gzip")
    
    # Save artifacts: scaler, kmeans, mappings
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(kmeans, os.path.join(output_dir, "kmeans.joblib"))
    joblib.dump(cat_mappings, os.path.join(output_dir, "categorical_mappings.joblib"))
    
    # Create metadata dictionary
    metadata = {
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
        "bin_cols": BINARY_COLS,
        "target_col": "Severity",
        "categorical_dims": categorical_dims,
        "embedding_dims": embedding_dims,
        "class_weights": class_weights,
        "num_features_count": len(NUM_COLS),
        "cat_features_count": len(CAT_COLS),
        "bin_features_count": len(BINARY_COLS),
        "total_features_count": len(NUM_COLS) + len(CAT_COLS) + len(BINARY_COLS)
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("\n==========================================")
    print("Preprocessing completed successfully!")
    print(f"Train dimensions: {train_final.shape}")
    print(f"Val dimensions:   {val_final.shape}")
    print(f"Test dimensions:  {test_final.shape}")
    print(f"Total features:   {metadata['total_features_count']}")
    print("Files saved to 'data/' folder.")
    print("==========================================")

if __name__ == "__main__":
    run_preprocessing()
