# ds_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

def detect_and_handle_outliers(df, report, cols_to_process, method='iqr', multiplier=1.5):
    """Detects and caps outliers in specified columns using IQR or Z-score."""
    if not cols_to_process:
        return df, report

    report.append(f"🔹 **Outlier Handling (Method: {method.upper()})**")
    for col in cols_to_process:
        if not pd.api.types.is_numeric_dtype(df[col]):
            report.append(f"  - ❗ Skipped '{col}': Not a numeric column.")
            continue

        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
        else:  # Z-score
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - multiplier * std
            upper_bound = mean + multiplier * std

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            df[col] = np.clip(df[col], lower_bound, upper_bound)
            report.append(f"  - Capped {len(outliers)} outliers in '{col}'.")
    return df, report

def scale_features(df, report, cols_to_scale, method='minmax'):
    """Scales specified numeric features."""
    if not cols_to_scale:
        return df, report

    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    report.append(f"🔹 **Feature Scaling (Method: {scaler.__class__.__name__})**")
    
    # Filter to only scale columns that are actually numeric
    valid_cols = [col for col in cols_to_scale if pd.api.types.is_numeric_dtype(df[col])]
    if not valid_cols:
        report.append("  - ❗ No valid numeric columns selected for scaling.")
        return df, report

    df[valid_cols] = df[valid_cols].astype('float64')
    df[valid_cols] = scaler.fit_transform(df[valid_cols])
    report.append(f"  - Scaled columns: {', '.join(valid_cols)}.")
    return df, report

def encode_categorical_features(df, report, cols_to_encode, one_hot_threshold=10):
    """Encodes specified categorical features using OneHot or Label Encoding."""
    if not cols_to_encode:
        return df, report
        
    df_encoded = df.copy()
    report.append(f"🔹 **Categorical Encoding (One-Hot Threshold: {one_hot_threshold})**")

    for col in cols_to_encode:
        if col not in df_encoded.columns:
            continue
        
        # Ensure the column is treated as a category
        df_encoded[col] = df_encoded[col].astype('category')
        
        if df_encoded[col].nunique() <= one_hot_threshold:
            # Using pd.get_dummies is safer and handles column names well
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True, dtype=float)
            df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
            report.append(f"  - One-Hot encoded '{col}'.")
        else:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            report.append(f"  - Label encoded '{col}' (as it has > {one_hot_threshold} unique values).")
    return df_encoded, report
    
def remove_low_variance_features(df, report, threshold=0.01):
    """Removes features with variance below a threshold (meant to be used after scaling)."""
    # This operates on all numeric columns by design.
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2:
        return df, report
    
    report.append(f"🔹 **Low Variance Feature Removal (Threshold: {threshold})**")
    
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df[numeric_cols])
    
    high_variance_cols = df[numeric_cols].columns[selector.get_support()]
    low_variance_cols = [col for col in numeric_cols if col not in high_variance_cols]
    
    if low_variance_cols:
        df.drop(columns=low_variance_cols, inplace=True)
        report.append(f"  - Removed low variance features: {', '.join(low_variance_cols)}.")
    else:
        report.append("  - No low variance features found below the threshold.")
    return df, report