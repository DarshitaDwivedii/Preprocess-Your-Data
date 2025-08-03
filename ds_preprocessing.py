import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

def _create_log_entry(icon, title, details=None, category="DS"):
    """Helper to create a structured log entry for DS tasks."""
    return {"category": category, "icon": icon, "title": title, "details": details}

def detect_and_handle_outliers(df, report, cols_to_process, method='iqr', multiplier=1.5):
    if not cols_to_process: return df, report
    details = []
    for col in cols_to_process:
        if not pd.api.types.is_numeric_dtype(df[col]): continue
        if method == 'iqr':
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            lower_bound, upper_bound = Q1 - multiplier * (Q3 - Q1), Q3 + multiplier * (Q3 - Q1)
        else:
            mean, std = df[col].mean(), df[col].std()
            lower_bound, upper_bound = mean - multiplier * std, mean + multiplier * std
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            df[col] = np.clip(df[col], lower_bound, upper_bound)
            details.append(f"Capped {len(outliers)} outliers in `{col}`.")
    if details:
        report.append(_create_log_entry("📈", f"Capped Outliers (Method: {method.upper()})", details))
    return df, report

def scale_features(df, report, cols_to_scale, method='minmax'):
    if not cols_to_scale: return df, report
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    valid_cols = [col for col in cols_to_scale if pd.api.types.is_numeric_dtype(df[col])]
    if not valid_cols: return df, report
    df[valid_cols] = df[valid_cols].astype('float64')
    df[valid_cols] = scaler.fit_transform(df[valid_cols])
    report.append(_create_log_entry("📏", f"Scaled Features (Method: {scaler.__class__.__name__})", [f"Scaled: `{', '.join(valid_cols)}`"]))
    return df, report

def encode_categorical_features(df, report, cols_to_encode, one_hot_threshold=10):
    if not cols_to_encode: return df, report
    df_encoded = df.copy()
    details = []
    for col in cols_to_encode:
        if col not in df_encoded.columns: continue
        if df_encoded[col].isnull().any():
            df_encoded[col].fillna('Unknown_Category', inplace=True)
        df_encoded[col] = df_encoded[col].astype('category')
        if df_encoded[col].nunique() <= one_hot_threshold:
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True, dtype=float)
            df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
            details.append(f"One-Hot encoded `{col}`.")
        else:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            details.append(f"Label encoded `{col}` (> {one_hot_threshold} unique values).")
    if details:
        report.append(_create_log_entry("🏷️", "Encoded Categorical Features", details))
    return df_encoded, report
    
def remove_low_variance_features(df, report, threshold=0.01):
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2: return df, report
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df[numeric_cols])
    high_variance_cols = df[numeric_cols].columns[selector.get_support()]
    low_variance_cols = [col for col in numeric_cols if col not in high_variance_cols]
    if low_variance_cols:
        df.drop(columns=low_variance_cols, inplace=True)
        report.append(_create_log_entry("📉", f"Removed Low Variance Features (Threshold: {threshold})", [f"Removed: `{', '.join(low_variance_cols)}`"]))
    else:
        report.append(_create_log_entry("✅", f"No low variance features found below threshold {threshold}."))
    return df, report