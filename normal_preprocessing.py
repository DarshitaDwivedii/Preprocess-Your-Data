import pandas as pd
import numpy as np
import re

def _create_log_entry(icon, title, details=None, category="Normal"):
    """Helper to create a structured log entry."""
    return {"category": category, "icon": icon, "title": title, "details": details}

def drop_duplicates(df, report):
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        report.append(_create_log_entry("🗑️", f"Removed {rows_removed} duplicate row(s)."))
    else:
        report.append(_create_log_entry("✅", "No duplicate rows found."))
    return df, report

# In normal_preprocessing.py

def handle_missing_values_per_column(df, report, strategies):
    if not strategies: return df, report
    
    details = []
    # Handle 'Drop Rows' strategy first and separately
    cols_to_drop_rows = [col for col, method in strategies.items() if method == 'Drop Rows']
    if cols_to_drop_rows:
        initial_rows = len(df)
        df.dropna(subset=cols_to_drop_rows, inplace=True)
        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
            details.append(f"Dropped {rows_dropped} rows due to missing values in: `{', '.join(cols_to_drop_rows)}`")

    # Process all other imputation strategies
    for col, strategy in strategies.items():
        # Skip columns that were part of 'Drop Rows' or have no missing values
        if strategy == 'Drop Rows' or col not in df.columns or df[col].isnull().sum() == 0:
            continue
            
        method = strategy[0] if isinstance(strategy, tuple) else strategy
        filler, success = None, True

        is_numeric = pd.api.types.is_numeric_dtype(df[col])

        if method in ['Mean', 'Median']:
            if not is_numeric:
                details.append(f"❗ Skipped `{col}`: Cannot use **{method}** on a non-numeric column.")
                success = False
            else:
                filler = df[col].mean() if method == 'Mean' else df[col].median()
        
        elif method == 'Mode':
            if not df[col].mode().empty:
                filler = df[col].mode()[0]
            else:
                # Fallback if mode is empty
                if is_numeric:
                    filler = df[col].median()
                    details.append(f"⚠️ Mode for `{col}` was empty; used **Median** as a fallback.")
                else:
                    filler = "Unknown"
        
        elif method == 'Constant':
            filler = strategy[1]
            if is_numeric:
                original_filler = filler
                # Attempt to convert the user's constant to a number
                filler = pd.to_numeric(filler, errors='coerce')
                if pd.isna(filler):
                    details.append(f"❗ Skipped `{col}`: Could not fill numeric column with non-numeric constant '{original_filler}'.")
                    success = False

        # Only perform the fill if a valid filler was determined
        if filler is not None and success:
            # Use a try-except block as a final safeguard against rare dtype issues
            try:
                df[col].fillna(filler, inplace=True)
                # Add to details only if not already handled by a special message (like mode fallback)
                if not any(f"`{col}`" in d for d in details):
                    details.append(f"Filled missing values in `{col}` using **{method}**.")
            except TypeError:
                details.append(f"❗ Skipped `{col}` due to a data type mismatch with filler '{filler}'.")

    if details:
        report.append(_create_log_entry("❓", "Handled Missing Values", details))
    return df, report

def convert_dtypes(df, report):
    initial_types = df.dtypes.to_dict()
    df = df.convert_dtypes()
    final_types = df.dtypes.to_dict()
    changes = [f"`{col}`: {initial_types[col]} → {final_types[col]}" for col in initial_types if initial_types[col] != final_types[col]]
    if changes:
        report.append(_create_log_entry("🔄", "Converted Column Data Types", changes))
    else:
        report.append(_create_log_entry("✅", "Column data types are already optimal."))
    return df, report

def drop_constant_columns(df, report):
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
    if constant_cols:
        df.drop(columns=constant_cols, inplace=True)
        report.append(_create_log_entry("🗑️", f"Dropped constant columns: `{', '.join(constant_cols)}`"))
    else:
        report.append(_create_log_entry("✅", "No constant columns found."))
    return df, report

def rename_columns_manual(df, report, rename_map):
    if not rename_map: return df, report
    df.rename(columns=rename_map, inplace=True)
    details = [f"`{old}` → `{new}`" for old, new in rename_map.items()]
    report.append(_create_log_entry("✏️", "Renamed Columns", details))
    return df, report

def strip_whitespace_from_strings(df, report):
    string_cols = df.select_dtypes(include=['string', 'object']).columns
    changed_cols = [col for col in string_cols if df[col].astype(str).str.strip().ne(df[col].astype(str)).any()]
    if changed_cols:
        for col in changed_cols: df[col] = df[col].str.strip()
        report.append(_create_log_entry("🧼", f"Stripped whitespace from: `{', '.join(changed_cols)}`"))
    else:
        report.append(_create_log_entry("✅", "No extraneous whitespace found."))
    return df, report

def drop_columns_manual(df, report, cols_to_drop):
    if not cols_to_drop: return df, report
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if not existing_cols_to_drop: return df, report
    df.drop(columns=existing_cols_to_drop, inplace=True)
    report.append(_create_log_entry("🗑️", f"Manually dropped columns: `{', '.join(existing_cols_to_drop)}`"))
    return df, report