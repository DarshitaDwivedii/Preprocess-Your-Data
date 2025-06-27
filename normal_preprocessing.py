# normal_preprocessing.py
import pandas as pd
import numpy as np
import re

def drop_duplicates(df, report):
    """Removes duplicate rows."""
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        report.append(f"✓ Removed {rows_removed} duplicate row(s).")
    else:
        report.append("✓ No duplicate rows found.")
    return df, report

# In normal_preprocessing.py -- CORRECTED VERSION

def handle_missing_values_per_column(df, report, strategies, fill_constant="Unknown"):
    """
    Handles missing values based on a dictionary of per-column strategies.
    - strategies: A dict like {'col_name': 'method', ...}
      Methods can be 'Mean', 'Median', 'Mode', 'Constant', or 'Drop Rows'.
    """
    if not strategies:
        return df, report

    report.append(f"🔹 **Missing Value Imputation**")

    cols_to_drop_rows = [col for col, method in strategies.items() if method == 'Drop Rows']
    if cols_to_drop_rows:
        initial_rows = len(df)
        df.dropna(subset=cols_to_drop_rows, inplace=True)
        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
            report.append(f"  - Dropped {rows_dropped} rows due to missing values in: {', '.join(cols_to_drop_rows)}.")
    
    for col, method in strategies.items():
        if method == 'Drop Rows' or col not in df.columns or df[col].isnull().sum() == 0:
            continue

        filler = None
        if method == 'Mean':
            filler = df[col].mean()
        elif method == 'Median':
            filler = df[col].median()
        elif method == 'Mode':
            filler = df[col].mode()[0] if not df[col].mode().empty else fill_constant
        elif method == 'Constant':
            # Handle numeric vs categorical constant fill
            filler = 0 if pd.api.types.is_numeric_dtype(df[col]) else fill_constant
        
        if filler is not None:
            df[col].fillna(filler, inplace=True)
            report.append(f"  - Filled missing values in '{col}' using **{method}**.")
            
    return df, report



def convert_dtypes(df, report):
    """Converts columns to best possible dtypes using pandas.convert_dtypes()."""
    initial_types = df.dtypes.to_dict()
    df = df.convert_dtypes()
    final_types = df.dtypes.to_dict()
    
    changes = [f"  - '{col}': {initial_types[col]} -> {final_types[col]}" for col in initial_types if initial_types[col] != final_types[col]]
    if changes:
        report.append("✓ Converted columns to more specific types:")
        report.extend(changes)
    else:
        report.append("✓ All columns already have appropriate dtypes.")
    return df, report

def drop_constant_columns(df, report):
    """Drops columns that have only one unique value."""
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
    if constant_cols:
        df.drop(columns=constant_cols, inplace=True)
        report.append(f"✓ Dropped constant columns: {', '.join(constant_cols)}.")
    else:
        report.append("✓ No constant columns found to drop.")
    return df, report

def rename_columns_manual(df, report, rename_map):
    """Renames columns based on a user-provided dictionary."""
    if not rename_map:
        report.append("✓ No columns were selected for manual renaming.")
        return df, report
        
    df.rename(columns=rename_map, inplace=True)
    report.append("✓ Manually renamed columns:")
    for old, new in rename_map.items():
        report.append(f"  - '{old}' -> '{new}'")
    return df, report

def strip_whitespace_from_strings(df, report):
    """Strips leading/trailing whitespace from all string/object columns."""
    string_cols = df.select_dtypes(include=['string', 'object']).columns
    changed_cols = []
    for col in string_cols:
        # Check if stripping would change anything to avoid unnecessary operations
        if df[col].str.strip().equals(df[col]):
            continue
        df[col] = df[col].str.strip()
        changed_cols.append(col)
        
    if changed_cols:
        report.append(f"✓ Stripped whitespace from string columns: {', '.join(changed_cols)}.")
    else:
        report.append("✓ No leading/trailing whitespace found in string columns.")
    return df, report

# Place this function with the others in normal_preprocessing.py

def drop_columns_manual(df, report, cols_to_drop):
    """Drops columns specified by the user."""
    if not cols_to_drop:
        # It's good practice to not add a report item if no action was taken
        return df, report
        
    # Ensure all selected columns actually exist to avoid errors
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    if not existing_cols_to_drop:
        report.append("! No columns were dropped (the selected columns might have been removed in a previous step).")
        return df, report

    df.drop(columns=existing_cols_to_drop, inplace=True)
    report.append(f"✓ Manually dropped columns: {', '.join(existing_cols_to_drop)}.")
    return df, report