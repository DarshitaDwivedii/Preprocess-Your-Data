import pandas as pd
import numpy as np

def generate_full_analysis(df):
    """
    Generates a complete analysis report, including detection of outliers and potential ID columns.
    """
    total_rows = len(df)
    
    quality_summary = {
        'constant_cols': [], 'highly_correlated_pairs': [], 'numeric_like_object_cols': [],
        'highly_imbalanced_cols': [], 'total_missing_cells': int(df.isnull().sum().sum()),
        'total_duplicate_rows': int(df.duplicated().sum()), 'inconsistent_col_names': [],
        'potential_datetime_cols': [], 'mixed_type_cols': [], 'potential_id_cols': []
    }

    column_details = {}
    summary_list = []
    id_keywords = ['id', 'no', 'number', 'code', 'key', 'identifier']

    for col in df.columns:
        if ' ' in str(col) or not str(col).isidentifier():
             quality_summary['inconsistent_col_names'].append(col)

        col_data = df[col]
        details = {
            'dtype': col_data.dtype, 'memory_usage': col_data.memory_usage(deep=True),
            'missing_count': int(col_data.isnull().sum()),
            'missing_percentage': (col_data.isnull().sum() / total_rows) * 100 if total_rows > 0 else 0,
            'unique_count': col_data.nunique(),
        }
        
        row_summary = {'Column': col, 'Data Type': str(details['dtype'])}

        if total_rows > 0:
            unique_ratio = details['unique_count'] / total_rows
            is_potential_id = (unique_ratio > 0.95) or \
                              (any(keyword in str(col).lower() for keyword in id_keywords) and unique_ratio > 0.85)
            if is_potential_id and details['unique_count'] > 20:
                quality_summary['potential_id_cols'].append(col)

        if pd.api.types.is_numeric_dtype(col_data):
            if details['unique_count'] == 1 and not col_data.isnull().all():
                quality_summary['constant_cols'].append(col)
            if details['unique_count'] > 1:
                details.update({
                    'skew': col_data.skew(), 'mean': col_data.mean(), 'median': col_data.median(),
                    'std': col_data.std(), 'min': col_data.min(), 'max': col_data.max(),
                    'zeros_count': int((col_data == 0).sum()),
                    'zeros_percentage': (int((col_data == 0).sum()) / total_rows) * 100 if total_rows > 0 else 0
                })
                Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                details['outlier_count'] = len(outliers)
                details['outlier_percentage'] = (len(outliers) / total_rows) * 100 if total_rows > 0 else 0
            row_summary.update({'Mean': details.get('mean'), 'Median': details.get('median'), 'Std Dev': details.get('std')})

        elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_string_dtype(col_data):
            if not col_data.isnull().all():
                non_null_count = col_data.count()
                if non_null_count > 0:
                    numeric_coerced = pd.to_numeric(col_data, errors='coerce')
                    if (numeric_coerced.count() / non_null_count) > 0.8:
                        quality_summary['numeric_like_object_cols'].append(col)
                        if numeric_coerced.isnull().sum() > 0: quality_summary['mixed_type_cols'].append(col)
                    try:
                        datetime_coerced = pd.to_datetime(col_data, errors='coerce', infer_datetime_format=True)
                        if (datetime_coerced.count() / non_null_count) > 0.8: quality_summary['potential_datetime_cols'].append(col)
                    except Exception: pass
            if details['unique_count'] == 1 and not col_data.isnull().all():
                quality_summary['constant_cols'].append(col)

        if not pd.api.types.is_numeric_dtype(col_data):
            if details['unique_count'] > 1 and details['unique_count'] <= 50:
                 details['value_counts'] = col_data.value_counts()
                 if total_rows > 0 and (details['value_counts'].iloc[0] / total_rows) * 100 > 95:
                     quality_summary['highly_imbalanced_cols'].append(col)
        
        column_details[col] = details
        row_summary.update({'Missing %': details['missing_percentage'], 'Unique Values': details['unique_count']})
        summary_list.append(row_summary)

    significant_outlier_cols = [name for name, details in column_details.items() if details.get('outlier_percentage', 0) > 1.0]
    quality_summary['significant_outlier_cols'] = significant_outlier_cols

    numeric_df = df.select_dtypes(include=np.number)
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        for col in upper_tri.columns:
            highly_corr_with = upper_tri.index[upper_tri[col] > 0.95].tolist()
            if highly_corr_with:
                for related_col in highly_corr_with:
                    pair = tuple(sorted((col, related_col)))
                    if not any(p[:2] == pair for p in quality_summary['highly_correlated_pairs']):
                        quality_summary['highly_correlated_pairs'].append((pair[0], pair[1], corr_matrix.loc[pair[0], pair[1]]))
    
    column_summary_df = pd.DataFrame(summary_list).set_index('Column')
    report = {
        'overview': {'rows': total_rows, 'columns': len(df.columns), 'memory_usage': df.memory_usage(deep=True).sum()},
        'quality_summary': quality_summary, 'column_details': column_details,
        'column_summary_df': column_summary_df,
        'characteristics': {'has_duplicates': quality_summary['total_duplicate_rows'] > 0}
    }
    return report

def format_analysis_to_text(analysis_report):
    """Converts the analysis dictionary into a nicely formatted string for a .txt file."""
    report_lines = []
    overview = analysis_report['overview']
    quality = analysis_report['quality_summary']
    mem_usage_mb = overview['memory_usage'] / 1024**2
    total_cells = overview['rows'] * overview['columns']
    missing_pct = (quality['total_missing_cells'] / total_cells) * 100 if total_cells > 0 else 0

    report_lines.append("=======================================\n     DATASET ANALYSIS REPORT\n=======================================")
    report_lines.append(f"\n--- Dataset Overview ---\nRows: {overview['rows']:,}\nColumns: {overview['columns']}\nTotal Cells: {total_cells:,}\nMemory Usage: {mem_usage_mb:.2f} MB")
    report_lines.append(f"\n--- Data Quality Warnings ---\nMissing Cells: {quality['total_missing_cells']:,} ({missing_pct:.2f}%)\nDuplicate Rows: {quality['total_duplicate_rows']:,}")
    
    if quality.get('potential_id_cols'): report_lines.append(f"[!] Potential ID Columns: {', '.join(quality['potential_id_cols'])}")
    if quality.get('significant_outlier_cols'): report_lines.append(f"[!] Significant Outlier Columns (>1%): {', '.join(quality['significant_outlier_cols'])}")
    if quality.get('constant_cols'): report_lines.append(f"[!] Constant Columns: {', '.join(quality['constant_cols'])}")
    if quality.get('highly_correlated_pairs'):
        report_lines.append("[!] Highly Correlated Column Pairs:")
        for pair in quality['highly_correlated_pairs']: report_lines.append(f"    - '{pair[0]}' & '{pair[1]}' (Corr: {pair[2]:.3f})")
    if quality.get('inconsistent_col_names'): report_lines.append(f"[!] Inconsistent Column Names (have spaces/symbols): {', '.join(quality['inconsistent_col_names'])}")
    if quality.get('numeric_like_object_cols'): report_lines.append(f"[!] Columns that look numeric but are text: {', '.join(quality['numeric_like_object_cols'])}")
    if quality.get('mixed_type_cols'): report_lines.append(f"[!] Columns with mixed data types (text and numbers): {', '.join(quality['mixed_type_cols'])}")
    if quality.get('potential_datetime_cols'): report_lines.append(f"[!] Potential Datetime Columns (stored as text): {', '.join(quality['potential_datetime_cols'])}")
    if quality.get('highly_imbalanced_cols'): report_lines.append(f"[!] Highly Imbalanced Columns (>95% one value): {', '.join(quality['highly_imbalanced_cols'])}")

    report_lines.append("\n\n--- Column-by-Column Details ---")
    col_details = analysis_report['column_details']
    for col_name, info in col_details.items():
        mem_kb = info.get('memory_usage', 0) / 1024
        report_lines.append(f"\n---------------------------------------\nColumn: '{col_name}'\n  Data Type: {info['dtype']}\n  Memory: {mem_kb:.2f} KB\n  Missing: {info['missing_count']} ({info['missing_percentage']:.2f}%)\n  Unique Values: {info['unique_count']:,}")
        if pd.api.types.is_numeric_dtype(info['dtype']) and 'mean' in info:
            report_lines.append(f"  Mean: {info.get('mean', 'N/A'):.2f}\n  Median: {info.get('median', 'N/A'):.2f}\n  Std Dev: {info.get('std', 'N/A'):.2f}\n  Skewness: {info.get('skew', 'N/A'):.2f}\n  Zeros: {info.get('zeros_count', 0)} ({info.get('zeros_percentage', 0):.2f}%)\n  Outliers (IQR): {info.get('outlier_count', 0)} ({info.get('outlier_percentage', 0):.2f}%)")
    
    return "\n".join(report_lines)