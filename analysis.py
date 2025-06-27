# analysis.py
import pandas as pd
import numpy as np

def analyze_missing_values(df):
    """Analyzes missing values in a DataFrame."""
    missing_info = {}
    total_rows = len(df)
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_percentage = (missing_count / total_rows) * 100
            
            # Suggest a method based on data type and skewness
            if pd.api.types.is_numeric_dtype(df[col]):
                suggested_method = 'Median' if abs(df[col].skew()) > 0.5 else 'Mean'
            else:
                suggested_method = 'Mode'
                
            missing_info[col] = {
                'missing_count': missing_count,
                'missing_percentage': missing_percentage,
                'suggested_method': suggested_method
            }
    return missing_info

def analyze_outliers(df):
    """Analyzes outliers using the IQR method."""
    outlier_info = {}
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        
        if outlier_count > 0:
            outlier_percentage = (outlier_count / len(df)) * 100
            suggestion = "Consider Capping/Winsorizing."
            if outlier_percentage > 10:
                suggestion = "High percentage of outliers detected. Investigate data source or consider robust scaling instead of removal."
            elif outlier_percentage < 1:
                suggestion = "Few outliers detected. Capping or removing them is likely safe."
            
            outlier_info[col] = {
                'outlier_count': outlier_count,
                'outlier_percentage': outlier_percentage,
                'suggestion': suggestion,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
    return outlier_info

def analyze_encoding(df):
    """Suggests encoding strategies for categorical columns."""
    encoding_suggestions = {
        'one_hot': [],
        'label_encode': [],
        'high_cardinality': []
    }
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count <= 2:
            encoding_suggestions['one_hot'].append(f"{col} (Binary, {unique_count} values)")
        elif 2 < unique_count <= 15:
            encoding_suggestions['one_hot'].append(f"{col} ({unique_count} values)")
        elif 15 < unique_count <= 50:
            encoding_suggestions['label_encode'].append(f"{col} ({unique_count} values)")
        else:
            encoding_suggestions['high_cardinality'].append(f"{col} ({unique_count} values)")
    return encoding_suggestions

def analyze_column_dropping(df):
    """Suggests columns that could potentially be dropped."""
    drop_suggestions = {
        'constant': [],
        'highly_correlated': []
    }
    # 1. Constant columns
    for col in df.columns:
        if df[col].nunique() == 1:
            drop_suggestions['constant'].append(col)
            
    # 2. Highly correlated numeric features
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    for col in upper_tri.columns:
        highly_corr_with = upper_tri.index[upper_tri[col] > 0.95].tolist()
        if highly_corr_with:
            for related_col in highly_corr_with:
                pair = tuple(sorted((col, related_col)))
                if pair not in [p[:2] for p in drop_suggestions['highly_correlated']]:
                     drop_suggestions['highly_correlated'].append(
                         (pair[0], pair[1], corr_matrix.loc[pair[0], pair[1]])
                     )
    return drop_suggestions

def generate_full_analysis(df):
    """Generates a complete analysis report for a DataFrame."""
    report = {
        'overview': {
            'rows': len(df),
            'columns': len(df.columns),
            'numeric_cols': len(df.select_dtypes(include=np.number).columns),
            'categorical_cols': len(df.select_dtypes(include=['object', 'category']).columns)
        },
        'missing_info': analyze_missing_values(df),
        'outlier_info': analyze_outliers(df),
        'encoding_suggestions': analyze_encoding(df),
        'drop_suggestions': analyze_column_dropping(df)
    }
    return report