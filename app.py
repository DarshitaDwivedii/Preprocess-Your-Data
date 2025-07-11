import streamlit as st
import pandas as pd
import numpy as np

import normal_preprocessing as normal_pp
import ds_preprocessing as ds_pp
import nlp_preprocessing as nlp_pp

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Data Preprocessor")

# --- Helper Functions ---
@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string."""
    return df.to_csv(index=False).encode('utf-8')

def load_csv_with_encoding_detection(uploaded_file):
    """Tries to load a CSV by attempting different common encodings."""
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings_to_try:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding)
            return df
        except (UnicodeDecodeError, Exception):
            continue
    st.error("Error: Could not decode the file with any of the common encodings.")
    return None

def detect_text_heavy_columns(df, threshold=30):
    """Heuristically detects columns that are likely to contain natural language text."""
    text_cols = []
    potential_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in potential_cols:
        # Calculate the average length of non-null entries
        avg_len = df[col].dropna().astype(str).str.len().mean()
        if avg_len > threshold:
            text_cols.append(col)
    return text_cols

# --- Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.processed_df = None
    st.session_state.report = []

# --- Main App UI ---
st.title("✨ The Intelligent Data Preprocessor ✨")
st.write("A one-stop tool to clean, transform, and enhance your data. Configure the steps below and click 'Run'.")

# --- Layout: Controls Panel and Display Panel ---
col1, col2 = st.columns([1, 2])

# --- Sidebar for Controls ---
with col1:
    st.header("🛠️ Controls Panel")
    uploaded_file = st.file_uploader("1. Upload your CSV file", type="csv")

    if uploaded_file:
        df = load_csv_with_encoding_detection(uploaded_file)
        if df is not None:
            st.session_state.df = df
            st.session_state.processed_df = None # Reset on new file upload
            st.session_state.report = []

    if st.session_state.df is not None:
        df = st.session_state.df
        
        # --- UI for Preprocessing Steps ---
        with st.expander("🧹 Normal Preprocessing", expanded=True):
            st.write("Basic cleaning and data wrangling.")
            with st.expander("🗑️ Column Removal & Renaming"):
                cols_to_drop = st.multiselect("Select columns to remove:", options=df.columns.tolist(), key="drop")
                available_cols_for_rename = [col for col in df.columns if col not in cols_to_drop]
                cols_to_rename = st.multiselect("Select columns to rename:", options=available_cols_for_rename, key="rename_select")
                rename_map = {}
                for col in cols_to_rename:
                    new_name = st.text_input(f"New name for '{col}':", value=col, key=f"rename_{col}")
                    if new_name and new_name != col: rename_map[col] = new_name
            with st.expander("🧼 Basic Cleaning Operations", expanded=True):
                do_strip_whitespace = st.checkbox("Strip Whitespace", value=True, key="strip")
                do_drop_duplicates = st.checkbox("Remove Duplicates", value=True, key="dupes")
                do_drop_constant = st.checkbox("Remove Constant Columns", value=True, key="const")
                do_convert_dtypes = st.checkbox("Optimize Data Types", value=True, key="dtypes")
            with st.expander("❓ Handle Missing Values"):
                do_handle_missing = st.checkbox("Enable Missing Value Handling", key="do_missing_values")
                cols_with_missing = df.columns[df.isnull().any()].tolist()
                if not cols_with_missing:
                    st.info("No missing values detected.")
                    imputation_strategies = {}
                elif not do_handle_missing:
                    st.info(f"Detected {len(cols_with_missing)} columns with missing values. Enable to configure.")
                    imputation_strategies = {}
                else:
                    imputation_strategies = {}
                    st.write("Configure a strategy for each column with missing data:")
                    for col in cols_with_missing:
                        with st.container(border=True):
                            col_type = "Numeric" if pd.api.types.is_numeric_dtype(df[col]) else "Categorical"
                            if col_type == "Numeric":
                                options = ['Mean', 'Median', 'Constant', 'Drop Rows']
                                default_method = 'Median' if abs(df[col].skew()) > 0.5 else 'Mean'
                                default_index = options.index(default_method)
                            else:
                                options = ['Mode', 'Constant', 'Drop Rows']
                                default_index = 0
                            st.markdown(f"**Column:** `{col}` ({col_type})")
                            selected_strategy = st.selectbox(f"Strategy for `{col}`", options, index=default_index, key=f"missing_{col}")
                            imputation_strategies[col] = selected_strategy
                    fill_constant_value = st.text_input("Constant value (if selected):", value="Unknown")
        
        with st.expander("🔬 Advanced DS Preprocessing"):
            st.write("Model-specific transformations.")
            potential_cols = [col for col in df.columns if col not in cols_to_drop]
            potential_cols = [rename_map.get(col, col) for col in potential_cols]
            temp_df = df.rename(columns=rename_map)[potential_cols]
            numeric_cols = temp_df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = temp_df.select_dtypes(include=['object', 'category']).columns.tolist()

            with st.expander("📈 Outlier Handling"):
                do_handle_outliers = st.checkbox("Enable", key="do_outliers")
                outlier_cols = st.multiselect("Apply to columns:", numeric_cols, default=numeric_cols, key="outlier_cols", disabled=not do_handle_outliers)
                outlier_method = st.selectbox("Method:", ["iqr", "z-score"], key="outlier_method", disabled=not do_handle_outliers)
                outlier_multiplier = st.slider("Multiplier:", 1.0, 5.0, 1.5, 0.1, key="outlier_mult", disabled=not do_handle_outliers)
            with st.expander("📏 Feature Scaling"):
                do_scale_features = st.checkbox("Enable", key="do_scaling")
                scaling_cols = st.multiselect("Columns to scale:", numeric_cols, default=numeric_cols, key="scaling_cols", disabled=not do_scale_features)
                scaling_method = st.selectbox("Scaling Method:", ["minmax", "standard"], key="scaling_method", disabled=not do_scale_features)
            with st.expander("🏷️ Feature Encoding"):
                do_encode_features = st.checkbox("Enable", key="do_encoding")
                encoding_cols = st.multiselect("Columns to encode:", categorical_cols, default=categorical_cols, key="encoding_cols", disabled=not do_encode_features)
                ohe_threshold = st.number_input("One-Hot Threshold:", 1, 50, 10, key="ohe_thresh", disabled=not do_encode_features)
            with st.expander("📉 Low Variance Removal"):
                do_remove_low_variance = st.checkbox("Enable", help="Run AFTER scaling for best results.", key="do_low_var")
                variance_threshold = st.number_input("Variance Threshold:", 0.0, 1.0, 0.01, format="%.3f", key="var_thresh", disabled=not do_remove_low_variance)
        
        with st.expander("📝 NLP Preprocessing"):
            st.write("Clean and process natural language text.")
            text_cols = detect_text_heavy_columns(df)
            if not text_cols:
                st.info("No text-heavy columns detected.")
                nlp_target_col = None
            else:
                st.success(f"Text column(s) detected: {', '.join(text_cols)}")
                nlp_target_col = st.selectbox("Select text column to process:", text_cols)
                nlp_operations = ['lowercase', 'remove_urls', 'remove_special_chars', 'remove_stopwords', 'lemmatize']
                nlp_funcs_to_apply = st.multiselect("Select NLP operations:", options=nlp_operations, default=['lowercase', 'remove_special_chars', 'remove_stopwords'])
        st.divider()

        # --- Execution Button and Pipeline Logic ---
        if st.button("🚀 Run All Preprocessing", type="primary"):
            with st.spinner("Processing... This may take a moment."):
                processed_df = df.copy()
                report = []
                # --- The order of operations is important! ---
                if cols_to_drop: processed_df, report = normal_pp.drop_columns_manual(processed_df, report, cols_to_drop)
                if rename_map: processed_df, report = normal_pp.rename_columns_manual(processed_df, report, rename_map)
                if do_strip_whitespace: processed_df, report = normal_pp.strip_whitespace_from_strings(processed_df, report)
                if do_drop_duplicates: processed_df, report = normal_pp.drop_duplicates(processed_df, report)
                if do_drop_constant: processed_df, report = normal_pp.drop_constant_columns(processed_df, report)
                if do_convert_dtypes: processed_df, report = normal_pp.convert_dtypes(processed_df, report)
                if do_handle_missing and 'imputation_strategies' in locals() and imputation_strategies:
                    corrected_strategies = {rename_map.get(k, k): v for k, v in imputation_strategies.items() if rename_map.get(k, k) in processed_df.columns}
                    processed_df, report = normal_pp.handle_missing_values_per_column(processed_df, report, strategies=corrected_strategies, fill_constant=fill_constant_value)
                if 'nlp_target_col' in locals() and nlp_target_col and 'nlp_funcs_to_apply' in locals() and nlp_funcs_to_apply:
                    corrected_nlp_col = rename_map.get(nlp_target_col, nlp_target_col)
                    if corrected_nlp_col in processed_df.columns:
                        processed_df, report = nlp_pp.process_text_column(processed_df, report, corrected_nlp_col, nlp_funcs_to_apply)
                if do_handle_outliers: processed_df, report = ds_pp.detect_and_handle_outliers(processed_df, report, outlier_cols, outlier_method, outlier_multiplier)
                if do_encode_features: processed_df, report = ds_pp.encode_categorical_features(processed_df, report, encoding_cols, ohe_threshold)
                if do_scale_features: processed_df, report = ds_pp.scale_features(processed_df, report, scaling_cols, scaling_method)
                if do_remove_low_variance: processed_df, report = ds_pp.remove_low_variance_features(processed_df, report, variance_threshold)
                
                st.session_state.processed_df = processed_df
                st.session_state.report = report

# --- Main Panel for Data Display ---
with col2:
    st.header("Data Display")
    if st.session_state.df is not None:
        st.subheader("Original Data")
        st.dataframe(st.session_state.df)
    else:
        st.info("Please upload a CSV file using the control panel on the left to begin.")
    
    if st.session_state.processed_df is not None:
        st.divider()
        st.subheader("✨ Final Processed Data")
        st.dataframe(st.session_state.processed_df)
        st.subheader("📄 Full Preprocessing Report")
        with st.container(height=300):
            for item in st.session_state.report:
                st.markdown(f"- {item}")
        
        csv_data = convert_df_to_csv(st.session_state.processed_df)
        st.download_button("📥 Download Processed CSV", csv_data, "processed_data.csv", "text/csv")