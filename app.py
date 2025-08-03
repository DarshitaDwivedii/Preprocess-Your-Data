import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Import your custom modules
import normal_preprocessing as normal_pp
import ds_preprocessing as ds_pp
import nlp_preprocessing as nlp_pp
import analysis as an

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Intelligent Data Preprocessor", page_icon="✨")

# --- Helper Functions ---
@st.cache_data
def convert_df_to_csv(df):
    """Converts a dataframe to a UTF-8 encoded CSV file for download."""
    return df.to_csv(index=False).encode('utf-8')

def load_csv_with_encoding_detection(uploaded_file):
    """Tries multiple common encodings to load a CSV file."""
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings_to_try:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding)
            st.session_state.file_encoding = encoding
            return df
        except Exception:
            continue
    st.error("Error: Could not decode the file with any of the common encodings.")
    st.session_state.file_encoding = "Unknown"
    return None

def detect_text_heavy_columns(df, threshold=30):
    """Identifies columns that likely contain long-form text."""
    text_cols = []
    potential_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in potential_cols:
        if df[col].isnull().all(): continue
        if df[col].dropna().astype(str).str.len().mean() > threshold: text_cols.append(col)
    return text_cols

def calculate_optimal_bins(data_series):
    """Calculates the optimal number of bins for a histogram using Freedman-Diaconis rule."""
    data_series = data_series.dropna()
    if data_series.empty: return 10
    q1 = data_series.quantile(0.25)
    q3 = data_series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0: return 20
    bin_width = 2 * iqr / (len(data_series) ** (1/3))
    if bin_width == 0: return 20
    num_bins = int(np.ceil((data_series.max() - data_series.min()) / bin_width))
    return min(max(num_bins, 10), 100)

# --- UI Display Functions ---
def display_interactive_analysis(df, analysis_report):
    """Renders the entire 'Interactive Analysis' tab with advanced visualizations."""
    col_details = analysis_report['column_details']; quality = analysis_report['quality_summary']
    overview = analysis_report['overview']; summary_df = analysis_report['column_summary_df']
    
    st.header("📋 Data Quality Health Report")
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        total_cells = overview['rows'] * overview['columns']
        missing_pct = (quality['total_missing_cells'] / total_cells) * 100 if total_cells > 0 else 0
        m1.metric("Rows", f"{overview['rows']:,}"); m2.metric("Columns", f"{overview['columns']:,}")
        m3.metric("Missing Cells", f"{missing_pct:.2f}%"); m4.metric("Duplicate Rows", f"{quality['total_duplicate_rows']:,}")
        if quality.get('constant_cols'): st.error(f"**Constant Columns Found:** These are pre-selected for removal.", icon="🗑️")
        if quality.get('highly_correlated_pairs'): st.error(f"**Highly Correlated Columns Found:** Consider dropping one from each pair.", icon="🔗")

    st.subheader("🤖 Smart Suggestions & Pre-selected Actions")
    suggestions = []
    
    def format_list(items, limit=4):
        if not items: return ""
        formatted_items = [f"`{item}`" for item in items]
        if len(formatted_items) > limit: return ", ".join(formatted_items[:limit]) + f", and {len(formatted_items) - limit} more"
        return ", ".join(formatted_items)

    id_cols = quality.get('potential_id_cols', []); outlier_cols = quality.get('significant_outlier_cols', [])
    if id_cols: suggestions.append({"icon": "🆔", "title": "Potential ID Columns", "text": f"Found **{len(id_cols)}** columns that appear to be unique identifiers.", "details": f"**Action:** Pre-selected for dropping: {format_list(id_cols)}", "color": "#FFF3CD"})
    if outlier_cols: suggestions.append({"icon": "📈", "title": "Significant Outliers", "text": f"Found **{len(outlier_cols)}** columns with >1% outlier data.", "details": f"**Action:** Pre-selected for capping: {format_list(outlier_cols)}", "color": "#D1E7DD"})
    if quality.get('has_duplicates', False): suggestions.append({"icon": "📋", "title": "Duplicate Rows", "text": f"The dataset contains **{quality['total_duplicate_rows']}** duplicate rows.", "details": f"**Action:** 'Remove Duplicates' option is pre-ticked.", "color": "#F8D7DA"})

    if suggestions:
        cols = st.columns(len(suggestions))
        for i, s in enumerate(suggestions):
            with cols[i]: st.markdown(f"""<div style="border: 1.5px solid {s['color']}; border-radius: 5px; padding: 15px; background-color: {s['color']}22;"><h5 style="margin-bottom: 10px;">{s['icon']} {s['title']}</h5><p style="font-size: 14px; margin-bottom: 12px;">{s['text']}</p><p style="font-size: 14px; color: #444;"><i>{s['details']}</i></p></div>""", unsafe_allow_html=True)
    else: st.success("✅ No major actions were automatically suggested. The data looks clean!")
    st.download_button("📥 Download Full Analysis Report (.txt)", an.format_analysis_to_text(analysis_report), f"analysis_report_{st.session_state.uploaded_filename}.txt", "text/plain")
    st.divider()

    st.header("📜 Column Summary Table")
    st.dataframe(summary_df.style.format({'Missing %': '{:.2f}%', 'Mean': '{:.2f}', 'Median': '{:.2f}', 'Std Dev': '{:.2f}'}).background_gradient(cmap='Reds', subset=['Missing %']).background_gradient(cmap='Blues', subset=['Unique Values']), use_container_width=True)
    st.divider()
    
    st.header("📊 Data Visualization Playground")
    st.markdown("Select one or two columns to explore their relationship and distribution. New plotting options will appear based on your selections.")
    
    c1, c2 = st.columns(2)
    with c1: primary_col = st.selectbox("Select primary column (X-axis)", options=df.columns, key="primary_col")
    with c2:
        secondary_options = ['(Single Column Plot)'] + [col for col in df.columns if col != primary_col]
        secondary_col = st.selectbox("Compare with (optional, Y-axis)", options=secondary_options, key="secondary_col")

    if primary_col in quality.get('potential_id_cols', []):
        st.info(f"💡 The column `{primary_col}` was identified as a potential ID and is not suitable for visualization.", icon="🆔")
    else:
        primary_type = 'numeric' if pd.api.types.is_numeric_dtype(df[primary_col]) else 'categorical'
        if secondary_col == '(Single Column Plot)':
            col_info = col_details[primary_col]
            st.subheader(f"Distribution of `{primary_col}`")
            if primary_type == 'numeric':
                default_bins = calculate_optimal_bins(df[primary_col])
                c1, c2 = st.columns([2, 1])
                with c1: num_bins = st.slider("Adjust number of histogram bins:", 10, 100, default_bins, key="bins")
                with c2: use_log_scale = st.checkbox("Use Logarithmic Scale", value=abs(col_info.get('skew', 0)) > 2)
                x_scale = alt.Scale(type='log') if use_log_scale else alt.Scale(type='linear')
                chart = alt.Chart(df).mark_bar().encode(alt.X(primary_col, bin=alt.Bin(maxbins=num_bins), title=primary_col, scale=x_scale), y=alt.Y('count()', title="Count"), tooltip=[alt.X(primary_col, bin=True), 'count()']).properties(title=f"Histogram of {primary_col}")
                st.altair_chart(chart, use_container_width=True)
            else:
                if 'value_counts' in col_info:
                    if 1 < col_info['unique_count'] <= 8:
                        chart_type = st.radio("Chart Type:", ["Bar Chart", "Pie Chart"], horizontal=True, key="chart_type")
                        vc_df = col_info['value_counts'].head(8).reset_index(); vc_df.columns = ['Category', 'Count']
                        if chart_type == "Bar Chart": chart = alt.Chart(vc_df).mark_bar().encode(x='Count:Q', y=alt.Y('Category:N', sort='-x'), tooltip=['Category', 'Count']).properties(title=f"Top Categories in {primary_col}")
                        else: chart = alt.Chart(vc_df).mark_arc(innerRadius=50).encode(theta=alt.Theta(field="Count", type="quantitative"), color=alt.Color(field="Category", type="nominal"), tooltip=['Category', 'Count']).properties(title=f"Composition of {primary_col}")
                    else:
                        vc = col_info['value_counts']; top_n = vc.head(15); other_sum = vc.iloc[15:].sum()
                        if other_sum > 0:
                            top_n_df = top_n.reset_index(); top_n_df.columns = ['Category', 'Count']
                            other_df = pd.DataFrame([{'Category': 'Other', 'Count': other_sum}])
                            vc_df = pd.concat([top_n_df, other_df], ignore_index=True)
                        else: vc_df = top_n.reset_index(); vc_df.columns = ['Category', 'Count']
                        chart = alt.Chart(vc_df).mark_bar().encode(x='Count:Q', y=alt.Y('Category:N', sort='-x'), tooltip=['Category', 'Count']).properties(title=f"Top 15 Categories (and 'Other') in {primary_col}")
                    st.altair_chart(chart, use_container_width=True)
                else: st.warning(f"Cannot plot `{primary_col}`. High cardinality columns with more than 50 unique values are not visualized for performance reasons.")
        else:
            st.subheader(f"Relationship between `{primary_col}` and `{secondary_col}`")
            secondary_type = 'numeric' if pd.api.types.is_numeric_dtype(df[secondary_col]) else 'categorical'
            if primary_type == 'numeric' and secondary_type == 'numeric':
                st.markdown("**Analysis:** A **scatter plot** shows correlation and clusters."); chart = alt.Chart(df).mark_circle(size=60, opacity=0.7).encode(x=alt.X(primary_col, scale=alt.Scale(zero=False)), y=alt.Y(secondary_col, scale=alt.Scale(zero=False)), tooltip=[primary_col, secondary_col]).interactive()
            elif primary_type == 'numeric' and secondary_type == 'categorical':
                st.markdown("**Analysis:** A **box plot** compares the numeric distribution across categories."); chart = alt.Chart(df).mark_boxplot(extent='min-max').encode(x=alt.X(secondary_col), y=alt.Y(primary_col), tooltip=[secondary_col, primary_col]).properties(title=f"Distribution of {primary_col} by {secondary_col}")
            elif primary_type == 'categorical' and secondary_type == 'numeric':
                st.markdown("**Analysis:** A **box plot** compares the numeric distribution across categories."); chart = alt.Chart(df).mark_boxplot(extent='min-max').encode(x=alt.X(primary_col), y=alt.Y(secondary_col), tooltip=[primary_col, secondary_col]).properties(title=f"Distribution of {secondary_col} by {primary_col}")
            else:
                st.markdown("**Analysis:** A **heatmap** shows the frequency of each category combination."); chart = alt.Chart(df).mark_rect().encode(x=alt.X(primary_col, title=primary_col), y=alt.Y(secondary_col, title=secondary_col), color=alt.Color('count()', scale=alt.Scale(scheme='viridis'), title='Frequency'), tooltip=[primary_col, secondary_col, 'count()']).properties(title=f"Frequency of {primary_col} vs. {secondary_col}")
            st.altair_chart(chart, use_container_width=True)

def display_processing_log(report):
    st.subheader("Log of All Actions Taken")
    if not report:
        st.info("No preprocessing actions have been run yet."); return
    def render_log(log_item):
        st.markdown(f"{log_item['icon']} **{log_item['title']}**")
        if log_item.get('details'):
            with st.container(border=True):
                for detail in log_item['details']: st.markdown(f"   • {detail}")
    categories = {"Normal": "🧹 Normal Preprocessing", "DS": "🔬 Advanced DS Preprocessing", "NLP": "📝 NLP Preprocessing"}
    for cat_code, cat_name in categories.items():
        logs = [log for log in report if log.get('category') == cat_code]
        if logs:
            with st.expander(f"**{cat_name} Log**", expanded=True):
                for log in logs: render_log(log); st.markdown("---")

# --- THIS IS THE MISSING FUNCTION THAT HAS BEEN RESTORED ---
def init_session_state():
    if 'df' not in st.session_state:
        st.session_state.df = None
        st.session_state.processed_df = None
        st.session_state.report = []
        st.session_state.initial_analysis = None
        st.session_state.uploaded_filename = None
        st.session_state.file_encoding = None

# --- Main App UI ---
init_session_state()

st.title("✨ The Intelligent Data Preprocessor")
st.markdown("Upload a CSV to get an instant, interactive analysis. Then, use the smart-configured controls to process your data with confidence.")
col1, col2 = st.columns([1.2, 2])

with col1:
    st.header("⚙️ Controls Panel")
    uploaded_file = st.file_uploader("1. Upload your CSV file", type="csv", key="file_uploader")
    if uploaded_file and uploaded_file.name != st.session_state.uploaded_filename:
        init_session_state(); st.session_state.uploaded_filename = uploaded_file.name
        df = load_csv_with_encoding_detection(uploaded_file)
        if df is not None:
            st.session_state.df = df
            with st.spinner("Performing deep analysis..."): st.session_state.initial_analysis = an.generate_full_analysis(df)
            st.rerun()
    if st.session_state.df is not None:
        df = st.session_state.df; analysis_report = st.session_state.initial_analysis
        quality_report = analysis_report.get('quality_summary', {}); char_report = analysis_report.get('characteristics', {})
        col_details = analysis_report['column_details']
        if st.button("🔄 Reset Selections to Default"):
            current_df, current_analysis, current_filename, current_encoding = st.session_state.df, st.session_state.initial_analysis, st.session_state.uploaded_filename, st.session_state.file_encoding
            init_session_state()
            st.session_state.df, st.session_state.initial_analysis, st.session_state.uploaded_filename, st.session_state.file_encoding = current_df, current_analysis, current_filename, current_encoding
            st.rerun()
        st.divider()
        with st.expander("🧹 Normal Preprocessing", expanded=True):
            with st.expander("🗑️ Column Removal & Renaming"):
                cols_to_drop = st.multiselect("Select columns to remove:", options=df.columns.tolist(), key="drop", default=quality_report.get('potential_id_cols', []))
                available_cols_for_rename = [col for col in df.columns if col not in cols_to_drop]
                cols_to_rename = st.multiselect("Select columns to rename:", options=available_cols_for_rename, key="rename_select")
                rename_map = {}
                for col in cols_to_rename:
                    new_name = st.text_input(f"New name for '{col}':", value=col, key=f"rename_{col}")
                    if new_name and new_name != col: rename_map[col] = new_name
            with st.expander("🧼 Basic Cleaning Operations", expanded=True):
                do_strip_whitespace = st.checkbox("Strip Whitespace from Text", value=True, key="strip")
                do_drop_duplicates = st.checkbox("Remove Duplicate Rows", value=char_report.get('has_duplicates', False), key="dupes")
                do_drop_constant = st.checkbox("Remove Constant Columns", value=bool(quality_report.get('constant_cols')), key="const")
                do_convert_dtypes = st.checkbox("Optimize Data Types (e.g., int, float)", value=True, key="dtypes")
            with st.expander("❓ Handle Missing Values (Per Column)"):
                do_handle_missing = st.checkbox("Enable Missing Value Handling", key="do_missing_values")
                missing_cols = [col for col, info in col_details.items() if info['missing_count'] > 0]
                if not missing_cols: st.info("No missing values detected.")
                elif do_handle_missing:
                    imputation_strategies = {}
                    for col in missing_cols:
                        info = col_details[col]
                        with st.container(border=True):
                            st.markdown(f"**Column:** `{col}` ({info['missing_count']} missing)"); options = ['Mean', 'Median', 'Mode', 'Constant', 'Drop Rows']
                            skew = info.get('skew'); default_method_index = 1 if skew and abs(skew) > 0.5 else (0 if skew is not None else 2)
                            selected_strategy = st.selectbox(f"Strategy for `{col}`", options, index=default_method_index, key=f"missing_{col}")
                            if selected_strategy == 'Constant':
                                constant_val = st.text_input(f"Constant value for `{col}`", key=f"constant_val_{col}")
                                imputation_strategies[col] = ('Constant', constant_val)
                            else: imputation_strategies[col] = selected_strategy
                else: st.info(f"Detected {len(missing_cols)} columns with missing values. Enable to configure.")
        with st.expander("🔬 Advanced DS Preprocessing"):
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist(); categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            with st.expander("📈 Outlier Handling"):
                outlier_cols_detected = quality_report.get('significant_outlier_cols', [])
                do_handle_outliers = st.checkbox("Enable Outlier Capping", value=bool(outlier_cols_detected), key="do_outliers")
                outlier_cols = st.multiselect("Apply to columns:", numeric_cols, default=outlier_cols_detected, key="outlier_cols", disabled=not do_handle_outliers)
                outlier_method = st.selectbox("Method:", ["iqr", "z-score"], key="outlier_method", disabled=not do_handle_outliers)
                outlier_multiplier = st.slider("Multiplier/Std Devs:", 1.0, 5.0, 1.5, 0.1, key="outlier_mult", disabled=not do_handle_outliers)
            with st.expander("📏 Feature Scaling"):
                do_scale_features = st.checkbox("Enable Feature Scaling", key="do_scaling")
                scaling_cols = st.multiselect("Columns to scale:", numeric_cols, default=numeric_cols, key="scaling_cols", disabled=not do_scale_features)
                scaling_method = st.selectbox("Scaling Method:", ["minmax", "standard"], key="scaling_method", disabled=not do_scale_features)
            with st.expander("🏷️ Feature Encoding"):
                do_encode_features = st.checkbox("Enable Feature Encoding", key="do_encoding")
                encoding_cols = st.multiselect("Columns to encode:", categorical_cols, default=categorical_cols, key="encoding_cols", disabled=not do_encode_features)
                ohe_threshold = st.number_input("One-Hot Threshold (max unique values):", 1, 50, 15, key="ohe_thresh", disabled=not do_encode_features)
            with st.expander("📉 Low Variance Removal"):
                do_remove_low_variance = st.checkbox("Enable Low Variance Removal", help="Run AFTER scaling for best results.", key="do_low_var")
                variance_threshold = st.number_input("Variance Threshold:", 0.0, 1.0, 0.01, format="%.3f", key="var_thresh", disabled=not do_remove_low_variance)
        with st.expander("📝 NLP Preprocessing"):
            text_cols = detect_text_heavy_columns(df)
            if not text_cols: st.info("No text-heavy columns detected.")
            else:
                st.success(f"Text column(s) detected: {', '.join(text_cols)}")
                nlp_target_col = st.selectbox("Select text column to process:", text_cols)
                nlp_operations = ['lowercase', 'remove_urls', 'remove_special_chars', 'remove_stopwords', 'lemmatize']
                default_nlp_ops = ['lowercase', 'remove_special_chars', 'remove_stopwords']
                if char_report.get('has_urls', False): default_nlp_ops.append('remove_urls')
                nlp_funcs_to_apply = st.multiselect("Select NLP operations:", options=nlp_operations, default=default_nlp_ops)
        st.divider()
        if st.button("🚀 Run All Preprocessing", type="primary"):
            with st.spinner("Processing... This may take a moment."):
                processed_df = df.copy(); report = []
                if 'imputation_strategies' in locals() and rename_map: imputation_strategies = {rename_map.get(k, k): v for k, v in imputation_strategies.items()}
                if 'outlier_cols' in locals() and rename_map: outlier_cols = [rename_map.get(c, c) for c in outlier_cols]
                if 'scaling_cols' in locals() and rename_map: scaling_cols = [rename_map.get(c, c) for c in scaling_cols]
                if 'encoding_cols' in locals() and rename_map: encoding_cols = [rename_map.get(c, c) for c in encoding_cols]
                if 'nlp_target_col' in locals() and rename_map: nlp_target_col = rename_map.get(nlp_target_col, nlp_target_col)
                processed_df, r = normal_pp.drop_columns_manual(processed_df, [], cols_to_drop); report.extend(r)
                processed_df, r = normal_pp.rename_columns_manual(processed_df, [], rename_map); report.extend(r)
                if do_strip_whitespace: processed_df, r = normal_pp.strip_whitespace_from_strings(processed_df, []); report.extend(r)
                if do_drop_duplicates: processed_df, r = normal_pp.drop_duplicates(processed_df, []); report.extend(r)
                if do_drop_constant: processed_df, r = normal_pp.drop_constant_columns(processed_df, []); report.extend(r)
                if do_convert_dtypes: processed_df, r = normal_pp.convert_dtypes(processed_df, []); report.extend(r)
                if do_handle_missing: processed_df, r = normal_pp.handle_missing_values_per_column(processed_df, [], strategies=imputation_strategies); report.extend(r)
                if 'nlp_target_col' in locals() and 'nlp_funcs_to_apply' in locals() and nlp_funcs_to_apply and nlp_target_col in processed_df.columns:
                    processed_df, r = nlp_pp.process_text_column(processed_df, [], nlp_target_col, nlp_funcs_to_apply); report.extend(r)
                if do_handle_outliers:
                    valid_cols = [c for c in outlier_cols if c in processed_df.columns and pd.api.types.is_numeric_dtype(processed_df[c])]
                    processed_df, r = ds_pp.detect_and_handle_outliers(processed_df, [], valid_cols, outlier_method, outlier_multiplier); report.extend(r)
                if do_encode_features:
                    valid_cols = [c for c in encoding_cols if c in processed_df.columns]
                    processed_df, r = ds_pp.encode_categorical_features(processed_df, [], valid_cols, ohe_threshold); report.extend(r)
                if do_scale_features:
                    valid_cols = [c for c in scaling_cols if c in processed_df.select_dtypes(include=np.number).columns]
                    processed_df, r = ds_pp.scale_features(processed_df, [], valid_cols, scaling_method); report.extend(r)
                if do_remove_low_variance:
                    processed_df, r = ds_pp.remove_low_variance_features(processed_df, [], variance_threshold); report.extend(r)
                st.session_state.processed_df = processed_df; st.session_state.report = report
                st.success("Preprocessing complete! View the results in the 'Data Viewer' and 'Processing Log' tabs.")

with col2:
    if st.session_state.df is None:
        st.info("👋 Welcome! Please upload a CSV file using the control panel on the left to begin.")
    else:
        active_tab = st.tabs(["**📊 Interactive Analysis**", "**🔬 Data Viewer**", "**📄 Processing Log**"])
        with active_tab[0]: display_interactive_analysis(st.session_state.df, st.session_state.initial_analysis)
        with active_tab[1]:
            st.subheader("Original Data Preview"); st.dataframe(st.session_state.df, use_container_width=True)
            if st.session_state.processed_df is not None:
                st.divider(); st.subheader("✨ Final Processed Data")
                st.dataframe(st.session_state.processed_df, use_container_width=True)
                csv_data = convert_df_to_csv(st.session_state.processed_df)
                st.download_button("📥 Download Processed CSV", csv_data, f"processed_{st.session_state.uploaded_filename}", "text/csv")
        with active_tab[2]: display_processing_log(st.session_state.report)