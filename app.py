import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import os
from io import StringIO, BytesIO
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import sweetviz
import tempfile
import sklearn.impute as impute

st.set_page_config(page_title="AutoEDA App", layout="wide")
st.title("🔍 AutoEDA Web App")

# ----------------------- Utility Functions -----------------------

def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

def show_file_info(uploaded_file):
    file_details = {"filename": uploaded_file.name, "size": uploaded_file.size}
    st.write("### File Info")
    st.json(file_details)

def preview_data(df):
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())
    if st.checkbox("Show Data Shape & Types"):
        st.write("Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())
        st.write("Data Types:", df.dtypes)

def generate_eda_report(df, method="ydata-profiling"):
    st.subheader("📑 Auto EDA Report")
    if method == "ydata-profiling":
        profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
        st_profile_report(profile)
    elif method == "sweetviz":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            report = sweetviz.analyze(df)
            report.show_html(filepath=tmp.name, open_browser=False)
            with open(tmp.name, 'r') as f:
                html = f.read()
            st.components.v1.html(html, height=800, scrolling=True)

def analyze_missing_values(df):
    st.subheader("🧩 Missing Values")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_percent})
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    st.write(missing_df)
    return missing_df

def handle_missing_values(df, col, method, value=None):
    if method == "Mean":
        df[col].fillna(df[col].mean(), inplace=True)
    elif method == "Median":
        df[col].fillna(df[col].median(), inplace=True)
    elif method == "Mode":
        df[col].fillna(df[col].mode()[0], inplace=True)
    elif method == "Constant" and value is not None:
        df[col].fillna(value, inplace=True)
    elif method == "Forward Fill":
        df[col].fillna(method='ffill', inplace=True)
    elif method == "Backward Fill":
        df[col].fillna(method='bfill', inplace=True)
    elif method == "Drop Rows":
        df.dropna(subset=[col], inplace=True)
    elif method == "Drop Column":
        df.drop(columns=[col], inplace=True)
    return df

def plot_distribution(df, column):
    fig, ax = plt.subplots()
    if column in df.columns:
        sns.histplot(df[column].dropna(), kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.warning(f"Column '{column}' is no longer in the dataset.")

def show_correlation(df):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def show_pairplot(df, cols):
    fig = sns.pairplot(df[cols].dropna())
    st.pyplot(fig)

def get_csv_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_data.csv">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

# ----------------------- Main App Logic -----------------------

if 'df' not in st.session_state:
    st.session_state.df = None

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx'])
if uploaded_file:
    show_file_info(uploaded_file)
    df = load_data(uploaded_file)
    st.session_state.df = df
    st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")

if st.session_state.df is not None:
    df = st.session_state.df
    preview_data(df)

    # EDA Report
    eda_choice = st.radio("Choose EDA tool:", ["ydata-profiling", "sweetviz"])
    generate_eda_report(df, eda_choice)

    # Missing Values
    missing_df = analyze_missing_values(df)
    if not missing_df.empty:
        st.markdown("### Handle Missing Values")
        col_to_fix = st.selectbox("Select column to fix", missing_df.index.tolist())
        method = st.selectbox("Imputation method", ["Mean", "Median", "Mode", "Constant", "Forward Fill", "Backward Fill", "Drop Rows", "Drop Column"])
        value = None
        if method == "Constant":
            value = st.text_input("Enter constant value")
        if st.button("Apply Fix"):
            df = handle_missing_values(df, col_to_fix, method, value)
            st.success(f"Missing values in {col_to_fix} handled using {method} method")
            st.session_state.df = df

    # Summary
    st.subheader("📈 Data Summary")
    st.write("### Numerical Summary")
    st.dataframe(df.describe())
    st.write("### Categorical Summary")
    st.dataframe(df.select_dtypes(include=['object']).describe())

    st.write("### Distribution of Variables")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:  # Ensure there are numeric columns
        col = st.selectbox("Select column to plot histogram", num_cols)
        plot_distribution(df, col)
    else:
        st.warning("No numeric columns available for plotting.")

    st.subheader("📊 Correlation & Pairplot")
    if st.checkbox("Show Correlation Heatmap"):
        show_correlation(df)

    if st.checkbox("Show Pairplot (slow for large datasets)"):
        subset = st.multiselect("Select up to 5 numeric columns", num_cols, default=num_cols[:2])
        if subset:
            show_pairplot(df, subset)

    st.subheader("💾 Download Cleaned Dataset")
    get_csv_download_link(df)

    st.markdown("### 🚀 Optional: Upload to GitHub (coming soon)")
    st.info("You can integrate GitHub upload using PyGithub or Git CLI in the future.")
