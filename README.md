# 🔍 AutoEDA Web App

A Streamlit-based AutoEDA (Automated Exploratory Data Analysis) application that helps you explore and clean datasets with ease. Upload a CSV or Excel file and interactively perform data preview, profiling, missing value handling, and visualizations.

## 🚀 Features

- **Dataset Upload**: Upload `.csv` or `.xlsx` files
- **Dataset Preview**: View first few rows, shape, column names, and data types
- **Auto EDA Report**:
  - `ydata-profiling` (formerly `pandas-profiling`)
  - `sweetviz`
- **Missing Value Analysis**:
  - View count and percentage of missing values
  - Handle with: mean, median, mode, constant, forward/backward fill, or drop
- **Summary Stats**: Numerical and categorical summaries
- **Visualizations**:
  - Histograms for distribution
  - Correlation heatmaps
  - Pairplots (optional for smaller datasets)
- **Download Cleaned Dataset**: Save your preprocessed data

## 📦 Installation

```bash
pip install -r requirements.txt
