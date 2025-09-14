import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import parallel_coordinates

st.set_page_config(page_title="Visualization Explorer", layout="wide")

st.title("📊 Visualization Explorer")
st.write("Choose the dimensionality and a visualization technique to explore.")

# Dataset selection
st.sidebar.header("Dataset Options")
data_choice = st.sidebar.radio("Choose Dataset", ["Default (Iris)", "Upload CSV"])

if data_choice == "Default (Iris)":
    iris = load_iris(as_frame=True)
    df = iris.frame
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file to proceed.")
        st.stop()

# Sidebar dimension selection
dimension = st.sidebar.selectbox("Select Dimension", ["1D", "2D", "3D", "Multi-Dimensional"])

# 1D Visualizations
if dimension == "1D":
    viz = st.sidebar.selectbox("Choose Visualization", ["Histogram", "Boxplot"])

    if viz == "Histogram":
        col = st.selectbox("Select Column", df.select_dtypes(include=np.number).columns)
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=20, color="skyblue", edgecolor="black")
        ax.set_title(f"Histogram of {col}")
        st.pyplot(fig)

    elif viz == "Boxplot":
        col = st.selectbox("Select Column", df.select_dtypes(include=np.number).columns)
        fig, ax = plt.subplots()
        sns.boxplot(y=df[col], ax=ax, color="lightgreen")
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)

# 2D Visualizations
elif dimension == "2D":
    viz = st.sidebar.selectbox("Choose Visualization", ["Scatter Plot", "Heatmap"])

    if viz == "Scatter Plot":
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) < 2:
            st.error("Need at least 2 numeric columns for scatter plot")
        else:
            x_col = st.selectbox("X-axis", num_cols)
            y_col = st.selectbox("Y-axis", num_cols)
            hue_col = st.selectbox("Color by (optional)", [None] + list(df.columns))
            fig, ax = plt.subplots()
            if hue_col and hue_col in df.columns:
                sns.scatterplot(x=df[x_col], y=df[y_col], hue=df[hue_col], palette="deep", ax=ax)
            else:
                sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
            ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
            st.pyplot(fig)

    elif viz == "Heatmap":
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

# 3D Visualizations
elif dimension == "3D":
    viz = st.sidebar.selectbox("Choose Visualization", ["3D Scatter Plot"])

    if viz == "3D Scatter Plot":
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) < 3:
            st.error("Need at least 3 numeric columns for 3D scatter plot")
        else:
            x_col = st.selectbox("X-axis", num_cols)
            y_col = st.selectbox("Y-axis", num_cols)
            z_col = st.selectbox("Z-axis", num_cols)
            color_col = st.selectbox("Color by (optional)", [None] + list(df.columns))

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            if color_col and color_col in df.columns:
                scatter = ax.scatter(
                    df[x_col], df[y_col], df[z_col],
                    c=pd.factorize(df[color_col])[0], cmap="viridis", s=50
                )
                fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
            else:
                ax.scatter(df[x_col], df[y_col], df[z_col], s=50, color="blue")

            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_zlabel(z_col)
            ax.set_title("3D Scatter Plot")
            st.pyplot(fig)

# Multi-Dimensional Visualizations
elif dimension == "Multi-Dimensional":
    viz = st.sidebar.selectbox("Choose Visualization", ["Pairplot", "Parallel Coordinates"])

    if viz == "Pairplot":
        try:
            fig = sns.pairplot(df, diag_kind="hist")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating pairplot: {e}")

    elif viz == "Parallel Coordinates":
        if "target" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            parallel_coordinates(df, "target", colormap=plt.cm.Set2)
            ax.set_title("Parallel Coordinates Plot")
            st.pyplot(fig)
        else:
            st.warning("Parallel coordinates require a categorical 'target' column.")
