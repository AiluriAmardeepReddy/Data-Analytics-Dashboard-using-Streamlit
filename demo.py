import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page setup
st.set_page_config(page_title="Data Analytics Dashboard", layout="wide")
sns.set_style("whitegrid")

# ================== UNIVERSAL CLEANING FUNCTION ==================
def clean_dataset(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mean(), inplace=True)

    # Convert to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    # Remove outliers using IQR method
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df

# ================== STREAMLIT APP ==================
st.title("📊 Advanced Data Analytics Dashboard")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "EDA", "Visualizations", "Insights"])
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# ================== MAIN APP ==================
if uploaded_file:
    raw_df = load_data(uploaded_file)
    cleaned_df = clean_dataset(raw_df)

    # Sidebar toggle
    use_cleaned = st.sidebar.checkbox("Use Cleaned Data", value=True)
    df = cleaned_df if use_cleaned else raw_df
    st.sidebar.write("✅ Using Cleaned Data" if use_cleaned else "⚠ Using Raw Data")

    # ================= OVERVIEW =================
    if page == "Overview":
        st.subheader("🔍 Dataset Overview")

        col1, col2 = st.columns(2)
        col1.metric("Raw Rows", raw_df.shape[0])
        col1.metric("Raw Columns", raw_df.shape[1])
        col2.metric("Cleaned Rows", cleaned_df.shape[0])
        col2.metric("Cleaned Columns", cleaned_df.shape[1])

        st.write("### Dataset Preview")
        st.dataframe(df.head(5))

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Dataset", csv, "dataset.csv", "text/csv")

    # ================= EDA =================
    elif page == "EDA":
        st.subheader("📋 Exploratory Data Analysis")

        st.write("### Missing Values")
        st.dataframe(df.isnull().sum())

        st.write("### Duplicate Rows")
        st.write(df.duplicated().sum())

        st.write("### Summary Statistics")
        st.dataframe(df.describe(include="all"))

        st.write("### Correlation Heatmap")
        num_df = df.select_dtypes(include=["int64", "float64"])
        if not num_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.info("No numeric columns available for correlation heatmap.")

    # ================= VISUALIZATIONS =================
    elif page == "Visualizations":
        st.subheader("📊 Create Visualizations")

        chart_type = st.selectbox(
            "Select Chart Type",
            ["Histogram", "Boxplot", "Scatter", "Bar", "Line", "Pie"]
        )

        all_columns = df.columns.tolist()
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = df.select_dtypes(include=["object"]).columns

        if chart_type == "Histogram":
            selected_num = st.selectbox("Select Numeric Column", num_cols)
            if selected_num:
                fig, ax = plt.subplots()
                sns.histplot(df[selected_num].dropna(), kde=True, ax=ax)
                st.pyplot(fig)

        elif chart_type == "Boxplot":
            x_axis = st.selectbox("X-axis (Categorical)", cat_cols)
            y_axis = st.selectbox("Y-axis (Numeric)", num_cols)
            if x_axis and y_axis:
                fig, ax = plt.subplots()
                sns.boxplot(x=df[x_axis], y=df[y_axis], ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

        elif chart_type == "Scatter":
            x_axis = st.selectbox("X-axis", num_cols)
            y_axis = st.selectbox("Y-axis", num_cols)
            if x_axis and y_axis:
                fig, ax = plt.subplots()
                ax.scatter(df[x_axis], df[y_axis], alpha=0.7)
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_title(f"{x_axis} vs {y_axis}")
                st.pyplot(fig)

        elif chart_type == "Bar":
            x_axis = st.selectbox("X-axis (Categorical)", cat_cols)
            y_axis = st.selectbox("Y-axis (Numeric)", num_cols)
            if x_axis and y_axis:
                fig, ax = plt.subplots()
                sns.barplot(x=df[x_axis], y=df[y_axis], ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

        elif chart_type == "Line":
            x_axis = st.selectbox("X-axis", all_columns)
            y_axis = st.selectbox("Y-axis", num_cols)
            if x_axis and y_axis:
                fig, ax = plt.subplots()
                sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

        elif chart_type == "Pie":
            selected_cat = st.selectbox("Select Categorical Column", cat_cols)
            if selected_cat:
                fig, ax = plt.subplots(figsize=(5, 5))
                df[selected_cat].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
                ax.set_ylabel("")
                st.pyplot(fig)

    # ================= INSIGHTS =================
    elif page == "Insights":
        st.subheader("📝 Write Your Insights")
        st.info("Use this space to note your key findings from analysis.")
        st.text_area("Write insights here...", height=200)

else:
    st.warning("⚠ Please upload a CSV file to get started.")
