import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dotenv import load_dotenv
from google import genai
import os
import io

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        gemini_client = None


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ==============================
       GLOBAL
    ============================== */

    .stApp {
        background: #f7f8fc;
        color: #172033;
    }

    .main .block-container {
        padding: 2rem 2.5rem 3rem 2.5rem;
        max-width: 1500px;
    }

    /* ==============================
       SIDEBAR
    ============================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #111827 0%,
            #172033 100%
        );
        min-width: 280px;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    .sidebar-logo {
        font-size: 23px;
        font-weight: 800;
        padding: 10px 5px 25px 5px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 25px;
    }

    .sidebar-logo span {
        background: linear-gradient(
            90deg,
            #60a5fa,
            #8b5cf6
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sidebar-section {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        color: #94a3b8 !important;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    /* ==============================
       HEADER
    ============================== */

    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }

    .page-title {
        font-size: 36px;
        font-weight: 800;
        color: #172033;
        margin-bottom: 5px;
    }

    .page-subtitle {
        font-size: 16px;
        color: #64748b;
        margin-bottom: 25px;
    }

    .gemini-online {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #ecfdf3;
        color: #16a34a;
        border: 1px solid #bbf7d0;
        padding: 10px 18px;
        border-radius: 25px;
        font-weight: 700;
    }

    /* ==============================
       METRIC CARDS
    ============================== */

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 15px;
        padding: 22px;
        min-height: 130px;
        box-shadow: 0 4px 15px rgba(15,23,42,0.05);
    }

    .metric-label {
        font-size: 14px;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #172033;
    }

    .metric-icon {
        font-size: 27px;
        margin-bottom: 10px;
    }

    /* ==============================
       CARDS
    ============================== */

    .content-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 15px;
        padding: 22px;
        box-shadow: 0 4px 15px rgba(15,23,42,0.04);
        margin-bottom: 20px;
    }

    .card-title {
        font-size: 19px;
        font-weight: 750;
        color: #172033;
        margin-bottom: 5px;
    }

    .card-description {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 15px;
    }

    /* ==============================
       AI CARD
    ============================== */

    .ai-card {
        background: linear-gradient(
            135deg,
            #ffffff 0%,
            #f5f3ff 100%
        );
        border: 1px solid #ddd6fe;
        border-radius: 18px;
        padding: 25px;
        box-shadow: 0 5px 20px rgba(124,58,237,0.08);
        margin-top: 20px;
    }

    .ai-title {
        color: #6d4aff;
        font-size: 22px;
        font-weight: 800;
    }

    .ai-description {
        color: #64748b;
        margin-bottom: 15px;
    }

    /* ==============================
       BUTTONS
    ============================== */

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #ddd6fe;
        background: white;
        color: #5b3cc4;
        font-weight: 650;
        min-height: 42px;
    }

    .stButton > button:hover {
        border-color: #7c3aed;
        color: #7c3aed;
        background: #f5f3ff;
    }

    /* ==============================
       FILE UPLOADER
    ============================== */

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
    }

    /* ==============================
       DATAFRAME
    ============================== */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ==============================
       TABS
    ============================== */

    button[data-baseweb="tab"] {
        font-weight: 650;
        color: #64748b;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #6d4aff;
    }

    /* ==============================
       INPUTS
    ============================== */

    textarea,
    input {
        color: #172033 !important;
    }

    textarea::placeholder,
    input::placeholder {
        color: #94a3b8 !important;
    }

    /* ==============================
       FOOTER
    ============================== */

    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 30px;
        font-size: 13px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-logo">🤖 <span>AI Data Analyst</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section">NAVIGATION</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "",
        [
            "🏠 Dashboard",
            "🤖 AI Analyst",
            "🗄️ Data Explorer",
            "📊 Statistics",
            "📈 Interactive Charts",
            "💻 SQL Query"
        ],
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="sidebar-section">DATA MANAGEMENT</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx", "xls"],
        help="Upload a CSV or Excel dataset"
    )

    if uploaded_file:

        try:

            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.df = df
            st.session_state.file_name = uploaded_file.name

            st.success("✓ File loaded successfully")

        except Exception as e:

            st.error(f"Unable to read file: {e}")

    if st.session_state.df is not None:

        st.markdown(
            f"""
            <div style="
                background:#1e293b;
                border:1px solid #334155;
                border-radius:12px;
                padding:15px;
                margin-top:10px;
            ">
                <b>📄 {st.session_state.file_name}</b>
                <br>
                <small style="color:#94a3b8;">
                    {len(st.session_state.df):,} rows ×
                    {len(st.session_state.df.columns)} columns
                </small>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="sidebar-section">CAPABILITIES</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ✓ AI-Powered Analysis  
    <br>
    ✓ Smart Visualizations  
    <br>
    ✓ Natural Language Queries  
    <br>
    ✓ Statistical Insights  
    <br>
    ✓ SQL Query Support
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.caption("Built with ❤️ using Streamlit")


# ============================================================
# DATA CHECK
# ============================================================

df = st.session_state.df

if df is None:

    st.markdown(
        '<div class="page-title">Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Upload a CSV or Excel dataset to begin your analysis.'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "👈 Upload your dataset using the sidebar to get started."
    )

    st.stop()


# ============================================================
# COMMON DATA INFORMATION
# ============================================================

rows = len(df)
columns = len(df.columns)
missing = int(df.isna().sum().sum())
duplicates = int(df.duplicated().sum())


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="page-title">Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Overview of your dataset and AI insights'
        '</div>',
        unsafe_allow_html=True
    )

    if gemini_client:
        st.markdown(
            '<div class="gemini-online">🟢 Gemini Online</div>',
            unsafe_allow_html=True
        )
    else:
        st.warning(
            "Gemini is not configured. Add GEMINI_API_KEY to Streamlit Secrets."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">📄</div>
                <div class="metric-label">TOTAL ROWS</div>
                <div class="metric-value">{rows:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">📋</div>
                <div class="metric-label">TOTAL COLUMNS</div>
                <div class="metric-value">{columns}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">⚠️</div>
                <div class="metric-label">MISSING VALUES</div>
                <div class="metric-value">{missing:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">♻️</div>
                <div class="metric-label">DUPLICATES</div>
                <div class="metric-value">{duplicates:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # DATASET PREVIEW + INFO
    # --------------------------------------------------------

    left, right = st.columns([2.1, 1])

    with left:

        st.markdown(
            '<div class="content-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="card-title">📋 Dataset Preview</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="card-description">'
            'Preview the first records of your dataset'
            '</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            df.head(8),
            use_container_width=True,
            hide_index=False
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:

        st.markdown(
            f"""
            <div class="content-card">

            <div class="card-title">ℹ️ Dataset Info</div>

            <br>

            <b>File Name</b>
            <p>{st.session_state.file_name}</p>

            <b>File Type</b>
            <p>{'Excel' if st.session_state.file_name.endswith(('xlsx','xls')) else 'CSV'}</p>

            <b>Rows</b>
            <p>{rows:,}</p>

            <b>Columns</b>
            <p>{columns}</p>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    st.markdown(
        '<div class="content-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card-title">📊 Quick Statistics</div>',
        unsafe_allow_html=True
    )

    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:

        stats = pd.DataFrame({
            "Mean": numeric_df.mean(),
            "Median": numeric_df.median(),
            "Mode": numeric_df.mode().iloc[0],
            "Minimum": numeric_df.min(),
            "Maximum": numeric_df.max()
        })

        st.dataframe(
            stats.round(2),
            use_container_width=True
        )

    else:

        st.info(
            "No numerical columns available for statistical analysis."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # ASK YOUR DATA
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="ai-card">

        <div class="ai-title">
        ✨ Ask Your Data
        </div>

        <div class="ai-description">
        Ask anything about your dataset using natural language.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    question = st.text_area(
        "Ask a question",
        placeholder=(
            "Example: How many students are there in each class?"
        ),
        height=120,
        label_visibility="collapsed"
    )

    if st.button(
        "✨ Analyze",
        use_container_width=True
    ):

        if not question.strip():

            st.warning("Please enter a question.")

        elif not gemini_client:

            st.error(
                "Gemini is not configured. "
                "Please add GEMINI_API_KEY to Streamlit Secrets."
            )

        else:

            with st.spinner("🤖 Gemini is analyzing your data..."):

                try:

                    data_sample = df.head(50).to_csv(index=False)

                    prompt = f"""
You are an expert data analyst.

Analyze the following dataset and answer the user's question.

Dataset:
{data_sample}

User question:
{question}

Give a clear and concise answer.
Use numbers from the dataset whenever possible.
"""

                    response = gemini_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )

                    st.markdown("### 🤖 Gemini Analysis")
                    st.write(response.text)

                except Exception as e:

                    st.error(
                        f"Gemini analysis failed: {e}"
                    )


# ============================================================
# AI ANALYST
# ============================================================

elif page == "🤖 AI Analyst":

    st.markdown(
        '<div class="page-title">🤖 AI Analyst</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Ask questions about your data using natural language'
        '</div>',
        unsafe_allow_html=True
    )

    question = st.text_area(
        "Your question",
        placeholder="Example: What is the average Math Score?",
        height=150
    )

    if st.button("✨ Analyze Data", use_container_width=True):

        if not question:

            st.warning("Please enter a question.")

        elif not gemini_client:

            st.error("Gemini API is not configured.")

        else:

            with st.spinner("Analyzing..."):

                try:

                    prompt = f"""
You are a professional data analyst.

Dataset columns:
{list(df.columns)}

Dataset summary:
{df.describe(include="all").to_string()}

Question:
{question}

Provide a useful answer.
"""

                    response = gemini_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )

                    st.markdown("### 🤖 Analysis Result")
                    st.write(response.text)

                except Exception as e:

                    st.error(str(e))


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "🗄️ Data Explorer":

    st.markdown(
        '<div class="page-title">🗄️ Data Explorer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Explore and inspect your complete dataset'
        '</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df,
        use_container_width=True,
        height=600
    )


# ============================================================
# STATISTICS
# ============================================================

elif page == "📊 Statistics":

    st.markdown(
        '<div class="page-title">📊 Statistics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Mean, Median, Mode and descriptive statistics'
        '</div>',
        unsafe_allow_html=True
    )

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:

        st.warning(
            "Your dataset does not contain numerical columns."
        )

    else:

        statistics = pd.DataFrame({
            "Mean": numeric_df.mean(),
            "Median": numeric_df.median(),
            "Mode": numeric_df.mode().iloc[0],
            "Std Dev": numeric_df.std(),
            "Minimum": numeric_df.min(),
            "Maximum": numeric_df.max()
        })

        st.markdown(
            '<div class="content-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="card-title">'
            '📈 Mean / Median / Mode'
            '</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            statistics.round(2),
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

        selected_column = st.selectbox(
            "Select numerical column",
            numeric_df.columns
        )

        fig = px.histogram(
            df,
            x=selected_column,
            title=f"Distribution of {selected_column}",
            marginal="box"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# INTERACTIVE CHARTS
# ============================================================

elif page == "📈 Interactive Charts":

    st.markdown(
        '<div class="page-title">📈 Interactive Charts</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Create interactive visualizations from your dataset'
        '</div>',
        unsafe_allow_html=True
    )

    numeric_columns = list(
        df.select_dtypes(include=np.number).columns
    )

    all_columns = list(df.columns)

    chart_type = st.selectbox(
        "Chart Type",
        [
            "Histogram",
            "Bar Chart",
            "Scatter Plot",
            "Box Plot",
            "Line Chart"
        ]
    )

    if chart_type == "Histogram":

        column = st.selectbox(
            "Select column",
            all_columns
        )

        fig = px.histogram(
            df,
            x=column,
            title=f"Distribution of {column}"
        )

    elif chart_type == "Bar Chart":

        x_column = st.selectbox(
            "X-axis",
            all_columns
        )

        if numeric_columns:

            y_column = st.selectbox(
                "Y-axis",
                numeric_columns
            )

            fig = px.bar(
                df,
                x=x_column,
                y=y_column,
                title=f"{y_column} by {x_column}"
            )

        else:

            st.warning("No numerical columns available.")
            st.stop()

    elif chart_type == "Scatter Plot":

        if len(numeric_columns) < 2:

            st.warning(
                "At least two numerical columns are required."
            )
            st.stop()

        x_column = st.selectbox(
            "X-axis",
            numeric_columns
        )

        y_column = st.selectbox(
            "Y-axis",
            numeric_columns,
            index=1
        )

        fig = px.scatter(
            df,
            x=x_column,
            y=y_column,
            title=f"{y_column} vs {x_column}"
        )

    elif chart_type == "Box Plot":

        column = st.selectbox(
            "Select numerical column",
            numeric_columns
        )

        fig = px.box(
            df,
            y=column,
            title=f"Box Plot - {column}"
        )

    else:

        if not numeric_columns:

            st.warning(
                "No numerical columns available."
            )
            st.stop()

        column = st.selectbox(
            "Select numerical column",
            numeric_columns
        )

        fig = px.line(
            df,
            y=column,
            title=f"{column} Trend"
        )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# SQL QUERY
# ============================================================

elif page == "💻 SQL Query":

    st.markdown(
        '<div class="page-title">💻 SQL Query</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Query your dataset using SQL'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "SQL execution can be connected to your existing MCP SQL server here."
    )

    sql_query = st.text_area(
        "Enter SQL query",
        placeholder="SELECT * FROM data LIMIT 10;",
        height=150
    )

    if st.button(
        "▶ Execute Query",
        use_container_width=True
    ):

        st.warning(
            "Connect this section to your existing MCP SQL tool."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        AI Data Analyst • Intelligent Data Analysis •
        Interactive Visualizations
    </div>
    """,
    unsafe_allow_html=True
)
