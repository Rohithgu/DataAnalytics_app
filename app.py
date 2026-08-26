import os
import re
import json
import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dotenv import load_dotenv
from google import genai


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# 3. GEMINI CLIENT
# ============================================================

if API_KEY:

    client = genai.Client(
        api_key=API_KEY
    )

else:

    client = None


# ============================================================
# 4. SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# 5. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       MAIN APPLICATION
       ====================================================== */

    .stApp {

        background:
            linear-gradient(
                135deg,
                #0b1020 0%,
                #111827 50%,
                #0f172a 100%
            );

        color: #f8fafc;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                #0f172a,
                #111827
            );

        border-right:
            1px solid
            rgba(
                255,
                255,
                255,
                0.08
            );
    }


    /* ======================================================
       HEADINGS
       ====================================================== */

    h1,
    h2,
    h3 {

        color: #f8fafc !important;
    }


    /* ======================================================
       MAIN TITLE
       ====================================================== */

    .main-title {

        font-size: 42px;

        font-weight: 800;

        margin-bottom: 4px;

        background:
            linear-gradient(
                90deg,
                #60a5fa,
                #a78bfa,
                #22d3ee
            );

        -webkit-background-clip: text;

        -webkit-text-fill-color:
            transparent;
    }


    .subtitle {

        color: #94a3b8;

        font-size: 16px;

        margin-bottom: 25px;
    }


    /* ======================================================
       STATUS
       ====================================================== */

    .status-online {

        display: inline-block;

        padding:
            7px 14px;

        border-radius: 20px;

        background:
            rgba(
                34,
                197,
                94,
                0.12
            );

        border:
            1px solid
            rgba(
                34,
                197,
                94,
                0.35
            );

        color: #4ade80;

        font-size: 13px;

        font-weight: 600;
    }


    .status-offline {

        display: inline-block;

        padding:
            7px 14px;

        border-radius: 20px;

        background:
            rgba(
                239,
                68,
                68,
                0.12
            );

        border:
            1px solid
            rgba(
                239,
                68,
                68,
                0.35
            );

        color: #f87171;

        font-size: 13px;

        font-weight: 600;
    }


    /* ======================================================
       KPI CARDS
       ====================================================== */

    .kpi-card {

        background:
            rgba(
                255,
                255,
                255,
                0.045
            );

        border:
            1px solid
            rgba(
                255,
                255,
                255,
                0.09
            );

        border-radius: 18px;

        padding: 20px;

        min-height: 120px;

        box-shadow:
            0 8px 30px
            rgba(
                0,
                0,
                0,
                0.20
            );

        transition:
            transform 0.2s ease;
    }


    .kpi-card:hover {

        transform:
            translateY(-3px);
    }


    .kpi-icon {

        font-size: 26px;

        margin-bottom: 10px;
    }


    .kpi-label {

        color: #94a3b8;

        font-size: 13px;

        text-transform:
            uppercase;

        letter-spacing:
            0.08em;
    }


    .kpi-value {

        color: #f8fafc;

        font-size: 30px;

        font-weight: 800;

        margin-top: 5px;
    }


    /* ======================================================
       PANELS
       ====================================================== */

    .glass-panel {

        background:
            rgba(
                255,
                255,
                255,
                0.035
            );

        border:
            1px solid
            rgba(
                255,
                255,
                255,
                0.08
            );

        border-radius: 20px;

        padding: 22px;

        margin-bottom: 20px;
    }


    /* ======================================================
       CHAT
       ====================================================== */

    .ai-message {

        background:
            rgba(
                255,
                255,
                255,
                0.06
            );

        border:
            1px solid
            rgba(
                255,
                255,
                255,
                0.08
            );

        padding:
            14px 18px;

        border-radius:
            18px 18px 18px 4px;

        margin:
            12px 0;

        color: #e2e8f0;

        max-width: 90%;
    }


    /* ======================================================
       VERIFIED BADGE
       ====================================================== */

    .exact-badge {

        display: inline-block;

        padding:
            6px 12px;

        border-radius:
            15px;

        background:
            rgba(
                34,
                197,
                94,
                0.12
            );

        color: #4ade80;

        border:
            1px solid
            rgba(
                34,
                197,
                94,
                0.3
            );

        font-size: 12px;

        font-weight: 700;
    }


    /* ======================================================
       FEATURE CARDS
       ====================================================== */

    .feature-card {

        background:
            rgba(
                255,
                255,
                255,
                0.035
            );

        border:
            1px solid
            rgba(
                255,
                255,
                255,
                0.08
            );

        border-radius: 16px;

        padding: 18px;

        height: 100%;
    }


    .feature-icon {

        font-size: 28px;

        margin-bottom: 8px;
    }


    .feature-title {

        font-weight: 700;

        color: #f8fafc;

        margin-bottom: 5px;
    }


    .feature-text {

        color: #94a3b8;

        font-size: 13px;

        line-height: 1.5;
    }


    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] {

        background:
            rgba(
                255,
                255,
                255,
                0.035
            );

        border-radius: 15px;

        padding: 8px;
    }


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {

        border-radius:
            10px;

        border:
            1px solid
            rgba(
                255,
                255,
                255,
                0.12
            );

        font-weight: 600;

        transition:
            all 0.2s ease;
    }


    .stButton > button:hover {

        transform:
            translateY(-1px);

        border-color:
            #60a5fa;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {

        text-align:
            center;

        color:
            #64748b;

        font-size:
            12px;

        padding:
            25px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 6. LOAD DATASET
# ============================================================

def load_dataset(uploaded_file):

    filename = uploaded_file.name.lower()


    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    if filename.endswith(".csv"):

        encodings = [
            "utf-8",
            "utf-8-sig",
            "cp1252",
            "latin1"
        ]

        for encoding in encodings:

            try:

                uploaded_file.seek(0)

                return pd.read_csv(
                    uploaded_file,
                    encoding=encoding
                )

            except UnicodeDecodeError:

                continue

        raise ValueError(
            "Could not decode the CSV file."
        )


    # --------------------------------------------------------
    # XLSX
    # --------------------------------------------------------

    if filename.endswith(".xlsx"):

        uploaded_file.seek(0)

        return pd.read_excel(
            uploaded_file
        )


    # --------------------------------------------------------
    # XLS
    # --------------------------------------------------------

    if filename.endswith(".xls"):

        uploaded_file.seek(0)

        return pd.read_excel(
            uploaded_file
        )


    raise ValueError(
        "Only CSV, XLSX and XLS files are supported."
    )


# ============================================================
# 7. COLUMN NAME NORMALIZATION
# ============================================================

def normalize_column_name(column):

    return (
        str(column)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
    )


# ============================================================
# 8. FIND COLUMN
# ============================================================

def find_column(
    df,
    names=None,
    contains=None
):

    names = names or set()

    contains = contains or []


    # Exact match

    for column in df.columns:

        normalized = (
            normalize_column_name(
                column
            )
        )

        if normalized in names:

            return column


    # Partial match

    for column in df.columns:

        normalized = (
            normalize_column_name(
                column
            )
        )

        for word in contains:

            if word in normalized:

                return column


    return None


# ============================================================
# 9. FIND CLASS COLUMN
# ============================================================

def find_class_column(df):

    return find_column(

        df,

        names={
            "class",
            "class name",
            "class number",
            "standard"
        },

        contains=[
            "class"
        ]

    )


# ============================================================
# 10. FIND GRADE COLUMN
# ============================================================

def find_grade_column(df):

    return find_column(

        df,

        names={
            "grade",
            "grades",
            "result grade",
            "final grade"
        },

        contains=[
            "grade"
        ]

    )


# ============================================================
# 11. FIND LOCATION COLUMN
# ============================================================

def find_location_column(df):

    return find_column(

        df,

        names={
            "city",
            "city name",
            "location",
            "place",
            "town",
            "district",
            "student city",
            "student location",
            "hometown",
            "home town",
            "address"
        },

        contains=[
            "city",
            "location",
            "district",
            "hometown"
        ]

    )


# ============================================================
# 12. NORMALIZE CLASS
# ============================================================

def normalize_class(value):

    if pd.isna(value):

        return None


    text = str(
        value
    ).strip()


    match = re.search(
        r"(\d+)",
        text
    )


    if match:

        return int(
            match.group(1)
        )


    return None


# ============================================================
# 13. NORMALIZE GRADE
# ============================================================

def normalize_grade(value):

    if pd.isna(value):

        return ""


    return (
        str(value)
        .strip()
        .upper()
    )


# ============================================================
# 14. EXTRACT GRADE
# ============================================================

def extract_grade(question):

    match = re.search(

        r"grade\s*"
        r"['\"]?"
        r"([a-z])"
        r"['\"]?",

        question,

        re.IGNORECASE

    )


    if match:

        return (
            match.group(1)
            .upper()
        )


    return None


# ============================================================
# 15. EXTRACT CLASS
# ============================================================

def extract_class_number(question):

    match = re.search(

        r"class\s*(\d+)",

        question,

        re.IGNORECASE

    )


    if match:

        return int(
            match.group(1)
        )


    return None


# ============================================================
# 16. EXTRACT LOCATION
# ============================================================

def extract_location(question):

    patterns = [

        r"(?:from|in|at|near)\s+"
        r"([A-Za-z][A-Za-z\s\-]+?)"
        r"\??$",

        r"(?:students from|students in)\s+"
        r"([A-Za-z][A-Za-z\s\-]+?)"
        r"\??$"

    ]


    for pattern in patterns:

        match = re.search(

            pattern,

            question.strip(),

            re.IGNORECASE

        )


        if match:

            location = (

                match.group(1)

                .strip()

                .rstrip("?")

                .strip()

            )


            return location


    return None


# ============================================================
# 17. COUNT REQUEST
# ============================================================

def is_count_request(question):

    text = question.lower()


    keywords = [

        "how many",

        "number of",

        "count",

        "total number"

    ]


    return any(

        word in text

        for word in keywords

    )


# ============================================================
# 18. EACH CLASS REQUEST
# ============================================================

def is_each_class(question):

    text = question.lower()


    phrases = [

        "each class",

        "every class",

        "from each class",

        "by class",

        "class wise",

        "class-wise"

    ]


    return any(

        phrase in text

        for phrase in phrases

    )


# ============================================================
# 19. EXACT ANALYSIS ENGINE
# ============================================================

def exact_analysis(
    df,
    question
):

    text = (
        question
        .lower()
        .strip()
    )


    result = {

        "exact":
            False,

        "calculated_by":
            "Pandas",

        "result":
            None

    }


    class_column = (
        find_class_column(df)
    )


    grade_column = (
        find_grade_column(df)
    )


    location_column = (
        find_location_column(df)
    )


    class_number = (
        extract_class_number(
            question
        )
    )


    grade = (
        extract_grade(
            question
        )
    )


    location = (
        extract_location(
            question
        )
    )


    # ========================================================
    # GRADE BY CLASS
    # ========================================================

    if (

        grade

        and

        is_each_class(
            question
        )

        and

        is_count_request(
            question
        )

        and

        class_column is not None

        and

        grade_column is not None

    ):

        temp = df.copy()


        temp["_class"] = (

            temp[class_column]

            .apply(
                normalize_class
            )

        )


        temp["_grade"] = (

            temp[grade_column]

            .apply(
                normalize_grade
            )

        )


        filtered = temp[

            temp["_grade"]

            == grade

        ]


        counts = (

            filtered

            .groupby(
                "_class"
            )

            .size()

            .sort_index()

        )


        result["exact"] = True


        result["result"] = {

            "type":
                "grade_by_class",

            "grade":
                grade,

            "counts": {

                f"Class {int(k)}":
                    int(v)

                for k, v
                in counts.items()

                if pd.notna(k)

            },

            "total":
                int(
                    len(filtered)
                )

        }


        return result


    # ========================================================
    # LOCATION COUNT
    # ========================================================

    if (

        location

        and

        is_count_request(
            question
        )

        and

        location_column is not None

    ):

        values = (

            df[location_column]

            .astype(str)

            .str.strip()

            .str.casefold()

        )


        filtered = df[

            values

            == location.casefold()

        ]


        result["exact"] = True


        result["result"] = {

            "type":
                "location_count",

            "location":
                location,

            "column":
                str(location_column),

            "count":
                int(
                    len(filtered)
                )

        }


        return result


    # ========================================================
    # CITY DISTRIBUTION
    # ========================================================

    if (

        location_column is not None

        and

        (

            "each city"
            in text

            or

            "every city"
            in text

            or

            "by city"
            in text

            or

            "city wise"
            in text

            or

            "city-wise"
            in text

        )

    ):

        values = (

            df[location_column]

            .astype(str)

            .str.strip()

        )


        counts = (

            values

            .value_counts()

        )


        result["exact"] = True


        result["result"] = {

            "type":
                "location_distribution",

            "counts": {

                str(k):
                    int(v)

                for k, v
                in counts.items()

            }

        }


        return result


    # ========================================================
    # CLASS + GRADE
    # ========================================================

    if (

        class_number is not None

        and

        grade is not None

        and

        is_count_request(
            question
        )

        and

        class_column is not None

        and

        grade_column is not None

    ):

        class_values = (

            df[class_column]

            .apply(
                normalize_class
            )

        )


        grade_values = (

            df[grade_column]

            .apply(
                normalize_grade
            )

        )


        filtered = df[

            (class_values == class_number)

            &

            (grade_values == grade)

        ]


        result["exact"] = True


        result["result"] = {

            "type":
                "class_grade_count",

            "class":
                f"Class {class_number}",

            "grade":
                grade,

            "count":
                int(
                    len(filtered)
                )

        }


        return result


    # ========================================================
    # CLASS COUNT
    # ========================================================

    if (

        class_number is not None

        and

        is_count_request(
            question
        )

        and

        class_column is not None

    ):

        values = (

            df[class_column]

            .apply(
                normalize_class
            )

        )


        filtered = df[

            values == class_number

        ]


        result["exact"] = True


        result["result"] = {

            "type":
                "class_count",

            "class":
                f"Class {class_number}",

            "count":
                int(
                    len(filtered)
                )

        }


        return result


    # ========================================================
    # GRADE COUNT
    # ========================================================

    if (

        grade is not None

        and

        is_count_request(
            question
        )

        and

        grade_column is not None

    ):

        values = (

            df[grade_column]

            .apply(
                normalize_grade
            )

        )


        filtered = df[

            values == grade

        ]


        result["exact"] = True


        result["result"] = {

            "type":
                "grade_count",

            "grade":
                grade,

            "count":
                int(
                    len(filtered)
                )

        }


        return result


    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    if (

        class_column is not None

        and

        (

            "each class"
            in text

            or

            "every class"
            in text

            or

            "by class"
            in text

            or

            "class wise"
            in text

            or

            "class-wise"
            in text

        )

    ):

        values = (

            df[class_column]

            .apply(
                normalize_class
            )

        )


        counts = (

            values

            .value_counts()

            .sort_index()

        )


        result["exact"] = True


        result["result"] = {

            "type":
                "class_distribution",

            "counts": {

                f"Class {int(k)}":
                    int(v)

                for k, v
                in counts.items()

                if pd.notna(k)

            }

        }


        return result


    # ========================================================
    # GRADE DISTRIBUTION
    # ========================================================

    if (

        grade_column is not None

        and

        (

            "grade distribution"
            in text

            or

            "each grade"
            in text

            or

            "by grade"
            in text

            or

            "grade wise"
            in text

            or

            "grade-wise"
            in text

        )

    ):

        values = (

            df[grade_column]

            .apply(
                normalize_grade
            )

        )


        counts = (

            values

            .value_counts()

            .sort_index()

        )


        result["exact"] = True


        result["result"] = {

            "type":
                "grade_distribution",

            "counts": {

                str(k):
                    int(v)

                for k, v
                in counts.items()

            }

        }


        return result


    # ========================================================
    # MISSING VALUES
    # ========================================================

    if (

        "missing"
        in text

        or

        "null"
        in text

    ):

        missing = (
            df.isna()
            .sum()
        )


        result["exact"] = True


        result["result"] = {

            "type":
                "missing_values",

            "total":
                int(
                    missing.sum()
                ),

            "by_column": {

                str(k):
                    int(v)

                for k, v
                in missing.items()

                if v > 0

            }

        }


        return result


    # ========================================================
    # DUPLICATES
    # ========================================================

    if "duplicate" in text:

        result["exact"] = True


        result["result"] = {

            "type":
                "duplicate_count",

            "count":
                int(
                    df.duplicated()
                    .sum()
                )

        }


        return result


    # ========================================================
    # TOTAL ROWS
    # ========================================================

    if (

        "total students"
        in text

        or

        "total records"
        in text

        or

        "how many rows"
        in text

        or

        "how many records"
        in text

    ):

        result["exact"] = True


        result["result"] = {

            "type":
                "total_rows",

            "count":
                int(
                    len(df)
                )

        }


        return result


    # ========================================================
    # FALLBACK
    # ========================================================

    result["result"] = {

        "type":
            "unsupported",

        "message":
            "No exact analysis operation detected.",

        "columns":
            list(
                df.columns
            )

    }


    return result


# ============================================================
# 20. DESCRIPTIVE STATISTICS
# ============================================================

def descriptive_statistics(df):

    numeric_df = (
        df.select_dtypes(
            include=np.number
        )
    )


    if numeric_df.empty:

        return pd.DataFrame()


    rows = []


    for column in numeric_df.columns:

        series = (
            pd.to_numeric(
                numeric_df[column],
                errors="coerce"
            )
            .dropna()
        )


        if series.empty:

            continue


        # ----------------------------------------------------
        # MODE
        # ----------------------------------------------------

        modes = (
            series
            .mode()
            .tolist()
        )


        if modes:

            if len(modes) <= 3:

                mode_value = ", ".join(
                    str(
                        round(
                            float(value),
                            2
                        )
                    )
                    for value in modes
                )

            else:

                mode_value = (
                    f"{len(modes)} modes"
                )

        else:

            mode_value = "No mode"


        # ----------------------------------------------------
        # CREATE ROW
        # ----------------------------------------------------

        rows.append({

            "Column":
                str(column),

            "Count":
                int(
                    series.count()
                ),

            "Mean":
                round(
                    float(
                        series.mean()
                    ),
                    2
                ),

            "Median":
                round(
                    float(
                        series.median()
                    ),
                    2
                ),

            "Mode":
                mode_value,

            "Std Dev":
                round(
                    float(
                        series.std()
                    ),
                    2
                ),

            "Minimum":
                round(
                    float(
                        series.min()
                    ),
                    2
                ),

            "25%":
                round(
                    float(
                        series.quantile(
                            0.25
                        )
                    ),
                    2
                ),

            "50%":
                round(
                    float(
                        series.quantile(
                            0.50
                        )
                    ),
                    2
                ),

            "75%":
                round(
                    float(
                        series.quantile(
                            0.75
                        )
                    ),
                    2
                ),

            "Maximum":
                round(
                    float(
                        series.max()
                    ),
                    2
                )

        })


    return pd.DataFrame(
        rows
    )


# ============================================================
# 21. CHART DETECTION
# ============================================================

def detect_chart(question):

    text = question.lower()


    if (

        "bar chart"
        in text

        or

        "bar graph"
        in text

        or

        "bar plot"
        in text

    ):

        return "bar"


    if (

        "pie chart"
        in text

        or

        "pie graph"
        in text

    ):

        return "pie"


    if (

        "line chart"
        in text

        or

        "line graph"
        in text

        or

        "trend"
        in text

    ):

        return "line"


    if (

        "scatter plot"
        in text

        or

        "scatter chart"
        in text

    ):

        return "scatter"


    return None


# ============================================================
# 22. CREATE CHART DATA
# ============================================================

def create_chart_data(
    df,
    question,
    chart_type
):

    text = question.lower()


    class_column = (
        find_class_column(df)
    )


    grade_column = (
        find_grade_column(df)
    )


    location_column = (
        find_location_column(df)
    )


    grade = (
        extract_grade(
            question
        )
    )


    # ========================================================
    # GRADE BY CLASS
    # ========================================================

    if (

        chart_type == "bar"

        and

        grade

        and

        is_each_class(
            question
        )

        and

        class_column is not None

        and

        grade_column is not None

    ):

        temp = df.copy()


        temp["_class"] = (

            temp[class_column]

            .apply(
                normalize_class
            )

        )


        temp["_grade"] = (

            temp[grade_column]

            .apply(
                normalize_grade
            )

        )


        filtered = temp[

            temp["_grade"]
            == grade

        ]


        counts = (

            filtered

            .groupby(
                "_class"
            )

            .size()

            .sort_index()

            .reset_index(
                name="Students"
            )

        )


        counts["Class"] = (

            counts["_class"]

            .apply(

                lambda x:
                f"Class {int(x)}"

            )

        )


        return counts[

            [
                "Class",
                "Students"
            ]

        ]


    # ========================================================
    # CITY
    # ========================================================

    if (

        chart_type == "bar"

        and

        location_column is not None

        and

        (

            "city"
            in text

            or

            "cities"
            in text

            or

            "location"
            in text

        )

    ):

        values = (

            df[location_column]

            .astype(str)

            .str.strip()

        )


        counts = (

            values

            .value_counts()

            .head(20)

            .reset_index()

        )


        counts.columns = [

            "City",
            "Students"

        ]


        return counts


    # ========================================================
    # CLASS
    # ========================================================

    if (

        chart_type == "bar"

        and

        class_column is not None

        and

        "class"
        in text

        and

        grade is None

    ):

        values = (

            df[class_column]

            .apply(
                normalize_class
            )

        )


        counts = (

            values

            .value_counts()

            .sort_index()

            .reset_index()

        )


        counts.columns = [

            "Class",
            "Students"

        ]


        counts["Class"] = (

            counts["Class"]

            .apply(

                lambda x:
                f"Class {int(x)}"

            )

        )


        return counts


    # ========================================================
    # GRADE
    # ========================================================

    if (

        chart_type == "bar"

        and

        grade_column is not None

        and

        "grade"
        in text

        and

        grade is None

    ):

        values = (

            df[grade_column]

            .apply(
                normalize_grade
            )

        )


        counts = (

            values

            .value_counts()

            .sort_index()

            .reset_index()

        )


        counts.columns = [

            "Grade",
            "Students"

        ]


        return counts


    # ========================================================
    # PIE - GRADE
    # ========================================================

    if (

        chart_type == "pie"

        and

        grade_column is not None

        and

        "grade"
        in text

    ):

        values = (

            df[grade_column]

            .apply(
                normalize_grade
            )

        )


        counts = (

            values

            .value_counts()

            .sort_index()

            .reset_index()

        )


        counts.columns = [

            "Grade",
            "Students"

        ]


        return counts


    # ========================================================
    # PIE - CLASS
    # ========================================================

    if (

        chart_type == "pie"

        and

        class_column is not None

        and

        "class"
        in text

    ):

        values = (

            df[class_column]

            .apply(
                normalize_class
            )

        )


        counts = (

            values

            .value_counts()

            .sort_index()

            .reset_index()

        )


        counts.columns = [

            "Class",
            "Students"

        ]


        counts["Class"] = (

            counts["Class"]

            .apply(

                lambda x:
                f"Class {int(x)}"

            )

        )


        return counts


    return None


# ============================================================
# 23. DISPLAY CHART
# ============================================================

def display_chart(
    data,
    chart_type,
    title
):

    if data is None:

        st.warning(
            "Unable to create the chart."
        )

        return


    if chart_type == "bar":

        fig = px.bar(

            data,

            x=data.columns[0],

            y=data.columns[1],

            text=data.columns[1],

            title=title

        )


    elif chart_type == "pie":

        fig = px.pie(

            data,

            names=data.columns[0],

            values=data.columns[1],

            title=title

        )


    elif chart_type == "line":

        fig = px.line(

            data,

            x=data.columns[0],

            y=data.columns[1],

            markers=True,

            title=title

        )


    elif chart_type == "scatter":

        fig = px.scatter(

            data,

            x=data.columns[0],

            y=data.columns[1],

            title=title

        )


    else:

        return


    fig.update_layout(

        template="plotly_dark",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="#e2e8f0"
        ),

        margin=dict(

            l=20,

            r=20,

            t=60,

            b=20

        )

    )


    st.plotly_chart(

        fig,

        use_container_width=True

    )


# ============================================================
# 24. GEMINI
# ============================================================

def ask_gemini(
    df,
    question,
    exact_result
):

    if client is None:

        return (
            "Gemini API key is not configured."
        )


    prompt = f"""

You are an expert AI Data Analyst.

USER QUESTION:

{question}


VERIFIED PYTHON RESULT:

{json.dumps(
    exact_result,
    indent=2,
    default=str
)}


IMPORTANT:

Python/Pandas is the authority for
numerical results.

Never invent numbers.

Never guess.

Never calculate from sample rows.

Never change numbers returned by Python.

Use the verified result exactly.

Explain the result clearly.

Keep the answer concise.

"""


    try:

        response = client.models.generate_content(

            model="gemini-2.5-flash",

            contents=prompt

        )


        return response.text


    except Exception as error:

        return (
            f"Gemini error: {error}"
        )


# ============================================================
# 25. SQL
# ============================================================

def execute_sql(
    df,
    query
):

    connection = sqlite3.connect(
        ":memory:"
    )


    try:

        df.to_sql(

            "data",

            connection,

            index=False,

            if_exists="replace"

        )


        return pd.read_sql_query(

            query,

            connection

        )


    finally:

        connection.close()


# ============================================================
# 26. HEADER
# ============================================================

header_left, header_right = st.columns(
    [7, 2]
)


with header_left:

    st.markdown(

        '<div class="main-title">'
        '🤖 AI Data Analyst'
        '</div>',

        unsafe_allow_html=True

    )


    st.markdown(

        '<div class="subtitle">'
        'Intelligent Excel analysis • '
        'AI insights • Interactive visualizations'
        '</div>',

        unsafe_allow_html=True

    )


with header_right:

    if client:

        st.markdown(

            '<div class="status-online">'
            '● Gemini Online'
            '</div>',

            unsafe_allow_html=True

        )

    else:

        st.markdown(

            '<div class="status-offline">'
            '● Gemini Offline'
            '</div>',

            unsafe_allow_html=True

        )


# ============================================================
# 27. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🤖 AI Data Analyst"
    )


    st.caption(
        "Your intelligent data assistant"
    )


    st.divider()


    st.markdown(
        "### 📁 Upload Dataset"
    )


    uploaded_file = st.file_uploader(

        "Choose Excel or CSV",

        type=[
            "xlsx",
            "xls",
            "csv"
        ]

    )


    st.divider()


    st.markdown(
        "### 🚀 Capabilities"
    )


    st.markdown(
        """
        📊 Data Analysis

        📈 Interactive Charts

        📐 Mean / Median / Mode

        🤖 Gemini AI

        🔎 Data Explorer

        🗃️ SQL Queries

        🎯 Exact Pandas Results
        """
    )


    st.divider()


    if client:

        st.success(
            "Gemini API connected"
        )

    else:

        st.warning(
            "Gemini API unavailable"
        )


# ============================================================
# 28. LOAD FILE
# ============================================================

if uploaded_file is not None:

    try:

        df = load_dataset(
            uploaded_file
        )


        st.session_state.df = df


        st.session_state.file_name = (

            uploaded_file.name

        )


    except Exception as error:

        st.error(
            f"File error: {error}"
        )

        st.stop()


# ============================================================
# 29. NO FILE SCREEN
# ============================================================

if st.session_state.df is None:

    st.markdown(

        """
        <div class="glass-panel">

        <h2>
        Welcome to your AI Data Analyst 👋
        </h2>

        <p style="color:#94a3b8;font-size:16px;">

        Upload an Excel or CSV file and ask
        questions about your data using
        natural language.

        </p>

        </div>
        """,

        unsafe_allow_html=True

    )


    c1, c2, c3, c4 = st.columns(4)


    features = [

        (
            "📊",
            "Smart Analysis",
            "Analyze your complete dataset using Pandas."
        ),

        (
            "📈",
            "Visualizations",
            "Generate interactive charts."
        ),

        (
            "📐",
            "Statistics",
            "Mean, median, mode and more."
        ),

        (
            "🤖",
            "Gemini AI",
            "Get natural-language explanations."
        )

    ]


    for column, feature in zip(

        [c1, c2, c3, c4],

        features

    ):

        with column:

            st.markdown(

                f"""
                <div class="feature-card">

                <div class="feature-icon">
                {feature[0]}
                </div>

                <div class="feature-title">
                {feature[1]}
                </div>

                <div class="feature-text">
                {feature[2]}
                </div>

                </div>
                """,

                unsafe_allow_html=True

            )


    st.stop()


# ============================================================
# 30. CURRENT DATASET
# ============================================================

df = st.session_state.df


# ============================================================
# 31. KPI CALCULATIONS
# ============================================================

rows = len(df)

columns = len(df.columns)

missing = int(

    df.isna()
    .sum()
    .sum()

)

duplicates = int(

    df.duplicated()
    .sum()

)


# ============================================================
# 32. DASHBOARD TITLE
# ============================================================

st.markdown(
    "## 📊 Dashboard"
)


st.caption(

    f"Currently loaded: "
    f"**{st.session_state.file_name}**"

)


# ============================================================
# 33. KPI CARDS
# ============================================================

k1, k2, k3, k4 = st.columns(4)


kpis = [

    (
        k1,
        "📄",
        "Rows",
        f"{rows:,}"
    ),

    (
        k2,
        "📋",
        "Columns",
        f"{columns:,}"
    ),

    (
        k3,
        "⚠️",
        "Missing Values",
        f"{missing:,}"
    ),

    (
        k4,
        "♻️",
        "Duplicates",
        f"{duplicates:,}"
    )

]


for column, icon, label, value in kpis:

    with column:

        st.markdown(

            f"""
            <div class="kpi-card">

            <div class="kpi-icon">
            {icon}
            </div>

            <div class="kpi-label">
            {label}
            </div>

            <div class="kpi-value">
            {value}
            </div>

            </div>
            """,

            unsafe_allow_html=True

        )


st.write("")


# ============================================================
# 34. MAIN TABS
# ============================================================

dashboard_tab, statistics_tab, ai_tab, data_tab, sql_tab = st.tabs(

    [

        "🏠 Dashboard",

        "📐 Statistics",

        "🤖 AI Analyst",

        "🔎 Data Explorer",

        "🗃️ SQL"

    ]

)


# ============================================================
# 35. DASHBOARD TAB
# ============================================================

with dashboard_tab:

    st.markdown(
        "### 📄 Dataset Preview"
    )


    st.dataframe(

        df.head(20),

        use_container_width=True,

        height=400

    )


    st.markdown(
        "### 📈 Quick Visualization"
    )


    numeric_columns = list(

        df.select_dtypes(
            include=np.number
        ).columns

    )


    if numeric_columns:

        selected_column = st.selectbox(

            "Select numerical column",

            numeric_columns

        )


        fig = px.histogram(

            df,

            x=selected_column,

            title=(

                f"Distribution of "
                f"{selected_column}"

            )

        )


        fig.update_layout(

            template="plotly_dark",

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)"

        )


        st.plotly_chart(

            fig,

            use_container_width=True

        )


    else:

        st.info(
            "No numerical columns available."
        )


# ============================================================
# 36. STATISTICS TAB
# ============================================================

with statistics_tab:

    st.markdown(
        "## 📐 Descriptive Statistics"
    )


    st.caption(

        "Statistical summary calculated "
        "directly from the complete dataset "
        "using Pandas."

    )


    statistics_df = (
        descriptive_statistics(
            df
        )
    )


    if statistics_df.empty:

        st.warning(

            "No numerical columns were found "
            "in the uploaded dataset."

        )

    else:

        # ----------------------------------------------------
        # Mean
        # ----------------------------------------------------

        st.markdown(
            "### 📊 Mean"
        )


        st.info(

            "Mean = Sum of all values ÷ "
            "Number of values."

        )


        # ----------------------------------------------------
        # Median
        # ----------------------------------------------------

        st.markdown(
            "### 📍 Median"
        )


        st.info(

            "Median = Middle value when "
            "the data is arranged in order."

        )


        # ----------------------------------------------------
        # Mode
        # ----------------------------------------------------

        st.markdown(
            "### 🔁 Mode"
        )


        st.info(

            "Mode = Most frequently occurring value."

        )


        # ----------------------------------------------------
        # COMPLETE TABLE
        # ----------------------------------------------------

        st.markdown(
            "### 📋 Complete Statistical Table"
        )


        st.dataframe(

            statistics_df,

            use_container_width=True,

            hide_index=True,

            height=450

        )


        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        statistics_csv = (

            statistics_df

            .to_csv(
                index=False
            )

            .encode(
                "utf-8"
            )

        )


        st.download_button(

            "⬇️ Download Statistics",

            data=statistics_csv,

            file_name=(
                "descriptive_statistics.csv"
            ),

            mime="text/csv"

        )


        # ----------------------------------------------------
        # OPTIONAL INDIVIDUAL COLUMN
        # ----------------------------------------------------

        st.markdown(
            "### 🔍 Analyze One Column"
        )


        selected_stat_column = st.selectbox(

            "Select a numerical column",

            statistics_df["Column"].tolist()

        )


        selected_row = statistics_df[

            statistics_df["Column"]
            == selected_stat_column

        ].iloc[0]


        s1, s2, s3, s4 = st.columns(4)


        with s1:

            st.metric(

                "Mean",

                selected_row["Mean"]

            )


        with s2:

            st.metric(

                "Median",

                selected_row["Median"]

            )


        with s3:

            st.metric(

                "Mode",

                selected_row["Mode"]

            )


        with s4:

            st.metric(

                "Std Dev",

                selected_row["Std Dev"]

            )


# ============================================================
# 37. AI ANALYST TAB
# ============================================================

with ai_tab:

    st.markdown(
        "## 🤖 Ask Your Data"
    )


    st.caption(
        "Ask questions in normal language."
    )


    st.markdown(

        """
        **Try asking:**

        `How many students are from Hyderabad?`

        `How many students are in Class 1?`
    """

    )


    question = st.text_area(

        "Ask a question",

        placeholder=(

            "Example: "
            "How many students are from Hyderabad?"

        ),

        height=100

    )


    if st.button(

        "✨ Analyze",

        type="primary",

        use_container_width=True

    ):

        if not question.strip():

            st.warning(

                "Please enter a question."

            )

        else:

            # =================================================
            # EXACT PANDAS ANALYSIS
            # =================================================

            exact_result = exact_analysis(

                df,

                question

            )


            # =================================================
            # CHART
            # =================================================

            chart_type = detect_chart(

                question

            )


            chart_data = None


            if chart_type:

                chart_data = create_chart_data(

                    df,

                    question,

                    chart_type

                )


            # =================================================
            # VERIFIED RESULT
            # =================================================

            if exact_result["exact"]:

                st.markdown(

                    '<span class="exact-badge">'
                    '✓ VERIFIED BY PANDAS'
                    '</span>',

                    unsafe_allow_html=True

                )


                result = (

                    exact_result["result"]

                )


                result_type = (

                    result["type"]

                )


                # ---------------------------------------------
                # LOCATION
                # ---------------------------------------------

                if (

                    result_type
                    == "location_count"

                ):

                    st.success(

                        f"There are "
                        f"**{result['count']:,}** "
                        f"students from "
                        f"**{result['location']}**."

                    )


                # ---------------------------------------------
                # CLASS
                # ---------------------------------------------

                elif (

                    result_type
                    == "class_count"

                ):

                    st.success(

                        f"There are "
                        f"**{result['count']:,}** "
                        f"students in "
                        f"**{result['class']}**."

                    )


                # ---------------------------------------------
                # GRADE
                # ---------------------------------------------

                elif (

                    result_type
                    == "grade_count"

                ):

                    st.success(

                        f"There are "
                        f"**{result['count']:,}** "
                        f"students with "
                        f"**Grade {result['grade']}**."

                    )


                # ---------------------------------------------
                # CLASS + GRADE
                # ---------------------------------------------

                elif (

                    result_type
                    == "class_grade_count"

                ):

                    st.success(

                        f"**{result['count']:,}** "
                        f"students from "
                        f"**{result['class']}** "
                        f"received "
                        f"**Grade {result['grade']}**."

                    )


                # ---------------------------------------------
                # GRADE BY CLASS
                # ---------------------------------------------

                elif (

                    result_type
                    == "grade_by_class"

                ):

                    st.markdown(

                        f"### Grade "
                        f"{result['grade']} "
                        f"Students by Class"

                    )


                    result_df = pd.DataFrame(

                        list(
                            result["counts"]
                            .items()
                        ),

                        columns=[

                            "Class",

                            "Students"

                        ]

                    )


                    st.dataframe(

                        result_df,

                        use_container_width=True,

                        hide_index=True

                    )


                    st.info(

                        f"Total Grade "
                        f"{result['grade']} "
                        f"students: "
                        f"**{result['total']:,}**"

                    )


                # ---------------------------------------------
                # CLASS DISTRIBUTION
                # ---------------------------------------------

                elif (

                    result_type
                    == "class_distribution"

                ):

                    result_df = pd.DataFrame(

                        list(
                            result["counts"]
                            .items()
                        ),

                        columns=[

                            "Class",

                            "Students"

                        ]

                    )


                    st.dataframe(

                        result_df,

                        use_container_width=True,

                        hide_index=True

                    )


                # ---------------------------------------------
                # GRADE DISTRIBUTION
                # ---------------------------------------------

                elif (

                    result_type
                    == "grade_distribution"

                ):

                    result_df = pd.DataFrame(

                        list(
                            result["counts"]
                            .items()
                        ),

                        columns=[

                            "Grade",

                            "Students"

                        ]

                    )


                    st.dataframe(

                        result_df,

                        use_container_width=True,

                        hide_index=True

                    )


                # ---------------------------------------------
                # VIEW JSON
                # ---------------------------------------------

                with st.expander(

                    "🔍 View verified calculation"

                ):

                    st.json(

                        result

                    )


            else:

                # =================================================
                # GEMINI ONLY FOR NON-EXACT QUESTIONS
                # =================================================

                with st.spinner(

                    "Gemini is analyzing..."

                ):

                    answer = ask_gemini(

                        df,

                        question,

                        exact_result

                    )


                st.markdown(

                    '<div class="ai-message">'
                    f'{answer}'
                    '</div>',

                    unsafe_allow_html=True

                )


            # =================================================
            # VISUALIZATION
            # =================================================

            if chart_type:

                st.markdown(
                    "### 📊 Visualization"
                )


                if chart_data is not None:

                    display_chart(

                        chart_data,

                        chart_type,

                        question

                    )


                    with st.expander(

                        "View chart data"

                    ):

                        st.dataframe(

                            chart_data,

                            use_container_width=True,

                            hide_index=True

                        )

                else:

                    st.warning(

                        "The visualization request "
                        "was detected, but the "
                        "required data could not "
                        "be identified."

                    )


# ============================================================
# 38. DATA EXPLORER
# ============================================================

with data_tab:

    st.markdown(
        "## 🔎 Data Explorer"
    )


    selected_column = st.selectbox(

        "Select column",

        list(
            df.columns
        )

    )


    search = st.text_input(

        "Search inside column"

    )


    if search:

        filtered_df = df[

            df[selected_column]

            .astype(str)

            .str.contains(

                search,

                case=False,

                na=False

            )

        ]

    else:

        filtered_df = df


    st.info(

        f"{len(filtered_df):,} rows found"

    )


    st.dataframe(

        filtered_df,

        use_container_width=True,

        height=500

    )


    csv_data = (

        filtered_df

        .to_csv(
            index=False
        )

        .encode(
            "utf-8"
        )

    )


    st.download_button(

        "⬇️ Download Filtered Data",

        data=csv_data,

        file_name="filtered_data.csv",

        mime="text/csv"

    )


# ============================================================
# 39. SQL
# ============================================================

with sql_tab:

    st.markdown(
        "## 🗃️ SQL Data Analysis"
    )


    st.info(

        "Your uploaded dataset is available "
        "as the SQL table `data`."

    )


    query = st.text_area(

        "SQL Query",

        value=(

            "SELECT * "
            "FROM data "
            "LIMIT 10"

        ),

        height=130

    )


    if st.button(

        "▶ Run SQL",

        use_container_width=True

    ):

        try:

            sql_result = execute_sql(

                df,

                query

            )


            st.success(

                "Query executed successfully."

            )


            st.dataframe(

                sql_result,

                use_container_width=True

            )


        except Exception as error:

            st.error(

                f"SQL Error: {error}"

            )


# ============================================================
# 40. FOOTER
# ============================================================

st.markdown(

    """
    <div class="footer">

    🤖 AI Data Analyst &nbsp;•&nbsp;
    Pandas &nbsp;•&nbsp;
    Plotly &nbsp;•&nbsp;
    Gemini &nbsp;•&nbsp;
    Streamlit

    </div>
    """,

    unsafe_allow_html=True

)
