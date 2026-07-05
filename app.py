
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AdVision AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    regression_model = joblib.load("adview_regression_model.pkl")
    segment_model = joblib.load("adview_segment_classifier.pkl")
    category_encoder = joblib.load("category_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    return (
        regression_model,
        segment_model,
        category_encoder,
        feature_columns
    )

(
    regression_model,
    segment_model,
    category_encoder,
    feature_columns
) = load_models()

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

/* Main page */
.stApp {
    background:
        radial-gradient(circle at top right,
        rgba(255, 75, 75, 0.08),
        transparent 30%),
        #f7f8fc;
}

/* Main container */
.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* Hide default Streamlit elements */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Hero */
.hero {
    padding: 55px 50px;
    border-radius: 24px;
    background:
        linear-gradient(
            135deg,
            #111827 0%,
            #1f2937 55%,
            #7f1d1d 100%
        );
    color: white;
    margin-bottom: 28px;
    box-shadow:
        0 20px 50px rgba(15, 23, 42, 0.15);
}

.hero-badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    font-size: 13px;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 52px;
    line-height: 1.05;
    font-weight: 850;
    letter-spacing: -1.5px;
    margin-bottom: 16px;
}

.hero-text {
    max-width: 760px;
    font-size: 18px;
    line-height: 1.7;
    color: #d1d5db;
}

/* Section titles */
.section-title {
    font-size: 30px;
    font-weight: 800;
    color: #111827;
    margin-top: 15px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #6b7280;
    font-size: 15px;
    margin-bottom: 22px;
}

/* Cards */
.metric-card {
    background: white;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    box-shadow:
        0 8px 25px rgba(15, 23, 42, 0.05);
    min-height: 145px;
}

.metric-icon {
    font-size: 25px;
    margin-bottom: 12px;
}

.metric-value {
    font-size: 29px;
    font-weight: 850;
    color: #111827;
}

.metric-label {
    color: #6b7280;
    font-size: 14px;
    margin-top: 5px;
}

/* Info cards */
.info-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    min-height: 190px;
    box-shadow:
        0 8px 25px rgba(15, 23, 42, 0.04);
}

.info-card h3 {
    margin-top: 5px;
    color: #111827;
}

.info-card p {
    color: #6b7280;
    line-height: 1.65;
}

/* Result cards */
.result-card {
    background:
        linear-gradient(
            145deg,
            #ffffff,
            #f8fafc
        );
    padding: 32px;
    border-radius: 20px;
    border: 1px solid #e5e7eb;
    text-align: center;
    box-shadow:
        0 12px 35px rgba(15, 23, 42, 0.08);
}

.result-value {
    font-size: 44px;
    font-weight: 900;
    color: #111827;
}

.result-label {
    color: #6b7280;
    font-size: 15px;
    margin-top: 8px;
}

/* Segment badge */
.segment-badge {
    display: inline-block;
    padding: 10px 22px;
    border-radius: 999px;
    background: #fee2e2;
    color: #b91c1c;
    font-weight: 800;
    font-size: 20px;
}

/* Workflow */
.workflow-step {
    background: white;
    border-left: 4px solid #ef4444;
    padding: 18px 20px;
    margin-bottom: 12px;
    border-radius: 10px;
    border-top: 1px solid #e5e7eb;
    border-right: 1px solid #e5e7eb;
    border-bottom: 1px solid #e5e7eb;
}

.workflow-number {
    font-weight: 850;
    color: #ef4444;
}

/* Footer */
.custom-footer {
    margin-top: 50px;
    padding: 28px;
    text-align: center;
    color: #6b7280;
    border-top: 1px solid #e5e7eb;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
}

section[data-testid="stSidebar"] * {
    color: white;
}

/* Buttons */
.stButton > button {
    border-radius: 12px;
    min-height: 50px;
    font-weight: 750;
}

/* Inputs */
div[data-baseweb="input"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("## 📈 AdVision AI")

    st.caption(
        "Intelligent YouTube Advertisement "
        "View Prediction Platform"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🎯 Predict",
            "📊 Model Insights",
            "🧠 How It Works",
            "ℹ️ About Project"
        ]
    )

    st.divider()

    st.markdown("### Model Status")
    st.success("● Models Loaded")

    st.markdown("### Engine")
    st.write("Random Forest")

    st.markdown("### Features")
    st.write("12 Predictive Features")

# =========================================================
# OVERVIEW PAGE
# =========================================================
if page == "🏠 Overview":

    st.markdown(
        """<div class="hero"><div class="hero-badge">MACHINE LEARNING • PREDICTIVE ANALYTICS</div><div class="hero-title">Predict YouTube Adviews<br>with Machine Learning</div><div class="hero-text">AdVision AI analyzes video engagement, publishing patterns, and audience interaction signals to estimate advertisement views and classify expected video performance.</div></div>""",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Platform Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'A complete machine learning pipeline from raw data '
        'to real-time web predictions.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🗂️</div>
            <div class="metric-value">14,637</div>
            <div class="metric-label">
                Processed video records
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">⚙️</div>
            <div class="metric-value">12</div>
            <div class="metric-label">
                Predictive input features
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🤖</div>
            <div class="metric-value">2</div>
            <div class="metric-label">
                Integrated ML models
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-value">4</div>
            <div class="metric-label">
                Performance segments
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    st.markdown(
        '<div class="section-title">Core Capabilities</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Designed as an end-to-end intelligent prediction system.'
        '</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="info-card">
            <h3>📈 Adview Estimation</h3>
            <p>
                Predicts advertisement view potential using
                a tuned Random Forest regression pipeline
                trained on log-transformed target values.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="info-card">
            <h3>🏷️ Segment Classification</h3>
            <p>
                Classifies expected performance into Low,
                Medium, High or Viral segments using a
                balanced classification workflow.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="info-card">
            <h3>🧪 Feature Engineering</h3>
            <p>
                Generates engagement rate, like ratio and
                comment rate to capture interaction quality
                beyond simple raw counts.
            </p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# PREDICT PAGE
# =========================================================
elif page == "🎯 Predict":

    st.markdown(
        '<div class="section-title">'
        'Real-Time Adview Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Enter video statistics below. The system automatically '
        'creates engineered features before prediction.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        st.subheader("Video Performance Inputs")

        col1, col2 = st.columns(2)

        with col1:

            views = st.number_input(
                "Total Views",
                min_value=0,
                value=100000,
                help="Current total video views"
            )

            likes = st.number_input(
                "Total Likes",
                min_value=0,
                value=5000
            )

            comments = st.number_input(
                "Total Comments",
                min_value=0,
                value=800
            )

        with col2:

            dislikes = st.number_input(
                "Total Dislikes",
                min_value=0,
                value=200
            )

            duration = st.number_input(
                "Video Duration (seconds)",
                min_value=1,
                value=300
            )

            category_name = st.selectbox(
                "Video Category",
                options=list(
                    category_encoder.classes_
                )
            )

        published_date = st.date_input(
            "Published Date",
            value=date.today()
        )

    st.write("")

    predict_button = st.button(
        "🚀 Generate Prediction",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        category_code = category_encoder.transform(
            [category_name]
        )[0]

        input_data = pd.DataFrame({
            "views": [views],
            "likes": [likes],
            "dislikes": [dislikes],
            "comment": [comments],
            "duration": [duration],
            "category": [category_code],
            "year": [published_date.year],
            "month": [published_date.month],
            "day": [published_date.day]
        })

        # Feature Engineering
        input_data["engagement_rate"] = (
            input_data["likes"]
            + input_data["dislikes"]
            + input_data["comment"]
        ) / (
            input_data["views"] + 1
        )

        input_data["like_ratio"] = (
            input_data["likes"]
            / (
                input_data["likes"]
                + input_data["dislikes"]
                + 1
            )
        )

        input_data["comment_rate"] = (
            input_data["comment"]
            / (
                input_data["views"] + 1
            )
        )

        input_data = input_data[
            feature_columns
        ]

        predicted_log = regression_model.predict(
            input_data
        )[0]

        predicted_adviews = max(
            0,
            np.expm1(predicted_log)
        )

        predicted_segment = segment_model.predict(
            input_data
        )[0]

        st.write("")
        st.markdown(
            '<div class="section-title">'
            'Prediction Results'
            '</div>',
            unsafe_allow_html=True
        )

        r1, r2 = st.columns(2)

        with r1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-value">
                        {round(predicted_adviews):,}
                    </div>
                    <div class="result-label">
                        Estimated Adviews
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with r2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="segment-badge">
                        {predicted_segment}
                    </div>
                    <div class="result-label">
                        Expected Performance Segment
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        # Input analytics
        engagement_rate = float(
            input_data["engagement_rate"].iloc[0]
        )

        like_ratio = float(
            input_data["like_ratio"].iloc[0]
        )

        comment_rate = float(
            input_data["comment_rate"].iloc[0]
        )

        st.markdown("### Derived Engagement Analytics")

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Engagement Rate",
            f"{engagement_rate * 100:.2f}%"
        )

        m2.metric(
            "Like Ratio",
            f"{like_ratio * 100:.2f}%"
        )

        m3.metric(
            "Comment Rate",
            f"{comment_rate * 100:.3f}%"
        )

        st.info(
            "Prediction is generated from historical patterns "
            "in the training dataset and should be interpreted "
            "as an analytical estimate rather than a guaranteed outcome."
        )

# =========================================================
# MODEL INSIGHTS PAGE
# =========================================================
elif page == "📊 Model Insights":

    st.markdown(
        '<div class="section-title">'
        'Model Performance Insights'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Comparison of regression approaches evaluated '
        'during model development.'
        '</div>',
        unsafe_allow_html=True
    )

    model_results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting"
        ],
        "R² Score": [
            0.016892,
            -0.533106,
            0.303416,
            0.177365
        ],
        "MAE": [
            1.194098,
            1.256118,
            1.015897,
            1.087230
        ],
        "RMSE": [
            1.896378,
            2.368157,
            1.596287,
            1.734714
        ]
    })

    st.dataframe(
        model_results,
        use_container_width=True,
        hide_index=True
    )

    st.write("")

    chart_data = model_results.set_index(
        "Model"
    )[["R² Score"]]

    st.bar_chart(chart_data)

    st.markdown("### Final Tuned Random Forest")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Log-Scale MAE",
        "1.0003"
    )

    c2.metric(
        "Log-Scale RMSE",
        "1.5938"
    )

    c3.metric(
        "Log-Scale R²",
        "0.3056"
    )

    st.warning(
        "The original adview target is extremely right-skewed. "
        "The dataset median is 2 adviews while the maximum exceeds "
        "5.4 million. A log transformation was therefore used to "
        "reduce the influence of extreme viral observations."
    )

    st.markdown("### Classification Performance")

    st.write(
        "The balanced segment classifier achieved approximately "
        "**82% accuracy**, with improved minority-class detection "
        "after random oversampling."
    )

    classification_df = pd.DataFrame({
        "Segment": [
            "Low",
            "Medium",
            "High",
            "Viral"
        ],
        "Precision": [
            0.86,
            0.43,
            0.41,
            0.67
        ],
        "Recall": [
            0.93,
            0.31,
            0.16,
            0.34
        ],
        "F1 Score": [
            0.90,
            0.36,
            0.23,
            0.46
        ]
    })

    st.dataframe(
        classification_df,
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# HOW IT WORKS PAGE
# =========================================================
elif page == "🧠 How It Works":

    st.markdown(
        '<div class="section-title">'
        'Machine Learning Workflow'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'The complete pipeline used to transform raw video '
        'records into real-time predictions.'
        '</div>',
        unsafe_allow_html=True
    )

    steps = [
        (
            "01",
            "Data Validation",
            "Duplicate records and missing values are checked "
            "before model preparation."
        ),
        (
            "02",
            "Data Preprocessing",
            "Numeric columns are converted, categories encoded, "
            "durations transformed to seconds and date features extracted."
        ),
        (
            "03",
            "Outlier Analysis",
            "IQR analysis identifies extreme adview observations "
            "without blindly deleting valid viral videos."
        ),
        (
            "04",
            "Target Transformation",
            "Log transformation reduces severe right skew while "
            "preserving all observations."
        ),
        (
            "05",
            "Feature Engineering",
            "Engagement rate, like ratio and comment rate are "
            "created from raw interaction statistics."
        ),
        (
            "06",
            "Model Comparison",
            "Linear Regression, Decision Tree, Random Forest and "
            "Gradient Boosting are evaluated."
        ),
        (
            "07",
            "Hyperparameter Tuning",
            "RandomizedSearchCV optimizes the selected Random Forest model."
        ),
        (
            "08",
            "Imbalance Handling",
            "Random oversampling improves detection of minority "
            "performance segments."
        ),
        (
            "09",
            "Deployment",
            "Saved models are integrated into a Streamlit application "
            "for interactive prediction."
        )
    ]

    for number, title, description in steps:

        st.markdown(
            f"""
            <div class="workflow-step">
                <span class="workflow-number">
                    STEP {number}
                </span>
                <h3>{title}</h3>
                <div style="color:#6b7280;">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================================================
# ABOUT PAGE
# =========================================================
elif page == "ℹ️ About Project":

    st.markdown(
        '<div class="section-title">'
        'About AdVision AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'An end-to-end machine learning project for '
        'YouTube advertisement view analytics.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### Project Objective

    The objective of this project is to estimate advertisement
    views from video-level engagement and publishing information,
    while also assigning videos to interpretable performance segments.

    ### Prediction Outputs

    The platform produces two outputs:

    - **Estimated Adviews** using tuned Random Forest regression
    - **Performance Segment** using balanced Random Forest classification

    ### Input Features

    The prediction pipeline uses:

    - Views
    - Likes
    - Dislikes
    - Comments
    - Duration
    - Category
    - Publication year
    - Publication month
    - Publication day
    - Engagement rate
    - Like ratio
    - Comment rate

    ### Technology Stack
    """)

    t1, t2, t3, t4 = st.columns(4)

    t1.metric("Language", "Python")
    t2.metric("ML", "Scikit-learn")
    t3.metric("Data", "Pandas")
    t4.metric("Web App", "Streamlit")

    st.write("")

    st.markdown("""
    ### Key Project Limitation

    The target variable contains extreme imbalance and severe
    right skew. Most observations have very low adviews, while
    a small number of viral observations reach extremely high values.

    The system therefore performs substantially better on the
    log-transformed target scale than on the raw original scale.
    This limitation is documented transparently as part of the
    model evaluation process.
    """)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="custom-footer">
    <strong>AdVision AI</strong><br>
    Machine Learning Based YouTube Adview Prediction Platform<br>
    Built with Python • Scikit-learn • Pandas • Streamlit
</div>
""", unsafe_allow_html=True)
