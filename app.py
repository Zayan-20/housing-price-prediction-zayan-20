"""
California Housing Price Prediction System
------------------------------------------
Course      : Machine Learning Fundamentals (AIC354)
Instructor  : Miss Ifra Saeed
Institution : COMSATS University Islamabad, Lahore Campus
Developer   : Zayan Amjad (FA24-BED-082)
GitHub      : https://github.com/Zayan-20
Live App    : https://housing-price-prediction-zayan-20.streamlit.app/
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="California Housing Price Predictor | Zayan Amjad",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .author-badge {
        background-color: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
        color: #1E40AF;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        color: white;
        padding: 1.6rem;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.18);
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.9;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.4rem 0;
    }
    .metric-sub {
        font-size: 0.88rem;
        opacity: 0.85;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Header & Developer Info
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🏡 California Housing Price Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header"><b>Assignment 3:</b> Applying, Evaluating, and Comparing Multiple ML Algorithms on Large Real-World Dataset</div>', unsafe_allow_html=True)
st.markdown("""
<div class="author-badge">
    👨‍💻 <b>Developer:</b> Zayan Amjad | <b>Roll No:</b> FA24-BED-082 | <b>GitHub:</b> <a href="https://github.com/Zayan-20" target="_blank">@Zayan-20</a><br>
    🏛 <b>Course:</b> Machine Learning Fundamentals (AIC354) | <b>Instructor:</b> Dr. Rao Muhammad Adeel Nawab | COMSATS University Islamabad, Lahore Campus
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Load Model
# -----------------------------------------------------------------------------
MODEL_FILE = os.path.join(os.path.dirname(__file__), "model.pkl")

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_FILE):
        st.error(f"Error: Model file '{MODEL_FILE}' was not found in the application directory.")
        return None
    try:
        return joblib.load(MODEL_FILE)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_trained_model()

# -----------------------------------------------------------------------------
# Sidebar: Property Inputs
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Property Specifications")
st.sidebar.write("Configure the neighborhood and property characteristics:")

# Presets
preset = st.sidebar.selectbox(
    "💡 Quick Presets",
    [
        "Custom Input",
        "Bay Area Luxury (Near Bay)",
        "Inland Suburban (Inland)",
        "San Diego Coastal (Near Ocean)",
        "Southern California Metro (<1H Ocean)"
    ]
)

# Default values based on preset
if preset == "Bay Area Luxury (Near Bay)":
    def_longitude, def_latitude = -122.23, 37.88
    def_age = 41.0
    def_rooms, def_bedrooms = 880.0, 129.0
    def_pop, def_households = 322.0, 126.0
    def_income = 8.3252
    def_ocean = "NEAR BAY"
elif preset == "Inland Suburban (Inland)":
    def_longitude, def_latitude = -119.75, 36.75
    def_age = 20.0
    def_rooms, def_bedrooms = 1800.0, 380.0
    def_pop, def_households = 1350.0, 400.0
    def_income = 2.8000
    def_ocean = "INLAND"
elif preset == "San Diego Coastal (Near Ocean)":
    def_longitude, def_latitude = -117.16, 32.72
    def_age = 35.0
    def_rooms, def_bedrooms = 2400.0, 480.0
    def_pop, def_households = 1100.0, 450.0
    def_income = 6.5000
    def_ocean = "NEAR OCEAN"
elif preset == "Southern California Metro (<1H Ocean)":
    def_longitude, def_latitude = -118.25, 34.05
    def_age = 28.0
    def_rooms, def_bedrooms = 2800.0, 560.0
    def_pop, def_households = 1600.0, 520.0
    def_income = 4.5000
    def_ocean = "<1H OCEAN"
else:
    def_longitude, def_latitude = -122.23, 37.88
    def_age = 28.0
    def_rooms, def_bedrooms = 2500.0, 500.0
    def_pop, def_households = 1400.0, 480.0
    def_income = 4.2000
    def_ocean = "<1H OCEAN"

median_income = st.sidebar.slider(
    "Median Household Income ($10,000s)",
    min_value=0.5,
    max_value=15.0,
    value=float(def_income),
    step=0.1,
    help="Median income of households in this block (e.g., 4.2 = $42,000/year)"
)

housing_median_age = st.sidebar.slider(
    "Housing Median Age (years)",
    min_value=1.0,
    max_value=52.0,
    value=float(def_age),
    step=1.0
)

ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    options=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
    index=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"].index(def_ocean)
)

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    total_rooms = st.number_input("Total Rooms", min_value=10.0, max_value=40000.0, value=float(def_rooms), step=50.0)
    population = st.number_input("Population", min_value=5.0, max_value=35000.0, value=float(def_pop), step=25.0)
    latitude = st.number_input("Latitude", min_value=32.0, max_value=42.5, value=float(def_latitude), step=0.01)

with col_s2:
    total_bedrooms = st.number_input("Total Bedrooms", min_value=1.0, max_value=6500.0, value=float(def_bedrooms), step=25.0)
    households = st.number_input("Households", min_value=1.0, max_value=6000.0, value=float(def_households), step=25.0)
    longitude = st.number_input("Longitude", min_value=-124.5, max_value=-114.0, value=float(def_longitude), step=0.01)

# Construct Feature Vector
feature_vector = pd.DataFrame([{
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity
}])

# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
tab_predict, tab_benchmark, tab_about = st.tabs(["🎯 House Price Prediction", "📊 10-Fold CV Benchmark", "ℹ️ About the Project"])

with tab_predict:
    col_left, col_right = st.columns([1.3, 1.0], gap="large")

    with col_left:
        st.subheader("Selected Property Feature Vector")
        st.dataframe(feature_vector, use_container_width=True)

        if model is not None:
            pred_price = float(model.predict(feature_vector)[0])
            pred_price = max(15000.0, pred_price)

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Estimated Median House Value</div>
                <div class="metric-value">${pred_price:,.2f} USD</div>
                <div class="metric-sub">Trained Pipeline: <b>Extreme Gradient Boosting (XGBoost)</b> | 10-Fold CV RMSE: ±$47,845</div>
            </div>
            """, unsafe_allow_html=True)

            lower_bound = max(0.0, pred_price - 47845.0)
            upper_bound = pred_price + 47845.0
            st.info(f"💡 **Estimated 68% Confidence Interval (±1 RMSE):** ${lower_bound:,.0f} — ${upper_bound:,.0f} USD")

        st.subheader("📍 Geographical Property Location")
        map_df = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
        st.map(map_df, zoom=7)

    with col_right:
        st.subheader("📐 Property Derived Metrics")
        rooms_per_hh = total_rooms / max(1.0, households)
        bdrms_per_room = total_bedrooms / max(1.0, total_rooms)
        pop_per_hh = population / max(1.0, households)

        st.metric("Rooms per Household", f"{rooms_per_hh:.2f}")
        st.metric("Bedrooms Ratio", f"{bdrms_per_room * 100:.1f}%")
        st.metric("Population per Household", f"{pop_per_hh:.2f}")

        st.markdown("""
        ---
        ### 🔍 Model Inference Notes:
        - **Pipeline Integration:** Missing values in `total_bedrooms` are automatically imputed via median.
        - **Feature Scaling:** Continuous numeric attributes are standardized via `StandardScaler`.
        - **Categorical Handling:** `ocean_proximity` is dynamically transformed into one-hot binary flags.
        """)

with tab_benchmark:
    st.subheader("📊 10-Fold Cross-Validation Performance Comparison")
    st.write("Results of evaluating 5 distinct regression algorithms on the full California Housing dataset (20,640 records) using 10-Fold Cross Validation:")

    benchmark_data = pd.DataFrame({
        "Algorithm": ["XGBoost Regressor (Best)", "Extra Trees", "Ridge Regression", "Lasso Regression", "AdaBoost Regressor"],
        "Mean RMSE ($)": [47844.55, 51464.71, 68904.26, 68912.40, 90215.29],
        "Std RMSE ($)": [2029.81, 1772.81, 2754.08, 2761.15, 2927.02],
        "% of Avg Price": ["23.13%", "24.88%", "33.31%", "33.32%", "43.61%"],
        "Training Time": ["~4s", "~35s", "~0.3s", "~1.5s", "~14s"]
    })
    st.dataframe(benchmark_data.style.highlight_min(subset=["Mean RMSE ($)"], color="#D1FAE5"), use_container_width=True)

    st.markdown("""
    ### 🏆 Why XGBoost Won:
    1. **Non-Linear Geographic Boundaries:** Real estate values jump sharply across coastal micro-climates and city limits. Tree boosting models this non-linearity effortlessly.
    2. **Regularization & Second-Order Gradients:** XGBoost penalizes model complexity via L1/L2 leaf weights, preventing overfitting while minimizing RMSE.
    3. **Consistency:** Low standard deviation (~$2,000) across all 10 folds proves strong generalization on unseen test blocks.
    """)

with tab_about:
    st.subheader("ℹ️ Project & Dataset Information")
    st.markdown("""
    - **Dataset:** 1990 California Census Block Groups (20,640 records, 9 input features, 1 target).
    - **Target Variable:** `median_house_value` (USD).
    - **Cross-Validation:** 10-Fold Cross-Validation (`n_splits=10, shuffle=True, random_state=42`).
    - **Academic Context:** Course Assignment 3 for Machine Learning Fundamentals (AIC354), COMSATS University Islamabad, Lahore Campus.
    - **Code Repository:** [GitHub (@Zayan-20)](https://github.com/Zayan-20)
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #6B7280; font-size: 0.85rem;'>Zayan Amjad (FA24-BED-082) | Machine Learning Assignment 3 | Fall 2026</div>", unsafe_allow_html=True)
