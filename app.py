"""
Sustainable Fertilizer Recommendation System
--------------------------------------------
A clean, beginner-friendly Streamlit web application that predicts optimal,
sustainable fertilizer recommendations based on soil N-P-K nutrients, pH,
and target crop type using a Decision Tree Classifier.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

# Import model helper functions
from model import (
    load_or_train_model,
    predict_fertilizer,
    SUPPORTED_CROPS,
    FERTILIZER_KNOWLEDGE_BASE,
    DATASET_PATH,
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sustainable Fertilizer Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# HIGH-CONTRAST GREEN & WHITE AGRICULTURE CSS
# Bulletproofed against browser/OS dark mode mismatches
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main Background & Base Font */
    .stApp {
        background-color: #f7faf7 !important;
        color: #1e293b !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Force all general text to high-contrast dark slate */
    .stMarkdown p, .stMarkdown span, .stMarkdown li, p, span, label {
        color: #1e293b;
    }

    /* Input Labels - High Contrast Forest Green */
    [data-testid="stWidgetLabel"] p, 
    [data-testid="stWidgetLabel"] label, 
    .stWidgetLabel p, 
    label {
        color: #1b5e20 !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }

    /* Captions and helper descriptions */
    [data-testid="stCaptionContainer"] p, .stCaption, small {
        color: #475569 !important;
        font-size: 0.88rem !important;
    }

    /* Input fields (Number input & Selectbox) */
    input[type="number"], 
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > input {
        color: #1e293b !important;
        background-color: #ffffff !important;
        border-color: #a5d6a7 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }

    /* Selectbox dropdown text */
    div[data-baseweb="select"] span {
        color: #1e293b !important;
        font-weight: 600 !important;
    }

    /* Table styling - High Contrast & Clean */
    table {
        width: 100% !important;
        border-collapse: collapse !important;
        margin-top: 0.5rem !important;
        margin-bottom: 1rem !important;
        background-color: #ffffff !important;
    }
    th {
        background-color: #e8f5e9 !important;
        color: #1b5e20 !important;
        font-weight: 700 !important;
        padding: 9px 12px !important;
        border: 1px solid #c8e6c9 !important;
        text-align: left !important;
        font-size: 0.9rem !important;
    }
    td {
        color: #1e293b !important;
        background-color: #ffffff !important;
        padding: 8px 12px !important;
        border: 1px solid #e0e7df !important;
        font-size: 0.88rem !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        background: transparent !important;
    }
    button[data-baseweb="tab"] p {
        color: #374151 !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1b5e20 !important;
        font-weight: 700 !important;
        border-bottom: 3px solid #2e7d32 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #1b5e20 !important;
        font-weight: 700 !important;
    }

    /* Header Container */
    .header-box {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        color: #ffffff !important;
        padding: 1.8rem 1.8rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(27, 94, 32, 0.18);
    }
    .header-box h1 {
        color: #ffffff !important;
        font-weight: 800;
        margin-bottom: 0.4rem;
        font-size: 2.1rem;
    }
    .header-box p {
        color: #e8f5e9 !important;
        font-size: 1.02rem;
        margin-bottom: 0;
        line-height: 1.5;
    }

    /* Custom White Cards with Subtle Shadow */
    .custom-card {
        background-color: #ffffff !important;
        border: 1px solid #dcdfdc !important;
        border-radius: 12px;
        padding: 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .custom-card h3, .custom-card h4 {
        color: #1b5e20 !important;
        font-weight: 700 !important;
    }
    .custom-card p {
        color: #334155 !important;
    }
    
    /* Recommendation Result Card */
    .result-card {
        background: linear-gradient(145deg, #ffffff, #f1f8e9) !important;
        border: 2px solid #66bb6a !important;
        border-radius: 14px;
        padding: 1.6rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(76, 175, 80, 0.18);
    }
    .fert-title {
        font-size: 2rem;
        font-weight: 800;
        color: #1b5e20 !important;
        margin-top: 0.2rem;
        margin-bottom: 0.5rem;
    }

    /* Status Badges */
    .badge-optimal {
        background-color: #e8f5e9 !important;
        color: #1b5e20 !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid #a5d6a7;
        display: inline-block;
    }
    .badge-deficient {
        background-color: #ffebee !important;
        color: #b71c1c !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid #ef9a9a;
        display: inline-block;
    }
    .badge-high {
        background-color: #fff3e0 !important;
        color: #e65100 !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid #ffcc80;
        display: inline-block;
    }

    /* Section Headings */
    .section-title {
        color: #1b5e20 !important;
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 0.8rem;
        border-left: 5px solid #2e7d32;
        padding-left: 10px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f1f8e9 !important;
        border-right: 1px solid #c8e6c9 !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #1b5e20 !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] li {
        color: #1e293b !important;
    }

    /* Sidebar preset action buttons */
    [data-testid="stSidebar"] div.stButton > button {
        background-color: #ffffff !important;
        color: #1b5e20 !important;
        border: 1px solid #a5d6a7 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.45rem 0.8rem !important;
        margin-bottom: 0.35rem !important;
        border-radius: 8px !important;
        width: 100% !important;
        text-align: left !important;
        display: flex !important;
        justify-content: flex-start !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] div.stButton > button p {
        color: #1b5e20 !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #e8f5e9 !important;
        border-color: #2e7d32 !important;
        color: #0d3b10 !important;
    }

    /* Main "Recommend Fertilizer" Button */
    [data-testid="stMain"] div.stButton > button:first-child {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        padding: 0.75rem 2rem !important;
        border-radius: 9px !important;
        border: none !important;
        width: 100% !important;
        box-shadow: 0 4px 12px rgba(27, 94, 32, 0.3) !important;
        transition: all 0.25s ease !important;
        cursor: pointer !important;
    }
    [data-testid="stMain"] div.stButton > button:first-child p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    [data-testid="stMain"] div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #144417 0%, #1b5e20 100%) !important;
        box-shadow: 0 6px 16px rgba(27, 94, 32, 0.4) !important;
    }

    /* Disclaimer alert styling */
    .disclaimer-box {
        background-color: #fffde7 !important;
        border-left: 4px solid #fbc02d !important;
        padding: 0.85rem 1rem !important;
        border-radius: 6px !important;
        font-size: 0.85rem !important;
        color: #422006 !important;
        margin-top: 1.2rem !important;
    }
    .disclaimer-box strong {
        color: #713f12 !important;
    }

    /* Streamlit native info callout box text */
    div[data-testid="stAlert"] {
        color: #1e293b !important;
        background-color: #e8f5e9 !important;
        border: 1px solid #a5d6a7 !important;
    }
    div[data-testid="stAlert"] p {
        color: #1b5e20 !important;
        font-weight: 500 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# LOAD MODEL PIPELINE (CACHED)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading machine learning model...")
def get_model():
    return load_or_train_model()

model_pkg = get_model()

# -----------------------------------------------------------------------------
# SIDEBAR: PRESETS & PROJECT CONTEXT
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1500937386664-56d1dfef3854?auto=format&fit=crop&w=600&q=80",
        caption="Sustainable Soil Stewardship",
        use_container_width=True,
    )
    
    st.markdown("### 🧪 Quick Demonstration Presets")
    st.caption("Click any preset below to instantly populate soil nutrient values:")

    # Initialize session state variables
    if "nitrogen_val" not in st.session_state:
        st.session_state.nitrogen_val = 20.0
    if "phosphorus_val" not in st.session_state:
        st.session_state.phosphorus_val = 45.0
    if "potassium_val" not in st.session_state:
        st.session_state.potassium_val = 40.0
    if "ph_val" not in st.session_state:
        st.session_state.ph_val = 6.5
    if "crop_val" not in st.session_state:
        st.session_state.crop_val = "Rice"

    # Full-width, clearly labeled preset buttons so text is NEVER truncated
    if st.button("🔴 Low Nitrogen (Urea Demo)"):
        st.session_state.nitrogen_val = 18.0
        st.session_state.phosphorus_val = 45.0
        st.session_state.potassium_val = 42.0
        st.session_state.ph_val = 6.5
        st.session_state.crop_val = "Rice"
        st.rerun()

    if st.button("🔴 Low Phosphorus (DAP Demo)"):
        st.session_state.nitrogen_val = 45.0
        st.session_state.phosphorus_val = 14.0
        st.session_state.potassium_val = 46.0
        st.session_state.ph_val = 6.7
        st.session_state.crop_val = "Wheat"
        st.rerun()

    if st.button("🔴 Low Potassium (MOP Demo)"):
        st.session_state.nitrogen_val = 48.0
        st.session_state.phosphorus_val = 44.0
        st.session_state.potassium_val = 15.0
        st.session_state.ph_val = 6.6
        st.session_state.crop_val = "Maize"
        st.rerun()

    if st.button("🟢 Balanced Soil (Compost Demo)"):
        st.session_state.nitrogen_val = 55.0
        st.session_state.phosphorus_val = 42.0
        st.session_state.potassium_val = 45.0
        st.session_state.ph_val = 6.8
        st.session_state.crop_val = "Tomato"
        st.rerun()

    if st.button("⚖️ Acidic Soil (Manure Demo)"):
        st.session_state.nitrogen_val = 36.0
        st.session_state.phosphorus_val = 36.0
        st.session_state.potassium_val = 38.0
        st.session_state.ph_val = 4.8
        st.session_state.crop_val = "Sugarcane"
        st.rerun()

    st.markdown("---")
    st.markdown("### 📌 About the System")
    st.markdown(
        """
        - **Algorithm:** Decision Tree Classifier (`scikit-learn`)
        - **Testing Accuracy:** **{acc:.1f}%**
        - **Dataset:** 160 balanced agro-profiles
        - **Framework:** Pure Python & Streamlit
        """.format(acc=model_pkg["accuracy"] * 100)
    )

    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Demonstration Notice:</strong><br>
            This application is built for educational demonstration and decision-support. 
            For commercial farming, always consult a certified local soil testing laboratory.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# MAIN CONTENT AREA
# -----------------------------------------------------------------------------

# Title & Description Box
st.markdown(
    """
    <div class="header-box">
        <h1>🌾 Sustainable Fertilizer Recommendation System</h1>
        <p>
            An AI-driven agronomic advisory platform helping farmers choose optimal, eco-friendly fertilizers 
            based on soil nutrients (N, P, K), pH balance, and crop requirements. Minimize excess chemical application, 
            protect groundwater, and nourish long-term soil fertility.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# App Navigation Tabs
tab_recommend, tab_model_info, tab_guide = st.tabs([
    "🌱 Fertilizer Recommendation",
    "📊 Model & Dataset Explorer",
    "📚 Sustainable Agriculture Guide"
])

# =============================================================================
# TAB 1: RECOMMENDATION ENGINE
# =============================================================================
with tab_recommend:
    col_input, col_info = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown('<div class="section-title">1. Enter Soil Nutrient & Crop Parameters</div>', unsafe_allow_html=True)
        st.caption("Input your latest soil health card or test readings:")

        # Crop Type Dropdown
        selected_crop = st.selectbox(
            "Select Crop Type",
            options=SUPPORTED_CROPS,
            index=SUPPORTED_CROPS.index(st.session_state.crop_val) if st.session_state.crop_val in SUPPORTED_CROPS else 0,
            help="Choose the primary crop intended for cultivation.",
        )

        col_n, col_p = st.columns(2)
        with col_n:
            val_nitrogen = st.number_input(
                "Nitrogen (N) - kg/ha or ppm",
                min_value=0.0,
                max_value=250.0,
                value=float(st.session_state.nitrogen_val),
                step=1.0,
                help="Available Nitrogen in soil (Optimal: 30 - 70 kg/ha).",
            )
        with col_p:
            val_phosphorus = st.number_input(
                "Phosphorus (P) - kg/ha or ppm",
                min_value=0.0,
                max_value=250.0,
                value=float(st.session_state.phosphorus_val),
                step=1.0,
                help="Available Phosphorus in soil (Optimal: 20 - 50 kg/ha).",
            )

        col_k, col_ph = st.columns(2)
        with col_k:
            val_potassium = st.number_input(
                "Potassium (K) - kg/ha or ppm",
                min_value=0.0,
                max_value=250.0,
                value=float(st.session_state.potassium_val),
                step=1.0,
                help="Available Potassium in soil (Optimal: 20 - 50 kg/ha).",
            )
        with col_ph:
            val_ph = st.number_input(
                "Soil pH (0 - 14 scale)",
                min_value=0.0,
                max_value=14.0,
                value=float(st.session_state.ph_val),
                step=0.1,
                help="Acidity or alkalinity measure (Optimal: 6.0 - 7.5).",
            )

        # Recommendation Trigger Button
        recommend_clicked = st.button("🌱 Recommend Fertilizer", use_container_width=True)

    with col_info:
        st.markdown('<div class="section-title">Quick Soil Health Benchmark Guide</div>', unsafe_allow_html=True)
        st.markdown(
            """
            | Parameter | Deficient / Low | Optimal Range | Excessive / High |
            | :--- | :--- | :--- | :--- |
            | **Nitrogen (N)** | < 30 kg/ha | 30 – 70 kg/ha | > 70 kg/ha |
            | **Phosphorus (P)** | < 20 kg/ha | 20 – 50 kg/ha | > 50 kg/ha |
            | **Potassium (K)** | < 20 kg/ha | 20 – 50 kg/ha | > 50 kg/ha |
            | **Soil pH** | < 6.0 (Acidic) | 6.0 – 7.5 (Neutral) | > 7.5 (Alkaline) |
            """
        )
        st.info(
            "💡 **Sustainable Farming Tip:** Precision fertilization avoids over-saturating soil with chemical nitrogen or phosphorus, "
            "protecting beneficial soil microbes and preserving local aquifers from nitrate pollution."
        )

    # -------------------------------------------------------------------------
    # RECOMMENDATION RESULTS SECTION
    # -------------------------------------------------------------------------
    if recommend_clicked:
        try:
            # Perform prediction
            result = predict_fertilizer(
                nitrogen=val_nitrogen,
                phosphorus=val_phosphorus,
                potassium=val_potassium,
                ph=val_ph,
                crop_type=selected_crop,
                model_package=model_pkg,
            )

            st.markdown("---")
            st.markdown('<div class="section-title">2. Recommendation & Diagnostic Analysis</div>', unsafe_allow_html=True)

            # Entered Values & Soil Status Metric Cards
            st.markdown("#### 🧪 Soil Nutrient Evaluation")
            status_cols = st.columns(4)

            metrics = [
                ("Nitrogen (N)", f"{val_nitrogen:.1f}", result["soil_status"]["Nitrogen"]),
                ("Phosphorus (P)", f"{val_phosphorus:.1f}", result["soil_status"]["Phosphorus"]),
                ("Potassium (K)", f"{val_potassium:.1f}", result["soil_status"]["Potassium"]),
                ("Soil pH", f"{val_ph:.1f}", result["soil_status"]["pH"]),
            ]

            for col, (label, val_str, s_dict) in zip(status_cols, metrics):
                with col:
                    badge_class = (
                        "badge-optimal" if s_dict["color"] == "green"
                        else "badge-deficient" if s_dict["color"] == "red"
                        else "badge-high"
                    )
                    st.markdown(
                        f"""
                        <div class="custom-card">
                            <small style="color: #475569; font-weight:700; text-transform:uppercase;">{label}</small>
                            <h3 style="margin: 4px 0 6px 0; color: #1b5e20;">{val_str}</h3>
                            <span class="{badge_class}">{s_dict['status']}</span>
                            <div style="font-size: 0.85rem; color: #334155; margin-top: 8px; line-height:1.4;">{s_dict['desc']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Main Result Banner
            st.markdown(
                f"""
                <div class="result-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <span style="background-color: #1b5e20; color: #ffffff; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">
                                Recommended Fertilizer
                            </span>
                            <div class="fert-title">🌿 {result['full_name']}</div>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 0.95rem; color: #1b5e20; font-weight: 700;">Crop: {selected_crop}</span><br>
                            <span style="font-size: 0.88rem; color: #475569; font-weight: 600;">Confidence: {result['confidence']*100:.1f}%</span>
                        </div>
                    </div>
                    <hr style="border: 0; border-top: 1px solid #c8e6c9; margin: 12px 0;">
                    <p style="font-size: 1.05rem; color: #1e293b; line-height: 1.5; margin-bottom: 0;">
                        <strong style="color: #1b5e20;">Agronomic Rationale:</strong> {result['reason']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Detailed Advice Cards
            card_col1, card_col2 = st.columns(2)

            with card_col1:
                st.markdown(
                    f"""
                    <div class="custom-card">
                        <h4 style="color: #1b5e20; margin-top: 0;">📋 Application Guidance</h4>
                        <p style="color: #334155; font-size: 0.95rem; line-height: 1.5;">{result['usage_guidance']}</p>
                        <small style="color: #64748b;">
                            <em>* Note: Exact dosage varies by soil texture, region, and irrigation. Consult local agricultural extension officers for dosage calculation.</em>
                        </small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with card_col2:
                st.markdown(
                    f"""
                    <div class="custom-card">
                        <h4 style="color: #1b5e20; margin-top: 0;">🌍 Environmental Benefits</h4>
                        <p style="color: #334155; font-size: 0.95rem; line-height: 1.5;">{result['environmental_benefits']}</p>
                        <small style="color: #1b5e20; font-weight:700;">
                            🌱 Reduces non-point source water pollution and optimizes soil carbon health.
                        </small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Sustainable Farming Best Practices Expander
            with st.expander("🌾 Sustainable Farming Practices & Tips for this Crop", expanded=True):
                st.markdown(f"**Recommended Sustainable Practice:** {result['sustainable_tips']}")
                st.markdown(
                    """
                    - **Split Doses:** Divide fertilizer applications into basal (planting) and top-dressing (vegetative/flowering) to maximize uptake.
                    - **Organic Buffers:** Apply 2–5 tonnes/acre of well-decomposed FYM or compost annually to maintain soil microbial ecology.
                    - **Moisture Management:** Ensure soil has adequate moisture when applying mineral fertilizers to prevent osmotic root shock.
                    """
                )

        except Exception as e:
            st.error(f"❌ Error evaluating fertilizer recommendation: {str(e)}")


# =============================================================================
# TAB 2: MODEL & DATASET EXPLORER
# =============================================================================
with tab_model_info:
    st.markdown('<div class="section-title">Machine Learning Model Architecture</div>', unsafe_allow_html=True)
    st.markdown(
        """
        The recommendation engine uses a **Decision Tree Classifier** from `scikit-learn`.
        Decision Trees are ideal for agronomic rule modeling because their hierarchical decision nodes
        mirror agricultural diagnostic flowcharts (e.g., checking if Nitrogen < 30, then testing Phosphorus, then pH).
        """
    )

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Model Algorithm", "Decision Tree Classifier")
    col_m2.metric("Testing Accuracy", f"{model_pkg['accuracy'] * 100:.2f}%")
    col_m3.metric("Splitting Criterion", "Gini Impurity (max_depth=6)")

    st.markdown("---")
    st.markdown("#### 🎯 Feature Importance")
    st.caption("Contribution of each soil and crop parameter to the Decision Tree's recommendation rules:")

    # Plot feature importance using Streamlit's built-in bar chart
    importance_df = pd.DataFrame(
        {
            "Feature": [f.replace("_Encoded", "") for f in model_pkg["feature_importance"].keys()],
            "Importance": list(model_pkg["feature_importance"].values()),
        }
    ).sort_values(by="Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"), color="#2e7d32")

    st.markdown("---")
    st.markdown("#### 📂 Sample Demonstration Dataset (`fertilizer_data.csv`)")
    st.caption("Showing the first 15 rows of the demonstration dataset:")
    try:
        sample_df = pd.read_csv(DATASET_PATH)
        st.dataframe(sample_df.head(15), use_container_width=True)
        st.caption(f"Total Dataset Records: {len(sample_df)} rows across {len(sample_df['Crop_Type'].unique())} crop varieties.")
    except Exception as e:
        st.warning(f"Could not load dataset table: {e}")


# =============================================================================
# TAB 3: SUSTAINABLE AGRICULTURE GUIDE
# =============================================================================
with tab_guide:
    st.markdown('<div class="section-title">Principles of Sustainable Nutrient Management</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        ### 🔄 The 4R Nutrient Stewardship Framework
        Sustainable fertilizer application rests upon four foundational tenets:
        1. **Right Source:** Match fertilizer type to soil deficiency and crop demand.
        2. **Right Rate:** Apply only the quantity required; avoid excess dosage.
        3. **Right Time:** Synchronize fertilizer delivery with the crop's maximum uptake window.
        4. **Right Place:** Position nutrients where roots can easily absorb them with minimal runoff.
        """
    )

    st.markdown("---")
    st.markdown("### 🧪 Macronutrients (N-P-K) & Their Roles")

    guide_c1, guide_c2, guide_c3 = st.columns(3)
    with guide_c1:
        st.markdown(
            """
            <div class="custom-card">
                <h4 style="color:#1b5e20;">🟢 Nitrogen (N)</h4>
                <p><strong>Primary Function:</strong> Chlorophyll synthesis, leaf development, and vegetative biomass.</p>
                <p><strong>Deficiency:</strong> Yellowing of older leaves (chlorosis), stunted growth.</p>
                <p><strong>Excess:</strong> Plant lodging, delayed fruiting, pest vulnerability, and nitrate leaching.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with guide_c2:
        st.markdown(
            """
            <div class="custom-card">
                <h4 style="color:#1b5e20;">🔵 Phosphorus (P)</h4>
                <p><strong>Primary Function:</strong> Deep root expansion, ATP energy transfer, and seed/fruit formation.</p>
                <p><strong>Deficiency:</strong> Purpling of leaves, poor root anchorage, delayed maturity.</p>
                <p><strong>Excess:</strong> Interferes with uptake of zinc, iron, and copper micronutrients.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with guide_c3:
        st.markdown(
            """
            <div class="custom-card">
                <h4 style="color:#1b5e20;">🟠 Potassium (K)</h4>
                <p><strong>Primary Function:</strong> Water regulation (stomatal opening), disease resistance, and stem rigidity.</p>
                <p><strong>Deficiency:</strong> Leaf margin scorch/browning, weak stalks, low drought tolerance.</p>
                <p><strong>Excess:</strong> Impedes plant absorption of magnesium and calcium ions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### ⚖️ The Impact of Soil pH on Nutrient Availability")
    st.markdown(
        """
        - **Strongly Acidic Soils (pH < 5.5):** Essential nutrients like Phosphorus become locked with aluminum and iron. Microbial decomposition of organic matter slows down.
        - **Optimal Soils (pH 6.0 – 7.5):** Almost all primary, secondary, and micronutrients achieve maximum solubility and biological bioavailability.
        - **Alkaline Soils (pH > 7.5):** Phosphorus precipitates with calcium, and micronutrients (Fe, Zn, Mn) become deficient. Organic manure and gypsum help buffer alkaline conditions.
        """
    )

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #475569; font-size: 0.9rem; padding: 1rem 0;">
        <strong>Sustainable Fertilizer Recommendation System</strong> | Built with Python, Streamlit & Scikit-Learn<br>
        🌱 Designed for College Demonstration & Sustainable Agriculture Education.
    </div>
    """,
    unsafe_allow_html=True,
)
