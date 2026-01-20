
# ============================================================================
# PLACENTA ACCRETA SPECTRUM (PAS) OUTCOMES CALCULATOR
# ============================================================================

import streamlit as st
import numpy as np
import pandas as pd

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="PAS Outcomes Calculator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODEL COEFFICIENTS
# ============================================================================

COEFFICIENTS = {
    "ICU Admission": {
        "Intercept": -3.1379,
        "pred_prior_cs": 0.6279,
        "pred_previa_us": 0.7813,
        "pred_ga_surgery": -0.0521,
        "pred_sdi_high": 0.7038,
        "pred_mri_anterior": -0.8758,
        "pred_mri_lowlying": 1.5022,
        "pred_mri_bands": 0.6798,
        "pred_mri_t2_loss": 0.4722,
        "pred_mri_heterogeneity": 1.1004,
        "pred_mri_invasion": -0.4709,
        "pred_mri_thinning": -0.9824,
        "pred_mri_bulge": 0.6713,
        "pred_mri_irregular_contour": 0.2077,
        "pred_mri_abnormal_vasculature": -0.1335,
        "pred_mri_vascular_recruitment": -0.6799,
        "pred_mri_previa": 0.2756
    },
    "Transfusion": {
        "Intercept": 0.0838,
        "pred_prior_cs": 0.0930,
        "pred_previa_us": 0.6564,
        "pred_ga_surgery": -0.0291,
        "pred_sdi_high": 0.4936,
        "pred_mri_anterior": -0.7826,
        "pred_mri_lowlying": 0.5232,
        "pred_mri_bands": -0.4968,
        "pred_mri_t2_loss": -0.3542,
        "pred_mri_heterogeneity": 0.3958,
        "pred_mri_invasion": 0.3785,
        "pred_mri_thinning": 0.0882,
        "pred_mri_bulge": -0.8050,
        "pred_mri_irregular_contour": 1.3126,
        "pred_mri_abnormal_vasculature": -0.2762,
        "pred_mri_vascular_recruitment": 0.2410,
        "pred_mri_previa": 0.2282
    },
    "Surgical Assistance": {
        "Intercept": -3.9781,
        "pred_prior_cs": 0.3841,
        "pred_previa_us": 0.5741,
        "pred_ga_surgery": 0.0258,
        "pred_sdi_high": -0.4429,
        "pred_mri_anterior": -0.0301,
        "pred_mri_lowlying": 1.0721,
        "pred_mri_bands": -0.1786,
        "pred_mri_t2_loss": 0.0614,
        "pred_mri_heterogeneity": 0.7378,
        "pred_mri_invasion": 0.7573,
        "pred_mri_thinning": -0.1853,
        "pred_mri_bulge": -0.6089,
        "pred_mri_irregular_contour": 0.7123,
        "pred_mri_abnormal_vasculature": -0.4504,
        "pred_mri_vascular_recruitment": -0.4626,
        "pred_mri_previa": 0.6349
    },
    "Hysterectomy": {
        "Intercept": -0.2708,
        "pred_prior_cs": 0.3953,
        "pred_previa_us": 1.4285,
        "pred_ga_surgery": -0.0445,
        "pred_sdi_high": 0.4568,
        "pred_mri_anterior": -0.7610,
        "pred_mri_lowlying": 0.5517,
        "pred_mri_bands": -0.2734,
        "pred_mri_t2_loss": -0.9483,
        "pred_mri_heterogeneity": 0.9005,
        "pred_mri_invasion": 0.1917,
        "pred_mri_thinning": -0.4091,
        "pred_mri_bulge": -0.1230,
        "pred_mri_irregular_contour": 2.1160,
        "pred_mri_abnormal_vasculature": -0.5238,
        "pred_mri_vascular_recruitment": 1.2680,
        "pred_mri_previa": 0.2106
    },
    "High Blood Loss (>2.6L)": {
        "Intercept": -1.5901,
        "pred_prior_cs": 0.2485,
        "pred_previa_us": 0.2878,
        "pred_ga_surgery": -0.0083,
        "pred_sdi_high": 0.1973,
        "pred_mri_anterior": -0.7941,
        "pred_mri_lowlying": 0.4556,
        "pred_mri_bands": -0.3982,
        "pred_mri_t2_loss": -2.1768,
        "pred_mri_heterogeneity": 0.6734,
        "pred_mri_invasion": 0.5361,
        "pred_mri_thinning": 0.0108,
        "pred_mri_bulge": 0.0179,
        "pred_mri_irregular_contour": 1.0763,
        "pred_mri_abnormal_vasculature": -0.5053,
        "pred_mri_vascular_recruitment": 0.0912,
        "pred_mri_previa": -0.4791
    },
    "Extended LOS (>5 days)": {
        "Intercept": -2.3592,
        "pred_prior_cs": 0.3606,
        "pred_previa_us": 0.4795,
        "pred_ga_surgery": -0.0318,
        "pred_sdi_high": 0.5325,
        "pred_mri_anterior": 0.4994,
        "pred_mri_lowlying": 1.4301,
        "pred_mri_bands": 0.5434,
        "pred_mri_t2_loss": -0.1203,
        "pred_mri_heterogeneity": 0.1065,
        "pred_mri_invasion": -0.2719,
        "pred_mri_thinning": -0.4766,
        "pred_mri_bulge": 0.1537,
        "pred_mri_irregular_contour": 0.2180,
        "pred_mri_abnormal_vasculature": 0.7065,
        "pred_mri_vascular_recruitment": 0.1362,
        "pred_mri_previa": -0.3436
    }
}

MODEL_AUC = {
    "ICU Admission": 0.865,
    "Transfusion": 0.738,
    "Surgical Assistance": 0.800,
    "Hysterectomy": 0.856,
    "High Blood Loss (>2.6L)": 0.718,
    "Extended LOS (>5 days)": 0.751
}

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def calculate_probability(coefficients, inputs):
    log_odds = coefficients["Intercept"]
    for var, value in inputs.items():
        if var in coefficients:
            log_odds += coefficients[var] * value
    probability = 1 / (1 + np.exp(-log_odds))
    return probability

def get_risk_category(probability):
    if probability >= 0.7:
        return "HIGH", "🔴"
    elif probability >= 0.3:
        return "MODERATE", "🟡"
    else:
        return "LOW", "🟢"

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("🏥 Placenta Accreta Spectrum (PAS) Outcomes Calculator")
    st.markdown("""
    This calculator predicts the probability of adverse outcomes in patients with 
    suspected Placenta Accreta Spectrum based on clinical characteristics and MRI findings.
    """)
    
    st.markdown("---")
    
    # Two columns for input
    col1, col2 = st.columns(2)
    
    # COLUMN 1: CLINICAL INFORMATION
    with col1:
        st.header("📋 Clinical Information")
        
        prior_cs = st.number_input(
            "Number of Prior Cesarean Sections",
            min_value=0, max_value=10, value=1, step=1,
            help="Enter the number of previous cesarean deliveries"
        )
        
        previa_us = st.selectbox(
            "Placenta Previa on Ultrasound",
            options=["No", "Yes"], index=0,
            help="Was placenta previa identified on ultrasound?"
        )
        previa_us_val = 1 if previa_us == "Yes" else 0
        
        ga_surgery = st.slider(
            "Gestational Age at Surgery (weeks)",
            min_value=20, max_value=42, value=34, step=1,
            help="Expected gestational age at time of delivery/surgery"
        )
        
        sdi_high = st.selectbox(
            "High Social Deprivation Index (SDI ≥ 50)",
            options=["No", "Yes", "Unknown"], index=0,
            help="Is the patient from a high social deprivation area? SDI can be looked up by ZIP code at https://www.graham-center.org/maps-data-tools/social-deprivation-index.html"
        )
        sdi_high_val = 1 if sdi_high == "Yes" else 0
    
    # COLUMN 2: MRI FINDINGS
    with col2:
        st.header("🔬 MRI Findings")
        
        mri_performed = st.selectbox(
            "Was MRI Performed?",
            options=["Yes", "No"], index=0,
            help="Select whether pelvic MRI was performed"
        )
        
        if mri_performed == "Yes":
            st.markdown("**Select all MRI findings that apply:**")
            
            mri_col1, mri_col2 = st.columns(2)
            
            with mri_col1:
                mri_anterior = st.checkbox("Anterior Placenta")
                mri_lowlying = st.checkbox("Low-lying Placenta")
                mri_bands = st.checkbox("Placental Bands (T2 dark)")
                mri_t2_loss = st.checkbox("Loss of T2 Hypointense Line")
                mri_heterogeneity = st.checkbox("Signal Heterogeneity")
                mri_invasion = st.checkbox("Myometrial Invasion")
            
            with mri_col2:
                mri_thinning = st.checkbox("Myometrial Thinning")
                mri_bulge = st.checkbox("Placental Bulge")
                mri_irregular = st.checkbox("Irregular Placental Contour")
                mri_abnormal_vasc = st.checkbox("Abnormal Vasculature")
                mri_vasc_recruit = st.checkbox("Vascular Recruitment")
                mri_previa = st.checkbox("Placenta Previa on MRI")
        else:
            mri_anterior = mri_lowlying = mri_bands = mri_t2_loss = False
            mri_heterogeneity = mri_invasion = mri_thinning = mri_bulge = False
            mri_irregular = mri_abnormal_vasc = mri_vasc_recruit = mri_previa = False
            st.info("💡 MRI findings will be set to 'No' for all variables.")
    
    # Prepare inputs
    inputs = {
        "pred_prior_cs": prior_cs,
        "pred_previa_us": previa_us_val,
        "pred_ga_surgery": ga_surgery,
        "pred_sdi_high": sdi_high_val,
        "pred_mri_anterior": 1 if mri_anterior else 0,
        "pred_mri_lowlying": 1 if mri_lowlying else 0,
        "pred_mri_bands": 1 if mri_bands else 0,
        "pred_mri_t2_loss": 1 if mri_t2_loss else 0,
        "pred_mri_heterogeneity": 1 if mri_heterogeneity else 0,
        "pred_mri_invasion": 1 if mri_invasion else 0,
        "pred_mri_thinning": 1 if mri_thinning else 0,
        "pred_mri_bulge": 1 if mri_bulge else 0,
        "pred_mri_irregular_contour": 1 if mri_irregular else 0,
        "pred_mri_abnormal_vasculature": 1 if mri_abnormal_vasc else 0,
        "pred_mri_vascular_recruitment": 1 if mri_vasc_recruit else 0,
        "pred_mri_previa": 1 if mri_previa else 0
    }
    
    # Calculate button
    st.markdown("---")
    
    if st.button("🔮 Calculate Risk Probabilities", type="primary", use_container_width=True):
        st.markdown("---")
        st.header("📊 Predicted Outcomes")
        
        # Calculate all probabilities
        results = []
        for outcome, coeffs in COEFFICIENTS.items():
            prob = calculate_probability(coeffs, inputs)
            risk_cat, risk_icon = get_risk_category(prob)
            auc = MODEL_AUC[outcome]
            results.append({
                "Outcome": outcome,
                "Probability": prob,
                "Risk Category": risk_cat,
                "Icon": risk_icon,
                "AUC": auc
            })
        
        # Display results
        result_cols = st.columns(3)
        
        for i, result in enumerate(results):
            with result_cols[i % 3]:
                # Color based on risk
                if result['Risk Category'] == 'HIGH':
                    bg_color = '#ffebee'
                    border_color = '#f44336'
                    text_color = '#c62828'
                elif result['Risk Category'] == 'MODERATE':
                    bg_color = '#fff8e1'
                    border_color = '#ff9800'
                    text_color = '#ef6c00'
                else:
                    bg_color = '#e8f5e9'
                    border_color = '#4caf50'
                    text_color = '#2e7d32'
                
                st.markdown(f"""
                <div style="
                    background-color: {bg_color};
                    padding: 20px;
                    border-radius: 10px;
                    margin: 10px 0;
                    text-align: center;
                    border: 2px solid {border_color};
                ">
                    <h4 style="margin: 0; color: #333;">{result['Outcome']}</h4>
                    <h1 style="margin: 10px 0; color: {text_color};">
                        {result['Probability']*100:.1f}%
                    </h1>
                    <p style="margin: 0; font-size: 1.2em;">
                        {result['Icon']} {result['Risk Category']} RISK
                    </p>
                    <p style="margin: 5px 0 0 0; font-size: 0.8em; color: #666;">
                        Model AUC: {result['AUC']:.3f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Summary table
        st.markdown("---")
        st.subheader("📋 Summary Table")
        
        summary_df = pd.DataFrame(results)
        summary_df["Probability (%)"] = summary_df["Probability"].apply(lambda x: f"{x*100:.1f}%")
        summary_df = summary_df[["Outcome", "Probability (%)", "Risk Category", "AUC"]]
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Input summary
        st.markdown("---")
        st.subheader("📝 Patient Input Summary")
        
        input_col1, input_col2 = st.columns(2)
        
        with input_col1:
            st.markdown("**Clinical Information:**")
            st.markdown(f"- Prior Cesarean Sections: **{prior_cs}**")
            st.markdown(f"- Placenta Previa on US: **{previa_us}**")
            st.markdown(f"- Gestational Age: **{ga_surgery} weeks**")
            st.markdown(f"- High SDI Area: **{sdi_high}**")
        
        with input_col2:
            st.markdown("**MRI Findings:**")
            if mri_performed == "Yes":
                mri_findings = []
                if mri_anterior: mri_findings.append("Anterior Placenta")
                if mri_lowlying: mri_findings.append("Low-lying")
                if mri_bands: mri_findings.append("Placental Bands")
                if mri_t2_loss: mri_findings.append("T2 Line Loss")
                if mri_heterogeneity: mri_findings.append("Heterogeneity")
                if mri_invasion: mri_findings.append("Myometrial Invasion")
                if mri_thinning: mri_findings.append("Myometrial Thinning")
                if mri_bulge: mri_findings.append("Placental Bulge")
                if mri_irregular: mri_findings.append("Irregular Contour")
                if mri_abnormal_vasc: mri_findings.append("Abnormal Vasculature")
                if mri_vasc_recruit: mri_findings.append("Vascular Recruitment")
                if mri_previa: mri_findings.append("Previa on MRI")
                
                if mri_findings:
                    for finding in mri_findings:
                        st.markdown(f"- ✓ {finding}")
                else:
                    st.markdown("- No positive MRI findings")
            else:
                st.markdown("- MRI not performed")
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #fff3e0; padding: 15px; border-radius: 5px; border-left: 4px solid #ff9800;">
        <strong>⚠️ Disclaimer:</strong> This calculator is intended for educational and research purposes only. 
        It should NOT be used as a substitute for clinical judgment. The predictions are based on 
        statistical models derived from a single-institution cohort and may not generalize to all populations.
        Always consult with qualified healthcare professionals for clinical decision-making.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About This Calculator")
        
        st.markdown("""
        **PAS Outcomes Calculator v1.0**
        
        Predicts probability of adverse outcomes in patients with suspected Placenta Accreta Spectrum.
        
        ---
        
        **Outcomes Predicted:**
        - 🏥 ICU Admission
        - 🩸 Blood Transfusion
        - 👨‍⚕️ Surgical Assistance
        - 🔪 Hysterectomy
        - 💉 High Blood Loss (>2.6L)
        - 🛏️ Extended Stay (>5 days)
        
        ---
        
        **Risk Categories:**
        - 🟢 LOW: < 30%
        - 🟡 MODERATE: 30-70%
        - 🔴 HIGH: > 70%
        
        ---
        
        **Model Performance:**
        """)
        
        for outcome, auc in MODEL_AUC.items():
            st.markdown(f"- {outcome}: AUC = {auc:.3f}")
        
        st.markdown("""
        ---
        
        **Developed by:**
        Johns Hopkins Hospital
        
        **Citation:**
        [Your publication reference here]
        """)

if __name__ == "__main__":
    main()
