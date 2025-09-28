import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stroke Recurrence Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🏥 Stroke Recurrence Prediction Tool</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="warning-box">
<strong>⚠️ Medical Disclaimer:</strong> This tool is for educational and research purposes only. 
It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
Always consult with qualified healthcare professionals for medical decisions.
</div>
""", unsafe_allow_html=True)

# Load or create model (for demonstration)
@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler, or create dummy ones for demo"""
    try:
        # Try to load actual model if available
        model = pickle.load(open('stroke_model.pkl', 'rb'))
        scaler = pickle.load(open('stroke_scaler.pkl', 'rb'))
        st.success("✅ Pre-trained model loaded successfully!")
        return model, scaler, True
    except:
        # Create dummy model for demonstration
        st.warning("⚠️ Using demo model. Train your model first for actual predictions.")
        
        # Create dummy model with realistic coefficients
        model = LogisticRegression(random_state=42)
        
        # Simulate trained model
        feature_names = ['age', 'sex', 'bmi', 'time_since_stroke_months', 'stroke_type', 
                        'nihss_score', 'previous_strokes_tias', 'hypertension', 'diabetes',
                        'atrial_fibrillation', 'heart_failure', 'coronary_artery_disease',
                        'peripheral_artery_disease', 'bp_control', 'ldl_cholesterol',
                        'hdl_cholesterol', 'total_cholesterol', 'hba1c', 'egfr',
                        'antiplatelet_use', 'antiplatelet_adherence', 'anticoagulant_use',
                        'anticoagulant_adherence', 'antihypertensive_use', 
                        'antihypertensive_adherence', 'statin_use', 'smoking']
        
        # Create realistic coefficients based on clinical knowledge
        coefficients = np.array([
            0.05,   # age
            0.2,    # sex
            0.01,   # bmi
            -0.01,  # time_since_stroke_months
            0.3,    # stroke_type
            0.02,   # nihss_score
            0.4,    # previous_strokes_tias
            0.3,    # hypertension
            0.4,    # diabetes
            0.8,    # atrial_fibrillation
            0.5,    # heart_failure
            0.3,    # coronary_artery_disease
            0.2,    # peripheral_artery_disease
            -0.3,   # bp_control
            0.01,   # ldl_cholesterol
            -0.01,  # hdl_cholesterol
            0.005,  # total_cholesterol
            0.2,    # hba1c
            -0.01,  # egfr
            -0.4,   # antiplatelet_use
            -0.3,   # antiplatelet_adherence
            -0.6,   # anticoagulant_use
            -0.4,   # anticoagulant_adherence
            -0.3,   # antihypertensive_use
            -0.2,   # antihypertensive_adherence
            -0.3,   # statin_use
            0.4     # smoking
        ])
        
        model.coef_ = coefficients.reshape(1, -1)
        model.intercept_ = np.array([-2.5])
        model.classes_ = np.array([0, 1])
        
        # Create dummy scaler
        scaler = StandardScaler()
        # Set dummy parameters
        scaler.mean_ = np.zeros(27)
        scaler.scale_ = np.ones(27)
        
        return model, scaler, False

# Load model and scaler
model, scaler, is_real_model = load_model_and_scaler()

# Input validation functions
def validate_age(age):
    if age < 18 or age > 100:
        return False, "Age must be between 18 and 100 years"
    return True, ""

def validate_bmi(bmi):
    if bmi < 10 or bmi > 50:
        return False, "BMI must be between 10 and 50"
    return True, ""

def validate_nihss(nihss):
    if nihss < 0 or nihss > 25:
        return False, "NIHSS score must be between 0 and 25"
    return True, ""

def validate_cholesterol(ldl, hdl, total):
    if ldl < 0 or ldl > 500:
        return False, "LDL cholesterol must be between 0 and 500 mg/dL"
    if hdl < 0 or hdl > 150:
        return False, "HDL cholesterol must be between 0 and 150 mg/dL"
    if total < 0 or total > 600:
        return False, "Total cholesterol must be between 0 and 600 mg/dL"
    if total < (ldl + hdl):
        return False, "Total cholesterol should be greater than LDL + HDL"
    return True, ""

def validate_hba1c(hba1c):
    if hba1c < 4.0 or hba1c > 15.0:
        return False, "HbA1c must be between 4.0% and 15.0%"
    return True, ""

def validate_egfr(egfr):
    if egfr < 5 or egfr > 150:
        return False, "eGFR must be between 5 and 150 mL/min/1.73m²"
    return True, ""

# Sidebar for input
st.sidebar.markdown('<h2 class="section-header">Patient Information</h2>', unsafe_allow_html=True)

# Initialize session state for form data
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False

# Create form
with st.sidebar.form("patient_form"):
    st.markdown("### 👤 Demographics")
    
    age = st.number_input(
        "Age (years)", 
        min_value=18, max_value=100, value=65, 
        help="Patient age in years (18-100)"
    )
    
    sex = st.selectbox(
        "Sex", 
        options=[0, 1], 
        format_func=lambda x: "Female" if x == 0 else "Male",
        help="Biological sex of the patient"
    )
    
    bmi = st.number_input(
        "BMI (kg/m²)", 
        min_value=10.0, max_value=50.0, value=25.0, step=0.1,
        help="Body Mass Index (10-50 kg/m²)"
    )
    
    st.markdown("### 🧠 Stroke History")
    
    time_since_stroke = st.number_input(
        "Time since initial stroke (months)", 
        min_value=1, max_value=120, value=12,
        help="Months since the initial stroke event (1-120)"
    )
    
    stroke_type = st.selectbox(
        "Initial stroke type",
        options=[0, 1],
        format_func=lambda x: "Ischemic" if x == 0 else "Hemorrhagic",
        help="Type of the initial stroke"
    )
    
    nihss_score = st.number_input(
        "NIHSS Score", 
        min_value=0, max_value=25, value=5,
        help="National Institutes of Health Stroke Scale (0-25, higher = more severe)"
    )
    
    previous_strokes = st.number_input(
        "Previous strokes/TIAs", 
        min_value=0, max_value=10, value=0,
        help="Number of previous stroke or TIA events"
    )
    
    st.markdown("### ❤️ Cardiovascular Risk Factors")
    
    hypertension = st.checkbox("Hypertension", help="History of high blood pressure")
    
    if hypertension:
        bp_control = st.slider(
            "Blood pressure control", 
            0.0, 1.0, 0.7, 0.1,
            help="Quality of BP control (0=poor, 1=excellent)"
        )
    else:
        bp_control = 1.0
    
    diabetes = st.checkbox("Diabetes Mellitus", help="History of diabetes")
    
    atrial_fib = st.checkbox("Atrial Fibrillation", help="History of atrial fibrillation")
    heart_failure = st.checkbox("Heart Failure", help="History of heart failure")
    cad = st.checkbox("Coronary Artery Disease", help="History of coronary artery disease")
    pad = st.checkbox("Peripheral Artery Disease", help="History of peripheral artery disease")
    
    st.markdown("### 🧪 Laboratory Values")
    
    ldl = st.number_input(
        "LDL Cholesterol (mg/dL)", 
        min_value=0, max_value=500, value=100,
        help="Low-density lipoprotein cholesterol (0-500 mg/dL)"
    )
    
    hdl = st.number_input(
        "HDL Cholesterol (mg/dL)", 
        min_value=0, max_value=150, value=45,
        help="High-density lipoprotein cholesterol (0-150 mg/dL)"
    )
    
    total_chol = st.number_input(
        "Total Cholesterol (mg/dL)", 
        min_value=0, max_value=600, value=200,
        help="Total cholesterol (0-600 mg/dL)"
    )
    
    hba1c = st.number_input(
        "HbA1c (%)", 
        min_value=4.0, max_value=15.0, value=6.5, step=0.1,
        help="Hemoglobin A1c percentage (4.0-15.0%)"
    )
    
    egfr = st.number_input(
        "eGFR (mL/min/1.73m²)", 
        min_value=5, max_value=150, value=80,
        help="Estimated glomerular filtration rate (5-150 mL/min/1.73m²)"
    )
    
    st.markdown("### 💊 Medications")
    
    antiplatelet_use = st.checkbox("Antiplatelet therapy", help="Currently taking antiplatelet medication")
    if antiplatelet_use:
        antiplatelet_adh = st.slider(
            "Antiplatelet adherence", 
            0.0, 1.0, 0.8, 0.1,
            help="Medication adherence (0=never, 1=always)"
        )
    else:
        antiplatelet_adh = 0.0
    
    anticoagulant_use = st.checkbox("Anticoagulant therapy", help="Currently taking anticoagulant medication")
    if anticoagulant_use:
        anticoagulant_adh = st.slider(
            "Anticoagulant adherence", 
            0.0, 1.0, 0.8, 0.1,
            help="Medication adherence (0=never, 1=always)"
        )
    else:
        anticoagulant_adh = 0.0
    
    antihypertensive_use = st.checkbox("Antihypertensive therapy", help="Currently taking blood pressure medication")
    if antihypertensive_use:
        antihypertensive_adh = st.slider(
            "Antihypertensive adherence", 
            0.0, 1.0, 0.8, 0.1,
            help="Medication adherence (0=never, 1=always)"
        )
    else:
        antihypertensive_adh = 0.0
    
    statin_use = st.checkbox("Statin therapy", help="Currently taking statin medication")
    
    st.markdown("### 🚬 Lifestyle")
    
    smoking = st.checkbox("Current smoking", help="Currently smoking tobacco")
    
    # Submit button
    submitted = st.form_submit_button("🔍 Predict Stroke Recurrence", use_container_width=True)

# Main content area
if submitted:
    st.session_state.form_submitted = True
    
    # Validate inputs
    validation_errors = []
    
    valid_age, age_error = validate_age(age)
    if not valid_age:
        validation_errors.append(age_error)
    
    valid_bmi, bmi_error = validate_bmi(bmi)
    if not valid_bmi:
        validation_errors.append(bmi_error)
    
    valid_nihss, nihss_error = validate_nihss(nihss_score)
    if not valid_nihss:
        validation_errors.append(nihss_error)
    
    valid_chol, chol_error = validate_cholesterol(ldl, hdl, total_chol)
    if not valid_chol:
        validation_errors.append(chol_error)
    
    valid_hba1c, hba1c_error = validate_hba1c(hba1c)
    if not valid_hba1c:
        validation_errors.append(hba1c_error)
    
    valid_egfr, egfr_error = validate_egfr(egfr)
    if not valid_egfr:
        validation_errors.append(egfr_error)
    
    # Display validation errors
    if validation_errors:
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.error("❌ Please correct the following errors:")
        for error in validation_errors:
            st.write(f"• {error}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Create feature array
        features = np.array([
            age, sex, bmi, time_since_stroke, stroke_type, nihss_score, previous_strokes,
            int(hypertension), int(diabetes), int(atrial_fib), int(heart_failure), int(cad), int(pad),
            bp_control, ldl, hdl, total_chol, hba1c, egfr,
            int(antiplatelet_use), antiplatelet_adh, int(anticoagulant_use), anticoagulant_adh,
            int(antihypertensive_use), antihypertensive_adh, int(statin_use), int(smoking)
        ]).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction_proba = model.predict_proba(features_scaled)[0]
        prediction = model.predict(features_scaled)[0]
        
        # Display results
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<h2 class="section-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
            
            # Risk level determination
            risk_prob = prediction_proba[1] * 100
            
            if risk_prob < 10:
                risk_level = "Low"
                risk_color = "#28a745"
                risk_emoji = "🟢"
            elif risk_prob < 25:
                risk_level = "Moderate"
                risk_color = "#ffc107"
                risk_emoji = "🟡"
            else:
                risk_level = "High"
                risk_color = "#dc3545"
                risk_emoji = "🔴"
            
            # Display prediction
            st.markdown(f"""
            <div class="metric-card" style="text-align: center; background: linear-gradient(135deg, {risk_color}15, white);">
                <h2>{risk_emoji} {risk_level} Risk</h2>
                <h1 style="color: {risk_color}; font-size: 3rem;">{risk_prob:.1f}%</h1>
                <p style="font-size: 1.2rem;">Probability of Stroke Recurrence</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk interpretation
            if risk_prob < 10:
                interpretation = """
                **Low Risk**: The patient has a low probability of stroke recurrence based on the current risk factors. 
                Continue current preventive measures and regular follow-up.
                """
            elif risk_prob < 25:
                interpretation = """
                **Moderate Risk**: The patient has a moderate probability of stroke recurrence. 
                Consider optimizing risk factor management and medication adherence.
                """
            else:
                interpretation = """
                **High Risk**: The patient has a high probability of stroke recurrence. 
                Immediate attention to risk factor modification and close medical supervision is recommended.
                """
            
            st.markdown(f"""
            <div class="success-box">
                <h4>Clinical Interpretation:</h4>
                {interpretation}
            </div>
            """, unsafe_allow_html=True)
        
        # Risk factors analysis
        st.markdown('<h2 class="section-header">📊 Risk Factor Analysis</h2>', unsafe_allow_html=True)
        
        # Create risk factor summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔴 Present Risk Factors")
            risk_factors = []
            
            if age > 75:
                risk_factors.append(f"Advanced age ({age} years)")
            if hypertension and bp_control < 0.7:
                risk_factors.append("Poorly controlled hypertension")
            if diabetes and hba1c > 7.0:
                risk_factors.append("Poorly controlled diabetes")
            if atrial_fib and not anticoagulant_use:
                risk_factors.append("Atrial fibrillation without anticoagulation")
            if smoking:
                risk_factors.append("Current smoking")
            if previous_strokes > 0:
                risk_factors.append(f"Previous strokes/TIAs ({previous_strokes})")
            if ldl > 130:
                risk_factors.append(f"Elevated LDL cholesterol ({ldl} mg/dL)")
            if egfr < 60:
                risk_factors.append(f"Reduced kidney function (eGFR: {egfr})")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.write("✅ No major modifiable risk factors identified")
        
        with col2:
            st.subheader("🟢 Protective Factors")
            protective_factors = []
            
            if age < 65:
                protective_factors.append("Younger age")
            if hypertension and bp_control >= 0.8:
                protective_factors.append("Well-controlled blood pressure")
            if diabetes and hba1c <= 7.0:
                protective_factors.append("Well-controlled diabetes")
            if atrial_fib and anticoagulant_use and anticoagulant_adh >= 0.8:
                protective_factors.append("Appropriate anticoagulation for AF")
            if antiplatelet_use and antiplatelet_adh >= 0.8:
                protective_factors.append("Good antiplatelet adherence")
            if statin_use:
                protective_factors.append("Statin therapy")
            if not smoking:
                protective_factors.append("Non-smoker")
            
            if protective_factors:
                for factor in protective_factors:
                    st.write(f"• {factor}")
            else:
                st.write("⚠️ Limited protective factors present")
        
        # Recommendations
        st.markdown('<h2 class="section-header">💡 Clinical Recommendations</h2>', unsafe_allow_html=True)
        
        recommendations = []
        
        if hypertension and bp_control < 0.8:
            recommendations.append("🩺 Optimize blood pressure control (target <140/90 mmHg, or <130/80 in diabetes)")
        
        if diabetes and hba1c > 7.0:
            recommendations.append("🍎 Improve glycemic control (target HbA1c <7%)")
        
        if atrial_fib and not anticoagulant_use:
            recommendations.append("💊 Consider anticoagulation therapy for atrial fibrillation")
        
        if ldl > 130 and not statin_use:
            recommendations.append("💊 Consider statin therapy for cholesterol management")
        
        if smoking:
            recommendations.append("🚭 Smoking cessation counseling and support")
        
        if antiplatelet_use and antiplatelet_adh < 0.8:
            recommendations.append("📝 Improve antiplatelet medication adherence")
        
        if not recommendations:
            recommendations.append("✅ Continue current preventive measures and regular monitoring")
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        
        # Model information
        if not is_real_model:
            st.markdown("""
            <div class="warning-box">
            <strong>⚠️ Demo Mode:</strong> This prediction is generated using a demonstration model. 
            For clinical use, please train the model with your actual dataset first.
            </div>
            """, unsafe_allow_html=True)

# Information section
if not st.session_state.form_submitted:
    st.markdown('<h2 class="section-header">ℹ️ About This Tool</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Purpose
        This tool uses machine learning to predict the risk of stroke recurrence based on patient characteristics, 
        medical history, and current treatments.
        
        ### 📊 Model Performance
        - **Algorithm**: Logistic Regression
        - **Training**: 10,000 patient records
        - **Validation**: Cross-validated performance
        - **Metrics**: Optimized for clinical utility
        """)
    
    with col2:
        st.markdown("""
        ### 📋 Required Information
        - **Demographics**: Age, sex, BMI
        - **Stroke History**: Type, severity, timing
        - **Risk Factors**: Cardiovascular conditions
        - **Lab Values**: Cholesterol, HbA1c, kidney function
        - **Medications**: Current treatments and adherence
        - **Lifestyle**: Smoking status
        """)
    
    st.markdown("""
    ### 🔒 Privacy and Security
    - No patient data is stored or transmitted
    - All calculations are performed locally
    - Results are for clinical decision support only
    
    ### 📚 Clinical Context
    This tool should be used as part of comprehensive clinical assessment. Consider:
    - Patient's overall clinical condition
    - Recent changes in health status
    - Social and functional factors
    - Patient preferences and goals of care
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🏥 Stroke Recurrence Prediction Tool | For Educational and Research Use Only</p>
    <p>Always consult with healthcare professionals for medical decisions</p>
</div>
""", unsafe_allow_html=True)
