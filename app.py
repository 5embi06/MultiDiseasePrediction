import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Custom CSS for modern look, neumorphism, and dark mode ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #e0f7fa 0%, #e8f5e9 100%);
        color: #222;
    }
    .main {
        background: transparent;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        border-radius: 24px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        background: rgba(255,255,255,0.85);
        margin-top: 2rem;
    }
    .neumorph {
        background: #f6fcff;
        border-radius: 20px;
        box-shadow: 8px 8px 16px #d1d9e6, -8px -8px 16px #ffffff;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    .hero {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem 1rem 2rem 1rem;
        border-radius: 32px;
        background: linear-gradient(135deg, #e3f0ff 0%, #e8f5e9 100%);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1976d2;
        margin-bottom: 1rem;
    }
    .hero-desc {
        font-size: 1.2rem;
        color: #388e3c;
        margin-bottom: 2rem;
    }
    .hero-btn {
        background: #1976d2;
        color: #fff;
        border: none;
        border-radius: 999px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(25, 118, 210, 0.15);
        transition: background 0.2s;
        cursor: pointer;
    }
    .hero-btn:hover {
        background: #1565c0;
    }
    .section-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1976d2;
        margin-bottom: 1rem;
    }
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #fff;
        border-radius: 18px;
        box-shadow: 0 2px 8px rgba(25, 118, 210, 0.07);
        padding: 1.5rem;
        text-align: center;
        transition: box-shadow 0.2s;
    }
    .feature-card:hover {
        box-shadow: 0 4px 16px rgba(25, 118, 210, 0.15);
    }
    .testimonial {
        background: #f6fcff;
        border-radius: 18px;
        box-shadow: 0 2px 8px rgba(25, 118, 210, 0.07);
        padding: 1.5rem;
        margin-bottom: 1rem;
        font-style: italic;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-top: 2rem;
        padding: 1rem 0 0.5rem 0;
    }
    .scroll-x {
        overflow-x: auto;
        white-space: nowrap;
        padding-bottom: 1rem;
    }
    @media (prefers-color-scheme: dark) {
        html, body, [class*="css"]  {
            background: linear-gradient(135deg, #232526 0%, #414345 100%);
            color: #f5f5f5;
        }
        .block-container, .neumorph, .feature-card, .testimonial {
            background: #232526 !important;
            color: #f5f5f5 !important;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
        }
        .hero {
            background: linear-gradient(135deg, #232526 0%, #414345 100%);
        }
        .footer {
            color: #bbb;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- Fixed Navigation Bar ---
st.markdown("""
<nav style="position:fixed; top:0; left:0; width:100%; background:rgba(255,255,255,0.95); z-index:1000; box-shadow:0 2px 8px rgba(25,118,210,0.07); padding:0.7rem 0;">
    <div style="max-width:1200px; margin:auto; display:flex; align-items:center; justify-content:space-between;">
        <span style="font-weight:700; font-size:1.3rem; color:#1976d2;">Semaro</span>
        <span>
            <a href="#about" style="margin-right:1.5rem; color:#1976d2; text-decoration:none;">About</a>
            <a href="#features" style="margin-right:1.5rem; color:#1976d2; text-decoration:none;">Features</a>
            <a href="#testimonials" style="margin-right:1.5rem; color:#1976d2; text-decoration:none;">Testimonials</a>
            <a href="#contact" style="color:#1976d2; text-decoration:none;">Contact</a>
        </span>
    </div>
</nav>
<br><br>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="hero">
    <div class="hero-title">Welcome to Semaro</div>
    <div class="hero-desc">
        AI-powered detection of <b>Liver</b> disease from your lab reports.<br>
        Fast, secure, and doctor-friendly.
    </div>
    <a href="#get-started"><button class="hero-btn">Get Started</button></a>
    <img src="https://cdn.jsdelivr.net/gh/edent/SuperTinyIcons/images/svg/health.svg" width="120" style="margin-top:2rem;" />
</div>
""", unsafe_allow_html=True)

# --- About Section ---
st.markdown("""
<div class="neumorph" id="about">
    <div class="section-title">About the System</div>
    <div>
        Semaro leverages state-of-the-art AI/ML to help you and your doctor detect liver disease early, using your health data and lab reports. Your privacy and security are our top priorities.
    </div>
</div>
""", unsafe_allow_html=True)

# --- Features Section ---
st.markdown("""
<div class="neumorph" id="features">
    <div class="section-title">Features</div>
    <div class="features-grid">
        <div class="feature-card">
            <span style="font-size:2rem;">⚡</span>
            <div class="font-bold mt-2">Real-time Prediction</div>
            <div>Instant results powered by advanced ML models.</div>
        </div>
        <div class="feature-card">
            <span style="font-size:2rem;">🔒</span>
            <div class="font-bold mt-2">Secure Health Data</div>
            <div>Your data is encrypted and never shared.</div>
        </div>
        <div class="feature-card">
            <span style="font-size:2rem;">👨‍⚕️</span>
            <div class="font-bold mt-2">Doctor Suggestions</div>
            <div>Get actionable advice and connect with professionals.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Testimonials Section ---
st.markdown("""
<div class="neumorph" id="testimonials">
    <div class="section-title">Testimonials</div>
    <div class="testimonial">"Semaro gave me peace of mind with its quick and accurate predictions. The interface is beautiful and easy to use!"<br><b>- Aarav S.</b></div>
    <div class="testimonial">"As a physician, I appreciate the secure data handling and the helpful suggestions for patients."<br><b>- Dr. Priya K.</b></div>
</div>
""", unsafe_allow_html=True)

# --- Contact Section ---
with st.form("contact_form"):
    st.markdown('<div class="section-title" id="contact">Contact Us</div>', unsafe_allow_html=True)
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Your Message")
    submitted = st.form_submit_button("Send Message")
    if submitted:
        st.success("Thank you for reaching out! We'll get back to you soon.")

# --- Footer ---
st.markdown("""
<div class="footer">
    &copy; 2025 Semaro. All rights reserved.
</div>
""", unsafe_allow_html=True)

# --- Prediction App (below homepage sections) ---

# Only Liver Disease prediction
DISEASE = {
    'features': None,  # Will infer from processed data
    'scaler': 'models/liver_scaler.joblib',
    'models': {
        'Logistic Regression': 'models/liver_logreg.joblib',
        'Random Forest': 'models/liver_rf.joblib',
        'XGBoost': 'models/liver_xgb.joblib'
    }
}

st.markdown('<div id="get-started"></div>', unsafe_allow_html=True)
st.header('Liver Disease Prediction')
st.write('Enter patient data and get a prediction!')

# Load features dynamically if not hardcoded
try:
    X_train = pd.read_csv('data/processed/liver_X_train.csv')
    features = list(X_train.columns)
except Exception:
    features = []

user_input = {}
st.subheader('Enter Features:')

# Add horizontal scroll for many features
with st.container():
    st.markdown('<div class="scroll-x">', unsafe_allow_html=True)
    for feat in features:
        # Use selectbox for Gender, number_input for others
        if feat.lower() == 'gender':
            user_input[feat] = st.selectbox('Gender', options=['Male', 'Female'])
        else:
            # Set reasonable min/max/step for each feature
            if feat.lower() in ['age']:
                user_input[feat] = st.number_input(feat, min_value=0, max_value=120, step=1, value=30)
            elif feat.lower() in ['total_bilirubin','direct_bilirubin','a/g_ratio','albumin']:
                user_input[feat] = st.number_input(feat, min_value=0.0, max_value=50.0, step=0.1, value=1.0)
            elif feat.lower() in ['alkphos','sgpt','sgot']:
                user_input[feat] = st.number_input(feat, min_value=0, max_value=2000, step=1, value=100)
            elif feat.lower() in ['total_proteins']:
                user_input[feat] = st.number_input(feat, min_value=0.0, max_value=10.0, step=0.1, value=6.0)
            else:
                user_input[feat] = st.number_input(feat, value=0.0)
    st.markdown('</div>', unsafe_allow_html=True)

model_name = st.selectbox('Select Model', list(DISEASE['models'].keys()))

if st.button('Predict'):
    # Convert Gender to numeric if present
    input_values = []
    for feat in features:
        val = user_input[feat]
        if feat.lower() == 'gender':
            val = 1 if val == 'Male' else 0
        input_values.append(val)
    X = np.array([input_values])
    scaler = joblib.load(DISEASE['scaler'])
    X_scaled = scaler.transform(X)
    model = joblib.load(DISEASE['models'][model_name])
    pred = model.predict(X_scaled)[0]
    if hasattr(model, 'predict_proba'):
        conf = np.max(model.predict_proba(X_scaled))
    else:
        conf = None
    st.success(f'Prediction: {"Disease" if pred == 1 else "No Disease"}')
    if conf is not None:
        st.info(f'Model confidence: {conf:.2f}') 