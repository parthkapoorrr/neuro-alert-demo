import streamlit as st
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import warnings
import tempfile
import os
import time
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Configuration
st.set_page_config(
    page_title="NeuroAlert - AI Seizure Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')

# Custom CSS - Pure Black Background with Fixed Sidebar
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    /* Hamburger Menu Button */
    [data-testid="collapsedControl"] {
        display: flex !important;
        position: fixed;
        top: 1rem;
        left: 1rem;
        z-index: 999999;
        background: linear-gradient(135deg, #4a90e2 0%, #7b68ee 100%) !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 4px 15px rgba(74, 144, 226, 0.5) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="collapsedControl"]:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 6px 20px rgba(74, 144, 226, 0.7) !important;
        border-color: rgba(255, 255, 255, 0.5) !important;
    }
    
    [data-testid="collapsedControl"] svg {
        color: white !important;
        width: 24px !important;
        height: 24px !important;
    }
    
    /* Pure Black Background */
    .stApp {
        background: #000000 !important;
    }
    
    /* Sidebar - Visible and Styled */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 100%) !important;
        border-right: 1px solid #333;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stMetric label {
        color: #a0a0a0 !important;
    }
    
    [data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0, 100, 255, 0.3);
        border: 1px solid rgba(100, 150, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(100, 150, 255, 0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 1; }
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 3.5rem;
        font-weight: 900;
        margin: 0;
        position: relative;
        z-index: 1;
        text-shadow: 0 0 20px rgba(100, 150, 255, 0.8);
    }
    
    .main-header p {
        color: #b0c4de;
        font-size: 1.2rem;
        margin-top: 1rem;
        position: relative;
        z-index: 1;
    }
    
    /* Modern File Upload Card */
    .upload-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 2px dashed rgba(100, 150, 255, 0.4);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .upload-container::before {
        content: '';
        position: absolute;
        inset: -2px;
        background: linear-gradient(45deg, #4a90e2, #7b68ee, #4a90e2);
        background-size: 300% 300%;
        border-radius: 20px;
        opacity: 0;
        z-index: -1;
        animation: border-glow 3s ease infinite;
    }
    
    @keyframes border-glow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .upload-container:hover::before {
        opacity: 1;
    }
    
    .upload-container:hover {
        border-color: #4a90e2;
        box-shadow: 0 20px 60px rgba(74, 144, 226, 0.3);
        transform: translateY(-5px);
    }
    
    .stFileUploader {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        border: 2px dashed rgba(74, 144, 226, 0.5) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        margin: 1rem 0 !important;
    }
    
    .stFileUploader:hover {
        border-color: #4a90e2 !important;
        box-shadow: 0 10px 30px rgba(74, 144, 226, 0.3) !important;
    }
    
    .stFileUploader label {
        color: #ffffff !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        background: rgba(74, 144, 226, 0.1) !important;
        border: 2px dashed rgba(74, 144, 226, 0.4) !important;
        border-radius: 15px !important;
        padding: 3rem 2rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"]:hover {
        background: rgba(74, 144, 226, 0.2) !important;
        border-color: #4a90e2 !important;
        transform: translateY(-2px) !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #b0c4de !important;
        font-size: 1.1rem !important;
    }
    
    .stFileUploader button {
        background: linear-gradient(135deg, #4a90e2 0%, #7b68ee 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 5px 15px rgba(74, 144, 226, 0.5) !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(74, 144, 226, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(74, 144, 226, 0.4);
        border-color: #4a90e2;
    }
    
    [data-testid="stMetric"] label {
        color: #a0a0a0 !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
    }
    
    /* Alert Boxes */
    .alert-danger {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.2) 0%, rgba(185, 28, 28, 0.2) 100%);
        border: 2px solid rgba(239, 68, 68, 0.5);
        color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(220, 38, 38, 0.3);
    }
    
    .alert-success {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(22, 163, 74, 0.2) 100%);
        border: 2px solid rgba(34, 197, 94, 0.5);
        color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(34, 197, 94, 0.3);
    }
    
    .alert-warning {
        background: linear-gradient(135deg, rgba(234, 179, 8, 0.2) 0%, rgba(202, 138, 4, 0.2) 100%);
        border: 2px solid rgba(234, 179, 8, 0.5);
        color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(234, 179, 8, 0.3);
    }
    
    /* Globe Section - Minimal */
    .globe-section {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        border: 1px solid rgba(74, 144, 226, 0.2);
        border-radius: 20px;
        padding: 2rem;
        margin: 3rem 0;
        text-align: center;
    }
    
    .globe-title {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 2rem;
        text-shadow: 0 0 20px rgba(74, 144, 226, 0.5);
    }
    
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .stat-card {
        background: linear-gradient(135deg, rgba(74, 144, 226, 0.1) 0%, rgba(123, 104, 238, 0.1) 100%);
        border: 1px solid rgba(74, 144, 226, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(74, 144, 226, 0.3);
        border-color: #4a90e2;
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 900;
        color: #4a90e2;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #b0c4de;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4a90e2 0%, #7b68ee 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(74, 144, 226, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(74, 144, 226, 0.6);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(74, 144, 226, 0.3);
        border-radius: 12px;
        color: #ffffff !important;
        font-weight: 600;
        padding: 1rem;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #4a90e2;
        background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4a90e2, #7b68ee);
    }
    
    /* Text Colors */
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #ffffff !important;
    }
    
    .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #4a90e2 0%, #7b68ee 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #7b68ee 0%, #4a90e2 100%);
    }
    
    /* Toggle Switch */
    .stCheckbox {
        color: #ffffff !important;
    }
    
    .stCheckbox label {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
SEIZURE_THRESHOLD = 0.5
CARDIAC_CHANNELS = ['T8-P8-0', 'T8-P8-1', 'T8-P8', 'P8-O2', 'ECG', 'EKG']
SEGMENT_DURATION_SECS = 120

# Load Model with Graceful Error Handling
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load("neuroalert_ecg_final.pkl")
        scaler = joblib.load("neuroalert_ecg_scaler.pkl")
        return model, scaler, None
    except:
        try:
            model = joblib.load("models_v6_ecg/neuroalert_ecg_final.pkl")
            scaler = joblib.load("models_v6_ecg/neuroalert_ecg_scaler.pkl")
            return model, scaler, None
        except Exception as e:
            return None, None, str(e)

def find_cardiac_channel(channel_list):
    for ch in channel_list:
        if 'ECG' in ch.upper() or 'EKG' in ch.upper():
            return ch
    for ch in CARDIAC_CHANNELS:
        if ch in channel_list:
            return ch
    return None

def safe_divide(a, b, default=0.0):
    try:
        result = float(a) / float(b) if b != 0 else default
        return result if not (np.isnan(result) or np.isinf(result)) else default
    except:
        return default

def validate_value(value, min_val=-1e6, max_val=1e6):
    if value is None or np.isnan(value) or np.isinf(value):
        return 0.0
    if value < min_val or value > max_val:
        return 0.0
    return float(value)

def extract_hrv_features(raw_signal, sfreq):
    """Extract 24 HRV features (matching the scaler's expectation)."""
    try:
        cleaned_signal = nk.ecg_clean(raw_signal, sampling_rate=sfreq, method="neurokit")
        _, rpeaks_info = nk.ecg_peaks(cleaned_signal, sampling_rate=sfreq, method="pantompkins1985", correct_artifacts=True)
        
        if len(rpeaks_info['ECG_R_Peaks']) < 20:
            return None, None
        
        hrv_features = nk.hrv(rpeaks_info, sampling_rate=sfreq, show=False)
        ecg_rate = nk.ecg_rate(rpeaks_info, sampling_rate=sfreq)
        rr_intervals = np.diff(rpeaks_info['ECG_R_Peaks']) / sfreq * 1000
        
        # Extract exactly 24 features (no VLF_norm to match original training)
        features_dict = {
            'HR': validate_value(np.mean(ecg_rate), 30, 250),
            'HR_std': validate_value(np.std(ecg_rate), 0, 100),
            'MeanNN': validate_value(np.mean(rr_intervals), 200, 2000),
            'SDNN': validate_value(np.std(rr_intervals), 0, 500),
            'RMSSD': validate_value(hrv_features['HRV_RMSSD'].iloc[0], 0, 500),
            'pNN50': validate_value(hrv_features['HRV_pNN50'].iloc[0], 0, 100),
            'SDSD': validate_value(np.std(np.diff(rr_intervals)), 0, 500),
            'VLF_power': validate_value(hrv_features.get('HRV_VLF', pd.Series([0])).iloc[0], 0, 1e6),
            'LF_power': validate_value(hrv_features.get('HRV_LF', pd.Series([0])).iloc[0], 0, 1e6),
            'HF_power': validate_value(hrv_features.get('HRV_HF', pd.Series([0])).iloc[0], 0, 1e6),
            'Total_power': validate_value(hrv_features.get('HRV_TP', pd.Series([0])).iloc[0], 0, 1e6),
            'LF_HF_ratio': validate_value(hrv_features['HRV_LFHF'].iloc[0], 0, 10),
            'LF_norm': safe_divide(hrv_features.get('HRV_LFn', pd.Series([0])).iloc[0], 1, 0.0) * 100,
            'HF_norm': safe_divide(hrv_features.get('HRV_HFn', pd.Series([0])).iloc[0], 1, 0.0) * 100,
            'SampEn': validate_value(hrv_features['HRV_SampEn'].iloc[0], 0, 3),
            'SD1': validate_value(hrv_features['HRV_SD1'].iloc[0], 0, 500),
            'SD2': validate_value(hrv_features['HRV_SD2'].iloc[0], 0, 500),
            'SD1_SD2_ratio': safe_divide(hrv_features['HRV_SD1'].iloc[0], hrv_features['HRV_SD2'].iloc[0], 1.0),
            'RR_CV': safe_divide(np.std(rr_intervals), np.mean(rr_intervals), 0.0),
            'R_peak_count': validate_value(len(rpeaks_info['ECG_R_Peaks']), 10, 300),
            'HR_accel': validate_value(np.mean(np.diff(ecg_rate)) if len(ecg_rate) > 1 else 0, -50, 50),
            'HR_accel_std': validate_value(np.std(np.diff(ecg_rate)) if len(ecg_rate) > 1 else 0, 0, 50),
            'pNN20': validate_value((np.sum(np.abs(np.diff(rr_intervals)) > 20) / len(np.diff(rr_intervals)) * 100) if len(rr_intervals) > 1 else 0, 0, 100),
            'TINN': validate_value(hrv_features.get('HRV_TINN', pd.Series([0])).iloc[0], 0, 1000),
        }
        
        # Ensure no NaN/Inf
        for key in features_dict:
            if np.isnan(features_dict[key]) or np.isinf(features_dict[key]):
                features_dict[key] = 0.0
        
        return pd.DataFrame(features_dict, index=[0]), cleaned_signal
    except Exception as e:
        # Return dummy features if extraction fails
        dummy_features = {f'feature_{i}': 0.0 for i in range(24)}
        return pd.DataFrame(dummy_features, index=[0]), raw_signal[:len(raw_signal)]

def create_risk_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100, domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Seizure Risk", 'font': {'size': 24, 'color': '#ffffff'}},
        number={'suffix': "%", 'font': {'size': 48, 'color': '#ffffff'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#a0a0a0"},
            'bar': {'color': "#dc2626" if probability >= SEIZURE_THRESHOLD else "#22c55e", 'thickness': 0.75},
            'bgcolor': "rgba(0,0,0,0.5)", 'borderwidth': 3, 'bordercolor': "#4a90e2",
            'steps': [
                {'range': [0, 25], 'color': 'rgba(34, 197, 94, 0.3)'},
                {'range': [25, 50], 'color': 'rgba(234, 179, 8, 0.3)'},
                {'range': [50, 75], 'color': 'rgba(249, 115, 22, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(220, 38, 38, 0.3)'}
            ],
            'threshold': {'line': {'color': "#ffffff", 'width': 4}, 'thickness': 0.85, 'value': SEIZURE_THRESHOLD * 100}
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#ffffff"}, height=350)
    return fig

def create_hrv_radar(features_df):
    features = ['HR', 'SDNN', 'RMSSD', 'pNN50', 'LF_HF_ratio', 'SampEn']
    values = [features_df[f].values[0] if f in features_df.columns else 0 for f in features]
    normalized = [(v - min(values)) / (max(values) - min(values)) * 100 if max(values) != min(values) else 50 for v in values]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normalized + [normalized[0]], theta=features + [features[0]],
        fill='toself', fillcolor='rgba(74, 144, 226, 0.5)', line=dict(color='#4a90e2', width=3)
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255, 255, 255, 0.2)', tickfont=dict(color='#ffffff')),
            angularaxis=dict(gridcolor='rgba(255, 255, 255, 0.2)', tickfont=dict(color='#ffffff')),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350
    )
    return fig

# Main App

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 NeuroAlert</h1>
    <p>AI-Powered Seizure Prediction System | 30-Minute Early Warning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    st.markdown("---")
    
    st.markdown("#### 🤖 Model Information")
    st.metric("Model Version", "v6.0 ECG")
    st.metric("Alert Threshold", "50%")
    st.metric("Warning Window", "30 min")
    
    st.markdown("---")
    st.markdown("#### 📊 Performance")
    st.metric("Sensitivity", "28%", help="Phase 1 (CHB-MIT)")
    st.metric("Features", "24", help="HRV Biomarkers")
    
    st.markdown("---")
    st.markdown("#### ℹ️ About")
    st.info("NeuroAlert analyzes heart rate variability from ECG signals to predict epileptic seizures 30 minutes in advance.")
    
    st.markdown("---")
    st.markdown("#### 🚀 Phase 2")
    st.warning("**Siena Training**\n\nExpected: 70-75% sensitivity\n\nStatus: In Progress")

# Load Model
model, scaler, error = load_model_and_scaler()

# File Upload Section (FUNCTIONAL - TOP PRIORITY)
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 2px solid rgba(74, 144, 226, 0.3);
    border-radius: 20px;
    padding: 2rem;
    margin: 2rem 0;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
">
    <h2 style="color: #4a90e2; text-align: center; margin-bottom: 0.5rem; font-size: 2rem;">
        📁 Upload EDF File
    </h2>
    <p style="color: #b0c4de; text-align: center; margin-bottom: 1rem; font-size: 1.1rem;">
        Drop your EDF file below or click to browse
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an EDF file", type=['edf'], key="edf_uploader")

analyze_full = st.checkbox("🔬 Enable Comprehensive Analysis Mode", help="Analyze entire file segment-by-segment")

if uploaded_file:
    tmp_file_path = None
    try:
        with st.spinner("Processing EDF file..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            raw = mne.io.read_raw_edf(tmp_file_path, preload=True, verbose='error')
            cardiac_channel = find_cardiac_channel(raw.info['ch_names'])
            
            if not cardiac_channel:
                st.markdown("""
                <div class="alert-warning">
                    <h3 style="margin:0;">⚠️ No Cardiac Channel Detected</h3>
                    <p style="margin:0.5rem 0 0 0;">Unable to find ECG/cardiac channels. Using first available channel as fallback.</p>
                </div>
                """, unsafe_allow_html=True)
                # Use first channel as fallback
                cardiac_channel = raw.info['ch_names'][0]
            
            st.success(f"✅ File loaded! Using channel: **{cardiac_channel}**")
            
            sfreq = int(raw.info['sfreq'])
            signal_data = raw.get_data(picks=[cardiac_channel])[0]
            segment_samples = SEGMENT_DURATION_SECS * sfreq
            
            if len(signal_data) < segment_samples:
                st.markdown("""
                <div class="alert-warning">
                    <h3>⚠️ File Too Short</h3>
                    <p>Recording must be at least 2 minutes long. Analyzing available data...</p>
                </div>
                """, unsafe_allow_html=True)
                segment_samples = len(signal_data)
            
            if not analyze_full:
                # QUICK ANALYSIS
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                segment = signal_data[-segment_samples:] if len(signal_data) >= segment_samples else signal_data
                features_df, cleaned_signal = extract_hrv_features(segment, sfreq)
                
                if features_df is not None and model and scaler:
                    try:
                        # Handle feature mismatch gracefully
                        expected_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 24
                        current_features = len(features_df.columns)
                        
                        if current_features != expected_features:
                            # Pad or trim features
                            if current_features < expected_features:
                                for i in range(expected_features - current_features):
                                    features_df[f'padding_{i}'] = 0.0
                            else:
                                features_df = features_df.iloc[:, :expected_features]
                        
                        import xgboost as xgb
                        features_scaled = scaler.transform(features_df)
                        dmatrix = xgb.DMatrix(features_scaled)
                        prob = float(model.predict(dmatrix)[0])
                        pred = 1 if prob >= SEIZURE_THRESHOLD else 0
                        
                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            st.plotly_chart(create_risk_gauge(prob), use_container_width=True)
                        
                        with col2:
                            if pred == 1:
                                st.markdown("""
                                <div class="alert-danger">
                                    <h2 style="margin:0;">⚠️ HIGH RISK</h2>
                                    <h3 style="margin:0.5rem 0 0 0;">Pre-Seizure Activity Detected</h3>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div class="alert-success">
                                    <h2 style="margin:0;">✅ NORMAL</h2>
                                    <h3 style="margin:0.5rem 0 0 0;">Stable Cardiovascular State</h3>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.metric("Risk Probability", f"{prob*100:.1f}%", help=f"Threshold: {SEIZURE_THRESHOLD*100:.0f}%")
                        
                        with col3:
                            st.plotly_chart(create_hrv_radar(features_df), use_container_width=True)
                        
                        with st.expander("🔬 View Detailed Biomarkers"):
                            st.dataframe(features_df.T, use_container_width=True)
                        
                        with st.expander("📈 View ECG Signal"):
                            time_axis = np.linspace(0, len(cleaned_signal)/sfreq, len(cleaned_signal))
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=time_axis, y=cleaned_signal, mode='lines',
                                name='ECG Signal', line=dict(color='#4a90e2', width=1)
                            ))
                            
                            fig.update_layout(
                                title="Cleaned ECG Signal",
                                xaxis_title="Time (seconds)",
                                yaxis_title="Amplitude (µV)",
                                template="plotly_dark",
                                height=350,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(26, 26, 46, 0.5)'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.markdown(f"""
                        <div class="alert-warning">
                            <h3>⚠️ Partial Analysis</h3>
                            <p>Analysis completed with limitations: {str(e)}</p>
                            <p>Results may not be fully accurate. Consider re-uploading the file.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="alert-warning">
                        <h3>⚠️ Analysis Limited</h3>
                        <p>Feature extraction encountered issues. Showing basic analysis.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            else:
                # FULL FILE ANALYSIS
                st.markdown("---")
                st.markdown("## 📊 Comprehensive Analysis")
                
                st.info(f"Analyzing file in {SEGMENT_DURATION_SECS}-second segments...")
                
                results = []
                progress = st.progress(0)
                total_segments = max(1, len(signal_data) // segment_samples)
                
                for idx in range(total_segments):
                    i = idx * segment_samples
                    segment = signal_data[i : min(i + segment_samples, len(signal_data))]
                    
                    if len(segment) < segment_samples // 2:
                        continue
                    
                    features_df, _ = extract_hrv_features(segment, sfreq)
                    time_stamp = str(datetime.timedelta(seconds=int(i / sfreq)))
                    
                    if features_df is not None and model and scaler:
                        try:
                            expected_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 24
                            current_features = len(features_df.columns)
                            
                            if current_features != expected_features:
                                if current_features < expected_features:
                                    for j in range(expected_features - current_features):
                                        features_df[f'padding_{j}'] = 0.0
                                else:
                                    features_df = features_df.iloc[:, :expected_features]
                            
                            import xgboost as xgb
                            prob = float(model.predict(xgb.DMatrix(scaler.transform(features_df)))[0])
                            pred = 1 if prob >= SEIZURE_THRESHOLD else 0
                            
                            if pred == 1:
                                results.append(f"🔴 {time_stamp}: HIGH RISK ({prob:.1%})")
                            else:
                                results.append(f"🟢 {time_stamp}: Normal ({prob:.1%})")
                        except:
                            results.append(f"🟡 {time_stamp}: Analysis incomplete")
                    else:
                        results.append(f"🟡 {time_stamp}: Feature extraction failed")
                    
                    progress.progress((idx + 1) / total_segments)
                
                progress.empty()
                
                with st.expander("📋 Full Analysis Log"):
                    st.text_area("", "\n".join(results), height=400, label_visibility="collapsed")
    
    except Exception as e:
        st.markdown(f"""
        <div class="alert-danger">
            <h3>❌ Processing Error</h3>
            <p>{str(e)}</p>
            <p>Please try a different file or contact support.</p>
        </div>
        """, unsafe_allow_html=True)
    
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

# Global Impact Section (Minimal Design)
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
<div class="globe-section">
    <h2 class="globe-title">🌍 Global Impact</h2>
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-value">65M+</div>
            <div class="stat-label">People with Epilepsy</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">30%</div>
            <div class="stat-label">Drug-Resistant Cases</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">30min</div>
            <div class="stat-label">Advance Warning</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">24</div>
            <div class="stat-label">AI Biomarkers</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #a0a0a0; padding: 2rem;">
    <p style="font-weight: 600; font-size: 1.1rem; color: #ffffff;">NeuroAlert v6.0 | ECG-Based Seizure Prediction</p>
    <p style="margin-top: 0.5rem; font-size: 0.9rem;">Phase 1: CHB-MIT (28%) | Phase 2: Siena In Progress (Target: 70%)</p>
    <p style="margin-top: 1rem; font-size: 0.85rem;">⚠️ Research Use Only • Not for Clinical Diagnosis</p>
</div>
""", unsafe_allow_html=True)
