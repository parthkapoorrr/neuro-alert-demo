import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfilt, find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings('ignore')

# For EDF file reading
try:
    import mne
    EDF_SUPPORT = True
except ImportError:
    EDF_SUPPORT = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="NeuroAlert - AI Seizure Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - DEEP PURPLE/SOFT LILAC THEME
# ============================================================================
st.markdown("""
<style>
    /* Color Palette Variables */
    :root {
        --primary-purple: #5B2E90;
        --secondary-lilac: #BFA2DB;
        --background-white: #F7F4FB;
        --accent-purple: #7B4FB8;
        --dark-purple: #3D1B5C;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #F7F4FB 0%, #E8DFF5 100%);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Title - WHITE COLOR */
    .main-title {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #5B2E90 0%, #7B4FB8 100%);
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(91, 46, 144, 0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    .main-title h1 {
        color: #FFFFFF !important;
        font-size: 4rem !important;
        font-weight: 900 !important;
        margin: 0 !important;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
        letter-spacing: 2px;
    }
    
    .main-title p {
        color: #F7F4FB !important;
        font-size: 1.3rem !important;
        margin: 10px 0 0 0 !important;
        font-weight: 300;
        opacity: 0.95;
    }
    
    /* Glowing Animation */
    @keyframes glow {
        from {
            box-shadow: 0 10px 40px rgba(91, 46, 144, 0.3),
                        0 0 20px rgba(191, 162, 219, 0.2);
        }
        to {
            box-shadow: 0 15px 60px rgba(91, 46, 144, 0.5),
                        0 0 40px rgba(191, 162, 219, 0.4);
        }
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #5B2E90 0%, #7B4FB8 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.15);
        border: 2px dashed #BFA2DB;
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        background: rgba(255, 255, 255, 0.25);
        border-color: #FFFFFF;
    }
    
    /* Drag and drop text - Dark purple for visibility */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span {
        color: #3D1B5C !important;
        font-weight: 600 !important;
    }
    
    /* Browse Files Button - Match Primary Purple */
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #5B2E90 0%, #7B4FB8 100%) !important;
        color: #FFFFFF !important;
        border: 2px solid #7B4FB8 !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    
    [data-testid="stFileUploader"] button:hover {
        background: linear-gradient(135deg, #7B4FB8 0%, #9B6FD8 100%) !important;
        border-color: #9B6FD8 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(91, 46, 144, 0.4);
    }
    
    /* Uploaded Filename - Dark Purple for Contrast */
    [data-testid="stFileUploader"] section {
        color: #3D1B5C !important;
        font-weight: 700 !important;
        background: rgba(255, 255, 255, 0.9) !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stFileUploader"] small {
        color: #5B2E90 !important;
        font-weight: 600 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #BFA2DB 0%, #D4C2E8 100%);
        color: #3D1B5C;
        border: none;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(191, 162, 219, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #D4C2E8 0%, #BFA2DB 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(191, 162, 219, 0.5);
    }
    
    /* Glowing Cards */
    .glow-card {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid #BFA2DB;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(91, 46, 144, 0.15);
        transition: all 0.3s ease;
    }
    
    .glow-card:hover {
        box-shadow: 0 12px 48px rgba(91, 46, 144, 0.25);
        transform: translateY(-5px);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(247, 244, 251, 0.95) 100%);
        border: 2px solid #BFA2DB;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(91, 46, 144, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 35px rgba(91, 46, 144, 0.2);
        border-color: #7B4FB8;
    }
    
    /* Risk Level Cards - Purple background, colored text */
    .risk-high {
        background: linear-gradient(135deg, #5B2E90 0%, #7B4FB8 100%);
        border: 3px solid #BFA2DB;
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(91, 46, 144, 0.4);
        animation: pulse-purple 2s ease-in-out infinite;
    }
    
    .risk-high h1 {
        color: #ff4757 !important;
        text-shadow: 0 0 20px rgba(255, 71, 87, 0.6);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #5B2E90 0%, #7B4FB8 100%);
        border: 3px solid #BFA2DB;
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(91, 46, 144, 0.4);
        animation: pulse-purple 2s ease-in-out infinite;
    }
    
    .risk-medium h1 {
        color: #ffc107 !important;
        text-shadow: 0 0 20px rgba(255, 193, 7, 0.6);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #5B2E90 0%, #7B4FB8 100%);
        border: 3px solid #BFA2DB;
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(91, 46, 144, 0.4);
        animation: pulse-purple 2s ease-in-out infinite;
    }
    
    .risk-low h1 {
        color: #28a745 !important;
        text-shadow: 0 0 20px rgba(40, 167, 69, 0.6);
    }
    
    @keyframes pulse-purple {
        0%, 100% { box-shadow: 0 10px 40px rgba(91, 46, 144, 0.4); }
        50% { box-shadow: 0 15px 60px rgba(91, 46, 144, 0.6); }
    }
    
    /* Info Messages */
    .stAlert {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid #BFA2DB !important;
        border-radius: 12px !important;
        color: #3D1B5C !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(191, 162, 219, 0.2) 0%, rgba(212, 194, 232, 0.2) 100%);
        border-radius: 10px;
        font-weight: 600;
        color: #5B2E90 !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #5B2E90 0%, #BFA2DB 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #5B2E90 !important;
        font-weight: 700;
    }
    
    p, li, span {
        color: #3D1B5C;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F7F4FB;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #5B2E90 0%, #BFA2DB 100%);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN TITLE
# ============================================================================
st.markdown("""
<div class='main-title'>
    <h1>🧠 NEUROALERT</h1>
    <p>AI-Powered Seizure Risk Prediction from ECG Signals</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Butterworth bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply bandpass filter"""
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def resample_signal(signal, original_fs, target_fs=256):
    """
    Resample signal to target frequency
    SIENA files: 512 Hz → CHB-MIT standard: 256 Hz
    """
    if original_fs == target_fs:
        return signal, original_fs
    
    # Calculate resampling ratio
    num_samples = int(len(signal) * target_fs / original_fs)
    
    # Resample using scipy
    from scipy import signal as sp_signal
    resampled = sp_signal.resample(signal, num_samples)
    
    return resampled, target_fs

def extract_comprehensive_hrv_features(ecg_signal, fs=250):
    """
    Extract comprehensive HRV biomarkers for medical analysis
    Returns: features array and detailed metrics dictionary
    """
    ecg_signal = np.array(ecg_signal).flatten()
    
    # Filter the signal (0.5-40 Hz for ECG)
    filtered_ecg = bandpass_filter(ecg_signal, 0.5, 40, fs, order=5)
    
    # Detect R-peaks
    peaks, properties = find_peaks(
        filtered_ecg,
        height=np.mean(filtered_ecg) + 0.5 * np.std(filtered_ecg),
        distance=int(0.6 * fs)
    )
    
    features = []
    metrics = {}
    
    if len(peaks) > 2:
        # RR intervals (in ms)
        rr_intervals = np.diff(peaks) / fs * 1000
        
        # ========== TIME DOMAIN HRV FEATURES ==========
        mean_rr = np.mean(rr_intervals)
        sdnn = np.std(rr_intervals)  # Standard deviation of NN intervals
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))  # Root mean square of successive differences
        sdsd = np.std(np.diff(rr_intervals))  # Standard deviation of successive differences
        nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)  # Number of pairs differing by >50ms
        pnn50 = (nn50 / len(rr_intervals)) * 100 if len(rr_intervals) > 0 else 0  # Percentage of NN50
        
        # Additional time domain metrics
        rr_range = np.max(rr_intervals) - np.min(rr_intervals)
        cv_rr = (sdnn / mean_rr) * 100 if mean_rr > 0 else 0  # Coefficient of variation
        median_rr = np.median(rr_intervals)
        mad_rr = np.median(np.abs(rr_intervals - median_rr))  # Median absolute deviation
        
        # Heart Rate metrics
        mean_hr = 60000 / mean_rr if mean_rr > 0 else 0
        min_hr = 60000 / np.max(rr_intervals) if np.max(rr_intervals) > 0 else 0
        max_hr = 60000 / np.min(rr_intervals) if np.min(rr_intervals) > 0 else 0
        
        # Store time domain metrics
        metrics['Time Domain'] = {
            'Mean RR (ms)': mean_rr,
            'SDNN (ms)': sdnn,
            'RMSSD (ms)': rmssd,
            'SDSD (ms)': sdsd,
            'NN50 (count)': nn50,
            'pNN50 (%)': pnn50,
            'RR Range (ms)': rr_range,
            'CV (%)': cv_rr,
            'Median RR (ms)': median_rr,
            'MAD (ms)': mad_rr,
            'Mean HR (bpm)': mean_hr,
            'Min HR (bpm)': min_hr,
            'Max HR (bpm)': max_hr,
        }
        
        # Add to features array
        features.extend([mean_rr, sdnn, rmssd, sdsd, pnn50, mean_hr])
        
        # ========== FREQUENCY DOMAIN HRV FEATURES ==========
        if len(rr_intervals) > 10:
            # Resample RR intervals to uniform time series
            time_rr = np.cumsum(rr_intervals) / 1000  # Convert to seconds
            
            # Ensure we have enough time points for frequency analysis
            if time_rr[-1] < 30:  # Less than 30 seconds of RR data
                # Not enough data for reliable frequency analysis
                metrics['Frequency Domain'] = {
                    'VLF Power (ms²)': 0,
                    'LF Power (ms²)': 0,
                    'HF Power (ms²)': 0,
                    'Total Power (ms²)': 0,
                    'LF norm (%)': 0,
                    'HF norm (%)': 0,
                    'LF/HF Ratio': 0,
                }
                features.extend([0, 0, 0])
            else:
                # Uniform sampling at 4 Hz
                time_uniform = np.arange(0, time_rr[-1], 0.25)
                
                # Need at least 8 points for frequency analysis
                if len(time_uniform) < 8:
                    metrics['Frequency Domain'] = {
                        'VLF Power (ms²)': 0,
                        'LF Power (ms²)': 0,
                        'HF Power (ms²)': 0,
                        'Total Power (ms²)': 0,
                        'LF norm (%)': 0,
                        'HF norm (%)': 0,
                        'LF/HF Ratio': 0,
                    }
                    features.extend([0, 0, 0])
                else:
                    rr_uniform = np.interp(time_uniform, time_rr, rr_intervals)
                    
                    # Compute power spectral density with appropriate nperseg
                    nperseg_val = min(256, len(rr_uniform) // 2)
                    if nperseg_val < 4:
                        nperseg_val = min(len(rr_uniform), 4)
                    
                    freqs, psd = signal.welch(rr_uniform, fs=4, nperseg=nperseg_val)
                    
                    # Frequency bands (Hz)
                    vlf_band = (freqs >= 0.003) & (freqs < 0.04)   # Very Low Frequency
                    lf_band = (freqs >= 0.04) & (freqs < 0.15)     # Low Frequency
                    hf_band = (freqs >= 0.15) & (freqs < 0.4)      # High Frequency
                    
                    # Power in each band (ms²)
                    vlf_power = np.trapz(psd[vlf_band], freqs[vlf_band]) if np.any(vlf_band) else 0
                    lf_power = np.trapz(psd[lf_band], freqs[lf_band]) if np.any(lf_band) else 0
                    hf_power = np.trapz(psd[hf_band], freqs[hf_band]) if np.any(hf_band) else 0
                    total_power = vlf_power + lf_power + hf_power
                    
                    # Normalized power
                    lf_norm = (lf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0
                    hf_norm = (hf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0
                    
                    # LF/HF ratio (autonomic balance)
                    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
                    
                    metrics['Frequency Domain'] = {
                        'VLF Power (ms²)': vlf_power,
                        'LF Power (ms²)': lf_power,
                        'HF Power (ms²)': hf_power,
                        'Total Power (ms²)': total_power,
                        'LF norm (%)': lf_norm,
                        'HF norm (%)': hf_norm,
                        'LF/HF Ratio': lf_hf_ratio,
                    }
                    
                    features.extend([lf_power, hf_power, lf_hf_ratio])
        else:
            metrics['Frequency Domain'] = {
                'VLF Power (ms²)': 0,
                'LF Power (ms²)': 0,
                'HF Power (ms²)': 0,
                'Total Power (ms²)': 0,
                'LF norm (%)': 0,
                'HF norm (%)': 0,
                'LF/HF Ratio': 0,
            }
            features.extend([0, 0, 0])
        
        # ========== NONLINEAR HRV FEATURES ==========
        # SD1 and SD2 (Poincaré plot)
        sd1 = np.sqrt(0.5 * rmssd**2)
        sd2 = np.sqrt(2 * sdnn**2 - 0.5 * rmssd**2)
        sd_ratio = sd1 / sd2 if sd2 > 0 else 0
        
        metrics['Nonlinear'] = {
            'SD1 (ms)': sd1,
            'SD2 (ms)': sd2,
            'SD1/SD2 Ratio': sd_ratio,
        }
        
        # ========== SIGNAL QUALITY METRICS ==========
        signal_std = np.std(filtered_ecg)
        signal_max = np.max(np.abs(filtered_ecg))
        signal_mad = np.mean(np.abs(np.diff(filtered_ecg)))
        
        metrics['Signal Quality'] = {
            'Signal Std (mV)': signal_std,
            'Peak Amplitude (mV)': signal_max,
            'Mean Derivative': signal_mad,
            'R-peaks Detected': len(peaks),
        }
        
        features.extend([signal_std, signal_max, signal_mad, len(peaks)])
        
        # Additional features for model
        features.extend([rr_range, median_rr])
        
    else:
        # Not enough peaks - return zeros
        features = [0] * 15
        metrics = {
            'Time Domain': {'Error': 'Insufficient R-peaks detected'},
            'Frequency Domain': {'Error': 'Insufficient data'},
            'Nonlinear': {'Error': 'Insufficient data'},
            'Signal Quality': {'R-peaks Detected': len(peaks)}
        }
    
    return np.array(features), metrics, filtered_ecg, peaks

def create_speedometer(value, title="Risk Level"):
    """Create speedometer gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24, 'color': '#5B2E90', 'family': 'Arial Black'}},
        number={'font': {'size': 50, 'color': '#5B2E90', 'family': 'Arial Black'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': '#5B2E90'},
            'bar': {'color': "#5B2E90", 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#BFA2DB",
            'steps': [
                {'range': [0, 30], 'color': '#ffcccc'},
                {'range': [30, 70], 'color': '#fff4cc'},
                {'range': [70, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "#ff4757", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#5B2E90', 'family': 'Arial'}
    )
    
    return fig

def create_ecg_plot(ecg_data, fs, peaks=None):
    """Create interactive ECG plot with R-peaks"""
    time = np.arange(len(ecg_data)) / fs
    
    fig = go.Figure()
    
    # ECG trace
    fig.add_trace(go.Scatter(
        x=time,
        y=ecg_data,
        mode='lines',
        name='ECG Signal',
        line=dict(color='#5B2E90', width=2),
        hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.3f} mV<extra></extra>'
    ))
    
    # Add R-peaks if provided
    if peaks is not None and len(peaks) > 0:
        fig.add_trace(go.Scatter(
            x=time[peaks],
            y=ecg_data[peaks],
            mode='markers',
            name='R-peaks',
            marker=dict(color='#ff4757', size=10, symbol='diamond'),
            hovertemplate='R-peak<br>Time: %{x:.2f}s<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text='📈 ECG Signal with R-peak Detection', font=dict(size=24, color='#5B2E90', family='Arial Black')),
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (mV)",
        hovermode='x unified',
        height=500,
        plot_bgcolor='rgba(247, 244, 251, 0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#5B2E90', family='Arial'),
        xaxis=dict(gridcolor='rgba(191, 162, 219, 0.3)', showgrid=True),
        yaxis=dict(gridcolor='rgba(191, 162, 219, 0.3)', showgrid=True),
        legend=dict(
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#BFA2DB',
            borderwidth=2
        )
    )
    
    return fig

def read_edf_file(uploaded_file):
    """Read EDF file and extract ECG channel with robust temp file handling"""
    if not EDF_SUPPORT:
        st.error("⚠️ MNE library not installed. Install with: pip install mne")
        return None, None, None
    
    import tempfile
    import os
    
    tmp_path = None
    try:
        # Create temporary file with guaranteed cleanup
        with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        # Read EDF file
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        
        # Get sampling frequency
        fs = raw.info['sfreq']
        
        # Get channel names
        channels = raw.ch_names
        
        # Try to find ECG channel
        ecg_channel = None
        ecg_keywords = ['ECG', 'EKG', 'CARDIO', 'HEART']
        
        for ch in channels:
            if any(keyword in ch.upper() for keyword in ecg_keywords):
                ecg_channel = ch
                break
        
        # If no ECG channel found, use first channel
        if ecg_channel is None:
            ecg_channel = channels[0]
            st.warning(f"⚠️ No ECG channel detected. Using first channel: {ecg_channel}")
        
        # Extract ECG data
        ecg_data = raw.get_data(picks=[ecg_channel])[0]
        
        return ecg_data, fs, channels
        
    except Exception as e:
        st.error(f"⚠️ Error reading EDF file: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None
        
    finally:
        # Guaranteed cleanup of temp file
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                st.warning(f"Could not delete temporary file: {e}")

# ============================================================================
# LOAD MODEL AND SCALER
# ============================================================================
@st.cache_resource
def load_model_and_scaler():
    """Load the trained seizure prediction model and scaler"""
    model = None
    scaler = None
    
    try:
        with open('nueroalert_hybrid_ecg_only.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        try:
            with open('neuroalert_hybrid_ecg_only.pkl', 'rb') as f:
                model = pickle.load(f)
        except FileNotFoundError:
            st.error("⚠️ Model file not found. Please ensure model .pkl file is in the same directory.")
    
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            st.sidebar.success("✅ Scaler loaded - predictions will be accurate!")
    except FileNotFoundError:
        st.sidebar.warning("⚠️ Scaler not found. Predictions may be less accurate.")
    
    return model, scaler

model, scaler = load_model_and_scaler()

# ============================================================================
# SESSION STATE FOR TEMPORAL SMOOTHING
# ============================================================================
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'probability_history' not in st.session_state:
    st.session_state.probability_history = []

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    st.markdown("---")
    
    # Segment duration slider
    segment_duration = st.select_slider(
        "Analysis Window Size",
        options=[30, 60, 90, 120, 150, 180],
        value=120,
        help="Duration of ECG segment to analyze (seconds). Use 120s if model trained on 2-min windows."
    )
    
    st.info(f"📏 Analyzing **{segment_duration}s** ({segment_duration//60}min {segment_duration%60}s) windows")
    
    # Window start position
    window_start = st.number_input(
        "Window Start Position (minutes)",
        min_value=0,
        max_value=60,
        value=0,
        step=1,
        help="Where to start analyzing in the recording (0 = beginning). Change this if seizure occurs later in file."
    )
    
    st.info(f"⏱️ Starting analysis at **{window_start}** minute(s)")
    
    with st.expander("ℹ️ Why adjust start position?"):
        st.markdown("""
        **Critical for seizure detection:**
        
        - If seizure occurs 20 minutes into recording, analyzing first 2 minutes won't detect it!
        - Check your seizure timestamp file
        - Set window start to just before seizure time
        
        **Example:** PN00-2.edf
        - Recording starts: 02:18:17
        - Seizure starts: 02:38:37 (20 min later)
        - Set window start to **18-20 minutes**
        
        **Default (0 min):** Analyzes from beginning
        """)
    
    with st.expander("ℹ️ Why 120s default?"):
        st.markdown("""
        **Window Duration Guidelines:**
        
        - **120s (2 min):** Standard for HRV analysis, recommended if model trained on 2-min windows
        - **60s (1 min):** Faster analysis, less stable HRV metrics
        - **180s (3 min):** More stable, but slower
        
        **Important:** Use the same duration your model was trained on!
        
        If predictions seem random, try adjusting this setting.
        """)
    
    st.markdown("---")
    st.markdown("### 🎯 Prediction Settings")
    
    # Confidence threshold slider
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=0.8,
        value=0.65,
        step=0.05,
        help="Higher threshold = fewer false alarms, but may miss some events"
    )
    
    st.info(f"🎯 Current threshold: **{confidence_threshold:.0%}**")
    
    # Temporal smoothing settings
    st.markdown("### 🔄 Temporal Smoothing")
    
    enable_smoothing = st.checkbox(
        "Enable Temporal Smoothing",
        value=True,
        help="Require multiple consecutive positive predictions to reduce false alarms"
    )
    
    if enable_smoothing:
        window_size = st.select_slider(
            "History Window Size",
            options=[2, 3, 4, 5],
            value=3,
            help="Number of recent predictions to consider"
        )
        
        required_positive = st.select_slider(
            "Required Positive Count",
            options=list(range(1, window_size + 1)),
            value=2,
            help="Minimum positive predictions needed to trigger alert"
        )
        
        st.info(f"📊 Alert if **{required_positive}/{window_size}** windows are positive")
    
    st.markdown("---")
    st.markdown("### 📊 About NeuroAlert")
    st.markdown("""
    **NeuroAlert** uses AI to predict seizure risk from ECG patterns.
    
    **Model Performance:**
    - Sensitivity: 40%
    - Specificity: 72%
    - Model: v12 Hybrid
    
    **Analyzed Biomarkers:**
    - Time Domain HRV
    - Frequency Domain HRV
    - Nonlinear Metrics
    - Signal Quality
    """)
    
    st.markdown("---")
    
    st.markdown("### ⚠️ Important Limitations")
    with st.expander("🧠 EEG vs ECG Reality Check"):
        st.markdown("""
        **Critical Understanding:**
        
        🧠 **Seizures = Brain (EEG) Events**
        - Occur in EEG channels (Fp1, F3, C3, etc.)
        - Measured by brain electrical activity
        
        ❤️ **This App = Heart (ECG) Analysis**
        - Analyzes EKG/ECG channels only
        - Measures heart rate variability
        
        **⚠️ The Problem:**
        Not all brain seizures cause detectable heart changes!
        
        - Some seizures: Clear HR/HRV changes ✅
        - Many seizures: No ECG signature ❌
        
        **What this means:**
        - If model says LOW RISK but EEG shows seizure → ECG didn't see it (no cardiac signature)
        - This is a fundamental limitation of ECG-based detection
        - For full seizure detection, EEG analysis needed
        
        **Use case:** ECG best for seizures with cardiac manifestations
        """)
    
    st.markdown("---")
    st.warning("⚠️ **Medical Disclaimer:** For research and educational purposes only. Not for clinical diagnosis.")

# ============================================================================
# MAIN CONTENT
# ============================================================================

# File Upload Section - Prominent on main page
st.markdown("""
<div class='glow-card' style='text-align: center;'>
    <h2 style='color: #5B2E90; margin: 0 0 10px 0;'>📂 Upload Your ECG Data</h2>
    <p style='color: #3D1B5C; font-size: 1.1rem; margin: 0 0 20px 0;'>
        Upload an EDF file containing ECG recording to begin analysis
    </p>
</div>
""", unsafe_allow_html=True)

# File uploader with custom styling
uploaded_file = st.file_uploader(
    "Choose EDF file",
    type=['edf'],
    help="Upload an EDF file containing ECG signal recording",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    st.success(f"✅ **File uploaded successfully!** - {uploaded_file.name}")

st.markdown("<br>", unsafe_allow_html=True)

if uploaded_file is None:
    # Welcome screen
    st.markdown("""
    <div class='glow-card'>
        <h2 style='color: #5B2E90; text-align: center; margin-top: 0;'>👋 Welcome to NeuroAlert</h2>
        <p style='text-align: center; font-size: 1.2rem; color: #3D1B5C;'>
            Upload your ECG EDF file to get AI-powered seizure risk prediction
        </p>
        <br>
        <h3 style='color: #5B2E90;'>📋 How to Use:</h3>
        <ol style='font-size: 1.1rem; line-height: 2; color: #3D1B5C;'>
            <li><strong>Upload EDF File:</strong> Use the sidebar to upload your ECG recording</li>
            <li><strong>Start Analysis:</strong> Click the "Start Analysis" button when ready</li>
            <li><strong>View Results:</strong> Review comprehensive seizure risk assessment</li>
            <li><strong>Check Biomarkers:</strong> Examine full HRV biomarker analysis</li>
            <li><strong>Follow Guidelines:</strong> Read safety recommendations if risk detected</li>
        </ol>
        <br>
        <h3 style='color: #5B2E90;'>🎯 What We Analyze:</h3>
        <ul style='font-size: 1.1rem; line-height: 2; color: #3D1B5C;'>
            <li><strong>Time Domain HRV:</strong> SDNN, RMSSD, pNN50, Heart Rate</li>
            <li><strong>Frequency Domain:</strong> VLF, LF, HF power, LF/HF ratio</li>
            <li><strong>Nonlinear Metrics:</strong> Poincaré plot (SD1, SD2)</li>
            <li><strong>Signal Quality:</strong> R-peak detection, signal integrity</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='glow-card' style='text-align: center;'>
            <h2 style='color: #5B2E90; margin: 0;'>🎯</h2>
            <h3 style='color: #5B2E90; margin: 10px 0;'>AI Prediction</h3>
            <p style='color: #3D1B5C;'>Advanced ML models for seizure risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='glow-card' style='text-align: center;'>
            <h2 style='color: #5B2E90; margin: 0;'>⚡</h2>
            <h3 style='color: #5B2E90; margin: 10px 0;'>Real-Time</h3>
            <p style='color: #3D1B5C;'>Instant comprehensive analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='glow-card' style='text-align: center;'>
            <h2 style='color: #5B2E90; margin: 0;'>🛡️</h2>
            <h3 style='color: #5B2E90; margin: 10px 0;'>Safety First</h3>
            <p style='color: #3D1B5C;'>Actionable safety recommendations</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # File uploaded - show Start Analysis button
    st.markdown("""
    <div class='glow-card' style='text-align: center;'>
        <h2 style='color: #5B2E90; margin: 0;'>✅ EDF File Ready for Analysis</h2>
        <p style='color: #3D1B5C; font-size: 1.1rem; margin: 15px 0;'>
            Click the button below to start comprehensive ECG analysis and seizure risk prediction
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        start_analysis = st.button("🚀 START ANALYSIS", use_container_width=True, type="primary")
    
    if start_analysis and model is not None:
        # Analysis workflow
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Read EDF file
            status_text.text("📂 Reading EDF file...")
            progress_bar.progress(10)
            
            ecg_data, fs, channels = read_edf_file(uploaded_file)
            
            if ecg_data is None:
                st.error("❌ Failed to read EDF file. Please check file format.")
                progress_bar.empty()
                status_text.empty()
            else:
                # Step 2: Process signal
                status_text.text("🔬 Processing ECG signal...")
                progress_bar.progress(30)
                
                # Calculate start sample based on user-selected start position
                start_sample = int(window_start * 60 * fs)  # Convert minutes to samples
                end_sample = start_sample + int(segment_duration * fs)
                
                # Check if requested window is within recording
                if start_sample >= len(ecg_data):
                    st.error(f"❌ Window start ({window_start} min) is beyond recording length!")
                    st.info(f"💡 Recording is only {len(ecg_data)/fs/60:.1f} minutes long")
                    progress_bar.empty()
                    status_text.empty()
                    st.stop()
                
                if end_sample > len(ecg_data):
                    st.warning(f"⚠️ Requested window extends beyond recording. Using available data.")
                    end_sample = len(ecg_data)
                
                # Extract the window
                ecg_data = ecg_data[start_sample:end_sample]
                actual_duration = len(ecg_data) / fs
                
                # Minimum duration check
                min_duration = 10  # seconds
                if actual_duration < min_duration:
                    st.error(f"❌ Window too short! Need at least {min_duration}s, got {actual_duration:.1f}s")
                    progress_bar.empty()
                    status_text.empty()
                    st.stop()
                
                # Check if resampling needed (SIENA files are 512 Hz, model trained on 256 Hz)
                original_fs = fs
                if fs > 256:
                    status_text.text(f"🔄 Resampling from {fs} Hz to 256 Hz...")
                    ecg_data, fs = resample_signal(ecg_data, fs, target_fs=256)
                    st.info(f"📊 Resampled signal from {original_fs} Hz → {fs} Hz (CHB-MIT standard)")
                
                # Step 3: Extract features and biomarkers
                status_text.text("🧮 Extracting HRV biomarkers...")
                progress_bar.progress(50)
                
                features, hrv_metrics, filtered_ecg, peaks = extract_comprehensive_hrv_features(ecg_data, fs)
                
                # CRITICAL: Minimum R-peak validation
                min_peaks_required = max(10, segment_duration // 6)  # At least 10 peaks, or ~1 peak per 6s
                if len(peaks) < min_peaks_required:
                    st.error(f"❌ **Insufficient Signal Quality**")
                    st.error(f"Detected only **{len(peaks)} R-peaks**, need at least **{min_peaks_required}** for reliable analysis.")
                    st.warning("⚠️ **Possible causes:**")
                    st.markdown("""
                    - Poor electrode contact
                    - Wrong ECG channel selected
                    - Excessive noise in recording
                    - Non-ECG signal uploaded
                    
                    **Recommendation:** Check signal quality and try again.
                    """)
                    progress_bar.empty()
                    status_text.empty()
                    st.stop()
                
                # Step 4: Make prediction
                status_text.text("🤖 Running AI prediction model...")
                progress_bar.progress(70)
                
                features_reshaped = features.reshape(1, -1)
                
                # Apply scaler if available (CRITICAL for accuracy!)
                if scaler is not None:
                    features_scaled = scaler.transform(features_reshaped)
                else:
                    features_scaled = features_reshaped
                    st.warning("⚠️ Scaler not loaded - using raw features")
                
                # Make prediction with scaled features
                raw_prediction = model.predict(features_scaled)[0]
                probability = model.predict_proba(features_scaled)[0]
                
                # Apply confidence threshold
                high_risk_prob = probability[1]
                threshold_prediction = 1 if high_risk_prob >= confidence_threshold else 0
                
                # Store in history
                st.session_state.prediction_history.append(threshold_prediction)
                st.session_state.probability_history.append(high_risk_prob)
                
                # Keep only recent history based on window size
                if enable_smoothing:
                    if len(st.session_state.prediction_history) > window_size:
                        st.session_state.prediction_history = st.session_state.prediction_history[-window_size:]
                        st.session_state.probability_history = st.session_state.probability_history[-window_size:]
                    
                    # Apply temporal smoothing
                    positive_count = sum(st.session_state.prediction_history)
                    smoothed_prediction = 1 if positive_count >= required_positive else 0
                    
                    # Use smoothed prediction
                    prediction = smoothed_prediction
                else:
                    # No smoothing - use threshold prediction directly
                    prediction = threshold_prediction
                
                # Step 5: Generate visualizations
                status_text.text("📊 Generating visualizations...")
                progress_bar.progress(90)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Clear progress
                import time
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # ==================== RESULTS SECTION ====================
                st.markdown("---")
                st.markdown("# 📊 ANALYSIS RESULTS")
                
                # ========== PREDICTION HISTORY DISPLAY ==========
                if len(st.session_state.prediction_history) > 1:
                    st.markdown("### 📜 Prediction History")
                    
                    # Create history display
                    history_display = []
                    for i, (pred, prob) in enumerate(zip(
                        st.session_state.prediction_history[-window_size if enable_smoothing else -5:],
                        st.session_state.probability_history[-window_size if enable_smoothing else -5:]
                    )):
                        if pred == 1:
                            history_display.append(f"🔴 HIGH ({prob*100:.1f}%)")
                        else:
                            history_display.append(f"🟢 LOW ({prob*100:.1f}%)")
                    
                    history_str = " → ".join(history_display)
                    
                    if enable_smoothing:
                        final_msg = f"**{positive_count}/{len(st.session_state.prediction_history)}** positive"
                        if prediction == 1:
                            alert_msg = "→ ⚠️ **ALERT TRIGGERED**"
                        else:
                            alert_msg = "→ ✅ **NO ALERT**"
                    else:
                        final_msg = ""
                        alert_msg = ""
                    
                    st.markdown(f"""
                    <div class='glow-card' style='background: rgba(255, 255, 255, 0.7);'>
                        <h4 style='color: #5B2E90; margin-top: 0;'>📊 Recent Predictions</h4>
                        <p style='color: #3D1B5C; font-size: 1.1rem; margin: 10px 0;'>{history_str}</p>
                        <p style='color: #5B2E90; font-size: 1.2rem; font-weight: 700; margin: 10px 0 0 0;'>{final_msg} {alert_msg}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                
                # ========== SEIZURE RISK ASSESSMENT ==========
                st.markdown("## 🎯 Seizure Risk Assessment")
                
                if prediction == 1:
                    # HIGH RISK
                    risk_prob = probability[1] * 100
                    
                    st.markdown(f"""
                    <div class='risk-high'>
                        <h1 style='color: white; margin: 0; font-size: 3rem;'>🚨 HIGH SEIZURE RISK DETECTED</h1>
                        <p style='color: white; font-size: 1.5rem; margin: 10px 0 0 0;'>Immediate precautions required</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk meter
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_risk = create_speedometer(risk_prob, "Seizure Risk Level (%)")
                        st.plotly_chart(fig_risk, use_container_width=True)
                    with col2:
                        st.markdown(f"""
                        <div class='glow-card'>
                            <h3 style='color: #ff4757; margin-top: 0;'>⚠️ Risk Probability</h3>
                            <p style='font-size: 3rem; font-weight: 800; color: #ff4757; margin: 10px 0;'>{risk_prob:.1f}%</p>
                            <p style='font-size: 1.2rem; color: #3D1B5C;'>Confidence Level</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Safety instructions
                    st.markdown("""
                    <div class='glow-card'>
                        <h2 style='color: #ff6b6b; margin-top: 0;'>🏥 IMMEDIATE SAFETY ACTIONS</h2>
                        <ul style='font-size: 1.1rem; line-height: 2; color: #3D1B5C;'>
                            <li><strong>🚫 Stop dangerous activities immediately</strong> - No driving, heights, water, machinery</li>
                            <li><strong>🛡️ Move to safe location</strong> - Sit or lie down on floor away from hazards</li>
                            <li><strong>📞 Alert someone immediately</strong> - Notify caregiver, family, or nearby person</li>
                            <li><strong>💊 Take rescue medication</strong> - If prescribed and available</li>
                            <li><strong>😌 Stay calm, breathe deeply</strong> - Stress can trigger seizures</li>
                            <li><strong>🏥 Prepare for emergency</strong> - Have emergency contacts ready</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    # LOW RISK
                    safety_score = probability[0] * 100
                    
                    st.markdown(f"""
                    <div class='risk-low'>
                        <h1 style='color: white; margin: 0; font-size: 3rem;'>🟢 LOW SEIZURE RISK</h1>
                        <p style='color: white; font-size: 1.5rem; margin: 10px 0 0 0;'>No immediate risk detected</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Safety meter
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_safety = create_speedometer(safety_score, "Safety Level (%)")
                        st.plotly_chart(fig_safety, use_container_width=True)
                    with col2:
                        st.markdown(f"""
                        <div class='glow-card'>
                            <h3 style='color: #28a745; margin-top: 0;'>✅ Safety Score</h3>
                            <p style='font-size: 3rem; font-weight: 800; color: #28a745; margin: 10px 0;'>{safety_score:.1f}%</p>
                            <p style='font-size: 1.2rem; color: #3D1B5C;'>Confidence Level</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Continue activities
                    st.markdown("""
                    <div class='glow-card'>
                        <h2 style='color: #51cf66; margin-top: 0;'>✅ SAFE TO CONTINUE ACTIVITIES</h2>
                        <ul style='font-size: 1.1rem; line-height: 2; color: #3D1B5C;'>
                            <li><strong>✅ Normal activities OK</strong> - Continue regular daily activities safely</li>
                            <li><strong>💊 Maintain medication schedule</strong> - Keep taking prescribed medications</li>
                            <li><strong>👁️ Stay vigilant</strong> - Continue monitoring if recommended by doctor</li>
                            <li><strong>🌟 Healthy habits</strong> - Good sleep, reduced stress, regular meals</li>
                            <li><strong>📊 Regular monitoring</strong> - Continue periodic ECG monitoring</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # ========== ECG SIGNAL VISUALIZATION ==========
                st.markdown("## 📈 ECG Signal Analysis")
                
                fig_ecg = create_ecg_plot(filtered_ecg, fs, peaks)
                st.plotly_chart(fig_ecg, use_container_width=True)
                
                # Signal info
                actual_duration = len(ecg_data) / fs
                window_end_min = window_start + (actual_duration / 60)
                duration_match = "✅" if abs(actual_duration - segment_duration) < 1 else "⚠️"
                st.info(f"📍 **Signal Info:** Window: {window_start:.1f}-{window_end_min:.1f} min | Duration: {actual_duration:.1f}s {duration_match} | Target: {segment_duration}s | Sampling: {fs:.0f} Hz | R-peaks: {len(peaks)}")
                
                st.markdown("---")
                
                # ========== COMPREHENSIVE HRV BIOMARKERS ==========
                st.markdown("## 🧬 Comprehensive HRV Biomarker Analysis")
                
                # Create tabs for different biomarker categories
                tab1, tab2, tab3, tab4 = st.tabs([
                    "⏱️ Time Domain",
                    "📊 Frequency Domain",
                    "🔄 Nonlinear Metrics",
                    "🔬 Signal Quality"
                ])
                
                with tab1:
                    st.markdown("### Time Domain HRV Metrics")
                    if 'Time Domain' in hrv_metrics and 'Error' not in hrv_metrics['Time Domain']:
                        td_metrics = hrv_metrics['Time Domain']
                        
                        # Display in grid
                        col1, col2, col3 = st.columns(3)
                        
                        metrics_list = list(td_metrics.items())
                        for idx, (key, value) in enumerate(metrics_list):
                            with [col1, col2, col3][idx % 3]:
                                # Determine color based on metric type
                                if 'HR' in key:
                                    color = "#28a745" if 60 <= value <= 100 else "#ff4757"
                                elif key == 'SDNN (ms)':
                                    color = "#28a745" if 20 <= value <= 100 else "#ff4757"
                                else:
                                    color = "#5B2E90"
                                
                                st.markdown(f"""
                                <div class='metric-card'>
                                    <h4 style='color: {color}; margin: 0; font-size: 0.9rem;'>{key}</h4>
                                    <p style='font-size: 2rem; font-weight: 800; margin: 10px 0; color: {color};'>{value:.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ Insufficient data for time domain analysis")
                
                with tab2:
                    st.markdown("### Frequency Domain HRV Metrics")
                    if 'Frequency Domain' in hrv_metrics and 'Error' not in hrv_metrics['Frequency Domain']:
                        fd_metrics = hrv_metrics['Frequency Domain']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        metrics_list = list(fd_metrics.items())
                        for idx, (key, value) in enumerate(metrics_list):
                            with [col1, col2, col3][idx % 3]:
                                st.markdown(f"""
                                <div class='metric-card'>
                                    <h4 style='color: #5B2E90; margin: 0; font-size: 0.9rem;'>{key}</h4>
                                    <p style='font-size: 2rem; font-weight: 800; margin: 10px 0; color: #5B2E90;'>{value:.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # LF/HF interpretation
                        lf_hf = fd_metrics.get('LF/HF Ratio', 0)
                        if lf_hf > 2:
                            balance = "High sympathetic activity (stress)"
                            color = "#ff4757"
                        elif lf_hf < 0.5:
                            balance = "High parasympathetic activity (relaxed)"
                            color = "#28a745"
                        else:
                            balance = "Balanced autonomic function"
                            color = "#28a745"
                        
                        st.markdown(f"""
                        <div class='glow-card' style='background: rgba(255, 255, 255, 0.7);'>
                            <h4 style='color: {color};'>💡 Autonomic Balance Interpretation</h4>
                            <p style='color: #3D1B5C; font-size: 1.1rem;'><strong>LF/HF Ratio: {lf_hf:.2f}</strong> → {balance}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ Insufficient data for frequency domain analysis")
                
                with tab3:
                    st.markdown("### Nonlinear HRV Metrics")
                    if 'Nonlinear' in hrv_metrics and 'Error' not in hrv_metrics['Nonlinear']:
                        nl_metrics = hrv_metrics['Nonlinear']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        metrics_list = list(nl_metrics.items())
                        for idx, (key, value) in enumerate(metrics_list):
                            with [col1, col2, col3][idx % 3]:
                                st.markdown(f"""
                                <div class='metric-card'>
                                    <h4 style='color: #5B2E90; margin: 0; font-size: 0.9rem;'>{key}</h4>
                                    <p style='font-size: 2rem; font-weight: 800; margin: 10px 0; color: #5B2E90;'>{value:.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class='glow-card' style='background: rgba(255, 255, 255, 0.7);'>
                            <h4 style='color: #5B2E90;'>💡 Poincaré Plot Explanation</h4>
                            <p style='color: #3D1B5C;'>
                            <strong>SD1:</strong> Short-term HRV variability (fast changes)<br>
                            <strong>SD2:</strong> Long-term HRV variability (slow changes)<br>
                            <strong>SD1/SD2:</strong> Balance between short and long-term variability
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ Insufficient data for nonlinear analysis")
                
                with tab4:
                    st.markdown("### Signal Quality Metrics")
                    if 'Signal Quality' in hrv_metrics:
                        sq_metrics = hrv_metrics['Signal Quality']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        metrics_list = list(sq_metrics.items())
                        for idx, (key, value) in enumerate(metrics_list):
                            with [col1, col2, col3, col4][idx % 4]:
                                # Special formatting for R-peaks
                                if 'R-peaks' in key:
                                    display_value = f"{int(value)}"
                                    # Color code based on quality
                                    if value >= 20:
                                        color = "#28a745"  # Green for excellent
                                    elif value >= 10:
                                        color = "#ffc107"  # Yellow for acceptable
                                    else:
                                        color = "#ff4757"  # Red for poor
                                else:
                                    display_value = f"{value:.3f}"
                                    color = "#5B2E90"  # Purple for other metrics
                                
                                st.markdown(f"""
                                <div class='metric-card'>
                                    <h4 style='color: {color}; margin: 0; font-size: 0.9rem;'>{key}</h4>
                                    <p style='font-size: 2rem; font-weight: 800; margin: 10px 0; color: {color};'>{display_value}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Signal quality assessment
                        r_peaks = sq_metrics.get('R-peaks Detected', 0)
                        if r_peaks >= 20:
                            quality_msg = "✅ Excellent signal quality - reliable analysis"
                            quality_color = "#28a745"
                        elif r_peaks >= 10:
                            quality_msg = "⚠️ Good signal quality - analysis acceptable"
                            quality_color = "#ffc107"
                        else:
                            quality_msg = "❌ Poor signal quality - results may be unreliable"
                            quality_color = "#ff4757"
                        
                        st.markdown(f"""
                        <div class='glow-card' style='background: rgba(255, 255, 255, 0.7);'>
                            <h4 style='color: {quality_color};'>💡 Signal Quality Assessment</h4>
                            <p style='color: #3D1B5C; font-size: 1.1rem;'>{quality_msg}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Model information
                st.markdown("""
                <div class='glow-card' style='text-align: center;'>
                    <h3 style='margin: 0 0 10px 0; color: #5B2E90;'>🤖 AI Model Information</h3>
                    <p style='color: #3D1B5C; margin: 5px 0;'><strong>Model Version:</strong> v12 Hybrid ECG</p>
                    <p style='color: #3D1B5C; margin: 5px 0;'><strong>Sensitivity:</strong> 40% | <strong>Specificity:</strong> 72%</p>
                    <p style='color: #3D1B5C; margin: 5px 0;'><strong>Features Analyzed:</strong> 15 biomarkers</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"⚠️ Analysis Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            progress_bar.empty()
            status_text.empty()

# Footer
st.markdown("---")
st.markdown("""
<div class='glow-card' style='text-align: center;'>
    <h3 style='margin: 0 0 15px 0; color: #5B2E90;'>🧠 NeuroAlert v12</h3>
    <p style='margin: 5px 0; color: #3D1B5C;'><strong>AI-Powered Seizure Risk Prediction System</strong></p>
    <p style='margin: 5px 0; font-size: 0.9rem; opacity: 0.8; color: #5B2E90;'>For research and educational purposes only • Not for clinical use</p>
    <p style='margin: 5px 0; font-size: 0.9rem; opacity: 0.8; color: #5B2E90;'>Comprehensive HRV biomarker analysis with AI prediction</p>
</div>
""", unsafe_allow_html=True)
