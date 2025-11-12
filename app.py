"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                               🧠 NEUROALERT - PRODUCTION v2.1                ║
║                      AI + Adaptive Heuristic Seizure Risk Prediction           ║
║                                                                              ║
║ HOW TO RUN:                                                                  ║
║   1. pip install -r requirements.txt                                         ║
║   2. streamlit run app.py                                                    ║
║                                                                              ║
║ REQUIREMENTS (see requirements.txt):                                         ║
║   - streamlit, numpy, pandas, scipy, plotly, mne, scikit-learn, joblib     ║
║   - Optional for PDF Export: reportlab                                       ║
║                                                                              ║
║ CONFIGURATION:                                                               ║
║   - This app is configured via the sidebar controls.                         ║
║   - Model files (neuroalert_hybrid_ecg_only.pkl, scaler.pkl) must be local.  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, sosfilt, find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import joblib  # Using joblib for robust model loading
import warnings
import tempfile
import os
import time
import re
import base64
from io import StringIO, BytesIO

warnings.filterwarnings('ignore')

# Try to import MNE for EDF reading
try:
    import mne
    EDF_SUPPORT = True
except ImportError:
    EDF_SUPPORT = False

# Try to import ReportLab for PDF export
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ============================================================================
# CONFIGURATION - Default values (now controlled by sidebar)
# ============================================================================
BASELINE_DURATION_MIN_DEFAULT = 5.0
ALPHA_AI_WEIGHT_DEFAULT = 0.6
ADAPTIVE_THRESHOLD_MULTIPLIER_DEFAULT = 2.0
MIN_BASELINE_PEAKS = 30
MIN_SEGMENT_PEAKS = 10
CONFIDENCE_THRESHOLD_DEFAULT = 0.65
DEMO_MODE_STEP_SIZE_MIN = 1.0
DEMO_MODE_DELAY_SEC = 3

# Color palette (Pastel Pink Theme)
PRIMARY_COLOR = '#db7093'    # Darker pink
SECONDARY_COLOR = '#ffd1dc'  # Pastel pink
BACKGROUND_COLOR = '#fffafc' # Pinkish white
ACCENT_COLOR = '#FF69B4'     # Hot pink
DARK_TEXT = '#2d1a23'        # Dark rose for text
LIGHT_BORDER = '#ffebf0'     # Lightest pink

# Risk status colors
COLOR_HIGH_RISK = '#ff6b6b'
COLOR_BORDERLINE = '#ffd166'
COLOR_LOW_RISK = '#98e37e'

# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="NeuroAlert - AI Seizure Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"  # Reverted to always expanded
)

# CRITICAL: Increase upload limit to 500MB
try:
    st.set_option('server.maxUploadSize', 500)
except Exception as e:
    # st.warning(f"Could not set maxUploadSize: {e}")
    pass

# ============================================================================
# EMBEDDED SEIZURE DATABASE (for ground truth validation)
# ============================================================================
SEIZURE_DATABASE = {
    'PN00-1.edf': [('19:58:36', '19:59:46', '19:39:33')],
    'PN00-2.edf': [('02:38:37', '02:39:31', '02:18:17')],
    'PN00-3.edf': [('18:28:29', '19:29:29', '18:15:44')],
    'PN00-4.edf': [('21:08:29', '21:09:43', '20:51:43')],
    'PN00-5.edf': [('22:37:08', '22:38:15', '22:22:04')],
    'PN01.edf': [('21:51:02', '21:51:56', '19:00:44'), ('07:53:17', '07:54:31', '19:00:44')],
    'PN03-1.edf': [('09:29:10', '09:31:01', '22:44:37')],
    'PN03-2.edf': [('07:13:05', '07:15:18', '21:31:04')],
    'PN05-2.edf': [('08:45:25', '08:46:00', '06:46:02')],
    'PN05-3.edf': [('07:55:19', '07:55:49', '06:01:23')],
    'PN05-4.edf': [('07:38:43', '07:39:22', '06:38:35')],
    'PN06-1.edf': [('05:54:25', '05:55:29', '04:21:22')],
    'PN06-2.edf': [('23:39:09', '23:40:18', '21:11:29')],
    'PN06-3.edf': [('08:10:26', '08:11:08', '06:25:51')],
    'PN06-4.edf': [('12:55:08', '12:56:11', '11:16:09')],
    'PN06-5.edf': [('14:44:24', '14:45:08', '13:24:41')],
    'PN07-1.edf': [('05:25:49', '05:26:51', '23:18:10')],
    'PN09-1.edf': [('11:26:42', '11:27:21', '10:29:48')],
    'PN09-2.edf': [('09:35:03', '09:36:25', '08:41:17')],
    'PN10-1.edf': [('12:08:33', '12:09:06', '11:30:54')],
    'PN10-2.3.edf': [('16:27:38', '16:28:37', '15:37:43'), ('18:45:31', '18:46:34', '15:37:43')],
    'PN10-4.5.edf': [('13:16:02', '13:17:07', '12:05:08'), ('16:01:57', '16:02:16', '12:05:08')],
    'PN10-6.edf': [('15:18:26', '15:19:23', '14:31:53')],
    'PN10-7.8.9.edf': [('17:35:13', '17:36:01', '16:49:25'), ('18:20:24', '18:20:42', '16:49:25'), ('20:24:48', '20:25:03', '16:49:25')],
    'PN10-10.edf': [('10:58:19', '10:58:33', '08:45:22')],
    'PN11.edf': [('13:37:19', '13:38:14', '11:31:25')],
    'PN12-1.2.edf': [('16:13:23', '16:14:26', '15:51:31'), ('18:31:01', '18:32:09', '15:51:31')],
    'PN12-3.edf': [('08:55:27', '08:57:03', '08:42:35')],
    'PN12-4.edf': [('18:42:51', '18:43:54', '15:59:19')],
    'PN13-1.edf': [('10:22:10', '10:22:58', '08:24:28')],
    'PN13-2.edf': [('08:55:51', '08:56:56', '06:55:02')],
    'PN13-3.edf': [('14:05:54', '14:08:25', '12:00:01')],
    'PN14-1.edf': [('13:46:00', '13:46:27', '11:44:58')],
    'PN14-2.edf': [('17:54:52', '17:55:04', '15:50:13')],
    'PN14-3.edf': [('21:10:05', '21:10:46', '16:17:45')],
    'PN14-4.edf': [('15:49:33', '15:50:56', '14:18:30')],
    'PN16-1.edf': [('22:45:05', '22:47:08', '20:45:21')],
    'PN16-2.edf': [('03:16:49', '03:18:36', '00:53:55')],
    'PN17-1.edf': [('22:34:48', '22:35:58', '20:14:28')],
    'PN17-2.edf': [('16:01:09', '16:02:32', '13:52:18')],
}

def parse_time_to_seconds(time_str):
    """Convert HH:MM:SS to seconds"""
    try:
        parts = time_str.strip().replace('.', ':').split(':')
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        return None
    except:
        return None

def get_seizure_intervals_for_file(filename):
    """Get seizure intervals in seconds from recording start"""
    # Normalize filename, e.g. "PN10-2.3.edf"
    simple_filename = os.path.basename(filename)
    
    if simple_filename not in SEIZURE_DATABASE:
        # Try matching pattern like "PN10-2.edf"
        match = re.match(r"(PN\d+-\d+)", simple_filename)
        if match:
            for key in SEIZURE_DATABASE.keys():
                if key.startswith(match.group(1)):
                    simple_filename = key
                    break
    
    if simple_filename not in SEIZURE_DATABASE:
        return []

    intervals = []
    for seizure_start, seizure_end, rec_start in SEIZURE_DATABASE[simple_filename]:
        start_sec = parse_time_to_seconds(seizure_start)
        end_sec = parse_time_to_seconds(seizure_end)
        rec_start_sec = parse_time_to_seconds(rec_start)
        if all(x is not None for x in [start_sec, end_sec, rec_start_sec]):
            # Handle day rollover
            if start_sec < rec_start_sec:
                start_sec += 24 * 3600
            if end_sec < start_sec:
                end_sec += 24 * 3600
                
            offset_start = start_sec - rec_start_sec
            offset_end = end_sec - rec_start_sec
            if offset_start >= 0:
                intervals.append((offset_start, offset_end))
    return intervals

# ============================================================================
# CSS STYLING (Pastel Pink Theme)
# ============================================================================
st.markdown(f"""
<style>
:root {{
    --primary-color: {PRIMARY_COLOR};
    --secondary-color: {SECONDARY_COLOR};
    --background-color: {BACKGROUND_COLOR};
    --accent-color: {ACCENT_COLOR};
    --dark-text: {DARK_TEXT};
    --light-border: {LIGHT_BORDER};
    --risk-high-color: {COLOR_HIGH_RISK};
    --risk-low-color: {COLOR_LOW_RISK};
    --risk-borderline-color: {COLOR_BORDERLINE};
}}

.stApp {{ background: linear-gradient(135deg, var(--background-color) 0%, var(--light-border) 100%); }}
#MainMenu, footer, header {{visibility: hidden;}}

.main-title {{
    text-align: center; padding: 2rem 0; margin-bottom: 2rem;
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%);
    border-radius: 20px; box-shadow: 0 10px 40px rgba(219, 112, 147, 0.3);
    animation: glow 2s ease-in-out infinite alternate;
}}
.main-title h1 {{ color: #FFFFFF !important; font-size: 4rem !important; font-weight: 900 !important; margin: 0 !important; text-shadow: 0 0 20px rgba(255, 255, 255, 0.5); letter-spacing: 2px; }}
.main-title p {{ color: var(--light-border) !important; font-size: 1.3rem !important; margin: 10px 0 0 0 !important; font-weight: 300; opacity: 0.95; }}

@keyframes glow {{
    from {{ box-shadow: 0 10px 40px rgba(219, 112, 147, 0.3), 0 0 20px rgba(255, 209, 220, 0.2); }}
    to {{ box-shadow: 0 15px 60px rgba(219, 112, 147, 0.5), 0 0 40px rgba(255, 209, 220, 0.4); }}
}}

@keyframes pulse-high-risk {{
    0%, 100% {{ box-shadow: 0 10px 40px rgba(255, 107, 107, 0.4); }}
    50% {{ box-shadow: 0 15px 60px rgba(255, 107, 107, 0.7); }}
}}

@keyframes pulse-low-risk {{
    0%, 100% {{ box-shadow: 0 10px 40px rgba(152, 227, 126, 0.4); }}
    50% {{ box-shadow: 0 15px 60px rgba(152, 227, 126, 0.6); }}
}}

[data-testid="stSidebar"] {{ background: linear-gradient(180deg, var(--primary-color) 0%, var(--accent-color) 100%); padding: 2rem 1rem; }}
[data-testid="stSidebar"] * {{ color: #FFFFFF !important; }}

[data-testid="stFileUploader"] {{ background: rgba(255, 255, 255, 0.15); border: 2px dashed var(--secondary-color); border-radius: 15px; padding: 2rem; }}
[data-testid="stFileUploader"] section {{ color: {DARK_TEXT} !important; font-weight: 800 !important; background: rgba(255, 255, 255, 0.95) !important; padding: 10px 15px !important; border-radius: 10px !important; }}
[data-testid="stFileUploader"] label, [data-testid="stFileUploader"] p, [data-testid="stFileUploader"] span {{ color: var(--dark-text) !important; font-weight: 600 !important; }}
[data-testid="stFileUploader"] button {{
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%) !important;
    color: #FFFFFF !important; border: 2px solid var(--accent-color) !important; font-weight: 700 !important;
}}

.stButton > button {{ background: linear-gradient(135deg, var(--secondary-color) 0%, #ffe0e9 100%); color: var(--dark-text); border: none; padding: 0.8rem 2rem; font-size: 1.1rem; font-weight: 700; border-radius: 12px; }}
.stButton > button:hover {{ background: linear-gradient(135deg, #ffe0e9 0%, var(--secondary-color) 100%); transform: translateY(-2px); box-shadow: 0 6px 25px rgba(255, 209, 220, 0.5); }}

.stSlider > div > div > div {{ background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important; }}
.stSlider > div > div > div > div {{ background: var(--primary-color) !important; border: 3px solid #FFFFFF !important; }}

.glow-card {{ background: rgba(255, 255, 255, 0.95); border: 2px solid var(--secondary-color); border-radius: 20px; padding: 2rem; margin: 1rem 0; box-shadow: 0 8px 32px rgba(219, 112, 147, 0.15); }}
.glow-card, .glow-card p, .glow-card li {{ color: {DARK_TEXT} !important; }}

.metric-card {{ background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, var(--background-color) 100%); border: 2px solid var(--secondary-color); border-radius: 15px; padding: 1rem; text-align: center; box-shadow: 0 4px 20px rgba(219, 112, 147, 0.1); height: 100%; }}
.metric-card p, .metric-card h4 {{ color: {DARK_TEXT} !important; }}

.stMetric {{ background: rgba(255, 250, 252, 0.5); padding: 10px; border-radius: 8px; }}
.stMetric label, .stMetric [data-testid="stMetricLabel"] {{ color: {DARK_TEXT} !important; font-weight: 700 !important; opacity: 0.8; }}
.stMetric [data-testid="stMetricValue"] {{ color: {DARK_TEXT} !important; font-weight: 800 !important; }}

/* Color fix for expander text */
.stExpander {{ border-color: var(--secondary-color) !important; }}
.stExpander p, .stExpander li, .stExpander .stMarkdown p, .stExpander .stMetric label, .stExpander [data-testid="stMetricValue"], .stExpander .stInfo p {{
    color: var(--dark-text) !important;
}}
.stExpander [data-testid="stMetricLabel"] {{ color: var(--dark-text) !important; opacity: 0.7; }}


.risk-high {{ background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%); border: 3px solid var(--secondary-color); border-radius: 20px; padding: 3rem 2rem; text-align: center; animation: pulse-high-risk 2s ease-in-out infinite; }}
.risk-high h1, .risk-high p {{ color: #FFFFFF !important; text-shadow: 0 0 20px rgba(255, 71, 87, 0.6); }}

.risk-low {{ background: linear-gradient(135deg, #A0E7A0 0%, #50C878 100%); border: 3px solid #C8E6C8; border-radius: 20px; padding: 3rem 2rem; text-align: center; animation: pulse-low-risk 2s ease-in-out infinite; }}
.risk-low h1, .risk-low p {{ color: #FFFFFF !important; text-shadow: 0 0 20px rgba(40, 167, 69, 0.6); }}

.baseline-info {{ background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%); border: 2px solid #81C784; border-radius: 12px; padding: 1rem; margin: 1rem 0; }}
.baseline-info h4 {{ color: #2E7D32 !important; margin: 0 0 8px 0 !important; }}
.baseline-info p {{ color: #1B5E20 !important; margin: 3px 0 !important; font-size: 0.95rem; }}

h1, h2, h3 {{ color: var(--primary-color) !important; font-weight: 700; }}
p, li, span, div {{ color: {DARK_TEXT}; }}

/* Fix for tab labels */
[data-testid="stTabs"] button {{ color: var(--dark-text); }}
[data-testid="stTabs"] button[aria-selected="true"] {{ color: var(--primary-color); }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TITLE
# ============================================================================
# Reverted to original title block
st.markdown("""
<div class='main-title'>
    <h1>🧠 NeuroAlert</h1>
    <p>AI + Adaptive Heuristic Seizure Risk Prediction</p>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SIGNAL PROCESSING FUNCTIONS
# ============================================================================

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def resample_signal(signal_data, original_fs, target_fs=256):
    if original_fs == target_fs:
        return signal_data, original_fs
    num_samples = int(len(signal_data) * target_fs / original_fs)
    from scipy import signal as sp_signal
    resampled = sp_signal.resample(signal_data, num_samples)
    return resampled, target_fs

# ============================================================================
# HRV FEATURE EXTRACTION (with LF/HF bug fix)
# ============================================================================

def extract_comprehensive_hrv_features(ecg_signal, fs=256):
    """
    Extract comprehensive HRV biomarkers from ECG segment.
    CRITICAL: Handles LF/HF=0 bug with epsilon and NaN marking.
    """
    ecg_signal = np.array(ecg_signal).flatten()
    filtered_ecg = bandpass_filter(ecg_signal, 0.5, 40, fs, order=5)
    peaks, _ = find_peaks(filtered_ecg, height=np.mean(filtered_ecg) + 0.5 * np.std(filtered_ecg), distance=int(0.6 * fs))
    
    features = []
    metrics = {}
    
    if len(peaks) > 2:
        rr_intervals = np.diff(peaks) / fs * 1000  # ms
        mean_rr = np.mean(rr_intervals)
        sdnn = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))
        sdsd = np.std(np.diff(rr_intervals))
        nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
        pnn50 = (nn50 / len(rr_intervals)) * 100 if len(rr_intervals) > 0 else 0
        rr_range = np.max(rr_intervals) - np.min(rr_intervals)
        cv_rr = (sdnn / mean_rr) * 100 if mean_rr > 0 else 0
        median_rr = np.median(rr_intervals)
        mad_rr = np.median(np.abs(rr_intervals - median_rr))
        mean_hr = 60000 / mean_rr if mean_rr > 0 else 0
        min_hr = 60000 / np.max(rr_intervals) if np.max(rr_intervals) > 0 else 0
        max_hr = 60000 / np.min(rr_intervals) if np.min(rr_intervals) > 0 else 0
        
        metrics['Time Domain'] = {
            'Mean RR (ms)': mean_rr, 'SDNN (ms)': sdnn, 'RMSSD (ms)': rmssd,
            'SDSD (ms)': sdsd, 'NN50 (count)': nn50, 'pNN50 (%)': pnn50,
            'RR Range (ms)': rr_range, 'CV (%)': cv_rr, 'Median RR (ms)': median_rr,
            'MAD (ms)': mad_rr, 'Mean HR (bpm)': mean_hr, 'Min HR (bpm)': min_hr,
            'Max HR (bpm)': max_hr
        }
        features.extend([mean_rr, sdnn, rmssd, sdsd, pnn50, mean_hr])
        
        # FREQUENCY DOMAIN with LF/HF bug fix
        if len(rr_intervals) > 10:
            time_rr = np.cumsum(rr_intervals) / 1000
            if time_rr[-1] >= 30:  # Need at least 30 seconds
                time_uniform = np.arange(0, time_rr[-1], 0.25)
                if len(time_uniform) >= 8:
                    rr_uniform = np.interp(time_uniform, time_rr, rr_intervals)
                    nperseg_val = min(256, len(rr_uniform) // 2)
                    if nperseg_val >= 4:
                        freqs, psd = signal.welch(rr_uniform, fs=4, nperseg=nperseg_val)
                        vlf_band = (freqs >= 0.003) & (freqs < 0.04)
                        lf_band = (freqs >= 0.04) & (freqs < 0.15)
                        hf_band = (freqs >= 0.15) & (freqs < 0.4)
                        vlf_power = np.trapz(psd[vlf_band], freqs[vlf_band]) if np.any(vlf_band) else 0
                        lf_power = np.trapz(psd[lf_band], freqs[lf_band]) if np.any(lf_band) else 0
                        hf_power = np.trapz(psd[hf_band], freqs[hf_band]) if np.any(hf_band) else 0
                        total_power = vlf_power + lf_power + hf_power
                        
                        # CRITICAL FIX: Safe LF/HF computation with epsilon
                        hf_eps = 1e-9
                        if (lf_power + hf_power) < 1e-6:
                            lf_hf_ratio = np.nan  # Mark as invalid if both very small
                        else:
                            lf_hf_ratio = lf_power / (hf_power + hf_eps)
                        
                        lf_norm = (lf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0
                        hf_norm = (hf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0
                        
                        metrics['Frequency Domain'] = {
                            'VLF Power (ms²)': vlf_power, 'LF Power (ms²)': lf_power,
                            'HF Power (ms²)': hf_power, 'Total Power (ms²)': total_power,
                            'LF norm (%)': lf_norm, 'HF norm (%)': hf_norm,
                            'LF/HF Ratio': lf_hf_ratio
                        }
                        features.extend([lf_power, hf_power, lf_hf_ratio if not np.isnan(lf_hf_ratio) else 0])
                    else:
                        metrics['Frequency Domain'] = {'Error': 'Insufficient samples'}
                        features.extend([0, 0, 0])
                else:
                    metrics['Frequency Domain'] = {'Error': 'Insufficient samples'}
                    features.extend([0, 0, 0])
            else:
                metrics['Frequency Domain'] = {'Error': 'Insufficient duration'}
                features.extend([0, 0, 0])
        else:
            metrics['Frequency Domain'] = {'Error': 'Insufficient RR intervals'}
            features.extend([0, 0, 0])
        
        # NONLINEAR
        sd1 = np.sqrt(0.5 * rmssd**2)
        sd2 = np.sqrt(2 * sdnn**2 - 0.5 * rmssd**2)
        sd_ratio = sd1 / sd2 if sd2 > 0 else 0
        metrics['Nonlinear'] = {'SD1 (ms)': sd1, 'SD2 (ms)': sd2, 'SD1/SD2 Ratio': sd_ratio}
        
        # SIGNAL QUALITY
        signal_std = np.std(filtered_ecg)
        signal_max = np.max(np.abs(filtered_ecg))
        signal_mad = np.mean(np.abs(np.diff(filtered_ecg)))
        metrics['Signal Quality'] = {
            'Signal Std (mV)': signal_std, 'Peak Amplitude (mV)': signal_max,
            'Mean Derivative': signal_mad, 'R-peaks Detected': len(peaks)
        }
        features.extend([signal_std, signal_max, signal_mad, len(peaks)])
        features.extend([rr_range, median_rr])
    else:
        features = [0] * 15
        metrics = {
            'Time Domain': {'Error': 'Insufficient R-peaks detected'},
            'Frequency Domain': {'Error': 'Insufficient data'},
            'Nonlinear': {'Error': 'Insufficient data'},
            'Signal Quality': {'R-peaks Detected': len(peaks)}
        }
    
    return np.array(features), metrics, filtered_ecg, peaks

# ============================================================================
# PATIENT BASELINE COMPUTATION (with LF/HF fix)
# ============================================================================

@st.cache_data
def compute_patient_baseline(_ecg_data, fs, baseline_minutes):
    """
    Compute patient-specific baseline from start of recording.
    HANDLES: LF/HF zero bug by using np.nanmean/nanstd and filtering invalid values.
    """
    baseline_seconds = baseline_minutes * 60
    baseline_samples = min(int(baseline_seconds * fs), len(_ecg_data))
    baseline_duration_actual = baseline_samples / fs / 60  # Actual duration in minutes
    baseline_ecg = _ecg_data[:baseline_samples]
    
    # Divide into 30-second windows for robust stats
    window_size = int(30 * fs)
    num_windows = max(1, baseline_samples // window_size)
    
    sdnn_vals, rmssd_vals, mean_rr_vals, mean_hr_vals, lfhf_vals = [], [], [], [], []
    total_peaks = 0
    
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = min(start_idx + window_size, baseline_samples)
        window_ecg_segment = baseline_ecg[start_idx:end_idx]
        
        if len(window_ecg_segment) < fs * 5: # Need at least 5 sec
            continue
            
        _, metrics, _, peaks = extract_comprehensive_hrv_features(window_ecg_segment, fs)
        
        # Only use segments with a valid number of peaks
        if 'Error' not in metrics.get('Time Domain', {}) and metrics['Signal Quality'].get('R-peaks Detected', 0) > MIN_SEGMENT_PEAKS / 4: # scale min peaks for 30s
            td = metrics['Time Domain']
            sdnn_vals.append(td.get('SDNN (ms)', np.nan))
            rmssd_vals.append(td.get('RMSSD (ms)', np.nan))
            mean_rr_vals.append(td.get('Mean RR (ms)', np.nan))
            mean_hr_vals.append(td.get('Mean HR (bpm)', np.nan))
            total_peaks += metrics['Signal Quality'].get('R-peaks Detected', 0)
            
            # CRITICAL: Only include valid LF/HF values (not NaN)
            if 'Error' not in metrics.get('Frequency Domain', {}):
                fd = metrics['Frequency Domain']
                lfhf = fd.get('LF/HF Ratio', np.nan)
                if not np.isnan(lfhf):
                    lfhf_vals.append(lfhf)
    
    baseline_ok = (total_peaks >= MIN_BASELINE_PEAKS and len(mean_hr_vals) > 0)
    
    if not baseline_ok:
        return {}, False, baseline_duration_actual, total_peaks
    
    # Use nanmean/nanstd to ignore NaN values
    baseline_metrics = {
        'SDNN': {'mean': np.nanmean(sdnn_vals), 'std': np.nanstd(sdnn_vals), 'median': np.nanmedian(sdnn_vals)},
        'RMSSD': {'mean': np.nanmean(rmssd_vals), 'std': np.nanstd(rmssd_vals), 'median': np.nanmedian(rmssd_vals)},
        'Mean_RR': {'mean': np.nanmean(mean_rr_vals), 'std': np.nanstd(mean_rr_vals), 'median': np.nanmedian(mean_rr_vals)},
        'Mean_HR': {'mean': np.nanmean(mean_hr_vals), 'std': np.nanstd(mean_hr_vals), 'median': np.nanmedian(mean_hr_vals)},
        'LFHF': {
            'mean': np.nanmean(lfhf_vals) if len(lfhf_vals) > 0 else np.nan,
            'std': np.nanstd(lfhf_vals) if len(lfhf_vals) > 0 else np.nan,
            'median': np.nanmedian(lfhf_vals) if len(lfhf_vals) > 0 else np.nan,
            'n_valid': len(lfhf_vals)
        }
    }
    
    return baseline_metrics, baseline_ok, baseline_duration_actual, total_peaks

# ============================================================================
# ADAPTIVE THRESHOLD DERIVATION
# ============================================================================

def derive_adaptive_thresholds(baseline_metrics, multiplier):
    """Derive patient-specific adaptive thresholds from baseline."""
    adaptive_thresholds = {
        'SDNN': {
            'threshold': max(40.0, baseline_metrics['SDNN']['mean'] + multiplier * baseline_metrics['SDNN']['std']),
            'baseline_mean': baseline_metrics['SDNN']['mean'],
            'baseline_std': baseline_metrics['SDNN']['std']
        },
        'RMSSD': {
            'threshold': baseline_metrics['RMSSD']['mean'] + multiplier * baseline_metrics['RMSSD']['std'],
            'baseline_mean': baseline_metrics['RMSSD']['mean'],
            'baseline_std': baseline_metrics['RMSSD']['std']
        },
        'Mean_HR': {
            'baseline_mean': baseline_metrics['Mean_HR']['mean'],
            'baseline_std': baseline_metrics['Mean_HR']['std'],
            'jump_threshold': max(20.0, multiplier * baseline_metrics['Mean_HR']['std']) # 20bpm or k*std
        }
    }
    
    # Handle LFHF carefully - may be NaN
    lfhf_mean = baseline_metrics['LFHF']['mean']
    lfhf_std = baseline_metrics['LFHF']['std']
    if not np.isnan(lfhf_mean) and not np.isnan(lfhf_std):
        if lfhf_std > 0:
            adaptive_thresholds['LFHF'] = {
                'threshold': lfhf_mean + multiplier * lfhf_std,
                'baseline_mean': lfhf_mean,
                'baseline_std': lfhf_std,
                'valid': True
            }
        else:
            adaptive_thresholds['LFHF'] = {
                'threshold': lfhf_mean + 1.0,  # Fallback if std=0
                'baseline_mean': lfhf_mean,
                'baseline_std': 0,
                'valid': True
            }
    else:
        adaptive_thresholds['LFHF'] = {'valid': False, 'threshold': 3.5} # Fallback to population
    
    return adaptive_thresholds

# ============================================================================
# ADAPTIVE HEURISTIC EVALUATION
# ============================================================================

def evaluate_with_adaptive_heuristics(hrv_metrics, adaptive_thresholds, prev_hrv_metrics):
    """
    Evaluate segment using adaptive patient-specific thresholds.
    Returns a heuristic score [0.0 - 1.0]
    """
    if not hrv_metrics or 'Time Domain' not in hrv_metrics or 'Error' in hrv_metrics['Time Domain']:
        return 0.0, "No data"
    
    td = hrv_metrics['Time Domain']
    fd = hrv_metrics.get('Frequency Domain', {})
    
    rules_triggered = []
    score = 0.0
    
    # Rule 1: SDNN elevation (weight: 0.35)
    curr_sdnn = td.get('SDNN (ms)', 0)
    thresh_sdnn = adaptive_thresholds['SDNN']['threshold']
    if curr_sdnn > thresh_sdnn:
        rules_triggered.append(f"SDNN spike ({curr_sdnn:.1f} > {thresh_sdnn:.1f})")
        score += 0.35
    
    # Rule 2: RMSSD elevation (weight: 0.15)
    curr_rmssd = td.get('RMSSD (ms)', 0)
    thresh_rmssd = adaptive_thresholds['RMSSD']['threshold']
    if curr_rmssd > thresh_rmssd:
        rules_triggered.append(f"RMSSD spike ({curr_rmssd:.1f} > {thresh_rmssd:.1f})")
        score += 0.15
    
    # Rule 3: LF/HF elevation (weight: 0.20)
    if 'Error' not in fd:
        curr_lfhf = fd.get('LF/HF Ratio', np.nan)
        thresh_lfhf = adaptive_thresholds['LFHF']['threshold']
        if not np.isnan(curr_lfhf) and curr_lfhf > thresh_lfhf:
            rules_triggered.append(f"LF/HF spike ({curr_lfhf:.2f} > {thresh_lfhf:.2f})")
            score += 0.20
    
    # Rule 4: HR jump (weight: 0.30)
    curr_hr = td.get('Mean HR (bpm)', 0)
    thresh_jump = adaptive_thresholds['Mean_HR']['jump_threshold']
    
    if prev_hrv_metrics and 'Time Domain' in prev_hrv_metrics and 'Error' not in prev_hrv_metrics['Time Domain']:
        prev_hr = prev_hrv_metrics['Time Domain'].get('Mean HR (bpm)', 0)
        hr_jump = curr_hr - prev_hr
        if hr_jump > max(20.0, thresh_jump): # At least 20bpm jump
            rules_triggered.append(f"HR jump (+{hr_jump:.1f} BPM)")
            score += 0.30
            
    # Rule 5: Bradycardia
    if curr_hr > 0 and (curr_hr < 45): # Hard floor
        rules_triggered.append(f"Bradycardia ({curr_hr:.1f} BPM)")
        score += 0.15
    
    heuristic_score = min(1.0, score)
    rules_string = ", ".join(rules_triggered) if rules_triggered else "Nominal (adaptive)"
    
    return heuristic_score, rules_string

# ============================================================================
# FALLBACK: Population-based heuristics
# ============================================================================

def check_biomarker_heuristics_population(hrv_metrics, prev_hrv_metrics):
    """FALLBACK: Original population-based heuristics when baseline unavailable."""
    if not hrv_metrics or 'Time Domain' not in hrv_metrics or 'Error' in hrv_metrics['Time Domain']:
        return 0.0, "No data"
    td = hrv_metrics['Time Domain']
    fd = hrv_metrics.get('Frequency Domain', {})
    rules_triggered = []
    risk_score = 0
    
    if td.get('SDNN (ms)', 0) > 100 or td.get('RMSSD (ms)', 0) > 80:
        risk_score += 1
        rules_triggered.append(f"High Volatility [pop] (SDNN: {td.get('SDNN (ms)', 0):.0f})")
    if 'Error' not in fd and fd.get('LF/HF Ratio', 0) > 3.5:
        risk_score += 1
        rules_triggered.append(f"High Stress [pop] (LF/HF: {fd.get('LF/HF Ratio', 0):.1f})")
    if td.get('Mean HR (bpm)', 0) > 0 and td.get('Mean HR (bpm)', 100) < 45:
        risk_score += 1
        rules_triggered.append(f"Bradycardia [pop] ({td.get('Mean HR (bpm)', 0):.0f} BPM)")
    
    if prev_hrv_metrics and 'Time Domain' in prev_hrv_metrics and 'Error' not in prev_hrv_metrics['Time Domain']:
        prev_hr = prev_hrv_metrics['Time Domain'].get('Mean HR (bpm)', 0)
        curr_hr = td.get('Mean HR (bpm)', 0)
        if curr_hr > 0 and prev_hr > 0 and (curr_hr - prev_hr) > 25:
             risk_score += 2
             rules_triggered.append(f"HR Jump [pop] (+{int(curr_hr - prev_hr)} BPM)")

    if risk_score >= 3:
        return 1.0, ", ".join(rules_triggered)
    elif risk_score >= 1:
        return 0.75, ", ".join(rules_triggered)
    else:
        return 0.0, "Nominal (population)"

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_mini_segment_plot(segment_ecg, fs):
    """Create small ECG plot for segment card (120px height)."""
    t = np.arange(len(segment_ecg)) / fs
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=t, y=segment_ecg,
        mode='lines',
        line=dict(width=1.5, color=PRIMARY_COLOR),
        hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.3f} mV<extra></extra>'
    ))
    fig.update_layout(
        height=120,
        margin=dict(t=5, b=5, l=5, r=5),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='#fff4f7',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

def create_dual_panel_ecg(full_ecg, fs, window_start_min, window_end_min, total_duration_min, seizure_intervals=None):
    """
    Create dual-panel ECG visualization:
    - Top: Full recording overview with window highlight
    - Bottom: Zoomed 10-minute window detail
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.3, 0.7],
        vertical_spacing=0.05,
        subplot_titles=("Full Recording Overview", "10-Minute Analysis Window (Zoomed)")
    )
    
    # TOP PANEL: Overview (downsampled for performance)
    display_fs = 10  # 10 Hz display
    step = max(1, int(fs / display_fs))
    display_signal = full_ecg[::step]
    time_display = np.arange(len(display_signal)) * step / fs / 60
    
    fig.add_trace(go.Scattergl(
        x=time_display, y=display_signal,
        mode='lines',
        name='Full ECG',
        line=dict(color='rgba(224, 109, 145, 0.4)', width=1.0),
        hovertemplate='Time: %{x:.2f} min<br>Amplitude: %{y:.3f} mV<extra></extra>'
    ), row=1, col=1)
    
    # Highlight current window
    fig.add_vrect(
        x0=window_start_min, x1=window_end_min,
        fillcolor=SECONDARY_COLOR, opacity=0.3,
        layer="below", line_width=2, line_color=PRIMARY_COLOR,
        row=1, col=1
    )
    
    # Add seizure overlays if present
    if seizure_intervals:
        for i, (s, e) in enumerate(seizure_intervals):
            s_min, e_min = s / 60, e / 60
            if s_min < total_duration_min:
                # BUG FIX: Removed 'hovertemplate' from add_vrect
                fig.add_vrect(x0=s_min, x1=e_min, fillcolor="rgba(255, 0, 0, 0.3)", layer="above", 
                            line_width=2, line_color="red", row=1, col=1,
                            name=f"Seizure {i+1}")
    
    # BOTTOM PANEL: Zoomed window
    window_start_sample = int(window_start_min * 60 * fs)
    window_end_sample = int(window_end_min * 60 * fs)
    zoom_step = max(1, int(fs / 50))  # 50 Hz display
    ecg_window = full_ecg[window_start_sample:window_end_sample:zoom_step]
    time_window = np.arange(len(ecg_window)) * zoom_step / fs / 60 + window_start_min
    
    fig.add_trace(go.Scattergl(
        x=time_window, y=ecg_window,
        mode='lines',
        name='ECG Waveform (Zoomed)',
        line=dict(color=PRIMARY_COLOR, width=1.5),
        hovertemplate='Time: %{x:.2f} min<br>Amplitude: %{y:.3f} mV<extra></extra>'
    ), row=2, col=1)
    
    # Add seizure overlays on zoom
    if seizure_intervals:
        for i, (s, e) in enumerate(seizure_intervals):
            s_min, e_min = s / 60, e / 60
            if s_min < window_end_min and e_min > window_start_min:
                # BUG FIX: Removed 'hovertemplate' from add_vrect
                fig.add_vrect(x0=max(s_min, window_start_min), x1=min(e_min, window_end_min), 
                            fillcolor="rgba(255, 0, 0, 0.25)", layer="above", line_width=2, line_color="red", row=2, col=1,
                            name=f"Seizure {i+1}")
    
    # Styling
    axis_font = dict(color=DARK_TEXT, family='Arial', size=11)
    
    fig.update_xaxes(
        title_text="Time (minutes)", gridcolor='rgba(45, 26, 35, 0.15)', showgrid=True,
        title_font=axis_font, tickfont=axis_font,
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="Time (minutes)", gridcolor='rgba(45, 26, 35, 0.2)', showgrid=True, dtick=1,
        title_font=axis_font, tickfont=axis_font,
        row=2, col=1
    )
    
    # BUG FIX: Corrected y-axis grid styling
    fig.update_yaxes(
        title_text="Amplitude (mV)", gridcolor='rgba(45, 26, 35, 0.15)', showgrid=True,
        title_font=axis_font, tickfont=axis_font,
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Amplitude (mV)", gridcolor='rgba(45, 26, 35, 0.2)', showgrid=True, 
        zerolinecolor='rgba(45, 26, 35, 0.4)', zerolinewidth=2,
        dtick=0.5,
        minor=dict(
            dtick=0.1,
            gridcolor='rgba(45, 26, 35, 0.1)',
            showgrid=True
        ),
        title_font=axis_font, tickfont=axis_font,
        row=2, col=1
    )
    
    fig.update_layout(
        height=700,
        hovermode='x unified',
        plot_bgcolor='rgba(255, 255, 255, 0.9)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=DARK_TEXT, family='Arial', size=11),
        legend=dict(
            orientation='h', y=1.02, x=1, xanchor='right', 
            bgcolor='rgba(255, 255, 255, 0.95)', 
            bordercolor=PRIMARY_COLOR, borderwidth=1,
            font=dict(color=DARK_TEXT)
        ),
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=14, color=DARK_TEXT, family='Arial Black')
    
    return fig

def create_mini_navigator(full_ecg, fs, window_start_min, window_end_min, total_duration_min):
    """Create mini signal navigator strip (80px height)."""
    display_fs = 5
    step = max(1, int(fs / display_fs))
    display_signal = full_ecg[::step]
    time_display = np.arange(len(display_signal)) * step / fs / 60
    
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=time_display, y=display_signal,
        mode='lines',
        line=dict(color='rgba(224, 109, 145, 0.5)', width=0.5),
        hoverinfo='skip'
    ))
    
    fig.add_vrect(x0=window_start_min, x1=window_end_min, fillcolor=PRIMARY_COLOR, opacity=0.4, line_width=1, line_color=PRIMARY_COLOR)
    
    fig.update_layout(
        height=80,
        margin=dict(t=5, b=25, l=40, r=40),
        xaxis=dict(
            title="Minutes", 
            tickfont=dict(size=9, color=DARK_TEXT), 
            title_font=dict(size=10, color=DARK_TEXT)
        ),
        yaxis=dict(visible=False),
        plot_bgcolor='rgba(255, 250, 252, 0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

def create_segment_status_timeline(segment_results):
    """Create color-coded segment status bar."""
    fig = go.Figure()
    
    for seg in segment_results:
        if seg is None:
            continue
        seg_idx = seg['segment_idx']
        final_risk = seg['final_risk']
        
        if final_risk == 1:
            color = COLOR_HIGH_RISK
            status = "High Risk"
        elif seg['final_score'] > 0.5:
            color = COLOR_BORDERLINE
            status = "Borderline"
        else:
            color = COLOR_LOW_RISK
            status = "Low Risk"
        
        hover_text = (
            f"<b>Segment {seg_idx}</b> ({seg['start_min']:.1f}-{seg['end_min']:.1f} min)<br>"
            f"<b>Status: {status}</b><br>"
            f"Final Score: {seg['final_score']*100:.1f}%<br>"
            f"AI Prob: {seg['ai_prob']*100:.1f}%<br>"
            f"Heuristic Score: {seg['heuristic_score']*100:.1f}%<br>"
            f"Rules: {seg['rules_triggered']}"
        )
        
        fig.add_trace(go.Bar(
            y=[1], x=[1], # Each segment has a width of 1
            orientation='h',
            marker=dict(color=color),
            hovertemplate=hover_text + '<extra></extra>',
            name=f"Seg {seg_idx}"
        ))
    
    fig.update_layout(
        height=80,
        margin=dict(t=20, b=40, l=20, r=20),
        xaxis=dict(visible=False, range=[0, 5]),
        yaxis=dict(visible=False),
        plot_bgcolor='rgba(255, 255, 255, 0.9)',
        paper_bgcolor='rgba(0,0,0,0)',
        barmode='stack',
        showlegend=False
    )
    return fig

def create_biomarker_trends(segment_results):
    """Create line chart showing biomarker trends across segments."""
    valid_segments = [s for s in segment_results if s is not None]
    if not valid_segments:
        return None
    
    seg_indices = [s['segment_idx'] for s in valid_segments]
    sdnn_vals = [s['hrv_metrics']['Time Domain'].get('SDNN (ms)', 0) for s in valid_segments]
    hr_vals = [s['hrv_metrics']['Time Domain'].get('Mean HR (bpm)', 0) for s in valid_segments]
    lfhf_vals = []
    for s in valid_segments:
        fd = s['hrv_metrics'].get('Frequency Domain', {})
        if 'Error' not in fd:
            lfhf = fd.get('LF/HF Ratio', np.nan)
            lfhf_vals.append(lfhf if not np.isnan(lfhf) else 0)
        else:
            lfhf_vals.append(0)
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                        subplot_titles=("SDNN (ms)", "Mean HR (bpm)", "LF/HF Ratio"))
    
    fig.add_trace(go.Scatter(x=seg_indices, y=sdnn_vals, mode='lines+markers', name='SDNN', line=dict(color=PRIMARY_COLOR, width=2), marker=dict(size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=seg_indices, y=hr_vals, mode='lines+markers', name='Mean HR', line=dict(color=ACCENT_COLOR, width=2), marker=dict(size=8)), row=2, col=1)
    fig.add_trace(go.Scatter(x=seg_indices, y=lfhf_vals, mode='lines+markers', name='LF/HF', line=dict(color='rgba(219, 112, 147, 0.6)', width=2), marker=dict(size=8)), row=3, col=1)
    
    axis_font = dict(color=DARK_TEXT)
    fig.update_xaxes(title_text="Segment", row=3, col=1, dtick=1, title_font=axis_font, tickfont=axis_font)
    fig.update_yaxes(title_font=axis_font, tickfont=axis_font)
    
    fig.update_layout(height=500, showlegend=False, margin=dict(t=60, b=40, l=60, r=40), 
                      plot_bgcolor='rgba(255, 255, 255, 0.9)', paper_bgcolor='rgba(0,0,0,0)',
                      font_color=DARK_TEXT)
    
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=14, color=DARK_TEXT, family='Arial Black')

    return fig

# ============================================================================
# PDF EXPORT FUNCTION
# ============================================================================
def generate_pdf_report(segment_results, baseline_metrics, window_start, window_end, filename, baseline_duration):
    """Generates a PDF report in memory using ReportLab."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    story.append(Paragraph("🧠 NeuroAlert Analysis Report", styles['h1']))
    story.append(Spacer(1, 12))
    
    # Summary
    story.append(Paragraph(f"<b>File:</b> {filename}", styles['Normal']))
    story.append(Paragraph(f"<b>Analyzed Window:</b> {window_start:.1f} - {window_end:.1f} minutes", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Baseline
    story.append(Paragraph("Patient Baseline", styles['h2']))
    if baseline_metrics:
        bm = baseline_metrics
        lfhf_mean = bm['LFHF']['mean']
        lfhf_std = bm['LFHF']['std']
        lfhf_valid = bm['LFHF']['n_valid']
        lfhf_str = f"{lfhf_mean:.2f} ± {lfhf_std:.2f} (n={lfhf_valid})" if not np.isnan(lfhf_mean) else "N/A"
        
        baseline_data = [
            ["Baseline Duration", f"{baseline_duration:.1f} minutes"],
            ["Mean SDNN", f"{bm['SDNN']['mean']:.1f} ± {bm['SDNN']['std']:.1f} ms"],
            ["Mean HR", f"{bm['Mean_HR']['mean']:.1f} ± {bm['Mean_HR']['std']:.1f} bpm"],
            ["Mean LF/HF", lfhf_str],
        ]
        tbl = Table(baseline_data, colWidths=[2*inch, 3*inch])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor(SECONDARY_COLOR)),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor(PRIMARY_COLOR)),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor(PRIMARY_COLOR)),
        ]))
        story.append(tbl)
    else:
        story.append(Paragraph("Baseline not computed or invalid.", styles['Normal']))
    
    story.append(Spacer(1, 12))
    
    # Segment Results
    story.append(Paragraph("Segment-by-Segment Analysis", styles['h2']))
    
    for seg in segment_results:
        if seg is None:
            continue
        
        story.append(Paragraph(f"Segment {seg['segment_idx']} ({seg['start_min']:.1f} - {seg['end_min']:.1f} min)", styles['h3']))
        
        # Risk Summary
        final_risk_str = "HIGH" if seg['final_risk'] == 1 else "LOW"
        risk_data = [
            ["Final Risk", final_risk_str],
            ["Final Score", f"{seg['final_score']*100:.1f}%"],
            ["AI Probability", f"{seg['ai_prob']*100:.1f}%"],
            ["Heuristic Score", f"{seg['heuristic_score']*100:.1f}%"],
            ["Rules Triggered", Paragraph(seg['rules_triggered'], styles['BodyText'])],
        ]
        tbl = Table(risk_data, colWidths=[2*inch, 5*inch])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor(LIGHT_BORDER)),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor(SECONDARY_COLOR)),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor(PRIMARY_COLOR)),
            ('VALIGN', (0, 4), (-1, -1), 'TOP'),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 6))

        # Biomarkers
        hrv = seg['hrv_metrics']
        
        def make_metric_table(domain_name, metrics):
            data = []
            keys = list(metrics.keys())
            for i in range(0, len(keys), 2):
                key1 = keys[i]
                val1 = f"{metrics[key1]:.2f}" if isinstance(metrics[key1], float) else str(metrics[key1])
                key2 = keys[i+1] if i+1 < len(keys) else ""
                val2 = f"{metrics[keys[i+1]]:.2f}" if i+1 < len(keys) and isinstance(metrics[keys[i+1]], float) else (str(metrics[keys[i+1]]) if i+1 < len(keys) else "")
                data.append([key1, val1, key2, val2])
            
            t = Table(data, colWidths=[1.75*inch, 1*inch, 1.75*inch, 1*inch])
            t.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor(LIGHT_BORDER)),
                ('BACKGROUND', (2, 0), (2, -1), colors.HexColor(LIGHT_BORDER)),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                ('ALIGN', (3, 0), (3, -1), 'RIGHT'),
            ]))
            return t
        
        if 'Error' not in hrv['Time Domain']:
            story.append(Paragraph("Time Domain", styles['h4']))
            story.append(make_metric_table("Time Domain", hrv['Time Domain']))
        if 'Error' not in hrv['Frequency Domain']:
            story.append(Paragraph("Frequency Domain", styles['h4']))
            story.append(make_metric_table("Frequency Domain", hrv['Frequency Domain']))
        if 'Error' not in hrv['Nonlinear']:
            story.append(Paragraph("Nonlinear", styles['h4']))
            story.append(make_metric_table("Nonlinear", hrv['Nonlinear']))
        
        story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ============================================================================
# FILE I/O
# ============================================================================

@st.cache_data
def read_edf_file(uploaded_file):
    """Read EDF file and extract ECG channel."""
    if not EDF_SUPPORT:
        st.error("⚠️ MNE library not installed")
        return None, None, None, None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        fs = raw.info['sfreq']
        channels = raw.ch_names
        filename = uploaded_file.name
        
        # Find ECG channel
        ecg_channel = None
        ecg_keywords = ['ECG', 'EKG', 'CARDIO', 'HEART']
        for ch in channels:
            if any(keyword in ch.upper() for keyword in ecg_keywords):
                ecg_channel = ch
                break
        if ecg_channel is None:
            ecg_channel = channels[0]
            st.warning(f"⚠️ No ECG channel detected. Using first channel: {ecg_channel}")
        
        ecg_data = raw.get_data(picks=[ecg_channel])[0]
        
        # Auto-resample 512Hz to 256Hz
        if fs > 256:
            st.info(f"📊 High sample rate detected ({fs} Hz) → Auto-resampling to 256 Hz")
            ecg_data, fs = resample_signal(ecg_data, fs, target_fs=256)
        
        return ecg_data, fs, channels, filename
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
        return None, None, None, None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

@st.cache_resource
def load_model_and_scaler():
    """Load ML model and scaler with XGBoost error handling."""
    model = None
    scaler = None
    model_error = None
    
    # Try to load model
    try:
        with open('neuroalert_hybrid_ecg_only.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        try:
            with open('nueroalert_hybrid_ecg_only.pkl', 'rb') as f:
                model = pickle.load(f)
        except FileNotFoundError:
            model_error = "Model file not found. Running in Heuristic-Only Mode."
    except ImportError as e:
        if 'xgboost' in str(e).lower() or 'xgb' in str(e).lower():
            model_error = "XGBoost library missing. Running in Heuristic-Only Mode."
        else:
            model_error = f"Error loading model: {str(e)}. Running in Heuristic-Only Mode."
    except Exception as e:
        model_error = f"Error loading model: {str(e)}. Running in Heuristic-Only Mode."
    
    # Try to load scaler
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except:
        try:
            scaler = joblib.load('scaler.pkl')
        except:
            if not model_error:
                st.sidebar.warning("⚠️ Scaler 'scaler.pkl' not found. AI predictions may be inaccurate.")
    
    return model, scaler, model_error

model, scaler, model_error = load_model_and_scaler()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'window_start_minutes' not in st.session_state:
    st.session_state.window_start_minutes = 0.0
if 'demo_mode_active' not in st.session_state:
    st.session_state.demo_mode_active = False
if 'analyzed_windows' not in st.session_state:
    st.session_state.analyzed_windows = []

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    st.markdown("---")
    
    if model_error:
        st.error(f"⚠️ {model_error}")
    else:
        st.success("✅ AI Model loaded")
    
    if scaler:
        st.success("✅ Scaler loaded")
    
    st.markdown("### 🎯 Prediction Settings")
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.3, max_value=0.9, value=CONFIDENCE_THRESHOLD_DEFAULT, step=0.05,
        help="The hybrid score must be above this value to flag a segment as 'High Risk'."
    )
    
    alpha_ai_weight = st.slider(
        "AI Weight (α)",
        min_value=0.0, max_value=1.0, value=ALPHA_AI_WEIGHT_DEFAULT, step=0.1,
        help="Balance: 1.0 = 100% AI, 0.0 = 100% Heuristic"
    )
    
    st.info(f"Final Score = ({alpha_ai_weight*100:.0f}% AI) + ({(1-alpha_ai_weight)*100:.0f}% Heuristic)")
    
    st.markdown("### 🔄 Temporal Smoothing")
    enable_smoothing = st.checkbox("Enable Temporal Smoothing", value=True)
    if enable_smoothing:
        required_positive = st.select_slider("Required Positive Segments (out of 5)", options=[1, 2, 3, 4, 5], value=2)
    
    st.markdown("### 🧬 Adaptive Thresholds")
    use_adaptive = st.checkbox("Use Adaptive Thresholds", value=True, help="Patient-specific baselines (recommended)")
    if use_adaptive:
        baseline_duration = st.slider("Baseline Duration (min)", min_value=1.0, max_value=10.0, value=BASELINE_DURATION_MIN_DEFAULT, step=0.5)
        threshold_multiplier = st.slider("Threshold Multiplier (k-value)", min_value=1.0, max_value=3.0, value=ADAPTIVE_THRESHOLD_MULTIPLIER_DEFAULT, step=0.1)
    
    st.markdown("### 🎨 Display Mode")
    display_mode = st.radio("Mode", ["Research", "Clinical"], help="Research: Full details | Clinical: Summary only", horizontal=True)
    
    st.markdown("### 🎬 Demo Mode")
    demo_mode = st.checkbox("Auto-Advance Demo", value=st.session_state.demo_mode_active)
    
    st.markdown("---")
    st.warning("⚠️ **Medical Disclaimer:** For research/educational use only")

# ============================================================================
# MAIN APPLICATION LOGIC
# ============================================================================
def main():
    st.markdown(f"""
    <div class='glow-card' style='text-align: center;'>
        <h2 style='margin: 0 0 10px 0;'>📂 1. Upload ECG Data</h2>
        <p style='font-size: 1.1rem; margin: 0 0 20px 0;'>
            Upload an EDF file (max 500MB) for seizure risk analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose EDF file", type=['edf'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if 'file_name' not in st.session_state or st.session_state.file_name != uploaded_file.name:
            # New file uploaded, clear old state
            for key in list(st.session_state.keys()):
                if key not in ['window_start_minutes', 'demo_mode_active', 'analyzed_windows']: # Reverted
                    del st.session_state[key]
            st.session_state.file_name = uploaded_file.name
        
        st.success(f"✅ **File uploaded!** - {uploaded_file.name}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if uploaded_file is None:
        st.markdown("""
        <div class='glow-card'>
            <h2 style='text-align: center; margin-top: 0;'>👋 Welcome to NeuroAlert</h2>
            <p style='text-align: center; font-size: 1.2rem;'>Upload your ECG EDF file to begin adaptive analysis</p>
            <br>
            <h3>✨ Features:</h3>
            <ul style='font-size: 1.1rem; line-height: 2;'>
                <li><strong>Dual-Panel ECG Display:</strong> Overview + Zoomed window</li>
                <li><strong>Mini Signal Navigator:</strong> Real-time window positioning</li>
                <li><strong>Patient-Specific Thresholds:</strong> Adaptive to individual physiology</li>
                <li><strong>Ground Truth Validation:</strong> Automatic seizure overlay from database</li>
                <li><strong>Export Reports:</strong> CSV & PDF download of all analysis data</li>
                <li><strong>Demo Mode:</strong> Auto-advance through recording</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        if 'full_ecg_data' not in st.session_state:
            st.markdown("""
            <div class='glow-card' style='text-align: center;'>
                <h2 style='margin: 0;'>✅ EDF File Ready</h2>
                <p style='font-size: 1.1rem; margin: 15px 0;'>Click below to load and compute patient baseline</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                load_recording = st.button("🚀 LOAD FULL ECG RECORDING", use_container_width=True, type="primary")
            
            if load_recording:
                progress_bar = st.progress(0, text="Loading recording...")
                try:
                    with st.spinner("📂 Reading EDF file..."):
                        ecg_data, fs, channels, filename = read_edf_file(uploaded_file)
                    
                    if ecg_data is not None:
                        seizure_intervals = get_seizure_intervals_for_file(filename)
                        if seizure_intervals:
                            st.session_state['seizure_intervals'] = seizure_intervals
                        
                        st.session_state['baseline_ok'] = False
                        if use_adaptive:
                            with st.spinner(f"🧬 Computing patient baseline (first {baseline_duration:.1f} min)..."):
                                progress_bar.progress(30, text="Computing baseline...")
                                baseline_metrics, baseline_ok, baseline_duration_actual, baseline_peaks = compute_patient_baseline(ecg_data, fs, baseline_duration)
                                if baseline_ok:
                                    adaptive_thresholds = derive_adaptive_thresholds(baseline_metrics, threshold_multiplier)
                                    st.session_state['adaptive_thresholds'] = adaptive_thresholds
                                    st.session_state['baseline_metrics'] = baseline_metrics
                                    st.session_state['baseline_ok'] = True
                                    st.session_state['baseline_duration'] = baseline_duration_actual
                                    st.session_state['baseline_peaks'] = baseline_peaks
                                else:
                                    st.session_state['baseline_duration'] = baseline_duration_actual
                                    st.session_state['baseline_peaks'] = baseline_peaks
                        
                        with st.spinner("🔬 Filtering full signal for plotting..."):
                            progress_bar.progress(60, text="Filtering signal...")
                            filtered_ecg_data = bandpass_filter(ecg_data, 0.5, 40, fs, order=5)
                        
                        st.session_state['full_ecg_data'] = ecg_data
                        st.session_state['filtered_ecg_data'] = filtered_ecg_data
                        st.session_state['ecg_fs'] = fs
                        st.session_state['recording_duration_minutes'] = len(ecg_data) / fs / 60
                        
                        progress_bar.progress(100, text="✅ Recording loaded!")
                        time.sleep(0.5)
                        progress_bar.empty()
                        
                        st.success(f"✅ **Loaded!** Duration: {st.session_state['recording_duration_minutes']:.1f} min")
                        st.rerun()
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
                    progress_bar.empty()
        
        if 'full_ecg_data' in st.session_state:
            st.markdown("---")
            
            # DISPLAY BASELINE INFO
            if st.session_state.get('baseline_ok', False):
                bm = st.session_state['baseline_metrics']
                bd = st.session_state['baseline_duration']
                bp = st.session_state.get('baseline_peaks', 0)
                lfhf_mean = bm['LFHF']['mean']
                lfhf_std = bm['LFHF']['std']
                lfhf_valid = bm['LFHF']['n_valid']
                
                if np.isnan(lfhf_mean):
                    lfhf_str = "N/A (low spectral energy)"
                else:
                    lfhf_str = f"{lfhf_mean:.2f} ± {lfhf_std:.2f} (n={lfhf_valid})"
                
                st.markdown(f"""
                <div class='baseline-info'>
                    <h4>✅ Patient Baseline Computed (first {bd:.1f} min, {int(bp)} R-peaks)</h4>
                    <p><strong>SDNN:</strong> {bm['SDNN']['mean']:.1f} ± {bm['SDNN']['std']:.1f} ms | 
                    <strong>Mean HR:</strong> {bm['Mean_HR']['mean']:.1f} ± {bm['Mean_HR']['std']:.1f} bpm | 
                    <strong>LF/HF:</strong> {lfhf_str}</p>
                    <p style='font-size: 0.9rem; opacity: 0.8;'>Using adaptive patient-specific thresholds</p>
                </div>
                """, unsafe_allow_html=True)
            elif use_adaptive:
                bd = st.session_state.get('baseline_duration', 0)
                bp = st.session_state.get('baseline_peaks', 0)
                st.markdown(f"""
                <div class='glow-card' style='background: rgba(255, 235, 230, 0.7); border-color: #ff9800;'>
                    <h4 style='color: #e65100; margin-top: 0;'>⚠️ Baseline Insufficient (duration: {bd:.1f} min, {int(bp)} R-peaks)</h4>
                    <p style='color: #bf360c;'>Falling back to population-based thresholds for heuristics.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("## 🎚️ 2. Select 10-Minute Analysis Window")
            
            full_ecg = st.session_state['full_ecg_data']
            filtered_ecg = st.session_state['filtered_ecg_data']
            fs = st.session_state['ecg_fs']
            total_duration_min = st.session_state['recording_duration_minutes']
            seizure_intervals = st.session_state.get('seizure_intervals', [])
            
            max_window_start = max(0.0, total_duration_min - 10.0)
            
            window_start_minutes = st.slider(
                "Window Start Position (minutes)",
                min_value=0.0,
                max_value=max_window_start,
                value=st.session_state.window_start_minutes,
                step=0.5,
                key="window_slider"
            )
            st.session_state.window_start_minutes = window_start_minutes
            window_end_minutes = window_start_minutes + 10.0
            
            st.info(f"🔍 **Current Window:** {window_start_minutes:.1f} - {window_end_minutes:.1f} minutes")
            
            # Check for seizures in window
            window_seizures = []
            for (s, e) in seizure_intervals:
                if not (e/60 <= window_start_minutes or s/60 >= window_end_minutes):
                    window_seizures.append((s, e))
            
            if window_seizures:
                st.markdown(f"""
                <div class='glow-card' style='background: linear-gradient(135deg, #ff4757 0%, #ff6348 100%); border: 3px solid #ffa502;'>
                    <h3 style='color: white; margin: 0;'>⚠️ GROUND TRUTH SEIZURE IN WINDOW</h3>
                    <p style='color: white; margin: 5px 0 0 0;'>{len(window_seizures)} known seizure interval(s) detected (marked in red)</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 📈 Dual-Panel ECG Visualization")
            fig_dual = create_dual_panel_ecg(filtered_ecg, fs, window_start_minutes, window_end_minutes, total_duration_min, seizure_intervals)
            st.plotly_chart(fig_dual, use_container_width=True)
            
            st.markdown("### 🔍 Mini Signal Navigator")
            fig_nav = create_mini_navigator(filtered_ecg, fs, window_start_minutes, window_end_minutes, total_duration_min)
            st.plotly_chart(fig_nav, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                analyze_button = st.button("🔬 ANALYZE THIS 10-MINUTE WINDOW", use_container_width=True, type="primary")
            
            if demo_mode:
                st.session_state.demo_mode_active = True
                if st.button("STOP DEMO"):
                    st.session_state.demo_mode_active = False
                    st.rerun()

            if analyze_button or (demo_mode and st.session_state.get('demo_mode_active', False)):
                progress_bar = st.progress(0, text="Initializing analysis...")
                
                try:
                    # Extract 10-min window from *raw* data
                    window_start_sample = int(window_start_minutes * 60 * fs)
                    window_end_sample = int(window_end_minutes * 60 * fs)
                    window_ecg = full_ecg[window_start_sample:window_end_sample]

                    segment_duration_sec = 120  # 2 minutes
                    num_segments = 5
                    
                    segment_results = []
                    prev_metrics = None
                    
                    # Set alpha for heuristic-only mode
                    current_alpha = alpha_ai_weight
                    if model is None:
                        current_alpha = 0.0
                    
                    for seg_idx in range(num_segments):
                        seg_start_sample = int(seg_idx * segment_duration_sec * fs)
                        seg_end_sample = min(int((seg_idx + 1) * segment_duration_sec * fs), len(window_ecg))
                        
                        if seg_start_sample >= len(window_ecg):
                            break
                        
                        segment_ecg = window_ecg[seg_start_sample:seg_end_sample]
                        
                        progress = 10 + int((seg_idx / num_segments) * 60)
                        progress_bar.progress(progress, text=f"Analyzing segment {seg_idx + 1}/5...")
                        
                        features, hrv_metrics, filtered_seg_ecg, peaks = extract_comprehensive_hrv_features(segment_ecg, fs)
                        
                        if hrv_metrics['Signal Quality'].get('R-peaks Detected', 0) < MIN_SEGMENT_PEAKS:
                            st.warning(f"⚠️ Segment {seg_idx + 1}: Poor signal quality ({len(peaks)} R-peaks). Skipping.")
                            segment_results.append(None)
                            continue
                        
                        # Get AI Model Prediction
                        ai_prob = 0.0
                        if model is not None and scaler is not None:
                            features_reshaped = features.reshape(1, -1)
                            features_scaled = scaler.transform(features_reshaped)
                            ai_prob = model.predict_proba(features_scaled)[0][1]
                        
                        # Get Heuristic Veto System Prediction
                        if use_adaptive and st.session_state.get('baseline_ok', False):
                            heuristic_score, rules_triggered = evaluate_with_adaptive_heuristics(hrv_metrics, st.session_state['adaptive_thresholds'], prev_metrics)
                        else:
                            heuristic_score, rules_triggered = check_biomarker_heuristics_population(hrv_metrics, prev_metrics)
                        
                        # Combine results using Weighted Average
                        final_score = (current_alpha * ai_prob) + ((1 - current_alpha) * heuristic_score)
                        final_risk = 1 if final_score >= confidence_threshold else 0
                        
                        segment_results.append({
                            'segment_idx': seg_idx + 1,
                            'start_min': window_start_minutes + (seg_idx * 2),
                            'end_min': window_start_minutes + ((seg_idx + 1) * 2),
                            'ai_prob': ai_prob,
                            'heuristic_score': heuristic_score,
                            'final_score': final_score,
                            'rules_triggered': rules_triggered,
                            'final_risk': final_risk,
                            'hrv_metrics': hrv_metrics,
                            'filtered_ecg': filtered_seg_ecg, # For mini-plot
                            'fs': fs
                        })
                        
                        prev_metrics = hrv_metrics
                    
                    valid_segments = [s for s in segment_results if s is not None]
                    
                    if len(valid_segments) == 0:
                        st.error("❌ No valid segments found. Please try a different window position.")
                        progress_bar.empty()
                    else:
                        progress_bar.progress(80, text="Aggregating results...")
                        
                        final_scores = [s['final_score'] for s in valid_segments]
                        avg_final_score = np.mean(final_scores)
                        max_final_score = np.max(final_scores)
                        
                        positive_segments = sum([1 for s in valid_segments if s['final_risk'] > 0])
                        
                        if enable_smoothing:
                            final_verdict = 1 if positive_segments >= required_positive else 0
                        else:
                            final_verdict = 1 if positive_segments > 0 else 0
                        
                        # Store for session summary
                        st.session_state.analyzed_windows.append({
                            'start_min': window_start_minutes,
                            'end_min': window_end_minutes,
                            'final_verdict': final_verdict,
                            'avg_score': avg_final_score,
                            'positive_segments': positive_segments,
                            'num_valid_segments': len(valid_segments)
                        })

                        # Check ground truth accuracy
                        ground_truth_hits = 0
                        seizure_segments = 0
                        if seizure_intervals:
                            for seg in valid_segments:
                                seg_start_sec, seg_end_sec = seg['start_min']*60, seg['end_min']*60
                                is_seizure_segment = False
                                for (s, e) in seizure_intervals:
                                    if not (seg_end_sec <= s or seg_start_sec >= e):
                                        is_seizure_segment = True
                                        break
                                if is_seizure_segment:
                                    seizure_segments += 1
                                    if seg['final_risk'] == 1:
                                        ground_truth_hits += 1

                        progress_bar.progress(100, text="✅ Analysis complete!")
                        time.sleep(0.5)
                        progress_bar.empty()
                        
                        # ==================== RESULTS SECTION ====================
                        st.markdown("---")
                        st.markdown("# 📊 10-MINUTE WINDOW ANALYSIS RESULTS")
                        
                        if final_verdict == 1:
                            st.markdown(f"""
                            <div class='risk-high'>
                                <h1>🚨 HIGH SEIZURE RISK DETECTED</h1>
                                <p style='font-size: 1.5rem;'>
                                    {positive_segments} out of {len(valid_segments)} segments flagged as high risk
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class='risk-low'>
                                <h1>🟢 LOW SEIZURE RISK</h1>
                                <p style='font-size: 1.5rem;'>
                                    No significant risk patterns detected in this window.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if display_mode == "Clinical":
                            if final_verdict == 1:
                                st.markdown("""
                                <div class='glow-card'>
                                    <h2 style='color: var(--risk-high-color); margin-top: 0;'>🏥 IMMEDIATE SAFETY ACTIONS</h2>
                                    <ul style='font-size: 1.1rem; line-height: 2;'>
                                        <li><strong>🚫 Stop dangerous activities immediately</strong> (driving, swimming, heights)</li>
                                        <li><strong>🛡️ Move to safe location</strong> (sit or lie down away from hazards)</li>
                                        <li><strong>📞 Alert someone nearby</strong> (caregiver, family, or friend)</li>
                                        <li><strong>💊 Prepare rescue medication</strong> (if prescribed)</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div class='glow-card'>
                                    <h2 style='color: var(--risk-low-color); margin-top: 0;'>✅ SAFE TO CONTINUE ACTIVITIES</h2>
                                    <ul style='font-size: 1.1rem; line-height: 2;'>
                                        <li><strong>💊 Maintain medication schedule</strong></li>
                                        <li><strong>👁️ Stay vigilant</strong> and monitor as recommended</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # --- RESEARCH MODE ---
                        if display_mode == "Research":
                            st.markdown("### 🚦 Segment Status Timeline")
                            fig_timeline = create_segment_status_timeline(valid_segments)
                            st.plotly_chart(fig_timeline, use_container_width=True)

                            st.markdown("### 🔬 Segment-by-Segment Results (with mini-ECG)")
                            num_cols = len(valid_segments)
                            if num_cols > 0:
                                cols = st.columns(num_cols)
                                for i, seg in enumerate(valid_segments):
                                    with cols[i]:
                                        seg_idx = seg['segment_idx']
                                        ai_prob_pct = seg['ai_prob'] * 100
                                        heuristic_score_pct = seg['heuristic_score'] * 100
                                        final_score_pct = seg['final_score'] * 100
                                        rules = seg['rules_triggered']
                                        
                                        if seg['final_risk'] == 1:
                                            if seg['heuristic_score'] > seg['ai_prob']:
                                                status = "🚨 HIGH (Rule)"
                                                color = COLOR_HIGH_RISK
                                            else:
                                                status = "⚠️ HIGH (AI)"
                                                color = COLOR_BORDERLINE
                                        else:
                                            status = "✅ LOW"
                                            color = COLOR_LOW_RISK
                                        
                                        st.markdown(f"""
                                        <div class='metric-card' style='border-color: {color};'>
                                            <h4 style='color: {color}; margin: 0;'>Seg {seg_idx} ({seg['start_min']:.0f}-{seg['end_min']:.0f} min)</h4>
                                            <p style='font-size: 1.3rem; font-weight: 800; margin: 5px 0; color: {color};'>{status}</p>
                                            <p style='font-size: 1rem; font-weight: 700; color: var(--dark-text); margin: 0 0 5px 0;'>Score: {final_score_pct:.1f}%</p>
                                            <hr style='border: 1px solid var(--secondary-color); margin: 5px 0;'>
                                            <p style='font-size: 0.85rem; color: var(--dark-text); margin: 0; text-align: left;'>AI Prob: {ai_prob_pct:.1f}%</p>
                                            <p style='font-size: 0.85rem; color: var(--dark-text); margin: 0; text-align: left;'>Heuristic: {heuristic_score_pct:.1f}%</p>
                                            <p style='font-size: 0.75rem; color: var(--dark-text); margin: 5px 0 0 0; text-align: left; opacity: 0.7;'><i>Rules: {rules}</i></p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Add the mini-plot
                                        mini_fig = create_mini_segment_plot(seg['filtered_ecg'], seg['fs'])
                                        st.plotly_chart(mini_fig, use_container_width=True)

                                        # --- NEW: BIOMARKER EXPANDER ---
                                        with st.expander(f"Biomarkers for Segment {seg_idx}"):
                                            hrv_metrics = seg['hrv_metrics']
                                            sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(["⏱️ Time", "📊 Freq", "🔄 Nonlin", "🔬 Signal"])
                                            
                                            with sub_tab1:
                                                if 'Error' not in hrv_metrics['Time Domain']:
                                                    metric_cols = st.columns(2)
                                                    metrics_list = list(hrv_metrics['Time Domain'].items())
                                                    for j, (key, value) in enumerate(metrics_list):
                                                        metric_cols[j % 2].metric(label=key, value=f"{value:.2f}")
                                                else:
                                                    st.warning("Insufficient data for Time Domain analysis.")
                                            
                                            with sub_tab2:
                                                if 'Error' not in hrv_metrics['Frequency Domain']:
                                                    metric_cols = st.columns(2)
                                                    metrics_list = list(hrv_metrics['Frequency Domain'].items())
                                                    for j, (key, value) in enumerate(metrics_list):
                                                        metric_cols[j % 2].metric(label=key, value=f"{value:.2f}")
                                                else:
                                                    st.warning("Insufficient data for Frequency Domain analysis.")

                                            with sub_tab3:
                                                if 'Error' not in hrv_metrics['Nonlinear']:
                                                    metric_cols = st.columns(2)
                                                    metrics_list = list(hrv_metrics['Nonlinear'].items())
                                                    for j, (key, value) in enumerate(metrics_list):
                                                        metric_cols[j % 2].metric(label=key, value=f"{value:.2f}")
                                                else:
                                                    st.warning("Insufficient data for Nonlinear analysis.")

                                            with sub_tab4:
                                                if 'Error' not in hrv_metrics['Signal Quality']:
                                                    metric_cols = st.columns(2)
                                                    metrics_list = list(hrv_metrics['Signal Quality'].items())
                                                    for j, (key, value) in enumerate(metrics_list):
                                                        metric_cols[j % 2].metric(label=key, value=f"{int(value)}" if "R-peaks" in key else f"{value:.3f}")
                                                else:
                                                    st.warning("Insufficient data for Signal Quality analysis.")

                            
                            st.markdown("<br>", unsafe_allow_html=True)

                            st.markdown("### 📈 Biomarker Trends (across segments)")
                            fig_trends = create_biomarker_trends(valid_segments)
                            if fig_trends:
                                st.plotly_chart(fig_trends, use_container_width=True)

                            with st.expander("🔬 Session & Export"):
                                # Session Summary
                                st.markdown("#### 📋 Session Summary")
                                total_windows = len(st.session_state.analyzed_windows)
                                high_risk_windows = sum(1 for w in st.session_state.analyzed_windows if w['final_verdict'] == 1)
                                if total_windows > 0:
                                    avg_score_all = np.mean([w['avg_score'] for w in st.session_state.analyzed_windows]) * 100
                                    high_risk_perc = high_risk_windows / total_windows * 100
                                else:
                                    avg_score_all = 0
                                    high_risk_perc = 0
                                    
                                st.metric(label="Total 10-min Windows Analyzed", value=total_windows)
                                st.metric(label="Windows Flagged as HIGH RISK", value=f"{high_risk_windows} ({high_risk_perc:.0f}% of total)")
                                st.metric(label="Average Final Score Across All Windows", value=f"{avg_score_all:.1f}%")

                                if seizure_segments > 0:
                                    st.metric(label="Ground Truth Validation", value=f"Detected {ground_truth_hits} of {seizure_segments} seizure segments")

                                # Export Button
                                st.markdown("#### 📥 Export Segment Data")
                                df_rows = []
                                for seg in valid_segments:
                                    row = {
                                        'segment_idx': seg['segment_idx'],
                                        'start_min': seg['start_min'],
                                        'end_min': seg['end_min'],
                                        'ai_prob': seg['ai_prob'],
                                        'heuristic_score': seg['heuristic_score'],
                                        'final_score': seg['final_score'],
                                        'final_risk': seg['final_risk'],
                                        'rules_triggered': seg['rules_triggered'],
                                    }
                                    # Add HRV metrics
                                    for domain, metrics in seg['hrv_metrics'].items():
                                        if 'Error' not in metrics:
                                            for key, val in metrics.items():
                                                row[f"{domain}_{key}"] = val
                                    df_rows.append(row)
                                
                                df_export = pd.DataFrame(df_rows)
                                csv = df_export.to_csv(index=False).encode('utf-8')
                                
                                st.download_button(
                                    label="Download Segment Report (CSV)",
                                    data=csv,
                                    file_name=f"NeuroAlert_Report_{uploaded_file.name.split('.')[0]}_{window_start_minutes:.0f}min.csv",
                                    mime='text/csv',
                                )
                                
                                # PDF Export
                                if PDF_SUPPORT:
                                    pdf_buffer = generate_pdf_report(
                                        valid_segments, 
                                        st.session_state.get('baseline_metrics', {}),
                                        window_start_minutes,
                                        window_end_minutes,
                                        uploaded_file.name,
                                        st.session_state.get('baseline_duration', 0)
                                    )
                                    st.download_button(
                                        label="Download Full Report (PDF)",
                                        data=pdf_buffer,
                                        file_name=f"NeuroAlert_Report_{uploaded_file.name.split('.')[0]}_{window_start_minutes:.0f}min.pdf",
                                        mime='application/pdf',
                                    )
                                else:
                                    st.info("Install 'reportlab' (`pip install reportlab`) to enable PDF exports.")
                        
                except Exception as e:
                    st.error(f"⚠️ Analysis Error: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
                    progress_bar.empty()
        
        # Auto-advance logic
        if demo_mode and st.session_state.get('demo_mode_active', False):
            st.warning("Demo Mode Active: Advancing window...")
            with st.spinner(f"Waiting {DEMO_MODE_DELAY_SEC}s..."):
                time.sleep(DEMO_MODE_DELAY_SEC)
            
            new_start = st.session_state.window_start_minutes + DEMO_MODE_STEP_SIZE_MIN
            if new_start > max_window_start:
                st.success("✅ Demo complete!")
                st.session_state.demo_mode_active = False
            else:
                st.session_state.window_start_minutes = new_start
            
            st.rerun()

    st.markdown("---")
    st.markdown(f"""
    <div class='glow-card' style='text-align: center;'>
        <h3 style='margin: 0 0 15px 0;'>🧠 NeuroAlert</h3>
        <p style='margin: 5px 0;'><strong>AI + Adaptive Heuristic Seizure Prediction</strong></p>
        <p style='font-size: 0.9rem; opacity: 0.8;'>Research/educational use only. Not for clinical diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    if not EDF_SUPPORT:
        st.error("Fatal Error: MNE Python library not found. Please install it: `pip install mne`")
    else:
        main()
