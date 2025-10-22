import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import altair as alt
import mne
import neurokit2 as nk
from scipy.stats import linregress
from scipy import signal
import re
import warnings
import tempfile  
import os        

# --- Page Configuration ---
st.set_page_config(
    page_title="NeuroAlert Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- App Title ---
st.title("🧠 NeuroAlert: Epileptic Seizure Prediction")
st.markdown(f"""
    *A proof-of-concept by **Rijjul Garg (Medical Lead)** and **Parth Kapoor (Tech Lead)**.*
    
    **Upload a raw `.edf` file** (up to 200MB) to run a full analysis. The system will extract
    Heart Rate Variability (HRV) features, analyze 10-minute trends, and use our **V7 AI Model** to predict the pre-ictal phase (the period before a seizure).
""")

# --- Global Settings & Model Files ---
MODEL_FILE = 'neuroalert_final_model_v7.pkl'
SCALER_FILE = 'neuroalert_scaler_v6.pkl'
PREDICTION_THRESHOLD = 0.5205 # The "sensitivity dial" we found
WINDOW_SIZE = 5 # 5 segments = 10 minutes (since each segment is 2 mins)
SEGMENT_DURATION_SECS = 120 # 2-minute windows

BASE_FEATURES = ['HR', 'MeanNN', 'SDNN', 'RMSSD', 'pNN50', 'SampEn', 
                 'HRV_HTI', 'LF/HF', 'SD1', 'SD2', 'SD1/SD2', 'CSI']

# --- Caching: Load Model & Scaler ---
@st.cache_resource
def load_model_and_scaler():
    """Loads the V7 model and V6 scaler from disk."""
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        return model, scaler
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Missing '{MODEL_FILE}' or '{SCALER_FILE}'.")
        st.error("Please ensure the .pkl files are in the same GitHub repository as app.py.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, scaler = load_model_and_scaler()
if model is None or scaler is None:
    st.stop()

# Get the exact 36-feature order the scaler expects
try:
    temporal_feature_names = scaler.get_feature_names_out()
except AttributeError:
    # Fallback for older scikit-learn versions
    st.warning("Could not get feature names from scaler. Proceeding with caution.")
    temporal_feature_names = [f'{col}_{stat}_{WINDOW_SIZE}' for col in BASE_FEATURES for stat in ['mean', 'std', 'slope']]


# --- Core Processing Functions ---

def find_ecg_channel(ch_names):
    """Find the correct ECG channel name."""
    for ch in ch_names:
        if "ECG" in ch.upper() or "EKG" in ch.upper():
            return ch
    # Fallback if no ECG channel is obvious
    for ch in ch_names:
        if "T8-P8" in ch: # A common channel in this dataset
            return ch
    return None

def get_hrv_features(rpeaks, sfreq):
    """
    Extracts the 12 key HRV features from a list of R-peaks.
    """
    # Need at least 10 R-peaks to calculate reliable features
    if len(rpeaks) < 10:
        return pd.DataFrame(columns=BASE_FEATURES) 

    try:
        hrv_features = nk.hrv(rpeaks, sampling_rate=sfreq, show=False)
        # Select only the 12 features our model was trained on
        hrv_features_selected = hrv_features[BASE_FEATURES]
        return hrv_features_selected
    except Exception as e:
        # Catch errors from neurokit (e.g., "ZeroDivisionError")
        return pd.DataFrame(columns=BASE_FEATURES)

def calculate_temporal_features(window_df):
    """
    Takes a dataframe window (most recent 10 mins) and calculates 
    the 36 temporal features.
    """
    features = {}
    
    # Calculate 10-minute trends (mean, std, slope)
    for col in BASE_FEATURES:
        window = window_df[col].fillna(0).replace([np.inf, -np.inf], 0)
        
        # 1. Mean
        features[f'{col}_mean_{WINDOW_SIZE}'] = window.mean()
        
        # 2. Standard Deviation (Volatility)
        std_val = window.std()
        features[f'{col}_std_{WINDOW_SIZE}'] = std_val if np.isfinite(std_val) else 0
        
        # 3. Slope (Trend)
        if len(window) > 1:
            # Use np.arange for the x-axis of the trend line
            slope = linregress(np.arange(len(window)), window).slope
        else:
            slope = 0
        features[f'{col}_slope_{WINDOW_SIZE}'] = slope if np.isfinite(slope) else 0

    # Return as a single-row DataFrame
    return pd.DataFrame([features])


# --- Main Application Logic ---
st.sidebar.title("Control Panel")
uploaded_file = st.sidebar.file_uploader(
    "Upload a raw .edf file", 
    type="edf",
    accept_multiple_files=False,
    help="Max file size: 200MB."
)

if uploaded_file is not None:
    # This is the 200MB limit
    if uploaded_file.size > 200 * 1024 * 1024:
        st.error("File is too large. Please upload an .edf file under 200MB.")
    else:
        st.sidebar.info(f"File '{uploaded_file.name}' received. Starting analysis.")
        
        tmp_file_path = None 
        try:
            # 1. Create a temporary file and write the uploaded data to it
            with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name  
            
            with st.spinner("Analyzing EDF file... This may take 1-3 minutes. Please wait."):
                
                # 1. LOAD AND PROCESS EDF
                raw = mne.io.read_raw_edf(tmp_file_path, preload=True, verbose='error')
                sfreq = int(raw.info['sfreq'])
                
                # 2. FIND ECG CHANNEL
                ecg_channel = find_ecg_channel(raw.info['ch_names'])
                if ecg_channel is None:
                    st.error("Could not automatically find an 'ECG' or 'EKG' channel in this file.")
                    st.stop()
                
                st.success(f"Found ECG data in channel: '{ecg_channel}' at {sfreq} Hz.")
                ecg_signal = raw.get_data(picks=[ecg_channel])[0]
                
                # 3. PROCESS ALL SEGMENTS (V2-style processing)
                segment_length_samples = SEGMENT_DURATION_SECS * sfreq
                all_hrv_features = []
                
                total_segments_indices = range(0, raw.n_times - segment_length_samples, segment_length_samples)
                
                for i in total_segments_indices:
                    segment_ecg = ecg_signal[i : i + segment_length_samples]
                    
                    # Clean the segment
                    ecg_cleaned = nk.ecg_clean(segment_ecg, sampling_rate=sfreq, method="neurokit")
                    
                    # Find R-peaks
                    # --- THIS IS THE FIX ---
                    # Switched from "neurokit" to "pantompkins1985" for better noise resistance
                    rpeaks_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sfreq, method="pantompkins1985", correct_artifacts=True)
                    # --- END OF THE FIX ---

                    rpeaks = rpeaks_info[0]['ECG_R_Peaks']
                    
                    # Get HRV features
                    hrv_features = get_hrv_features(rpeaks, sfreq)
                    
                    if not hrv_features.empty:
                        all_hrv_features.append(hrv_features)
                
                if not all_hrv_features:
                    st.error("Could not extract any valid HRV data from this file. The signal may be too noisy.")
                    st.stop()

                hrv_df = pd.concat(all_hrv_features, ignore_index=True)
                hrv_df = hrv_df.fillna(0).replace([np.inf, -np.inf], 0)
                
                st.success(f"Successfully processed file into {len(hrv_df)} 2-minute segments.")
                
                # 4. CALCULATE TEMPORAL FEATURES (V6-style processing)
                temporal_features_list = []
                
                for i in range(len(hrv_df)):
                    # Get the start index for the window
                    start_idx = max(0, i - (WINDOW_SIZE - 1))
                    # Get the current window of data
                    window_df = hrv_df.iloc[start_idx : i + 1]
                    
                    temporal_features = calculate_temporal_features(window_df)
                    temporal_features_list.append(temporal_features)
                
                temporal_df = pd.concat(temporal_features_list, ignore_index=True)
                
                # 5. SCALE AND PREDICT (V7-style modeling)
                
                # Re-order columns to match scaler and model
                temporal_df_ordered = temporal_df[temporal_feature_names]
                
                # Scale the features
                X_final = scaler.transform(temporal_df_ordered)
                
                # Get probabilities from our V7 model
                final_probs = model.predict_proba(X_final)[:, 1]
                
                # Apply our chosen threshold
                final_predictions = (final_probs >= PREDICTION_THRESHOLD).astype(int)
                
                # --- 6. DISPLAY FINAL REPORT ---
                st.header(f"Analysis Complete: {uploaded_file.name}")
                
                total_alerts = np.sum(final_predictions)
                
                st.metric("Total Pre-Ictal Alerts Detected", f"{total_alerts} segments")
                if total_alerts > 0:
                    st.warning("Pre-ictal patterns were detected in this file.", icon="🚨")
                else:
                    st.success("No pre-ictal patterns were detected in this file.", icon="✅")
                    
                st.markdown("---")
                
                # Create a results dataframe for charting
                results_df = pd.DataFrame({
                    # Time starts at 2 mins (end of first segment)
                    'Time (minutes)': (np.arange(len(final_probs)) * 2) + 2, 
                    'Seizure Probability': final_probs,
                    'Threshold': PREDICTION_THRESHOLD,
                    'Alert': final_predictions
                })
                
                st.subheader("Prediction Confidence Over Time")
                
                # Melt for Altair
                prob_chart_data = results_df.melt(
                    'Time (minutes)', 
                    var_name='Metric', 
                    value_name='Probability',
                    value_vars=['Seizure Probability', 'Threshold']
                )

                prob_chart = alt.Chart(prob_chart_data).mark_line(point=False).encode(
                    x=alt.X('Time (minutes)', axis=alt.Axis(title='Time (minutes)')),
                    y=alt.Y('Probability', min=0, max=1, axis=alt.Axis(format='%')),
                    color=alt.Color('Metric', legend=alt.Legend(title="Metric"))
                ).interactive()
                
                # Add red dots for alerts
                alert_points = alt.Chart(results_df[results_df['Alert'] == 1]).mark_point(
                    color='red',
                    size=100,
                    filled=True,
                    opacity=1
                ).encode(
                    x=alt.X('Time (minutes)'),
                    y=alt.Y('Seizure Probability', axis=alt.Axis(format='%')),
                    tooltip=[
                        alt.Tooltip('Time (minutes)'),
                        alt.Tooltip('Seizure Probability', format='.1%')
                    ]
                )
                
                final_prob_chart = prob_chart + alert_points
                st.altair_chart(final_prob_chart, use_container_width=True)
                
                # Show key biomarker charts
                st.subheader("HRV Biomarker Trends (Raw Data)")
                
                # Add time column to original HRV data
                hrv_df['Time (minutes)'] = (hrv_df.index * 2) + 2
                
                col1, col2 = st.columns(2)
                with col1:
                    # SDNN Chart (Volatility)
                    sdnn_chart = alt.Chart(hrv_df).mark_line(point=False, color='blue').encode(
                        x=alt.X('Time (minutes)'),
                        y=alt.Y('SDNN', title='SDNN (ms)')
                    ).properties(
                        title="HRV Volatility (SDNN)"
                    ).interactive()
                    st.altair_chart(sdnn_chart, use_container_width=True)
                
                with col2:
                    # LF/HF Chart (Balance)
                    lfhf_chart = alt.Chart(hrv_df).mark_line(point=False, color='red').encode(
                        x=alt.X('Time (minutes)'),
                        y=alt.Y('LF/HF', title='LF/HF Ratio')
                    ).properties(
                        title="Autonomic Balance (LF/HF)"
                    ).interactive()
                    st.altair_chart(lfhf_chart, use_container_width=True)

                # Show the raw features data
                with st.expander("Show Raw Data and Predictions"):
                    st.dataframe(pd.concat([hrv_df, results_df.drop(columns='Time (minutes)')], axis=1))

        except Exception as e:
            # This is our custom error block
            st.error(f"An error occurred during processing: {e}", icon="⚠️")
            st.error("This file may be corrupted, have no valid ECG channel, or contain a signal that is too noisy for analysis.")
            # This line is for debugging, you can comment it out later
            st.exception(e) 
        
        finally:
            # --- CLEANUP ---
            # Always delete the temporary file, even if the app crashes
            if tmp_file_path is not None and os.path.exists(tmp_file_path):
                try:
                    os.remove(tmp_file_path)
                    # print(f"Removed temp file: {tmp_file_path}")
                except Exception as e:
                    # print(f"Could not remove temp file: {e}")
                    pass

else:
    st.info("Upload an .edf file to begin analysis.")
