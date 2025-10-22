import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import mne
import neurokit2 as nk
from scipy.stats import linregress
import altair as alt
import time
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

# --- Global Settings & Model Files ---
MODEL_FILE = 'neuroalert_final_model_v7.pkl'
SCALER_FILE = 'neuroalert_scaler_v6.pkl'
PREDICTION_THRESHOLD = 0.5205 # Our "sensitivity dial"
WINDOW_SIZE = 5 # 5 segments = 10 minutes
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
    possible_names = ['ECG', 'EKG', 'T8-P8-0', 'T8-P8-1', 'T8-P8', 'P8-O2']
    for name in possible_names:
        if name in ch_names: 
            return name
    return None

def calculate_temporal_features(window_df):
    """
    Takes a dataframe window (most recent 10 mins) and calculates 
    the 36 temporal features.
    """
    features = {}
    for col in BASE_FEATURES:
        window = window_df[col].fillna(0).replace([np.inf, -np.inf], 0)
        
        features[f'{col}_mean_{WINDOW_SIZE}'] = window.mean()
        std_val = window.std()
        features[f'{col}_std_{WINDOW_SIZE}'] = std_val if np.isfinite(std_val) else 0
        
        if len(window) > 1:
            slope = linregress(np.arange(len(window)), window).slope
        else:
            slope = 0
        features[f'{col}_slope_{WINDOW_SIZE}'] = slope if np.isfinite(slope) else 0
    return pd.DataFrame([features])

# ======================================================================================
# MAIN APP
# ======================================================================================
st.title("🧠 NeuroAlert: Seizure Risk Prediction")
st.markdown(f"""
    *A proof-of-concept by **Rijjul Garg (Medical Lead)** and **Parth Kapoor (Tech Lead)**.*
    
    **Upload a raw `.edf` file** (up to 200MB) to run a full analysis. The system will extract
    Heart Rate Variability (HRV) features, analyze 10-minute trends, and use our **V7 AI Model** to predict the pre-ictal phase.
""")

uploaded_file = st.file_uploader("Choose an .edf file", type="edf", help="Max file size: 200MB.")

if uploaded_file is not None:
    if uploaded_file.size > 200 * 1024 * 1024:
        st.error("File is too large. Please upload an .edf file under 200MB.")
    else:
        
        if st.button("Analyze Full Recording"):
            tmp_file_path = None
            try:
                # Use tempfile to write the in-memory file to disk for MNE
                with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                with st.spinner("Step 1/4: Loading and Reading EDF file..."):
                    raw = mne.io.read_raw_edf(tmp_file_path, preload=False, verbose='error') # preload=False is SAFER
                    sampling_rate = int(raw.info['sfreq'])
                    ecg_channel = find_ecg_channel(raw.info['ch_names'])
                    
                    if not ecg_channel:
                        st.error("Could not find a standard ECG channel (ECG, EKG, T8-P8) in this file."); st.stop()

                    st.success(f"File loaded. Analyzing ECG channel: '{ecg_channel}' @ {sampling_rate} Hz.")
                    # Load ONLY the one channel we need. This is memory-efficient.
                    ecg_signal = raw.get_data(picks=[ecg_channel])[0]
                    segment_length_samples = SEGMENT_DURATION_SECS * sampling_rate

                with st.spinner(f"Step 2/4: Analyzing {len(range(0, len(ecg_signal) - segment_length_samples, segment_length_samples))} 2-min segments..."):
                    base_features_list = []
                    
                    # --- LOOP 1: V1-style Robust Feature Extraction ---
                    for i in range(0, len(ecg_signal) - segment_length_samples, segment_length_samples):
                        segment_ecg = ecg_signal[i : i + segment_length_samples]
                        
                        try:
                            # Use the robust, all-in-one 'ecg_process' on the 2-min chunk
                            df_feat, info = nk.ecg_process(segment_ecg, sampling_rate=sampling_rate)
                            rpeaks = info['ECG_R_Peaks']
                            
                            if len(rpeaks) < 10: # Need enough heartbeats
                                continue # Skip this noisy segment

                            # Calculate HRV features
                            hrv_features = nk.hrv(rpeaks, sampling_rate=sampling_rate, show=False)
                            
                            # Build the feature dictionary (this is robust)
                            features = {
                                'HR': np.mean(df_feat['ECG_Rate']),
                                'MeanNN': hrv_features['HRV_MeanNN'].iloc[0],
                                'SDNN': hrv_features['HRV_SDNN'].iloc[0],
                                'RMSSD': hrv_features['HRV_RMSSD'].iloc[0],
                                'pNN50': hrv_features['HRV_pNN50'].iloc[0],
                                'SampEn': hrv_features['HRV_SampEn'].iloc[0],
                                'HRV_HTI': hrv_features['HRV_HTI'].iloc[0],
                                'LF/HF': hrv_features['HRV_LFHF'].iloc[0],
                                'SD1': hrv_features['HRV_SD1'].iloc[0],
                                'SD2': hrv_features['HRV_SD2'].iloc[0],
                                'SD1/SD2': hrv_features['HRV_SD1'].iloc[0] / hrv_features['HRV_SD2'].iloc[0] if hrv_features['HRV_SD2'].iloc[0] != 0 else 0,
                                'CSI': hrv_features['HRV_CSI'].iloc[0]
                            }
                            base_features_list.append(features)
                            
                        except Exception as e:
                            # This segment was noisy, just skip it and continue
                            pass

                if not base_features_list:
                    st.error("Could not extract any valid HRV data. The signal is likely too noisy or the selected channel is not ECG.")
                    st.stop()
                
                hrv_df = pd.DataFrame(base_features_list).fillna(0).replace([np.inf, -np.inf], 0)
                st.success(f"Step 2 Complete: Extracted 12 HRV features from {len(hrv_df)} valid segments.")
                
                # --- LOOP 2: V6-style Temporal Feature Generation ---
                with st.spinner("Step 3/4: Analyzing 10-minute trends..."):
                    temporal_features_list = []
                    for i in range(len(hrv_df)):
                        start_idx = max(0, i - (WINDOW_SIZE - 1))
                        window_df = hrv_df.iloc[start_idx : i + 1]
                        temporal_features = calculate_temporal_features(window_df)
                        temporal_features_list.append(temporal_features)
                    
                    temporal_df = pd.concat(temporal_features_list, ignore_index=True)

                # --- STEP 3: V7-style Prediction ---
                with st.spinner("Step 4/4: Running V7 AI Model..."):
                    temporal_df_ordered = temporal_df[temporal_feature_names]
                    X_final = scaler.transform(temporal_df_ordered)
                    final_probs = model.predict_proba(X_final)[:, 1]
                    final_predictions = (final_probs >= PREDICTION_THRESHOLD).astype(int)

                st.success("Analysis Complete!")

                # --- 4. DISPLAY FINAL REPORT ---
                st.header(f"Analysis Report: {uploaded_file.name}")
                
                total_alerts = np.sum(final_predictions)
                
                st.metric("Total Pre-Ictal Alerts Detected", f"{total_alerts} segments")
                if total_alerts > 0:
                    st.warning("High-risk pre-ictal patterns were detected in this file.", icon="🚨")
                else:
                    st.success("No significant pre-ictal patterns were detected.", icon="✅")
                    
                st.markdown("---")
                
                # Create a results dataframe for charting
                results_df = pd.DataFrame({
                    'Time (minutes)': (np.arange(len(final_probs)) * 2) + 2, # Time starts at 2 mins
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
                    color='red', size=100, filled=True, opacity=1
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
                st.subheader("HRV Biomarker Trends (Raw 2-min data)")
                
                # Add time column to original HRV data
                hrv_df['Time (minutes)'] = (hrv_df.index * 2) + 2
                
                col1, col2 = st.columns(2)
                with col1:
                    sdnn_chart = alt.Chart(hrv_df).mark_line(point=False, color='blue').encode(
                        x=alt.X('Time (minutes)'),
                        y=alt.Y('SDNN', title='SDNN (ms)')
                    ).properties(title="HRV Volatility (SDNN)").interactive()
                    st.altair_chart(sdnn_chart, use_container_width=True)
                
                with col2:
                    lfhf_chart = alt.Chart(hrv_df).mark_line(point=False, color='red').encode(
                        x=alt.X('Time (minutes)'),
                        y=alt.Y('LF/HF', title='LF/HF Ratio')
                    ).properties(title="Autonomic Balance (LF/HF)").interactive()
                    st.altair_chart(lfhf_chart, use_container_width=True)

                with st.expander("Show Raw Data and Predictions"):
                    st.dataframe(pd.concat([hrv_df, results_df.drop(columns='Time (minutes)')], axis=1))

            except Exception as e:
                st.error(f"An error occurred during processing: {e}", icon="⚠️")
                st.error("This file may be corrupted, have no valid ECG channel, or contain a signal that is too noisy for analysis.")
                st.exception(e) 
            
            finally:
                # Always delete the temporary file
                if tmp_file_path is not None and os.path.exists(tmp_file_path):
                    try:
                        os.remove(tmp_file_path)
                    except Exception as e:
                        pass # Ignore cleanup errors
