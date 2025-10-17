import streamlit as st
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
import joblib
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="NeuroAlert",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 1. Helper Functions (From our Colab Notebook) ---
# These functions are now part of the app

@st.cache_resource
def load_model():
    """Loads the trained XGBoost model."""
    try:
        model = joblib.load('neuroalert_final_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file ('neuroalert_final_model.pkl') not found. Please upload it to the GitHub repo.")
        return None

def find_ecg_channel(channel_list):
    """Finds the correct ECG channel from a list of possible names."""
    possible_names = ['T8-P8-0', 'T8-P8-1', 'T8-P8', 'P8-O2', 'ECG']
    for name in possible_names:
        if name in channel_list:
            return name
    return None

def extract_features_from_signal(segment_ecg, sampling_rate):
    """
    Extracts our 12 key biomarkers from a raw ECG signal segment.
    Returns a dictionary of features or None if processing fails.
    """
    try:
        df_feat, info = nk.ecg_process(segment_ecg, sampling_rate=sampling_rate)
        
        # Quality check: Need at least 20 heartbeats to get reliable HRV
        if len(info['ECG_R_Peaks']) < 20:
            st.write(f"Skipping segment: Not enough R-peaks found ({len(info['ECG_R_Peaks'])}).")
            return None
            
        hrv_features = nk.hrv(info['ECG_R_Peaks'], sampling_rate=sampling_rate)
        
        # Calculate SD1/SD2 ratio safely
        sd1 = hrv_features['HRV_SD1'].iloc[0]
        sd2 = hrv_features['HRV_SD2'].iloc[0]
        sd1_sd2_ratio = sd1 / sd2 if sd2 != 0 else 0

        features = {
            'HR': np.mean(df_feat['ECG_Rate']),
            'MeanNN': hrv_features['HRV_MeanNN'].iloc[0],
            'SDNN': hrv_features['HRV_SDNN'].iloc[0],
            'RMSSD': hrv_features['HRV_RMSSD'].iloc[0],
            'pNN50': hrv_features['HRV_pNN50'].iloc[0],
            'SampEn': hrv_features['HRV_SampEn'].iloc[0],
            'HRV_HTI': hrv_features['HRV_HTI'].iloc[0],
            'LF/HF': hrv_features['HRV_LFHF'].iloc[0],
            'SD1': sd1,
            'SD2': sd2,
            'SD1/SD2': sd1_sd2_ratio,
            'CSI': hrv_features['HRV_CSI'].iloc[0],
        }
        return features
        
    except Exception as e:
        st.write(f"Error during feature extraction: {e}")
        return None

# --- 2. Page Definitions ---

def research_page():
    """Page 1: Displays the research and medical foundation."""
    st.title("Our Research: The 10-Minute Warning 💓")
    st.markdown("This page details the medical science behind NeuroAlert's predictive technology.")

    st.header("The Pre-Ictal Phase: A Window of Opportunity")
    st.markdown(
        """
        An epileptic seizure is not an instant event. For many patients, it is preceded by a distinct physiological state 
        known as the **pre-ictal phase**. This phase can begin minutes to hours before the observable seizure.
        
        During this window, the body's **Autonomic Nervous System (ANS)**, which controls involuntary functions like
        heart rate and breathing, becomes dysregulated.
        """
    )

    st.header("Heart Rate Variability (HRV) as a Biomarker")
    st.markdown(
        """
        Instead of trying to detect the seizure in the brain (with EEG), our approach is to detect the *body's reaction* to the pre-ictal phase. We do this by analyzing **Heart Rate Variability (HRV)** from a simple ECG signal.
        
        HRV is not just the heart rate, but the tiny, millisecond-level variations *between* heartbeats. A healthy
        heart is not a perfect metronome; it has high variability. During the pre-ictal phase, this variability
        changes in predictable ways.
        
        **Our model was trained on 12 key HRV-derived biomarkers:**
        - **Time-Domain:** MeanNN, SDNN, RMSSD, pNN50
        - **Frequency-Domain:** LF/HF ratio
        - **Non-Linear (Poincaré):** SD1, SD2, SD1/SD2 ratio, CSI, HRV_HTI
        - **Entropy:** SampEn
        - **Basic:** Heart Rate (HR)
        
        By feeding these 12 biomarkers from 2-minute windows of ECG data into an XGBoost machine learning model,
        NeuroAlert learns the complex "signature" of the pre-ictal phase.
        """
    )
    
    st.success("Our system is designed to identify this signature and provide a **10-minute warning** *before* the seizure begins.")


def analysis_page():
    """Page 2: The analysis tool for new files."""
    st.title("Analyze New Patient File 🔬")
    st.markdown("Upload a new `.edf` file containing an ECG channel to begin analysis.")

    model = load_model()
    if model is None:
        st.stop()

    uploaded_file = st.file_uploader("Upload an .edf file", type=["edf"])

    if uploaded_file is not None:
        st.info("File uploaded. Reading and processing... This may take a moment.")
        
        # --- THIS IS THE FIX for the BytesIO error ---
        # We pass the 'uploaded_file' object directly to MNE.
        # We do not save it to disk.
        try:
            raw = mne.io.read_raw_edf(uploaded_file, preload=True, verbose='error')
        except Exception as e:
            st.error(f"Error reading .edf file: {e}")
            st.stop()

        sampling_rate = int(raw.info['sfreq'])
        ecg_channel = find_ecg_channel(raw.info['ch_names'])

        if not ecg_channel:
            st.error(f"Could not find a valid ECG channel in this file. Looked for: {['T8-P8-0', 'T8-P8-1', 'T8-P8', 'P8-O2', 'ECG']}")
            st.stop()
        
        st.success(f"Found ECG channel: '{ecg_channel}' with sampling rate: {sampling_rate} Hz.")
        ecg_signal = raw.get_data(picks=[ecg_channel])[0]
        
        # --- Analysis Loop ---
        SEGMENT_DURATION_SECS = 120 # Our 2-minute window
        segment_length_samples = SEGMENT_DURATION_SECS * sampling_rate
        total_seconds = len(ecg_signal) / sampling_rate
        
        st.write(f"File duration: {total_seconds:.0f} seconds. Analyzing in {SEGMENT_DURATION_SECS}-second segments...")
        
        all_features_list = []
        for i in range(0, len(ecg_signal) - segment_length_samples, segment_length_samples):
            segment_start_time_sec = i / sampling_rate
            segment_ecg = ecg_signal[i:i + segment_length_samples]
            
            # Extract features from this 2-minute segment
            features = extract_features_from_signal(segment_ecg, sampling_rate)
            
            if features:
                features['timestamp_sec'] = segment_start_time_sec
                all_features_list.append(features)
        
        if not all_features_list:
            st.error("No valid data segments could be processed from this file. The file may be too noisy or too short.")
            st.stop()

        # --- Prediction ---
        st.subheader("Analysis Complete: Results")
        feature_df = pd.DataFrame(all_features_list)
        
        # Get the feature columns in the exact order the model was trained on
        feature_columns = [
            'HR', 'MeanNN', 'SDNN', 'RMSSD', 'pNN50', 'SampEn',
            'HRV_HTI', 'LF/HF', 'SD1', 'SD2', 'SD1/SD2', 'CSI'
        ]
        X_predict = feature_df[feature_columns]
        
        predictions = model.predict(X_predict)
        probabilities = model.predict_proba(X_predict)[:, 1] # Get probability of class 1
        
        # --- Display Results ---
        results_df = feature_df[['timestamp_sec']].copy()
        results_df['Prediction'] = predictions
        results_df['Confidence'] = probabilities
        
        st.markdown("Below is the prediction for each 2-minute segment of the file:")
        
        for _, row in results_df.iterrows():
            timestamp = time.strftime('%H:%M:%S', time.gmtime(row['timestamp_sec']))
            confidence_pct = row['Confidence'] * 100
            
            if row['Prediction'] == 1:
                st.metric(
                    label=f"Segment starting at: {timestamp}",
                    value="🔴 PRE-ICTAL (WARNING)",
                    delta=f"{confidence_pct:.1f}% Confidence",
                    delta_color="inverse"
                )
            else:
                st.metric(
                    label=f"Segment starting at: {timestamp}",
                    value="🟢 BASELINE (NORMAL)",
                    delta=f"{100-confidence_pct:.1f}% Confidence",
                    delta_color="normal"
                )
        
        with st.expander("Show Raw Feature Data"):
            st.dataframe(results_df)

# --- 3. Main App Navigation ---

# --- 3. Main App Navigation ---

st.sidebar.title("NeuroAlert Navigation")
page = st.sidebar.radio(
    "Go to:",
    ("Analyze New File", "Our Research"),  # <-- REORDERED
    captions=[
        "Analyze a new patient .edf file.", # <-- REORDERED
        "The science behind our project."
    ]
)

if page == "Our Research":
    research_page()
elif page == "Analyze New File":
    analysis_page()
