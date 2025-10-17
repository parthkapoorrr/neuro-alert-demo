import streamlit as st
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
import joblib
import time
import os  # <-- Ensure this is at the top
import requests # <-- For model download

# --- Page Configuration ---
st.set_page_config(
    page_title="NeuroAlert",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 1. Asset Loading (With Download Fix) ---

@st.cache_resource
def load_model():
    """
    Downloads the model from an external source if it's not already present
    and then loads it.
    """
    MODEL_FILE_NAME = 'neuroalert_final_model.pkl'
    # --- PASTE YOUR DIRECT DOWNLOAD LINK HERE ---
    MODEL_DOWNLOAD_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID_HERE" # <-- IMPORTANT

    if not os.path.exists(MODEL_FILE_NAME):
        st.info("Model not found locally. Downloading from remote... ☁️")
        try:
            with requests.get(MODEL_DOWNLOAD_URL, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_FILE_NAME, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return None

    try:
        model = joblib.load(MODEL_FILE_NAME)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 2. Helper Functions (Unchanged) ---

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
        sd1_sd2_ratio = sd1 / sd2 if sd2 != 0 else 0 # <-- Variable defined here

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
            'SD1/SD2': sd1_sd2_ratio, # <-- FIX: Changed 'sd1_sd2_route' to 'sd1_sd2_ratio'
            'CSI': hrv_features['HRV_CSI'].iloc[0],
        }
        return features
        
    except Exception as e:
        # This will now correctly show the NameError from the line above if it fails
        st.error(f"Error during feature extraction: {e}")
        return None

# --- 3. Page Definitions ---

def research_page():
    """Page 1: Displays the research and medical foundation."""
    st.title("Our Research: The 10-Minute Warning 💓")
    st.markdown("This page details the medical science behind NeuroAlert's predictive technology.")
    # (Rest of the research page content is unchanged)
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
        st.info(f"File '{uploaded_file.name}' uploaded. Click button to analyze.")
        
        if st.button("Analyze Full Recording"):
            
            # --- THIS IS THE CORRECTED FILE-READING LOGIC (FROM YOUR OLD CODE) ---
            # This code should be inside the 'if st.button("Analyze Full Recording"):' block

            temp_file_path = None # Initialize variable
            try:
                # 1. Save the uploaded file to a temporary location on disk
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                temp_file_path = uploaded_file.name
                
                # 2. Now, pass the FILENAME (a string path) to mne.
                raw = mne.io.read_raw_edf(temp_file_path, preload=True, verbose='error')

                sampling_rate = int(raw.info['sfreq'])
                ecg_channel = find_ecg_channel(raw.info['ch_names'])

                if not ecg_channel:
                    st.error(f"Could not find a valid ECG channel in this file. Looked for: {['T8-P8-0', 'T8-P8-1', 'T8-P8', 'P8-O2', 'ECG']}")
                    st.stop()
                
                st.success(f"Found ECG channel: '{ecg_channel}'. Analyzing...")
                ecg_signal = raw.get_data(picks=[ecg_channel])[0]
                
                # --- Analysis Loop ---
                SEGMENT_DURATION_SECS = 120
                segment_length_samples = SEGMENT_DURATION_SECS * sampling_rate
                
                all_features_list = []
                for i in range(0, len(ecg_signal) - segment_length_samples, segment_length_samples):
                    segment_start_time_sec = i / sampling_rate
                    segment_ecg = ecg_signal[i:i + segment_length_samples]
                    features = extract_features_from_signal(segment_ecg, sampling_rate)
                    if features:
                        features['timestamp_sec'] = segment_start_time_sec
                        all_features_list.append(features)
                
                if not all_features_list:
                    st.error("No valid data segments could be processed. File may be too noisy or short.")
                    st.stop()

                # --- Prediction (Code from before) ---
                # --- Prediction ---
                st.subheader("Analysis Complete: Results")
                feature_df = pd.DataFrame(all_features_list)
                feature_columns = [
                    'HR', 'MeanNN', 'SDNN', 'RMSSD', 'pNN50', 'SampEn',
                    'HRV_HTI', 'LF/HF', 'SD1', 'SD2', 'SD1/SD2', 'CSI'
                ]
                
                # We need a dataframe with just these columns for prediction
                X_predict = feature_df[feature_columns]
                
                predictions = model.predict(X_predict)
                probabilities = model.predict_proba(X_predict)[:, 1]
                
                # Create the results_df for metrics and the final expander
                results_df = feature_df[['timestamp_sec']].copy()
                results_df['Prediction'] = predictions
                results_df['Confidence'] = probabilities
                
                # --- 1. FINAL VERDICT ---
                st.subheader("Final Summary Verdict", divider="rainbow")
                
                prediction_counts = results_df['Prediction'].value_counts()
                normal_count = prediction_counts.get(0, 0)
                pre_ictal_count = prediction_counts.get(1, 0)
                total_segments = len(results_df)

                st.write(f"**Analysis Breakdown:**")
                st.write(f"- **Normal Segments:** {normal_count} out of {total_segments}")
                st.write(f"- **Pre-ictal Segments:** {pre_ictal_count} out of {total_segments}")

                if pre_ictal_count > normal_count:
                    st.error(
                        f"**Verdict: 🔴 PRE-ICTAL RISK DETECTED**\n\n"
                        f"The majority of analyzed segments ({pre_ictal_count}/{total_segments}) were "
                        f"flagged as pre-ictal. Please review."
                    )
                else:
                    st.success(
                        f"**Verdict: 🟢 MAJORITY NORMAL**\n\n"
                        f"The majority of analyzed segments ({normal_count}/{total_segments}) appear to be in a normal baseline state."
                    )

                # --- 2. NEW: BIOMARKER TABLE ---
                st.subheader("Biomarker Levels per Segment", divider="gray")
                
                # Create a clean dataframe for display
                # We use the original 'feature_df' which has all the biomarker data
                display_df = feature_df.copy()
                
                # Format the timestamp column to be human-readable
                display_df['Timestamp'] = display_df['timestamp_sec'].apply(
                    lambda x: time.strftime('%H:%M:%S', time.gmtime(x))
                )
                
                # Reorder columns so Timestamp is first, then the 12 biomarkers
                display_columns = ['Timestamp'] + feature_columns 
                
                # Display the biomarker data in a table
                st.dataframe(display_df[display_columns])


                # --- 3. SEGMENT-BY-SEGMENT LOG ---
                st.subheader("Segment-by-Segment Log", divider="gray")

                # This loop uses 'results_df' to show the individual verdicts
                for _, row in results_df.iterrows():
                    timestamp = time.strftime('%H:%M:%S', time.gmtime(row['timestamp_sec']))
                    confidence_pct = row['Confidence'] * 100
                    if row['Prediction'] == 1:
                        st.metric(
                            label=f"Segment at: {timestamp}", value="🔴 PRE-ICTAL (WARNING)",
                            delta=f"{confidence_pct:.1f}% Confidence", delta_color="inverse"
                        )
                    else:
                        st.metric(
                            label=f"Segment at: {timestamp}", value="🟢 BASELINE (NORMAL)",
                            delta=f"{100-confidence_pct:.1f}% Confidence", delta_color="normal"
                        )
                
                # --- 4. RAW PREDICTION DATA (Expander) ---
                # I've renamed this expander to be clearer
                with st.expander("Show Raw Prediction Data (Timestamp, Prediction, Confidence)"):
                    st.dataframe(results_df)

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
            
            finally:
                # --- Cleanup ---
                # This ensures the temporary file is deleted even if an error occurs
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

# --- 4. Main App Navigation (Set to your preference) ---

st.sidebar.title("NeuroAlert Navigation")
page = st.sidebar.radio(
    "Go to:",
    ("Analyze New File", "Our Research"),  # <-- "Analyze" is the default page
    captions=[
        "Analyze a new patient .edf file.",
        "The science behind our project."
    ]
)

if page == "Our Research":
    research_page()
elif page == "Analyze New File":
    analysis_page()
