import streamlit as st
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
import joblib
import time
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="NeuroAlert",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 1. Asset Loading (Using your first model) ---
@st.cache_resource
def load_model():
    """
    Loads the original trained XGBoost model directly from the repository.
    """
    MODEL_FILE_NAME = 'neuroalert_final_model.pkl'
    try:
        model = joblib.load(MODEL_FILE_NAME)
        return model
    except FileNotFoundError:
        st.error(f"Model file ('{MODEL_FILE_NAME}') not found. Please ensure it is in the GitHub repository.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 2. Helper Functions ---
def find_ecg_channel(channel_list):
    """
    Finds the correct ECG channel, prioritizing 'ECG' over other fallback names.
    Ignores case and whitespace for robustness.
    """
    # Create a mapping of normalized names (uppercase, stripped) to original names
    normalized_channels = {ch.strip().upper(): ch for ch in channel_list}

    # --- 1. PRIORITIZE 'ECG' ---
    # Explicitly check for the 'ECG' channel first.
    if 'ECG' in normalized_channels:
        return normalized_channels['ECG'] # Return the original-cased name

    # --- 2. If 'ECG' is not found, check for other possible fallbacks ---
    fallback_names = ['T8-P8-0', 'T8-P8-1', 'T8-P8', 'P8-O2']
    for name in fallback_names:
        if name.upper() in normalized_channels:
            # If a fallback is found, return its original-cased name
            return normalized_channels[name.upper()]
            
    return None # Return None if no suitable channel is found

def extract_features_from_signal(segment_ecg, sampling_rate):
    """Extracts our 12 key biomarkers from a raw ECG signal segment."""
    try:
        df_feat, info = nk.ecg_process(segment_ecg, sampling_rate=sampling_rate)
        if len(info['ECG_R_Peaks']) < 20:
            return None
        hrv_features = nk.hrv(info['ECG_R_Peaks'], sampling_rate=sampling_rate)
        sd1 = hrv_features['HRV_SD1'].iloc[0]
        sd2 = hrv_features['HRV_SD2'].iloc[0]
        sd1_sd2_ratio = sd1 / sd2 if sd2 != 0 else 0
        features = {
            'HR': np.mean(df_feat['ECG_Rate']), 'MeanNN': hrv_features['HRV_MeanNN'].iloc[0],
            'SDNN': hrv_features['HRV_SDNN'].iloc[0], 'RMSSD': hrv_features['HRV_RMSSD'].iloc[0],
            'pNN50': hrv_features['HRV_pNN50'].iloc[0], 'SampEn': hrv_features['HRV_SampEn'].iloc[0],
            'HRV_HTI': hrv_features['HRV_HTI'].iloc[0], 'LF/HF': hrv_features['HRV_LFHF'].iloc[0],
            'SD1': sd1, 'SD2': sd2, 'SD1/SD2': sd1_sd2_ratio, 'CSI': hrv_features['HRV_CSI'].iloc[0],
        }
        return features
    except Exception:
        return None

# --- 3. Page Definitions ---
def research_page():
    """Displays the research, medical foundation, and team info."""
    st.title("Our Research: The 10-Minute Warning 💓")
    st.markdown("This page details the medical science behind NeuroAlert's predictive technology.")

    st.header("The Pre-Ictal Phase: A Window of Opportunity")
    st.markdown("""
        An epileptic seizure is not an instant event. For many patients, it is preceded by a distinct physiological state 
        known as the **pre-ictal phase**. This phase can begin minutes to hours before the observable seizure. During this window, the body's **Autonomic Nervous System (ANS)**, which controls involuntary functions like heart rate, becomes dysregulated.
        """)

    st.header("ECG as the Source for ANS Biomarkers")
    st.markdown("""
        Instead of detecting the seizure in the brain (EEG), our approach is to detect the *body's reaction* to the pre-ictal phase by analyzing the **ECG (Electrocardiogram) signal**. From the ECG, we can measure both **Heart Rate (HR)** and **Heart Rate Variability (HRV)**.
        
        HRV is not just the heart rate, but the tiny variations *between* heartbeats. During the pre-ictal phase, this variability changes in predictable ways.
        
        **Our model was trained on 12 key biomarkers, all derived from the ECG signal:**
        - **Basic:** Heart Rate (HR)
        - **HRV Time-Domain:** MeanNN, SDNN, RMSSD, pNN50
        - **HRV Frequency-Domain:** LF/HF ratio
        - **HRV Non-Linear (Poincaré):** SD1, SD2, SD1/SD2 ratio, CSI, HRV_HTI
        - **HRV Entropy:** SampEn
        
        By feeding these biomarkers from 2-minute windows of ECG data into an XGBoost machine learning model, NeuroAlert learns the complex "signature" of the pre-ictal phase.
        """)
    st.success("Our system is designed to identify this signature and provide a **10-minute warning** *before* the seizure begins.")

    st.header("The NeuroAlert Team")
    st.markdown("""
        This project is a collaboration between technology and medicine:
        - **Medical Lead:** Rijjul Garg
        - **Tech Lead:** Parth Kapoor
        
        
        Our unique combination of expertise allows us to build a tool that is not only
        technically advanced but also medically sound.
        """)

def analysis_page():
    """The analysis tool for new files."""
    st.title("Analyze New Patient File 🔬")
    st.markdown("Upload a new `.edf` file containing an ECG channel to begin analysis.")

    model = load_model()
    if model is None:
        st.stop()

    uploaded_file = st.file_uploader("Upload an .edf file", type=["edf"])

    if uploaded_file is not None:
        st.info(f"File '{uploaded_file.name}' uploaded. Click button to analyze.")
        
        if st.button("Analyze Full Recording"):
            with st.spinner('Analyzing... 💓 This may take a moment for large files.'):
                temp_file_path = None
                try:
                    with open(uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_file_path = uploaded_file.name
                    
                    raw = mne.io.read_raw_edf(temp_file_path, preload=True, verbose='error')
                    sampling_rate = int(raw.info['sfreq'])
                    ecg_channel = find_ecg_channel(raw.info['ch_names'])

                    if not ecg_channel:
                        st.error(f"Could not find a valid ECG channel. Looked for: {['ECG', 'T8-P8-0', 'T8-P8-1', 'T8-P8', 'P8-O2']}")
                        st.stop()
                    
                    st.success(f"Found and using signal from channel: '{ecg_channel}'.")
                    ecg_signal = raw.get_data(picks=[ecg_channel])[0]

                    # --- DYNAMIC PLOT TITLE ---
                    st.subheader(f"Raw Signal from Channel: '{ecg_channel}' (First 10 Seconds)", divider="gray")
                    plot_seconds = 10
                    plot_samples = int(plot_seconds * sampling_rate)
                    ecg_plot_df = pd.DataFrame({
                        "Time (s)": np.linspace(0, plot_seconds, plot_samples),
                        "Signal (μV)": ecg_signal[:plot_samples] # Changed y-axis label to be more generic
                    })
                    st.line_chart(ecg_plot_df.set_index("Time (s)"))
                    
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

                    st.subheader("Analysis Complete: Results", divider="rainbow")
                    feature_df = pd.DataFrame(all_features_list)
                    feature_columns = ['HR', 'MeanNN', 'SDNN', 'RMSSD', 'pNN50', 'SampEn', 'HRV_HTI', 'LF/HF', 'SD1', 'SD2', 'SD1/SD2', 'CSI']
                    X_predict = feature_df[feature_columns]
                    
                    predictions = model.predict(X_predict)
                    probabilities = model.predict_proba(X_predict)[:, 1]
                    
                    results_df = feature_df[['timestamp_sec']].copy()
                    results_df['Prediction'] = predictions
                    results_df['Confidence'] = probabilities
                    
                    st.subheader("Final Summary Verdict", divider="gray")
                    prediction_counts = results_df['Prediction'].value_counts()
                    normal_count = prediction_counts.get(0, 0)
                    pre_ictal_count = prediction_counts.get(1, 0)
                    total_segments = len(results_df)

                    st.write(f"**Analysis Breakdown:**")
                    st.metric("Normal Segments", f"{normal_count}/{total_segments}")
                    st.metric("Pre-ictal Segments", f"{pre_ictal_count}/{total_segments}")

                    if pre_ictal_count > normal_count:
                        st.error(f"**Verdict: 🔴 PRE-ICTAL RISK DETECTED**\nThe majority of segments ({pre_ictal_count}/{total_segments}) were flagged as pre-ictal.")
                    else:
                        st.success(f"**Verdict: 🟢 MAJORITY NORMAL**\nThe majority of segments ({normal_count}/{total_segments}) appear normal.")

                    st.subheader("Biomarker Levels per Segment", divider="gray")
                    display_df = feature_df.copy()
                    display_df['Timestamp'] = display_df['timestamp_sec'].apply(lambda x: time.strftime('%H:%M:%S', time.gmtime(x)))
                    display_columns = ['Timestamp'] + feature_columns
                    st.dataframe(display_df[display_columns])

                    st.subheader("Segment-by-Segment Log", divider="gray")
                    for _, row in results_df.iterrows():
                        timestamp = time.strftime('%H:%M:%S', time.gmtime(row['timestamp_sec']))
                        confidence_pct = row['Confidence'] * 100
                        if row['Prediction'] == 1:
                            st.metric(label=f"Segment at: {timestamp}", value="🔴 PRE-ICTAL (WARNING)", delta=f"{confidence_pct:.1f}% Confidence", delta_color="inverse")
                        else:
                            st.metric(label=f"Segment at: {timestamp}", value="🟢 BASELINE (NORMAL)", delta=f"{100-confidence_pct:.1f}% Confidence", delta_color="normal")
                    
                    report_content = f"--- NeuroAlert Analysis Report ---\n\n"
                    report_content += f"File Name: {uploaded_file.name}\nAnalysis Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    report_content += "--- FINAL VERDICT ---\n"
                    if pre_ictal_count > normal_count:
                        report_content += f"Verdict: PRE-ICTAL RISK DETECTED\nDetails: {pre_ictal_count}/{total_segments} segments flagged as pre-ictal.\n\n"
                    else:
                        report_content += f"Verdict: MAJORITY NORMAL\nDetails: {normal_count}/{total_segments} segments appear normal.\n\n"
                    report_content += "--- BIOMARKER DATA ---\n"
                    report_content += display_df[display_columns].to_string(index=False)
                    
                    st.download_button(
                        label="⬇️ Download Full Report (.txt)",
                        data=report_content,
                        file_name=f"NeuroAlert_Report_{uploaded_file.name}.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                
                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

# --- 4. Main App Navigation ---
st.sidebar.title("NeuroAlert Navigation")
page = st.sidebar.radio(
    "Go to:",
    ("Analyze New File", "Our Research"),
    captions=["Analyze a new patient .edf file.", "The science behind our project."]
)

if page == "Our Research":
    research_page()
elif page == "Analyze New File":
    analysis_page()

