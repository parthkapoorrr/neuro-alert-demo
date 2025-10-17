import streamlit as st
import pandas as pd
import joblib
import io
import mne
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# --- Page Configuration ---
st.set_page_config(page_title="NeuroAlert", page_icon="🧠", layout="wide")

# --- Asset Loading ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('neuroalert_model_2min.pkl')
        master_df = pd.read_csv('neuroalert_dataset_2min_window.csv')
        return model, master_df
    except FileNotFoundError:
        return None, None

model, master_df = load_assets()

# --- Helper Function ---
def find_ecg_channel(channel_list):
    possible_names = ['ECG', 'T8-P8-0', 'T8-P8-1', 'T8-P8', 'P8-O2']
    for name in possible_names:
        if name in channel_list: return name
    return None

# ======================================================================================
# PAGE 1: FILE UPLOAD AND PREDICTION (The Final Product)
# ======================================================================================
def file_upload_page():
    st.title("NeuroAlert: Seizure Risk Prediction")
    st.markdown("Upload a standard `.edf` file with an ECG channel to analyze it for pre-ictal seizure risk, one 2-minute segment at a time.")

    uploaded_file = st.file_uploader("Choose an .edf file", type="edf")

    if uploaded_file is not None:
        if model is None:
            st.error("Model file ('neuroalert_model_2min.pkl') not found in the repository."); return

        st.success(f"File '{uploaded_file.name}' uploaded successfully.")
        
        if st.button("Analyze Full Recording"):
            try:
                raw = mne.io.read_raw_edf(io.BytesIO(uploaded_file.read()), preload=True, verbose='error')
                sampling_rate = int(raw.info['sfreq'])
                ecg_channel = find_ecg_channel(raw.info['ch_names'])
                if not ecg_channel:
                    st.error("Could not find a standard ECG channel in this file."); return

                st.info(f"Analyzing file using ECG channel: '{ecg_channel}'.")
                ecg_signal = raw.get_data(picks=[ecg_channel])[0]
                segment_length = 120 * sampling_rate

                results_placeholder = st.empty()
                results = []
                feature_columns = ['HR', 'MeanNN', 'SDNN', 'RMSSD', 'pNN50', 'SampEn', 'HRV_HTI', 'LF/HF', 'SD1', 'SD2', 'SD1/SD2', 'CSI']

                for i in range(0, len(ecg_signal) - segment_length, segment_length):
                    segment_ecg = ecg_signal[i : i + segment_length]
                    time_stamp = str(datetime.timedelta(seconds=int(i / sampling_rate)))
                    
                    try:
                        df_feat, info = nk.ecg_process(segment_ecg, sampling_rate=sampling_rate)
                        hrv_features = nk.hrv(info['ECG_R_Peaks'], sampling_rate=sampling_rate)
                        
                        features = {
                            'HR': np.mean(df_feat['ECG_Rate']), 'MeanNN': hrv_features['HRV_MeanNN'].iloc[0],
                            'SDNN': hrv_features['HRV_SDNN'].iloc[0], 'RMSSD': hrv_features['HRV_RMSSD'].iloc[0],
                            'pNN50': hrv_features['HRV_pNN50'].iloc[0], 'SampEn': hrv_features['HRV_SampEn'].iloc[0],
                            'HRV_HTI': hrv_features['HRV_HTI'].iloc[0], 'LF/HF': hrv_features['HRV_LFHF'].iloc[0],
                            'SD1': hrv_features['HRV_SD1'].iloc[0], 'SD2': hrv_features['HRV_SD2'].iloc[0],
                            'SD1/SD2': hrv_features['HRV_SD1'].iloc[0] / hrv_features['HRV_SD2'].iloc[0] if hrv_features['HRV_SD2'].iloc[0] != 0 else 0,
                            'CSI': hrv_features['HRV_CSI'].iloc[0]
                        }
                        features_df = pd.DataFrame([features])[feature_columns]
                        prediction = model.predict(features_df)[0]
                        prediction_proba = model.predict_proba(features_df)[0]
                        
                        if prediction == 1:
                            results.append(f"🔴 {time_stamp}: SEIZURE RISK DETECTED (Confidence: {prediction_proba[1]:.1%})")
                        else:
                            results.append(f"🟢 {time_stamp}: Normal (Confidence: {prediction_proba[0]:.1%})")
                        
                        results_placeholder.text_area("Analysis Log", "\n".join(results), height=300)
                        time.sleep(0.1)
                    except Exception:
                        results.append(f"🟡 {time_stamp}: Could not analyze segment (noisy data).")
                        results_placeholder.text_area("Analysis Log", "\n".join(results), height=300)
                        time.sleep(0.1)
                        continue
                st.success("Full file analysis complete.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# ======================================================================================
# PAGE 2: OUR RESEARCH AND FINDINGS
# ======================================================================================
def research_findings_page():
    st.title("Our Research & Findings")
    st.markdown("""
    This project began with a single question: **Can we find a reliable, measurable signal in the body's cardiovascular system that precedes an epileptic seizure?** This page documents our journey from raw clinical data to the discovery of a predictive signature.
    """)

    st.header("Step 1: The Hypothesis")
    st.markdown("""
    Based on medical literature, we hypothesized that the **pre-ictal phase**—the period up to 10 minutes before a seizure—causes dysregulation in the autonomic nervous system. This should be detectable in Heart Rate Variability (HRV) metrics.
    """)

    st.header("Step 2: The Data")
    st.markdown("""
    We processed over 12GB of raw clinical data from the CHB-MIT dataset, extracting 2-minute segments of pure ECG signals from 24 different patients. We labeled these segments as either 'Normal' or 'Pre-ictal'.
    """)
    
    st.header("Step 3: The Findings")
    st.markdown("""
    Our analysis revealed a clear and statistically significant difference in key biomarkers between the two states.
    """)

    if master_df is None:
        st.error("Dataset ('neuroalert_dataset_2min_window.csv') not found."); return
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(ax=axes[0], x='label', y='HR', data=master_df).set(title='Finding #1: Heart Rate Increases', xticklabels=['Normal', 'Pre-ictal'])
    sns.boxplot(ax=axes[1], x='label', y='LF/HF', data=master_df).set(title='Finding #2: HRV Balance Shifts', xticklabels=['Normal', 'Pre-ictal'])
    st.pyplot(fig)
    
    st.header("Step 4: The Conclusion")
    st.success("""
    **Our findings confirmed the hypothesis.** A predictive signature does exist. These results gave us the confidence to build our final AI model, which is now available for you to use on the "Analyze New File" page.
    """)

# ======================================================================================
# MAIN APP NAVIGATION
# ======================================================================================
st.sidebar.title("App Navigation")
page = st.sidebar.selectbox("Choose a page", ["Analyze New File", "Our Research & Findings"])

if page == "Analyze New File":
    file_upload_page()
elif page == "Our Research & Findings":
    research_findings_page()
