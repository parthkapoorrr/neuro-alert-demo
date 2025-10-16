import streamlit as st
import pandas as pd
import joblib
import io
import mne
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(page_title="NeuroAlert", page_icon="🧠", layout="wide")

# --- Asset Loading ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('neuroalert_model_tuned.pkl')
        master_df = pd.read_csv('neuroalert_dataset.csv')
        return model, master_df
    except FileNotFoundError:
        return None, None

model, master_df = load_assets()

# ======================================================================================
# PAGE FUNCTIONS
# ======================================================================================

def file_upload_page():
    st.title("Analyze New ECG Recording")
    st.markdown("Upload an `.edf` file containing an ECG channel to predict seizure risk for the first 60 seconds.")

    uploaded_file = st.file_uploader("Choose an .edf file", type="edf")

    if uploaded_file is not None:
        if model is None:
            st.error("Model file ('neuroalert_model_tuned.pkl') not found in the repository. Please upload it.")
            return

        try:
            # Read the uploaded file in memory
            raw = mne.io.read_raw_edf(io.BytesIO(uploaded_file.read()), preload=True, verbose='error')
            sampling_rate = int(raw.info['sfreq'])
            
            ecg_channel = find_ecg_channel(raw.info['ch_names'])
            if not ecg_channel:
                st.error("Could not find a standard ECG channel (e.g., 'ECG', 'T8-P8-0') in this file.")
                return

            st.success(f"File loaded. Using ECG channel: '{ecg_channel}'")
            ecg_signal = raw.get_data(picks=[ecg_channel])[0]
            segment_ecg = ecg_signal[:60 * sampling_rate]

            # --- Feature Extraction ---
            df_feat, info = nk.ecg_process(segment_ecg, sampling_rate=sampling_rate)
            hrv_features = nk.hrv(info['ECG_R_Peaks'], sampling_rate=sampling_rate)
            
            features = {
                'HR': np.mean(df_feat['ECG_Rate']),
                'LF_HF_Ratio': hrv_features['HRV_LFHF'].iloc[0],
                'SD1': hrv_features['HRV_SD1'].iloc[0],
                'SD2': hrv_features['HRV_SD2'].iloc[0],
                'SD2_SD1_Ratio': hrv_features['HRV_SD2'].iloc[0] / hrv_features['HRV_SD1'].iloc[0] if hrv_features['HRV_SD1'].iloc[0] != 0 else 0
            }
            features_df = pd.DataFrame([features])

            st.write("Extracted Features (first 60s):")
            st.dataframe(features_df)

            # --- Prediction ---
            prediction = model.predict(features_df)[0]
            prediction_proba = model.predict_proba(features_df)[0]

            st.markdown("---")
            st.header("Prediction Result")
            if prediction == 1:
                st.error(f"STATUS: SEIZURE RISK DETECTED (Confidence: {prediction_proba[1]:.2%})")
            else:
                st.success(f"STATUS: NORMAL (Confidence: {prediction_proba[0]:.2%})")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

def clinical_analysis_page():
    st.title("Clinical Biomarker Validation")
    if master_df is None:
        st.error("Dataset ('neuroalert_dataset.csv') not found in the repository. Please upload it.")
        return
        
    st.markdown("Biomarker distributions for Normal vs. Pre-ictal states from the training data.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(ax=axes[0], x='label', y='HR', data=master_df).set(title='Heart Rate Distribution', xticklabels=['Normal', 'Pre-ictal'])
    sns.boxplot(ax=axes[1], x='label', y='LF_HF_Ratio', data=master_df).set(title='HRV (LF/HF Ratio) Distribution', xticklabels=['Normal', 'Pre-ictal'])
    st.pyplot(fig)

# --- Helper Function ---
def find_ecg_channel(channel_list):
    possible_names = ['ECG', 'T8-P8-0', 'T8-P8-1', 'T8-P8', 'P8-O2']
    for name in possible_names:
        if name in channel_list: return name
    return None

# ======================================================================================
# MAIN APP LOGIC & NAVIGATION
# ======================================================================================
st.sidebar.title("App Navigation")
page = st.sidebar.selectbox("Choose a page", ["Analyze New File", "Clinical Analysis"])

if page == "Analyze New File":
    file_upload_page()
elif page == "Clinical Analysis":
    clinical_analysis_page()
