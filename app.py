import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from scipy.stats import linregress
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
st.set_page_config(
    page_title="NeuroAlert Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --- CONSTANTS ---
# This is our "Sensitivity Dial" found from the V7 tuning step!
FINAL_THRESHOLD = 0.5205
FEATURE_COLUMNS = ['HR', 'MeanNN', 'SDNN', 'RMSSD', 'pNN50', 'SampEn', 
                     'HRV_HTI', 'LF/HF', 'SD1', 'SD2', 'SD1/SD2', 'CSI']
WINDOW_SIZE = 5  # 5 segments = 10 minutes (5 x 2-min windows)

# --- MODEL & SCALER LOADING ---
@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler from disk."""
    try:
        model = joblib.load('neuroalert_final_model_v7.pkl')
        scaler = joblib.load('neuroalert_scaler_v6.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("FATAL ERROR: Model or Scaler file not found.")
        st.error("Please make sure 'neuroalert_final_model_v7.pkl' and 'neuroalert_scaler_v6.pkl' are in your GitHub repository.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading files: {e}")
        st.error("Ensure the files are not corrupted and are accessible in the repository.")
        return None, None


model, scaler = load_model_and_scaler()

# --- HELPER FUNCTION: V6 FEATURE ENGINEERING ---
def create_temporal_features(df):
    """
    Takes the V2 (12-feature) dataframe and creates the V6 (36-feature)
    dataframe by calculating rolling statistics.
    """
    st.write("Processing data: Calculating 10-minute temporal features...")
    df_out = df.copy()
    
    # We must group by patient to prevent data leakage between patients
    grouped = df_out.groupby('patient')
    new_features_list = []
    
    # Create a progress bar for the user
    progress_bar = st.progress(0)
    total_patients = len(grouped)
    
    for i, (patient_id, patient_df) in enumerate(grouped):
        temp_df = patient_df.copy()
        for col in FEATURE_COLUMNS:
            # 1. Rolling Mean
            temp_df[f'{col}_mean_{WINDOW_SIZE}'] = temp_df[col].rolling(window=WINDOW_SIZE, min_periods=1).mean()
            # 2. Rolling Standard Deviation (Volatility)
            temp_df[f'{col}_std_{WINDOW_SIZE}'] = temp_df[col].rolling(window=WINDOW_SIZE, min_periods=1).std()
            # 3. Rolling Slope (Trend)
            rolling_slope = temp_df[col].rolling(window=WINDOW_SIZE, min_periods=1).apply(
                lambda x: linregress(np.arange(len(x)), x).slope if len(x) > 1 else 0,
                raw=False
            )
            temp_df[f'{col}_slope_{WINDOW_SIZE}'] = rolling_slope
        
        new_features_list.append(temp_df)
        progress_bar.progress((i + 1) / total_patients)
    
    # Combine back together
    df_out = pd.concat(new_features_list)
    
    # Clean up NaNs from rolling/slope functions and 'LF/HF'
    df_out = df_out.fillna(0)
    df_out = df_out.replace([np.inf, -np.inf], 0)
    
    # Define the final 36 feature columns
    temporal_features = [col for col in df_out.columns if '_mean_' in col or '_std_' in col or '_slope_' in col]
    
    progress_bar.empty() # Clear the progress bar
    return df_out, temporal_features

# --- UI LAYOUT ---
st.title("🧠 NeuroAlert: Real-Time Seizure Prediction Dashboard")
st.markdown(f"""
This dashboard simulates the NeuroAlert system in real-time. It uses our final V7 model, which analyzes 10-minute trends in 12 HRV biomarkers to provide a risk score.
- **Model:** `EasyEnsembleClassifier` (V7)
- **Data:** CHB-MIT Scalp EEG Database (ECG Channel)
- **Core Features:** 36 temporal features (10-min rolling mean, std, and slope)
- **Recall (Sensitivity):** **~71%** (Proven ability to detect 24/34 seizures)
- **Alert Threshold:** `{FINAL_THRESHOLD}` (Alerts if confidence is > 52.05%)
""")

# --- 1. FILE UPLOADER ---
uploaded_file = st.file_uploader(
    "Upload Test Feed (Requires 'neuroalert_final_dataset_v2.csv')",
    type="csv"
)

if uploaded_file is not None and model is not None:
    data = pd.read_csv(uploaded_file)
    st.success(f"Loaded test feed '{uploaded_file.name}' with {len(data)} 2-minute segments.")

    # --- 2. PRE-PROCESSING STEP ---
    with st.spinner("Analyzing patient data and calculating temporal features... This may take a moment."):
        v6_data, temporal_features = create_temporal_features(data)
        X_live = v6_data[temporal_features]
        X_live_scaled = scaler.transform(X_live)
        
        # Store true labels for comparison
        y_true_labels = v6_data['label'].values
        patient_ids = v6_data['patient'].values

    st.success("Patient data processed. Ready for simulation.")
    
    # --- 3. SIMULATION CONTROLS ---
    if 'simulation_started' not in st.session_state:
        st.session_state.simulation_started = False
    
    start_button = st.button("Start Real-Time Simulation")

    if start_button:
        st.session_state.simulation_started = True

    if st.session_state.simulation_started:
        st.subheader("🔴 Live Simulation Feed")
        
        # --- 4. DASHBOARD LAYOUT ---
        col_status, col_confidence = st.columns([2, 1])
        
        with col_status:
            status_placeholder = st.empty()
        
        with col_confidence:
            confidence_placeholder = st.empty()

        st.markdown("---")
        
        # --- 5. LIVE CHARTS ---
        st.markdown("##### Live Biomarker Trends (10-min rolling window)")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.text("HRV Volatility (Rolling RMSSD)")
            rmssd_chart_placeholder = st.empty()
        
        with chart_col2:
            st.text("Autonomic Balance (Rolling LF/HF Ratio)")
            lfhf_chart_placeholder = st.empty()
        
        # Initialize chart data
        chart_data_rmssd = pd.DataFrame(columns=['Time', 'RMSSD_mean_5'])
        chart_data_lfhf = pd.DataFrame(columns=['Time', 'LF/HF_mean_5'])

        # --- 6. SIMULATION LOOP ---
        # We iterate through our processed data, row by row
        for i in range(len(X_live_scaled)):
            # Get the features for this 2-minute window
            current_features = X_live_scaled[i].reshape(1, -1)
            
            # --- PREDICTION ---
            # Get the probability from the model
            probability = model.predict_proba(current_features)[0][1]
            # Apply our custom threshold
            prediction = 1 if probability >= FINAL_THRESHOLD else 0
            
            # Get true label for comparison
            true_label = y_true_labels[i]
            
            # --- UPDATE DASHBOARD ---
            
            # 1. Update Status Box
            if prediction == 1:
                status_placeholder.error(f"""
                ## 🚨 !!! PRE-ICTAL ALERT !!! 🚨
                **Patient:** `{patient_ids[i]}` (Segment {i})
                
                **Model has detected a high-risk pattern consistent with a pre-ictal state.**
                """)
            else:
                status_placeholder.success(f"""
                ## ✅ STATUS: NORMAL
                **Patient:** `{patient_ids[i]}` (Segment {i})
                
                **All biomarkers are within normal parameters.**
                """)
            
            # 2. Update Confidence Gauge
            confidence_placeholder.metric(
                label="Risk Score (Seizure Confidence)",
                value=f"{(probability * 100):.2f}%",
                delta=f"{(probability - FINAL_THRESHOLD):.2f} vs. Threshold"
            )

            # 3. Update Charts
            # We add new data and keep the last 30 points (1 hour)
            new_rmssd_data = pd.DataFrame({'Time': [i], 'RMSSD_mean_5': [v6_data['RMSSD_mean_5'].iloc[i]]})
            new_lfhf_data = pd.DataFrame({'Time': [i], 'LF/HF_mean_5': [v6_data['LF/HF_mean_5'].iloc[i]]})
            
            chart_data_rmssd = pd.concat([chart_data_rmssd, new_rmssd_data]).tail(30)
            chart_data_lfhf = pd.concat([chart_data_lfhf, new_lfhf_data]).tail(30)

            rmssd_chart_placeholder.line_chart(chart_data_rmssd, x='Time', y='RMSSD_mean_5')
            lfhf_chart_placeholder.line_chart(chart_data_lfhf, x='Time', y='LF/HF_mean_5')
            
            # --- 7. DEBUG / GROUND TRUTH ---
            with st.expander("Show Ground Truth (For Hackathon Demo)"):
                if true_label == 1:
                    st.warning(f"**GROUND TRUTH:** This segment was **ACTUALLY PRE-ICTAL**. ")
                else:
                    st.info(f"**GROUND TRUTH:** This segment was **ACTUALLY NORMAL**.")
                
                st.write(f"Model Prediction: {prediction}, True Label: {true_label}")
                if prediction == 1 and true_label == 1:
                    st.success("Result: TRUE POSITIVE (Correct Alert!)")
                elif prediction == 1 and true_label == 0:
                    st.error("Result: FALSE POSITIVE (False Alarm)")
                elif prediction == 0 and true_label == 1:
                    st.error("Result: FALSE NEGATIVE (Missed Seizure!)")
                elif prediction == 0 and true_label == 0:
                    st.success("Result: TRUE NEGATIVE (Correctly Normal)")

            # Pause to simulate real-time
            time.sleep(0.5) # Speed up simulation for demo
            
        st.session_state.simulation_started = False
        st.balloons()
        st.success("Simulation Complete!")
        
# --- Footer ---
st.markdown("---")
st.markdown("*This prototype was developed by Rijjul Garg (Medical Lead) and Parth Kapoor (Tech Lead).*")

