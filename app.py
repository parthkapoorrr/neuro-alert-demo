import streamlit as st
import pandas as pd
import joblib
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import io
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(page_title="NeuroAlert", page_icon="🧠", layout="wide")
plt.style.use('dark_background')

# --- Asset Loading ---
try:
    model = joblib.load('neuroalert_realistic_model.pkl')
    sim_df = pd.read_csv('simulation_data.csv')
    master_df = pd.read_csv('neuroalert_dataset.csv')
except FileNotFoundError:
    st.error("Required data files not found. Ensure all .pkl and .csv files are in the GitHub repository.")
    st.stop()

# ======================================================================================
# PAGE FUNCTIONS
# ======================================================================================

def live_monitor_page():
    st.title("NeuroAlert: Live Patient Monitor")
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### Physiological Signals")
        chart_container = st.container()
        st.markdown("---")
        st.header("AI STATUS")
        status_placeholder = st.empty()

    with col2:
        st.markdown("### Simulation Control")
        playback_speed = st.selectbox("Playback Speed (x real-time)", [1, 60, 360, 720], index=2)

    # Placeholders for charts
    c1, c2, c3 = chart_container.columns(3)
    chart1_placeholder = c1.empty()
    chart2_placeholder = c2.empty()
    chart3_placeholder = c3.empty()
    
    # Simulation data
    hr_data, hrv_data, eda_data = [], [], []
    SEIZURE_START_SEC = 1238
    PRE_ICTAL_START_SEC = SEIZURE_START_SEC - (10 * 60)

    for index, row in sim_df.iterrows():
        current_features = row[['HR', 'LF_HF_Ratio']].to_frame().T
        prediction = model.predict(current_features)[0]
        
        # THE CRITICAL TRIGGER
        if prediction == 1:
            st.session_state.page = 'alert'
            st.rerun() # Immediately stop and rerun the script to show the alert page

        sim_time = datetime.timedelta(seconds=int(row['timestamp']))
        
        with status_placeholder.container():
            st.subheader(f"Patient Time Elapsed: {sim_time}")
            st.success("STATUS: NORMAL")

        hr_data.append(row['HR']); hrv_data.append(row['LF_HF_Ratio']); eda_data.append(row['EDA_Mean'])
        if len(hr_data) > 30:
            hr_data.pop(0); hrv_data.pop(0); eda_data.pop(0)

        chart1_placeholder.image(create_plot_image(hr_data, "Heart Rate (HR)"), use_container_width=True)
        chart2_placeholder.image(create_plot_image(hrv_data, "HRV (LF/HF)"), use_container_width=True)
        chart3_placeholder.image(create_plot_image(eda_data, "EDA (Simulated)"), use_container_width=True)
        
        time.sleep(1 / playback_speed)

    st.balloons()

def alert_page():
    st.title("🚨 SEIZURE RISK DETECTED 🚨")
    timer_placeholder = st.empty()
    st.markdown("---")

    for seconds in range(10, 0, -1): # Short 10s timer for the demo
        timer_placeholder.header(f"**00:{seconds:02d}** MINUTE WARNING")
        time.sleep(1)
    timer_placeholder.header("ALERT TIME OVER")

    st.subheader("Please confirm the event:")
    col1, col2 = st.columns(2)
    if col1.button("✅ Confirmed Seizure"):
        st.success("Feedback recorded. We hope you are safe.")
    if col2.button("❌ Dismissed - False Alarm"):
        st.warning("Feedback recorded. Your algorithm will be adjusted.")

def clinical_analysis_page():
    st.title("Clinical Biomarker Validation")
    st.markdown("These plots show the biomarker distributions for Normal vs. Pre-ictal states from the training data.")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(ax=axes[0], x='label', y='HR', data=master_df).set(title='Heart Rate Distribution', xticklabels=['Normal', 'Pre-ictal'])
    sns.boxplot(ax=axes[1], x='label', y='LF_HF_Ratio', data=master_df).set(title='HRV (LF/HF Ratio) Distribution', xticklabels=['Normal', 'Pre-ictal'])
    st.pyplot(fig)

# --- Helper function for plotting ---
def create_plot_image(data, title):
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(data, color='cyan'); ax.set_title(title, fontsize=10)
    ax.tick_params(axis='x', labelsize=6); ax.tick_params(axis='y', labelsize=6)
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); buf.seek(0)
    plt.close(fig)
    return buf

# ======================================================================================
# MAIN APP LOGIC & NAVIGATION
# ======================================================================================
if 'page' not in st.session_state:
    st.session_state.page = 'monitoring'

st.sidebar.title("App Navigation")
page = st.sidebar.selectbox("Choose a page", ["Live Monitor", "Clinical Analysis", "Alert Page (Manual)"])

# Set session state based on sidebar selection
if page == "Live Monitor":
    st.session_state.page = 'monitoring'
elif page == "Clinical Analysis":
    st.session_state.page = 'analysis'
elif page == "Alert Page (Manual)":
    st.session_state.page = 'alert'

# Render the correct page based on session state
if st.session_state.page == 'monitoring':
    live_monitor_page()
elif st.session_state.page == 'alert':
    alert_page()
elif st.session_state.page == 'analysis':
    clinical_analysis_page()
