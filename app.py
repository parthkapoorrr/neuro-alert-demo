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
# This block now loads all necessary files at the start.
try:
    model = joblib.load('neuroalert_realistic_model.pkl')
    sim_df = pd.read_csv('simulation_data.csv')
    master_df = pd.read_csv('neuroalert_dataset.csv') # <-- BUG FIX: This line was added.
except FileNotFoundError:
    st.error("Required data files not found. Ensure app.py, .pkl, and .csv files are all in the GitHub repository.")
    st.stop()

# ======================================================================================
# LIVE MONITOR PAGE FUNCTION
# ======================================================================================
def live_monitor_page():
    # --- UI Layout ---
    st.title("NeuroAlert: Live Patient Monitor")
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### Physiological Signals")
        chart_placeholder_1 = st.empty()
        chart_placeholder_2 = st.empty()
        chart_placeholder_3 = st.empty()
        st.markdown("---")
        st.header("AI STATUS")
        status_placeholder = st.empty()

    with col2:
        st.markdown("### Simulation Control")
        playback_speed = st.selectbox("Playback Speed (x real-time)", [1, 60, 360, 720], index=2)
        st.markdown("---")
        st.markdown("### System Event Log")
        log_placeholder = st.empty()

    # --- Live Simulation ---
    hr_data, hrv_data, eda_data = [], [], []
    event_log = ["System Initialized..."]
    status_placeholder.success("STATUS: NORMAL")
    log_placeholder.text_area("Log", value='\n'.join(event_log), height=200)

    # (Simulation parameters from your data files)
    SEIZURE_START_SEC = 1238
    PRE_ICTAL_WINDOW_MINS = 10
    PRE_ICTAL_START_SEC = SEIZURE_START_SEC - (PRE_ICTAL_WINDOW_MINS * 60)

    for index, row in sim_df.iterrows():
        current_features = row[['HR', 'LF_HF_Ratio']].to_frame().T
        
        if PRE_ICTAL_START_SEC <= row['timestamp'] < SEIZURE_START_SEC:
            eda_value = 1.0 + np.random.uniform(-0.2, 0.2)
        else:
            eda_value = row['EDA_Mean']
        
        prediction = model.predict(current_features)[0]
        
        sim_time = datetime.timedelta(seconds=int(row['timestamp']))
        
        with status_placeholder.container():
            st.subheader(f"Patient Time Elapsed: {sim_time}")
            if prediction == 1:
                st.error("STATUS: SEIZURE RISK DETECTED")
                if "RISK DETECTED" not in event_log[-1]:
                    event_log.append(f"{sim_time}: AI detected potential pre-ictal signature.")
            else:
                st.success("STATUS: NORMAL")
                if "NORMAL" not in event_log[-1]:
                     event_log.append(f"{sim_time}: System status is normal.")

        hr_data.append(row['HR']); hrv_data.append(row['LF_HF_Ratio']); eda_data.append(eda_value)
        if len(hr_data) > 30:
            hr_data.pop(0); hrv_data.pop(0); eda_data.pop(0)

        c1, c2, c3 = st.columns(3)
        c1.image(create_plot_image(hr_data, "Heart Rate (HR)"), use_container_width=True)
        c2.image(create_plot_image(hrv_data, "HRV (LF/HF)"), use_container_width=True)
        c3.image(create_plot_image(eda_data, "EDA (Simulated)"), use_container_width=True)
        
        log_placeholder.text_area("Log", value='\n'.join(event_log), height=200, key=f"log_{index}")
        time.sleep(1 / playback_speed)

    st.balloons()
    event_log.append("Simulation Finished.")
    log_placeholder.text_area("Log", value='\n'.join(event_log), height=200)


# ======================================================================================
# CLINICAL ANALYSIS PAGE FUNCTION
# ======================================================================================
def clinical_analysis_page():
    st.title("Clinical Biomarker Validation")
    st.markdown("These plots show the distribution of key biomarkers for Normal vs. Pre-ictal states, based on the entire training dataset.")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Heart Rate distribution
    sns.boxplot(ax=axes[0], x='label', y='HR', data=master_df)
    axes[0].set_title('Heart Rate Distribution')
    axes[0].set_xticklabels(['Normal (0)', 'Pre-ictal (1)'])
    
    # Plot HRV distribution
    sns.boxplot(ax=axes[1], x='label', y='LF_HF_Ratio', data=master_df)
    axes[1].set_title('HRV (LF/HF Ratio) Distribution')
    axes[1].set_xticklabels(['Normal (0)', 'Pre-ictal (1)'])
    
    st.pyplot(fig)

    # st.markdown("---")
    # st.header("Discussion Points for Medical Lead")
    # st.markdown("""
    # 1.  **Does this data make sense?** Do the differences between the Normal and Pre-ictal groups align with clinical expectations?
    # 2.  **Is our model's trade-off acceptable?** Our AI is very good at catching seizures (high recall) but produces many false alarms. Is this a reasonable starting point for a prototype?
    # 3.  **What other physiological signs could we investigate?** Are there other biomarkers we could extract to improve the model's precision and reduce false alarms?
    # """)

# --- Helper function for plotting ---
def create_plot_image(data, title):
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(data, color='cyan')
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='x', labelsize=6); ax.tick_params(axis='y', labelsize=6)
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); buf.seek(0)
    plt.close(fig)
    return buf

# ======================================================================================
# SIDEBAR NAVIGATION
# ======================================================================================
st.sidebar.title("App Navigation")
page_options = ["Live Monitor", "Clinical Analysis"]
page = st.sidebar.selectbox("Choose a page", page_options)

if page == "Live Monitor":
    live_monitor_page()
elif page == "Clinical Analysis":
    clinical_analysis_page()
