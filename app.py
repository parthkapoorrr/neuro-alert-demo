import streamlit as st
import pandas as pd
import joblib
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import io

# --- Page Configuration and Style ---
st.set_page_config(page_title="NeuroAlert Live Monitor", page_icon="🧠", layout="wide")
# Use a dark theme for Matplotlib charts
plt.style.use('dark_background')

# --- Load Assets ---
try:
    model = joblib.load('neuroalert_realistic_model.pkl')
    sim_df = pd.read_csv('simulation_data.csv')
except FileNotFoundError:
    st.error("Model or simulation data not found. Please ensure all preparation scripts have run.")
    st.stop()

# --- Simulation Parameters ---
SEIZURE_START_SEC = 1238
PRE_ICTAL_WINDOW_MINS = 10
PRE_ICTAL_START_SEC = SEIZURE_START_SEC - (PRE_ICTAL_WINDOW_MINS * 60)

# --- UI Layout ---
st.title("NeuroAlert: Live Patient Monitor")
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### Physiological Signals")
    # Placeholders for our Matplotlib images
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

# --- Helper function to create a plot image ---
def create_plot_image(data, title):
    fig, ax = plt.subplots(figsize=(5, 2)) # Create a figure and axis
    ax.plot(data, color='cyan')
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    plt.tight_layout()
    
    # Save the plot to an in-memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    plt.close(fig) # Close the figure to save memory
    return buf

# --- Live Simulation ---
hr_data, hrv_data, eda_data = [], [], []
event_log = ["System Initialized..."]

status_placeholder.success("STATUS: NORMAL")
log_placeholder.text_area("Log", value='\n'.join(event_log), height=200)

for index, row in sim_df.iterrows():
    current_features = row[['HR', 'LF_HF_Ratio']].to_frame().T
    
    if PRE_ICTAL_START_SEC <= row['timestamp'] < SEIZURE_START_SEC:
        eda_value = 1.0 + np.random.uniform(-0.2, 0.2)
    else:
        eda_value = row['EDA_Mean']
    
    prediction = model.predict(current_features)[0]
    
    sim_time = datetime.timedelta(seconds=int(row['timestamp']))
    
    # Update AI status
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

    # Update data lists (keep only the last 30 points)
    hr_data.append(row['HR'])
    hrv_data.append(row['LF_HF_Ratio'])
    eda_data.append(eda_value)
    if len(hr_data) > 30:
        hr_data.pop(0)
        hrv_data.pop(0)
        eda_data.pop(0)

    # Create and display new plot images
    c1, c2, c3 = st.columns(3)
    with c1:
        c1.image(create_plot_image(hr_data, "Heart Rate (HR)"), use_container_width=True)
    with c2:
        c2.image(create_plot_image(hrv_data, "HRV (LF/HF)"), use_container_width=True)
    with c3:
        c3.image(create_plot_image(eda_data, "EDA (Simulated)"), use_container_width=True)
    
    # Update log
    log_placeholder.text_area("Log", value='\n'.join(event_log), height=200, key=f"log_{index}")
    time.sleep(1 / playback_speed)

st.balloons()
