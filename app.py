import streamlit as st
import pandas as pd
import joblib
import time
import datetime
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="NeuroAlert", page_icon="🧠", layout="wide")

# --- Asset Loading ---
# Use @st.cache_data for efficient loading
@st.cache_data
def load_data():
    try:
        model = joblib.load('neuroalert_realistic_model.pkl')
        sim_df = pd.read_csv('simulation_data.csv')
        master_df = pd.read_csv('neuroalert_dataset.csv')
        return model, sim_df, master_df
    except FileNotFoundError as e:
        st.error(f"Error loading data file: {e}. Ensure all .pkl and .csv files are present.")
        st.stop()

model, sim_df, master_df = load_data()

# ======================================================================================
# PAGE FUNCTIONS
# ======================================================================================

def live_monitor_page():
    st.title("NeuroAlert: Live Patient Monitor")
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### Physiological Signals")
        # Create placeholders for the charts
        chart_container = st.container()
        c1, c2, c3 = chart_container.columns(3)
        chart1_placeholder = c1.empty()
        chart2_placeholder = c2.empty()
        chart3_placeholder = c3.empty()
        st.markdown("---")

    with col2:
        st.markdown("### Simulation Control")
        playback_speed = st.selectbox("Playback Speed (x real-time)", [1, 60, 360, 720, 1440], index=3)

    # Placeholders for status text below the charts
    st.header("AI STATUS")
    status_placeholder = st.empty()

    # --- Simulation Loop ---
    # Initialize empty dataframes for plotting
    window_size = 30
    plot_data = pd.DataFrame({
        'HR': np.zeros(window_size),
        'LF_HF_Ratio': np.zeros(window_size),
        'EDA_Mean': np.zeros(window_size)
    })

    for index, row in sim_df.iterrows():
        # Prepare current data for prediction
        current_features = row[['HR', 'LF_HF_Ratio']].to_frame().T
        prediction = model.predict(current_features)[0]
        
        # --- THE CRITICAL TRIGGER ---
        if prediction == 1:
            st.session_state.page_key = 'alert'
            st.rerun() # Immediately stop and rerun to show the alert page

        # Update status text
        sim_time = datetime.timedelta(seconds=int(row['timestamp']))
        status_placeholder.success(f"STATUS: NORMAL | Patient Time Elapsed: {sim_time}")

        # Update data for charts by rolling the window
        new_row = pd.DataFrame([row[['HR', 'LF_HF_Ratio', 'EDA_Mean']].to_dict()])
        plot_data = pd.concat([plot_data.iloc[1:], new_row], ignore_index=True)

        # Update charts efficiently with st.line_chart
        chart1_placeholder.line_chart(plot_data['HR'], use_container_width=True)
        chart2_placeholder.line_chart(plot_data['LF_HF_Ratio'], use_container_width=True)
        chart3_placeholder.line_chart(plot_data['EDA_Mean'], use_container_width=True)

        # Control simulation speed
        time.sleep(1 / playback_speed)

    st.balloons()
    st.success("Simulation finished without detecting any seizure risk.")


def alert_page():
    st.title("🚨 SEIZURE RISK DETECTED 🚨")
    st.header("A pre-ictal state has been identified by the AI.")
    st.markdown("---")

    timer_placeholder = st.empty()
    for seconds in range(10, -1, -1): # Countdown from 10
        timer_placeholder.header(f"**Predicted Onset In: ~{seconds:02d}** minutes (Simulated)")
        time.sleep(1) # Use 1s sleep for demo purposes
    timer_placeholder.warning("CRITICAL ALERT TIME")

    st.subheader("Please confirm the event after it occurs:")
    col1, col2 = st.columns(2)
    if col1.button("✅ Confirmed Seizure", use_container_width=True):
        st.success("Feedback recorded. We hope the patient is safe.")
    if col2.button("❌ Dismissed - False Alarm", use_container_width=True):
        st.warning("Feedback recorded. The model will be fine-tuned with this data.")


def clinical_analysis_page():
    st.title("Clinical Biomarker Validation")
    st.markdown("These plots show the biomarker distributions for Normal vs. Pre-ictal states from the training dataset. This helps validate the features used by the AI model.")
    
    # Import plotting libraries here to keep other pages light
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 💡 The problematic line below has been removed.
    # st.set_option('deprecation.showPyplotGlobalUse', False) <--- DELETE THIS LINE

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(ax=axes[0], x='label', y='HR', data=master_df, palette="viridis").set(
        title='Heart Rate Distribution', xlabel='Patient State', ylabel='Heart Rate (bpm)', xticklabels=['Normal', 'Pre-ictal']
    )
    sns.boxplot(ax=axes[1], x='label', y='LF_HF_Ratio', data=master_df, palette="plasma").set(
        title='HRV (LF/HF Ratio) Distribution', xlabel='Patient State', ylabel='LF/HF Ratio', xticklabels=['Normal', 'Pre-ictal']
    )
    plt.tight_layout()
    st.pyplot(fig)


# ======================================================================================
# MAIN APP LOGIC & NAVIGATION
# ======================================================================================

# Initialize session state for page navigation if it doesn't exist.
if 'page_key' not in st.session_state:
    st.session_state.page_key = 'monitoring' # Default page

# --- Sidebar to control the page ---
st.sidebar.title("App Navigation")

# When a button is clicked, it updates the session state to the corresponding page key.
if st.sidebar.button("Live Monitor", use_container_width=True):
    st.session_state.page_key = 'monitoring'
if st.sidebar.button("Clinical Analysis", use_container_width=True):
    st.session_state.page_key = 'analysis'
if st.sidebar.button("Alert Page (Manual)", use_container_width=True):
    st.session_state.page_key = 'alert'

# --- Render the correct page based ONLY on session state ---
# This is the single source of truth for which page is displayed.
if st.session_state.page_key == 'monitoring':
    live_monitor_page()
elif st.session_state.page_key == 'analysis':
    clinical_analysis_page()
elif st.session_state.page_key == 'alert':
    alert_page()
