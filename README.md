🧠 NeuroAlert: An HRV-Based Epileptic Seizure Prediction System

NeuroAlert is a proof-of-concept prototype developed for a 7-day hackathon. It is a software-based system that analyzes Heart Rate Variability (HRV) biomarkers to predict the onset of an epileptic seizure minutes in advance, providing a life-saving warning.

This project is built on the medical hypothesis that the pre-ictal phase—a distinct period 10-20 minutes before a seizure—is marked by significant dysregulation of the autonomic nervous system (ANS). These changes are subtle but measurable in HRV data.

📈 Final Model Performance (V7)

This prototype proves the hypothesis is viable. The final model analyzes 36 temporal features (10-minute trends, volatility, and slope) derived from 12 core HRV biomarkers.

Recall (Sensitivity): ~71%

Correctly detected 24 out of 34 oncoming seizures in the test data.

Precision: ~2%

To achieve high sensitivity, the model generates a high number of false positives.

Accuracy: 82%

The model is highly accurate in identifying normal brain states.

Model: EasyEnsembleClassifier

An advanced model specifically designed to handle extreme 158:1 class imbalance.

The high recall demonstrates a successful proof-of-concept. The low precision is a known trade-off, which would be the primary focus of future R&D (e.g., by adding EEG features).

🚀 How to Run the Prototype

This project is a Streamlit dashboard that simulates the NeuroAlert system in real-time.

1. Setup:

# Clone the repository
git clone [your-github-repo-url]
cd NeuroAlert_App

# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install the required libraries
pip install -r requirements.txt



2. Run the App:

streamlit run app.py



3. Run the Simulation:

The app will open in your browser.

Upload the neuroalert_final_dataset_v2.csv file when prompted.

Click the "Start Real-Time Simulation" button.

Watch the dashboard as the model analyzes each 2-minute segment, calculates 10-minute trends, and raises alerts when it detects a pre-ictal pattern.

This prototype was developed by Rijjul Garg (Medical Lead) and Parth Kapoor (Tech Lead).
