# 🧠 NeuroAlert  
### Predicting Seizures Before They Strike  

> *Transforming epilepsy care from reactive detection to predictive prevention.*

---

## 🚀 Overview

**NeuroAlert** is a first-of-its-kind **AI + Heuristic ECG-based system** that predicts epileptic seizures **up to 10 minutes before onset**.  
Built on the principle that seizures disrupt both brain and heart function, NeuroAlert analyzes **Heart Rate Variability (HRV)** biomarkers extracted from ECG signals to detect early autonomic changes.

Unlike traditional EEG-based systems that are invasive and impractical for continuous use, NeuroAlert is **non-invasive, explainable, and personalized** — turning everyday ECG data into actionable clinical foresight.

---

## 🎯 Key Features

- ⚙️ **Hybrid Adaptive Model** – Combines ML (XGBoost, 60%) with a Heuristic Engine (40%) for reliability and interpretability.  
- 💓 **HRV Biomarker Analysis** – Extracts and evaluates 15 biomarkers (SDNN, RMSSD, LF/HF, etc.) from 2-minute ECG windows.  
- 🧠 **Patient-Specific Baselines** – Automatically learns and adapts thresholds per patient using the first 10 minutes of ECG.  
- 🔍 **Explainable AI** – Each alert cites the biomarker deviations that triggered it (e.g., LF/HF spike, HR surge).  
- ⏱️ **10-Minute Predictive Window** – Forecasts pre-ictal activity ahead of seizure onset.  
- 💻 **Streamlit Web App** – Intuitive dashboard with dual ECG visualization, biomarker trends, and risk timeline.  
- 🧩 **Clinically Deployable** – Runs on CPU, 200 MB file upload limit, low-resource compatible.

---

## 🧬 Technical Stack

| Stage | Core Techniques | Tools |
|--------|----------------|-------|
| **Data Acquisition** | EDF import, channel extraction | `MNE`, `NeuroKit2` |
| **Feature Extraction** | 15 HRV biomarkers + normalization | `NeuroKit2`, `NumPy`, `pandas` |
| **ML Model Training** | Gradient Boosting, GroupKFold CV | `XGBoost`, `scikit-learn` |
| **Calibration** | Isotonic regression + temporal smoothing | `scikit-learn` |
| **Heuristic Engine** | Adaptive, patient-specific rule set | `NumPy` |
| **Visualization** | ECG plots, biomarker dashboard | `Streamlit`, `Plotly` |
| **Deployment** | Local or cloud web app | `Streamlit`, `joblib`, `Git` |

---

## ⚡ Installation

```bash
# Clone repository
git clone https://github.com/<yourusername>/NeuroAlert.git
cd NeuroAlert

# Create environment (recommended)
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
📁 Repository Structure
bash
Copy code
NeuroAlert/
│
├── app.py                     # Main Streamlit web application
├── model/
│   ├── neuroalert_hybrid_ecg_only.pkl   # Trained hybrid model
│   ├── scaler.pkl                       # StandardScaler for normalization
│
├── data/
│   ├── sample_edf_files/                # Example EDF inputs
│   └── Seizures-list-PNxx.txt           # Ground truth annotations
│
├── assets/
│   └── visuals/                         # UI assets and reference diagrams
│
├── requirements.txt
└── README.md
🧩 How It Works
Upload an .edf ECG file.

Signal Parsing: MNE reads and preprocesses ECG data.

HRV Extraction: 15 biomarkers computed for 2-minute windows.

AI Inference: XGBoost model predicts seizure probability.

Heuristic Check: Physiological rules verify anomalies.

Fusion Logic: Weighted blend (60% AI + 40% Heuristic).

Output: 10-minute pre-ictal risk score, visualization, and biomarker trends.

🔒 Intellectual Property
The NeuroAlert architecture is protected under proprietary innovation in:

Adaptive HRV baseline calibration

AI + Heuristic fusion for seizure prediction

Explainable biomarker-driven alerts

Temporal risk consensus algorithm

Innovation Class: Non-invasive ECG-based predictive neurocardiology.

⚖️ Competitive Edge
Competitor	Modality	Focus	Limitation	NeuroAlert Advantage
Empatica	EDA + motion	Detection	No prediction	ECG-based, predictive
BioSerenity	EEG	Monitoring	Complex setup	Simple, scalable
Seer Medical	EEG + video	Diagnosis	Offline	Real-time
NeuroAlert	ECG HRV	Prediction	—	Adaptive, explainable

🧭 Features:

10-minute sliding analysis window

Real-time biomarker visualization

Seizure event overlays from annotation files

📊 Performance Metrics
Metric	Value (v12 Hybrid Model)
AUC (ROC)	0.74
Specificity	72%
Sensitivity	44%
F1 Score	0.56
False Alarm Reduction	60–80%

🌍 Future Roadmap
LSTM-based temporal modeling for continuous trends.

Integration into ECG wearable hardware.

Clinical trials with affiliated hospitals.

Regulatory validation (CDSCO / FDA).

Publication and IP filing.

👥 Contributors
Name - Role
Parth Kapoor - Technical Lead — AI architecture, Streamlit app, integration
Rijjul Garg -	Medical Lead — Clinical heuristics, HRV validation, interpretation
