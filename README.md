# AI-Enabled Multi-Modal Triage Framework

**Risk Prediction and Urgency Assessment in Resource-Constrained Rural Clinics**

## Team
- **Sarthak Gupta** (RA2311003010003)
- **Dhruv** (RA2311003010044)

## Guide
- **Dr R. Madhura**, Department of Computing Technologies, SRM Institute of Science and Technology, Kattankulathur

## About
An AI-powered multi-modal triage system that analyzes patient vitals and symptom descriptions to deliver real-time clinical decision support for rural healthcare settings. Aligned with **SDG 3: Good Health and Well-Being**.

## Project Structure
```
MINOR_PROJECT_WORK/
│
├── data/                          # Dataset and generation scripts
│   ├── triage_dataset_10k.csv     # 10,000 synthetic patient records (40 columns)
│   └── generate_triage_dataset.py # Script to regenerate/customize dataset
│
├── notebooks/                     # Jupyter notebooks for EDA and experiments
│
├── models/                        # Saved trained model files (.pkl, .pt)
│
├── src/                           # Source code
│   ├── preprocessing.py           # Data cleaning, feature engineering, splits
│   ├── vitals_model.py            # Gradient Boosting model for vitals (tabular)
│   ├── nlp_model.py               # Transformer/NLP model for symptom text
│   ├── fusion_model.py            # Multi-modal fusion (vitals + NLP ensemble)
│   └── red_flag_engine.py         # Rule-based red flag override system
│
├── app/                           # Android/Web app (offline-first deployment)
│
├── reports/                       # Project report, charts, presentation
│
└── README.md
```

## Tech Stack
- **ML/Vitals**: Gradient Boosting (XGBoost/LightGBM), Random Forest
- **NLP/Symptoms**: TF-IDF / Transformer-based text classification
- **Fusion**: Weighted ensemble combining tabular + text predictions
- **Rule Engine**: Hard-coded clinical red flag detection (NEWS/MEWS protocols)
- **Deployment**: Offline-first Android application (lightweight inference)

## Dataset
- 10,000 synthetic patient records
- 40 features across 7 categories (demographics, vitals, symptoms, history, red flags, context)
- Labels derived from NEWS scoring protocol (not random)
- 3 target variables: risk_level, urgency, disease_category

## How to Run
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# Regenerate dataset (optional)
python data/generate_triage_dataset.py

# Run preprocessing
python src/preprocessing.py

# Train models
python src/vitals_model.py
python src/nlp_model.py
python src/fusion_model.py
```

## References
1. Kwon, J. M., et al. (2019). Deep Learning for Early Triage. Nature Scientific Reports.
2. Chen, Y., Zhang, L., & Wang, H. (2021). Multi-Modal ML for Clinical Decision Support. Computers in Biology and Medicine.
3. Rajkomar, A., Dean, J., & Kohane, I. (2022). Explainable AI for Medical Triage. IEEE Access.
4. Patel, V., & Sharma, R. (2020). Mobile Health Decision Support for Rural Healthcare. Health Informatics Journal.
5. Liu, X., et al. (2021). Automated Syndromic Surveillance Using ML. Journal of Biomedical Informatics.
