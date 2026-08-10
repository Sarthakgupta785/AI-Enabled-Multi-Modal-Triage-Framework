<div align="center">

# 🚨 TRIAGE.AI — _Because Guessing Isn't a Medical Protocol_

### An AI That Triages Patients Faster Than Your Hospital's Wi-Fi

<br>

```
   _____ ____  _____          _____ ______            _____ 
  |_   _|  _ \|_   _|   /\   / ____|  ____|     /\   |_   _|
    | | | |_) | | |    /  \ | |  __| |__       /  \    | |  
    | | |  _ <  | |   / /\ \| | |_ |  __|     / /\ \   | |  
    | | | |_) |_| |_ / ____ \ |__| | |____   / ____ \ _| |_ 
    |_| |____/|_____/_/    \_\_____|______| /_/    \_\_____|
```

<br>

![Status](https://img.shields.io/badge/STATUS-COOKING_🔥-orange?style=for-the-badge)
![Saves](https://img.shields.io/badge/LIVES_SAVED-hopefully_many-green?style=for-the-badge)
![Vibe](https://img.shields.io/badge/VIBE_CHECK-PASSED_✅-blue?style=for-the-badge)
![SDG](https://img.shields.io/badge/UN_SDG-3_Good_Health-brightgreen?style=for-the-badge)

<br>

> _"We taught a machine to look at your vitals and your complaints and tell you if you're fine or if you should maybe, you know, **panic**."_

---

**Built by [Sarthak Gupta](# ) & [Dhruv](# )** · Guided by **Dr R. Madhura**
<br>
Department of Computing Technologies · SRM Institute of Science and Technology, Kattankulathur

</div>

---

## 🧠 WTF Is This?

Imagine you're a healthcare worker in a rural clinic. You've got 47 patients waiting, one stethoscope, and zero specialists in a 50km radius. Who do you see first? The guy with the cough or the lady who's _quietly_ having a stroke?

**TRIAGE.AI** answers that question. It's a multi-modal AI system that inhales patient vitals and symptom descriptions and exhales a clinical risk prediction — in real time, on a phone, _without internet_.

### The Pitch (in 30 seconds)
```
📊 Vitals go in          →  Gradient Boosting crunches the numbers
📝 Symptoms described    →  NLP model reads between the lines  
🔥 Red flags detected    →  Rule engine overrides everything (as it should)
🏥 Triage level comes out →  "See this patient NOW" or "They can wait"
```

No cloud required. No PhD required. Just a phone and a pulse oximeter.

---

## 🎯 The Problem We're Solving

| The Reality | Our Response |
|:---|:---|
| 60% of India's population relies on rural healthcare | Build AI that runs on a ₹10,000 phone |
| Triage in rural clinics is often _"whoever shouts loudest"_ | Replace chaos with clinical scoring protocols |
| Specialists are hours away | Put a decision-support copilot in every CHW's pocket |
| Internet is a luxury, not a given | Offline-first. Always. No exceptions. |

---

## 🏗️ Architecture — How the Magic Happens

```
                        ┌──────────────────┐
                        │   PATIENT INPUT   │
                        │  vitals + symptoms │
                        └────────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
              ┌─────▼──────┐          ┌───────▼───────┐
              │  VITALS     │          │  SYMPTOM TEXT  │
              │  Pipeline   │          │  Pipeline      │
              │             │          │                │
              │ XGBoost /   │          │ TF-IDF /       │
              │ LightGBM    │          │ Transformer    │
              └─────┬──────┘          └───────┬───────┘
                    │                         │
                    └────────────┬────────────┘
                                 │
                        ┌────────▼─────────┐
                        │  FUSION ENGINE    │
                        │  Weighted Ensemble │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  🚩 RED FLAG      │
                        │  OVERRIDE ENGINE  │
                        │  (NEWS/MEWS rules)│
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │   TRIAGE OUTPUT   │
                        │                   │
                        │  🟢 Low Risk      │
                        │  🟡 Medium Risk   │
                        │  🔴 HIGH RISK     │
                        │  ⚫ CRITICAL      │
                        └──────────────────┘
```

**Key insight:** The red flag engine has _veto power_. If your SpO₂ is below 90%, we don't care what the ML model thinks — you're critical. Period. AI should _assist_ clinical judgment, never override basic physiology.

---

## 📁 Project Map

```
MINOR_PROJECT_WORK/
│
├── 📊 data/
│   ├── triage_dataset_10k.csv          # 10K synthetic patients (40 columns of chaos)
│   └── generate_triage_dataset.py      # The patient factory
│
├── 📓 notebooks/                       # Where we broke things until they worked
│
├── 🤖 models/                          # The trained brains (.pkl, .pt)
│
├── ⚙️ src/
│   ├── preprocessing.py                # Data janitor (cleans up the mess)
│   ├── vitals_model.py                 # Gradient Boosting goes brrr
│   ├── nlp_model.py                    # Reads symptoms like a doctor reads handwriting
│   ├── fusion_model.py                 # Two models enter, one prediction leaves
│   └── red_flag_engine.py              # The "I don't care what the AI says" module
│
├── 📱 app/                             # Offline Android app (works in a field, literally)
│
├── 📄 reports/                         # The paperwork that proves we did things
│
└── README.md                           # You are here 📍
```

---

## 🛠️ Tech Arsenal

| Layer | What We Used | Why |
|:---|:---|:---|
| **Vitals ML** | XGBoost / LightGBM / Random Forest | Fast, interpretable, works on tabular data like a charm |
| **Symptom NLP** | TF-IDF + Transformer-based classification | Because "chest pain" and "my chest feels tight" mean the same thing |
| **Fusion** | Weighted ensemble | Two brains > one brain |
| **Red Flags** | Hard-coded rules (NEWS/MEWS) | Some things are too important for probability |
| **Deployment** | Offline Android app | No signal? No problem. |
| **Languages** | Python, probably some tears | The usual suspects |

---

## 📊 The Dataset — 10,000 Patients Who Don't Exist (But Could)

Our synthetic dataset is clinically grounded, not random noise:

- **40 features** across 7 categories: demographics, vitals, symptoms, medical history, red flags, environmental context, and more
- **Labels derived from NEWS scoring** — the same protocol used in actual emergency departments
- **3 prediction targets:**
  - `risk_level` — How bad is it? (Low / Medium / High / Critical)
  - `urgency` — How fast should they be seen?
  - `disease_category` — What's likely going on?

> _"Why synthetic?"_ — Because real patient data is (rightfully) locked behind ethics boards, HIPAA, and common sense. Our synthetic data maintains clinical distributions without compromising anyone's privacy.

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-repo/MINOR_PROJECT_WORK.git
cd MINOR_PROJECT_WORK

# 2. Install the goods
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# 3. (Optional) Regenerate the dataset — feel the power of creation
python data/generate_triage_dataset.py

# 4. Clean the data
python src/preprocessing.py

# 5. Train the models — grab a coffee, this takes a minute
python src/vitals_model.py      # Tabular model
python src/nlp_model.py         # NLP model  
python src/fusion_model.py      # The Voltron step — combine everything

# 6. Done. You just built a triage system. Tell your mom.
```

---

## 🔮 Roadmap — What's Next

- [ ] **Federated Learning** — Train across clinics without sharing patient data
- [ ] **Explainability Dashboard** — Show _why_ the AI made each decision (SHAP/LIME)
- [ ] **Voice Input** — Because typing is hard when you're treating patients
- [ ] **Multi-language Support** — Healthcare shouldn't require English
- [ ] **Edge Deployment** — Run inference on Raspberry Pi for even cheaper hardware
- [ ] **Integration with ABDM** — Plug into India's Ayushman Bharat Digital Mission

---

## 📚 References — Standing on the Shoulders of Giants

1. Kwon, J. M., et al. (2019). _Deep Learning for Early Triage._ Nature Scientific Reports.
2. Chen, Y., Zhang, L., & Wang, H. (2021). _Multi-Modal ML for Clinical Decision Support._ Computers in Biology and Medicine.
3. Rajkomar, A., Dean, J., & Kohane, I. (2022). _Explainable AI for Medical Triage._ IEEE Access.
4. Patel, V., & Sharma, R. (2020). _Mobile Health Decision Support for Rural Healthcare._ Health Informatics Journal.
5. Liu, X., et al. (2021). _Automated Syndromic Surveillance Using ML._ Journal of Biomedical Informatics.

---

<div align="center">

### 🌍 Aligned with UN Sustainable Development Goal 3
**Good Health and Well-Being**

_Because access to healthcare shouldn't depend on your postal code._

---

Made with ☕, 🧠, and occasional existential dread at **SRM Institute of Science and Technology**

<br>

```
If this project saves even one life, every late night was worth it.
```

</div>
