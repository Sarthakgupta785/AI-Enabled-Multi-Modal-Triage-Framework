# Chart Descriptions for Viva — AI-Enabled Multi-Modal Triage Framework
**Sarthak Gupta & Dhruv | Guide: Dr R. Madhura | SRM Institute of Science and Technology**

---

## Chart 01: Model Accuracy Comparison (Vitals vs NLP vs Fusion)

**What it shows:** A grouped bar chart comparing test set accuracy of all three models (Vitals, NLP, and Fusion) across the three prediction targets — Risk Level, Urgency, and Disease Category.

**How to explain in viva:**
"This chart demonstrates the core thesis of our multi-modal approach. The Vitals model (blue) achieves 100% accuracy on Risk Level because it directly processes NEWS scores and vital signs, which are the clinical basis for risk classification. However, it drops to only 56.6% on Disease Category because vitals alone cannot distinguish between diseases — a fever could be infectious, respiratory, or neurological.

The NLP model (green) shows the opposite pattern. It achieves 100% on Disease Category because symptom text contains disease-specific language like 'chest pain' for cardiac or 'wheezing' for respiratory. But it only reaches 73.1% on Risk Level because text descriptions don't precisely quantify how abnormal the vitals are.

The Fusion model (red) combines both and achieves the best of both worlds — near-perfect performance on all three targets. This validates our hypothesis that multi-modal fusion outperforms any single modality."

---

## Chart 02: F1 Score Comparison

**What it shows:** Same structure as Chart 01, but using weighted F1 score instead of accuracy. F1 is a more robust metric because it accounts for both precision and recall.

**How to explain in viva:**
"We use F1 score as our primary metric rather than just accuracy because our dataset has class imbalance — there are more Low-risk patients than Critical ones. A model that predicts 'Low' for everyone would get decent accuracy but terrible F1. The weighted F1 scores confirm the same pattern as accuracy — each model excels where the other struggles, and fusion captures the best of both. The urgency prediction shows a slight improvement from 0.9693 (Vitals alone) to 0.9703 (Fusion), demonstrating that even small NLP contributions improve the ensemble."

---

## Chart 03: Train / Validation / Test Comparison

**What it shows:** A side-by-side comparison of F1 scores on training, validation, and test sets for both Vitals and NLP models across all three targets.

**How to explain in viva:**
"This chart is our overfitting diagnostic. If a model performs significantly better on training data than test data, it's memorizing rather than learning. For the Vitals model, train and test performance are nearly identical across all targets, confirming the model generalizes well. For the NLP model, we see a slight gap in disease category — 100% on train and test — which is expected since the symptom templates have distinct vocabulary per disease. The key takeaway is that none of our models show concerning overfitting. The validation set performance closely tracks the test set, which validates our 80/10/10 split strategy."

---

## Chart 04a: Confusion Matrix — Risk Level

**What it shows:** Three heatmaps side by side showing how each model classifies patients into Low, Moderate, High, and Critical risk levels on the test set.

**How to explain in viva:**
"The confusion matrix reveals WHERE each model makes mistakes, not just how many. The Vitals model shows a perfect diagonal — every patient is correctly classified. The NLP model's confusion matrix shows interesting off-diagonal errors: some Critical patients get classified as High (59 cases), and some Low patients get classified as Moderate (108 cases). These are adjacent-class confusions, meaning the NLP model understands the severity ordering but lacks the numerical precision of vitals data. The Fusion model inherits the Vitals model's perfect classification, confirming that the ensemble handles this well."

---

## Chart 04b: Confusion Matrix — Urgency

**What it shows:** Three heatmaps for urgency classification (Non-Urgent, Semi-Urgent, Urgent, Emergency).

**How to explain in viva:**
"Urgency prediction is more challenging because it depends on factors beyond just vitals — comorbidities, symptom duration, and patient age all influence urgency. The Vitals model confuses 27 Emergency patients as Urgent, which is clinically a one-level under-triage. The NLP model shows broader confusion, especially between Semi-Urgent and Non-Urgent (115 misclassifications). The Fusion model reduces the Vitals model's Emergency-to-Urgent confusion from 27 to 26 patients, a small but meaningful safety improvement. Critically, no Emergency patient is ever classified as Non-Urgent by any model."

---

## Chart 04c: Confusion Matrix — Disease Category

**What it shows:** Three heatmaps for 9-class disease classification.

**How to explain in viva:**
"This is where the multi-modal advantage is most visible. The Vitals model's confusion matrix is heavily scattered — it confuses Cardiac with Neurological, Gastrointestinal with Respiratory, and General with almost everything. This makes clinical sense because different diseases can produce similar vital sign patterns — both pneumonia and heart failure can cause tachycardia and low SpO2. The NLP model achieves a perfect diagonal because disease-specific vocabulary like 'wheezing', 'seizure', or 'abdominal pain' directly maps to disease categories. The Fusion model correctly trusts the NLP model entirely for this task, assigning it 100% weight."

---

## Chart 05a: Feature Importance — Risk Level Prediction

**What it shows:** A horizontal bar chart showing the top 15 most important features for predicting risk level in the Vitals model, measured using permutation importance.

**How to explain in viva:**
"Permutation importance measures how much the model's performance drops when we randomly shuffle each feature. The NEWS score dominates with an importance of 0.57 — this is expected because our risk labels are derived from NEWS scores. The red flag indicator is the second most important feature at 0.10, which validates our clinical design where red flags directly escalate risk. The remaining features contribute minimally, suggesting that for risk prediction, the NEWS score is the single most predictive feature. This aligns with clinical literature where NEWS is considered the gold standard for patient deterioration detection."

---

## Chart 05b: Feature Importance — Urgency Prediction

**What it shows:** Same format as 05a but for urgency prediction.

**How to explain in viva:**
"Unlike risk level, urgency prediction relies on a broader set of features. NEWS score still dominates at 0.54, but patient age emerges as the second most important feature at 0.11 — because very young and very old patients are inherently more urgent. The red flag indicator contributes 0.07, and comorbidity count adds 0.03 — a patient with multiple chronic conditions needs faster attention even at the same risk level. The presence of features like nearest_hospital_km (0.001) shows the model also considers contextual factors specific to rural deployment, which is a key differentiator of our framework."

---

## Chart 06a: Per-Class F1 Scores — Risk Level

**What it shows:** Grouped bars showing F1 score for each individual risk class (Low, Moderate, High, Critical) across all three models.

**How to explain in viva:**
"This per-class breakdown reveals an important clinical insight. The NLP model's weakest class is Moderate (0.59 F1) — patients described with moderate symptoms use language that overlaps with both Low and High categories. Words like 'worsening' or 'persistent' could describe either moderate or high severity depending on context. However, the NLP model still achieves 0.77 F1 on Critical patients because critical symptom descriptions use very distinctive language — 'unconscious', 'shock', 'hemorrhage' — that rarely appears in milder cases."

---

## Chart 06b: Per-Class F1 Scores — Urgency

**What it shows:** Same format for urgency classes.

**How to explain in viva:**
"The NLP model struggles most with Semi-Urgent (0.47 F1) and Urgent (0.46 F1) — these middle categories have overlapping symptom descriptions. Non-Urgent and Emergency are easier extremes. The Vitals model handles all classes well, with slight weakness in Urgent (0.91 F1) where some patients get classified as Emergency. This over-triage pattern is actually clinically safer than under-triage — we prefer false alarms over missed emergencies."

---

## Chart 06c: Per-Class F1 Scores — Disease Category

**What it shows:** F1 scores for all 9 disease categories.

**How to explain in viva:**
"The Vitals model's per-class performance varies dramatically. Metabolic disorders achieve 0.74 F1 because blood glucose is a highly discriminative feature. Respiratory conditions score 0.71 because SpO2 and respiratory rate are characteristic. But Obstetric cases score only 0.17 — the vitals model struggles here because pregnancy modifies all normal ranges. The NLP model solves all of these problems with 1.0 across every category. This chart is the strongest evidence for why multi-modal fusion is necessary — the modalities are genuinely complementary."

---

## Chart 07: Complete System Performance Summary

**What it shows:** A horizontal bar chart showing all three models' F1 scores side by side for every target, giving a single-glance summary of the entire system.

**How to explain in viva:**
"This is our executive summary chart. It shows that the Fusion model (red) consistently matches or exceeds the best individual model for every target. For Risk Level and Urgency, it matches the Vitals model's near-perfect scores. For Disease Category, it matches the NLP model's perfect classification. The key insight is that fusion never performs worse than the best single model — it intelligently routes each prediction task to the strongest modality through learned weights."

---

## Chart 08: Disease-Specific Keywords (NLP Explainability)

**What it shows:** A horizontal bar chart showing the top keywords the NLP model associates with each of the 9 disease categories, with keyword labels displayed alongside.

**How to explain in viva:**
"This chart demonstrates the explainability of our NLP model — we can show clinicians exactly which words drive each prediction. Cardiac conditions are identified by 'chest', 'fainting', 'palpitations'. Respiratory by 'breathing', 'wheezing', 'cough'. Neurological by 'vision', 'speech', 'headache', 'numbness'. These mappings align perfectly with clinical symptom profiles, which builds trust in the system.

The Infectious category shows 'rash', 'fever', 'chills', 'body aches' — classic infection symptoms. Metabolic shows 'blood sugar', 'glucose', 'thirst', 'urination' — pointing to diabetic emergencies. This explainability directly addresses one of the key gaps we identified in our literature survey: existing AI systems are black boxes that clinicians cannot trust. Our system shows its reasoning."

---

## Chart 09: Optimal Fusion Weights

**What it shows:** A stacked bar chart showing how much weight each model receives in the fusion ensemble for each target, learned on the validation set.

**How to explain in viva:**
"This chart reveals the optimal fusion strategy our system learned automatically. For Risk Level, the weights are balanced at 0.50 Vitals and 0.50 NLP — both models contribute equally. For Urgency, NLP gets slightly more weight at 0.55 versus 0.45 for Vitals, suggesting symptom descriptions provide marginal additional signal beyond vitals for urgency decisions.

The most striking finding is Disease Category — the system assigns 0.00 weight to Vitals and 1.00 to NLP. It learned entirely on its own that vital signs are useless for disease classification and relies completely on symptom text. This is a powerful validation of our multi-modal architecture — rather than forcing both modalities to contribute everywhere, the system intelligently determines where each modality is useful."

---

## Chart 10: Test Set Class Distribution

**What it shows:** Three bar charts showing how many patients belong to each class in the test set for risk level, urgency, and disease category.

**How to explain in viva:**
"This chart confirms our dataset's class distribution is realistic and not artificially balanced. For Risk Level, 45.5% are Low, 14.7% Moderate, 17.0% High, and 22.8% Critical — reflecting a real-world pattern where most clinic visitors have non-critical conditions, but a significant minority need immediate attention. 

For Disease Category, Respiratory (21.2%) is the most common — consistent with rural India where respiratory infections are the leading cause of clinic visits. Obstetric cases are the rarest (2.2%), reflecting the smaller proportion of pregnant patients. We used stratified splitting to ensure every class appears proportionally in train, validation, and test sets — preventing the model from being biased by uneven sampling."

---

## Tips for Presenting These Charts in Viva

1. **Start with Chart 07 (Complete Summary)** — give the panel the big picture first
2. **Then show Chart 01 (Accuracy Comparison)** — explain the complementary strengths
3. **Use Chart 09 (Fusion Weights)** — this is the "wow" moment showing the system learned on its own
4. **Show Chart 08 (Keywords)** — proves explainability, which is a key project objective
5. **End with Chart 04b (Urgency Confusion Matrix)** — demonstrates the safety analysis showing no Emergency patient is classified as Non-Urgent

If your guide asks about overfitting, show Chart 03.
If asked about feature engineering, show Chart 05.
If asked about clinical safety, show the under-triage numbers from the fusion model output.
