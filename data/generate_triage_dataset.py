"""
=============================================================================
AI-Enabled Multi-Modal Triage Framework - Synthetic Dataset Generator
=============================================================================
Authors: Sarthak Gupta, Dhruv
Guide: Dr R. Madhura
Dept: Computing Technology, SRM Institute of Science and Technology

This script generates a clinically-grounded synthetic dataset of 10,000
patient records for training a multi-modal triage prediction model.

All vital sign ranges, scoring, and labeling are based on:
- NEWS (National Early Warning Score) protocol
- MEWS (Modified Early Warning Score)
- ESI (Emergency Severity Index) guidelines
- WHO IMCI protocols
=============================================================================
"""

import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

NUM_RECORDS = 10000

print("=" * 60)
print("  TRIAGE DATASET GENERATOR")
print("  Generating {:,} synthetic patient records...".format(NUM_RECORDS))
print("=" * 60)


# =====================================================================
# MODULE 1: PATIENT DEMOGRAPHICS
# =====================================================================
print("\n[1/8] Generating patient demographics...")

patient_ids = [f"P{str(i).zfill(5)}" for i in range(1, NUM_RECORDS + 1)]

# Age distribution — skewed toward adults but includes pediatric & elderly
# Rural clinics see more children and elderly
age_groups_dist = {
    'Neonate': (0, 0.083, 0.03),    # 0-1 month, 3%
    'Infant': (0.083, 1, 0.05),      # 1 month - 1 year, 5%
    'Child': (1, 12, 0.12),          # 1-12 years, 12%
    'Adolescent': (12, 18, 0.08),    # 12-18, 8%
    'Adult': (18, 60, 0.45),         # 18-60, 45%
    'Elderly': (60, 90, 0.27),       # 60-90, 27%
}

ages = []
age_groups = []
for _ in range(NUM_RECORDS):
    r = np.random.random()
    cumulative = 0
    for group, (low, high, prob) in age_groups_dist.items():
        cumulative += prob
        if r <= cumulative:
            age = round(np.random.uniform(low, high), 1)
            ages.append(age)
            age_groups.append(group)
            break

# Hybrid age formatting: decimals for <1yr (months matter), whole numbers for rest
ages = [round(a, 2) if a < 1 else int(round(a)) for a in ages]
ages = np.array(ages)
genders = np.random.choice(['M', 'F'], NUM_RECORDS, p=[0.52, 0.48])

# Weight based on age (realistic ranges)
weights = []
for age, gender in zip(ages, genders):
    if age < 0.083:  # neonate
        w = np.random.normal(3.2, 0.5)
    elif age < 1:
        w = np.random.normal(7, 2)
    elif age < 5:
        w = np.random.normal(15, 3)
    elif age < 12:
        w = np.random.normal(30, 8)
    elif age < 18:
        w = np.random.normal(50, 10)
    elif age < 60:
        base = 70 if gender == 'M' else 60
        w = np.random.normal(base, 12)
    else:
        base = 65 if gender == 'M' else 55
        w = np.random.normal(base, 10)
    weights.append(max(2.0, round(w, 1)))

weights = np.array(weights)

# Pregnancy — only females aged 15-45
is_pregnant = np.zeros(NUM_RECORDS, dtype=bool)
for i in range(NUM_RECORDS):
    if genders[i] == 'F' and 15 <= ages[i] <= 45:
        is_pregnant[i] = np.random.random() < 0.08  # 8% of eligible females

print(f"   ✓ Ages: {ages.min():.1f} – {ages.max():.1f} years")
print(f"   ✓ Gender split: M={sum(genders=='M')}, F={sum(genders=='F')}")
print(f"   ✓ Pregnant patients: {sum(is_pregnant)}")


# =====================================================================
# MODULE 2: DISEASE CATEGORY ASSIGNMENT
# =====================================================================
print("\n[2/8] Assigning disease categories...")

disease_categories = [
    'Respiratory', 'Cardiac', 'Neurological', 'Gastrointestinal',
    'Infectious', 'Trauma', 'Metabolic', 'Obstetric', 'General'
]

# Base distribution — varies by age group
def assign_disease(age, gender, pregnant):
    if pregnant:
        return np.random.choice(
            ['Obstetric', 'General', 'Infectious', 'Cardiac'],
            p=[0.45, 0.25, 0.20, 0.10]
        )
    if age < 1:
        return np.random.choice(
            ['Respiratory', 'Infectious', 'Gastrointestinal', 'General', 'Neurological'],
            p=[0.30, 0.30, 0.20, 0.15, 0.05]
        )
    if age < 12:
        return np.random.choice(
            ['Respiratory', 'Infectious', 'Gastrointestinal', 'Trauma', 'General', 'Neurological'],
            p=[0.25, 0.25, 0.15, 0.15, 0.10, 0.10]
        )
    if age < 18:
        return np.random.choice(
            ['Trauma', 'Respiratory', 'Infectious', 'Gastrointestinal', 'General', 'Neurological'],
            p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
        )
    if age >= 60:
        return np.random.choice(
            ['Cardiac', 'Respiratory', 'Neurological', 'Metabolic', 'Infectious', 'General', 'Gastrointestinal'],
            p=[0.25, 0.20, 0.15, 0.15, 0.10, 0.08, 0.07]
        )
    # Adult 18-60
    return np.random.choice(
        disease_categories,
        p=[0.15, 0.12, 0.10, 0.15, 0.15, 0.12, 0.08, 0.03, 0.10]
    )

diseases = [assign_disease(a, g, p) for a, g, p in zip(ages, genders, is_pregnant)]

print(f"   ✓ Disease categories assigned")
for d in disease_categories:
    count = diseases.count(d)
    print(f"     {d}: {count} ({count/NUM_RECORDS*100:.1f}%)")


# =====================================================================
# MODULE 3: SEVERITY ASSIGNMENT (drives vital generation)
# =====================================================================
print("\n[3/8] Assigning severity levels...")

# Target distribution: 40% low, 28% moderate, 20% high, 12% critical
severity_levels = np.random.choice(
    ['low', 'moderate', 'high', 'critical'],
    NUM_RECORDS,
    p=[0.40, 0.28, 0.20, 0.12]
)

print(f"   ✓ Low: {sum(severity_levels=='low')}")
print(f"   ✓ Moderate: {sum(severity_levels=='moderate')}")
print(f"   ✓ High: {sum(severity_levels=='high')}")
print(f"   ✓ Critical: {sum(severity_levels=='critical')}")


# =====================================================================
# MODULE 4: CORE VITAL SIGNS (based on severity + age + disease)
# =====================================================================
print("\n[4/8] Generating vital signs based on clinical protocols...")

def generate_vitals(age, severity, disease, gender, pregnant):
    """Generate physiologically consistent vital signs."""
    vitals = {}

    # --- HEART RATE ---
    # Normal ranges vary by age
    if age < 1:
        base_hr = {'low': 125, 'moderate': 150, 'high': 165, 'critical': 180}
    elif age < 5:
        base_hr = {'low': 100, 'moderate': 120, 'high': 140, 'critical': 160}
    elif age < 12:
        base_hr = {'low': 85, 'moderate': 105, 'high': 125, 'critical': 150}
    else:
        base_hr = {'low': 75, 'moderate': 95, 'high': 120, 'critical': 145}

    if disease == 'Cardiac':
        base_hr[severity] += np.random.choice([-15, 15])  # tachy or brady
    if pregnant:
        base_hr[severity] += 10  # pregnancy raises HR

    vitals['heart_rate_bpm'] = int(np.clip(
        np.random.normal(base_hr[severity], 8), 35, 200
    ))

    # --- BLOOD PRESSURE ---
    if age < 1:
        base_sbp = {'low': 75, 'moderate': 65, 'high': 55, 'critical': 45}
    elif age < 12:
        base_sbp = {'low': 100, 'moderate': 90, 'high': 80, 'critical': 70}
    else:
        base_sbp = {'low': 120, 'moderate': 135, 'high': 150, 'critical': 170}

    if disease == 'Cardiac' and severity in ['high', 'critical']:
        # Could be hypertensive crisis OR hypotensive shock
        if np.random.random() < 0.4:
            base_sbp[severity] = np.random.choice([200, 210, 220])  # crisis
        else:
            base_sbp[severity] = np.random.choice([75, 80, 85])  # shock

    vitals['bp_systolic_mmhg'] = int(np.clip(
        np.random.normal(base_sbp[severity], 10), 60, 240
    ))

    # Diastolic must be physiologically consistent (pulse pressure 25-80 normally)
    pulse_pressure = np.random.normal(40, 10)
    if severity == 'critical':
        pulse_pressure = np.random.normal(25, 8)  # narrowing
    vitals['bp_diastolic_mmhg'] = int(np.clip(
        vitals['bp_systolic_mmhg'] - abs(pulse_pressure), 30, 140
    ))

    # --- TEMPERATURE ---
    base_temp = {'low': 36.8, 'moderate': 37.5, 'high': 38.8, 'critical': 39.8}
    if disease in ['Infectious', 'Respiratory']:
        base_temp[severity] += 0.5
    if severity == 'critical' and np.random.random() < 0.15:
        base_temp[severity] = 35.0  # hypothermia in critical patients

    vitals['temperature_c'] = round(np.clip(
        np.random.normal(base_temp[severity], 0.4), 34.0, 42.5
    ), 1)

    # --- SpO2 ---
    base_spo2 = {'low': 97, 'moderate': 94, 'high': 90, 'critical': 82}
    if disease == 'Respiratory':
        base_spo2[severity] -= 2
    if age >= 70:
        base_spo2[severity] -= 1

    vitals['spo2_percent'] = int(np.clip(
        np.random.normal(base_spo2[severity], 2), 60, 100
    ))

    # --- RESPIRATORY RATE ---
    if age < 1:
        base_rr = {'low': 35, 'moderate': 45, 'high': 55, 'critical': 65}
    elif age < 5:
        base_rr = {'low': 25, 'moderate': 32, 'high': 40, 'critical': 50}
    elif age < 12:
        base_rr = {'low': 20, 'moderate': 26, 'high': 32, 'critical': 42}
    else:
        base_rr = {'low': 16, 'moderate': 22, 'high': 28, 'critical': 36}

    if disease == 'Respiratory':
        base_rr[severity] += 4

    vitals['respiratory_rate'] = int(np.clip(
        np.random.normal(base_rr[severity], 3), 6, 60
    ))

    # --- BLOOD GLUCOSE ---
    base_glu = {'low': 100, 'moderate': 130, 'high': 180, 'critical': 280}
    if disease == 'Metabolic':
        if np.random.random() < 0.3:
            base_glu[severity] = np.random.choice([45, 50, 55, 60])  # hypoglycemia
        else:
            base_glu[severity] = np.random.choice([350, 400, 450, 500])  # DKA
    vitals['blood_glucose_mg'] = int(np.clip(
        np.random.normal(base_glu[severity], 20), 30, 550
    ))
    # ~30% missing (not always measured in rural clinics)
    if np.random.random() < 0.30:
        vitals['blood_glucose_mg'] = np.nan

    # --- PAIN SCORE ---
    base_pain = {'low': 2, 'moderate': 4, 'high': 6, 'critical': 8}
    if disease == 'Trauma':
        base_pain[severity] += 2
    vitals['pain_score'] = int(np.clip(
        np.random.normal(base_pain[severity], 1.5), 0, 10
    ))
    # ~10% missing (infants, unconscious patients)
    if age < 1 or (severity == 'critical' and np.random.random() < 0.3):
        vitals['pain_score'] = np.nan

    # --- CONSCIOUSNESS (AVPU) ---
    avpu_probs = {
        'low':      [0.95, 0.04, 0.01, 0.00],
        'moderate': [0.80, 0.15, 0.04, 0.01],
        'high':     [0.55, 0.25, 0.15, 0.05],
        'critical': [0.15, 0.25, 0.30, 0.30],
    }
    if disease == 'Neurological' and severity in ['high', 'critical']:
        avpu_probs[severity] = [0.10, 0.20, 0.30, 0.40]

    vitals['consciousness_level'] = np.random.choice(
        ['Alert', 'Voice', 'Pain', 'Unresponsive'],
        p=avpu_probs[severity]
    )

    return vitals

# Generate vitals for all patients
all_vitals = [
    generate_vitals(ages[i], severity_levels[i], diseases[i], genders[i], is_pregnant[i])
    for i in range(NUM_RECORDS)
]

# Extract into arrays
heart_rates = [v['heart_rate_bpm'] for v in all_vitals]
bp_sys = [v['bp_systolic_mmhg'] for v in all_vitals]
bp_dia = [v['bp_diastolic_mmhg'] for v in all_vitals]
temps = [v['temperature_c'] for v in all_vitals]
spo2s = [v['spo2_percent'] for v in all_vitals]
resp_rates = [v['respiratory_rate'] for v in all_vitals]
glucoses = [v['blood_glucose_mg'] for v in all_vitals]
pain_scores = [v['pain_score'] for v in all_vitals]
consciousness = [v['consciousness_level'] for v in all_vitals]

print(f"   ✓ Heart rates: {min(heart_rates)} – {max(heart_rates)} bpm")
print(f"   ✓ BP systolic: {min(bp_sys)} – {max(bp_sys)} mmHg")
print(f"   ✓ SpO2: {min(spo2s)} – {max(spo2s)}%")
print(f"   ✓ Temperature: {min(temps)} – {max(temps)} °C")


# =====================================================================
# MODULE 5: DERIVED CLINICAL SCORES
# =====================================================================
print("\n[5/8] Computing derived clinical scores...")

def compute_news_score(hr, sbp, rr, temp, spo2, consciousness):
    """Compute NEWS (National Early Warning Score) from vitals."""
    score = 0

    # Respiratory rate
    if rr <= 8: score += 3
    elif rr <= 11: score += 1
    elif rr <= 20: score += 0
    elif rr <= 24: score += 2
    else: score += 3

    # SpO2
    if spo2 <= 83: score += 3
    elif spo2 <= 85: score += 3
    elif spo2 <= 87: score += 2
    elif spo2 <= 89: score += 2
    elif spo2 <= 91: score += 1
    elif spo2 <= 93: score += 1
    else: score += 0

    # Temperature
    if temp <= 35.0: score += 3
    elif temp <= 35.9: score += 1
    elif temp <= 38.0: score += 0
    elif temp <= 38.9: score += 1
    elif temp <= 39.0: score += 1
    else: score += 2

    # Systolic BP
    if sbp <= 70: score += 3
    elif sbp <= 80: score += 3
    elif sbp <= 90: score += 2
    elif sbp <= 100: score += 1
    elif sbp <= 159: score += 0  # Normal range widens from actual NEWS
    elif sbp <= 199: score += 1  # Mildly elevated
    elif sbp <= 219: score += 2
    else: score += 3

    # Heart rate
    if hr <= 40: score += 3
    elif hr <= 50: score += 1
    elif hr <= 90: score += 0
    elif hr <= 110: score += 1
    elif hr <= 130: score += 2
    else: score += 3

    # Consciousness
    if consciousness == 'Alert': score += 0
    elif consciousness == 'Voice': score += 1
    elif consciousness == 'Pain': score += 2
    else: score += 3  # Unresponsive

    return score

news_scores = [
    compute_news_score(heart_rates[i], bp_sys[i], resp_rates[i],
                       temps[i], spo2s[i], consciousness[i])
    for i in range(NUM_RECORDS)
]

# Mean Arterial Pressure
map_values = [round((bp_sys[i] + 2 * bp_dia[i]) / 3, 1) for i in range(NUM_RECORDS)]

# Shock Index
shock_indices = [round(heart_rates[i] / max(bp_sys[i], 1), 2) for i in range(NUM_RECORDS)]

# Pulse Pressure
pulse_pressures = [bp_sys[i] - bp_dia[i] for i in range(NUM_RECORDS)]

print(f"   ✓ NEWS scores: {min(news_scores)} – {max(news_scores)}")
print(f"   ✓ MAP: {min(map_values)} – {max(map_values)} mmHg")
print(f"   ✓ Shock Index: {min(shock_indices)} – {max(shock_indices)}")


# =====================================================================
# MODULE 6: MEDICAL HISTORY FLAGS
# =====================================================================
print("\n[6/8] Generating medical history...")

has_diabetes = np.zeros(NUM_RECORDS, dtype=bool)
has_hypertension = np.zeros(NUM_RECORDS, dtype=bool)
has_heart_disease = np.zeros(NUM_RECORDS, dtype=bool)
has_asthma_copd = np.zeros(NUM_RECORDS, dtype=bool)
has_immunocompromised = np.zeros(NUM_RECORDS, dtype=bool)

for i in range(NUM_RECORDS):
    age = ages[i]
    # Prevalence increases with age
    if age >= 30:
        has_diabetes[i] = np.random.random() < (0.05 + (age - 30) * 0.004)
        has_hypertension[i] = np.random.random() < (0.08 + (age - 30) * 0.005)
    if age >= 40:
        has_heart_disease[i] = np.random.random() < (0.03 + (age - 40) * 0.004)
    # Asthma can occur at any age
    has_asthma_copd[i] = np.random.random() < (0.06 if age < 50 else 0.12)
    # Immunocompromised is rarer
    has_immunocompromised[i] = np.random.random() < 0.03

comorbidity_counts = (has_diabetes.astype(int) + has_hypertension.astype(int) +
                      has_heart_disease.astype(int) + has_asthma_copd.astype(int) +
                      has_immunocompromised.astype(int))

print(f"   ✓ Diabetes: {sum(has_diabetes)}")
print(f"   ✓ Hypertension: {sum(has_hypertension)}")
print(f"   ✓ Heart disease: {sum(has_heart_disease)}")
print(f"   ✓ Asthma/COPD: {sum(has_asthma_copd)}")
print(f"   ✓ Immunocompromised: {sum(has_immunocompromised)}")


# =====================================================================
# MODULE 7: SYMPTOM TEXT GENERATION (NLP data)
# =====================================================================
print("\n[7/8] Generating symptom descriptions...")

# Chief complaints mapped to disease categories
chief_complaints_map = {
    'Respiratory': ['cough', 'breathlessness', 'wheezing', 'chest_tightness',
                    'sore_throat', 'cold_and_flu', 'difficulty_breathing'],
    'Cardiac': ['chest_pain', 'palpitations', 'dizziness', 'fainting',
                'swollen_legs', 'chest_discomfort'],
    'Neurological': ['headache', 'seizure', 'numbness', 'confusion',
                     'blurred_vision', 'weakness_one_side', 'neck_stiffness'],
    'Gastrointestinal': ['abdominal_pain', 'vomiting', 'diarrhea', 'nausea',
                         'blood_in_stool', 'constipation', 'loss_of_appetite'],
    'Infectious': ['fever', 'body_aches', 'chills', 'rash', 'night_sweats',
                   'fatigue', 'swollen_lymph_nodes'],
    'Trauma': ['injury', 'fracture', 'wound', 'fall', 'burn',
               'bleeding', 'sprain'],
    'Metabolic': ['excessive_thirst', 'frequent_urination', 'weight_loss',
                  'fatigue', 'confusion', 'blurred_vision'],
    'Obstetric': ['abdominal_pain', 'vaginal_bleeding', 'contractions',
                  'headache', 'swollen_feet', 'reduced_fetal_movement'],
    'General': ['fatigue', 'weakness', 'fever', 'body_pain',
                'loss_of_appetite', 'weight_loss', 'malaise']
}

# Symptom description templates per disease category and severity
symptom_templates = {
    'Respiratory': {
        'low': [
            "Patient presents with mild {complaint} for {duration}. Reports occasional sneezing and runny nose. No difficulty breathing at rest. General condition appears stable.",
            "Complains of {complaint} that started {duration} ago. Mild sore throat and nasal congestion present. No wheezing or chest tightness. Patient is eating and drinking normally.",
            "Reports mild {complaint} with slight nasal congestion since {duration}. No fever reported. Breathing comfortable at rest. Patient able to perform daily activities without difficulty.",
        ],
        'moderate': [
            "Patient complains of persistent {complaint} for {duration}. Reports mild breathlessness during physical activity. Occasional wheezing heard. Cough is productive with yellowish sputum.",
            "Presents with worsening {complaint} over {duration}. Experiencing difficulty breathing when walking. Productive cough with greenish sputum. Mild chest tightness reported.",
            "Reports {complaint} with increasing severity over {duration}. Breathing becomes labored with minimal exertion. Using accessory muscles slightly. Patient appears tired and uncomfortable.",
        ],
        'high': [
            "Patient in respiratory distress with severe {complaint} for {duration}. Breathing is rapid and labored. Visible use of accessory muscles. Unable to complete full sentences without pausing for breath.",
            "Acute worsening of {complaint} since {duration}. Severe breathlessness at rest with audible wheezing. Patient sitting upright and leaning forward to breathe. Appears anxious and distressed.",
            "Severe {complaint} with significant respiratory difficulty for {duration}. Nasal flaring and intercostal retractions visible. Cannot lie flat. Speaking in short phrases only.",
        ],
        'critical': [
            "Patient in acute respiratory failure with severe {complaint}. Gasping for breath, cyanotic lips and fingertips. Barely conscious, unable to speak. Oxygen saturation critically low despite positioning.",
            "Critically ill with {complaint} and impending respiratory arrest. Agonal breathing pattern observed. Skin appears mottled and blue. Patient unresponsive to verbal commands. Immediate intervention required.",
            "Severe respiratory emergency with {complaint}. Patient is drowsy, breathing shallow and irregular. Central cyanosis present. Silent chest on auscultation. Life-threatening condition.",
        ]
    },
    'Cardiac': {
        'low': [
            "Patient reports mild {complaint} occurring occasionally over {duration}. No radiation of pain. Heart rhythm feels regular. No associated breathlessness or sweating.",
            "Complains of intermittent {complaint} for {duration}. Occurs mainly after heavy meals. No dizziness or fainting episodes. Patient appears comfortable and well.",
            "Mild {complaint} reported since {duration}. No chest pain at rest. Blood pressure has been slightly elevated recently. No history of sudden fainting.",
        ],
        'moderate': [
            "Patient presents with {complaint} for {duration}. Pain described as squeezing sensation in chest. Some radiation to left shoulder. Mild breathlessness with exertion.",
            "Reports {complaint} worsening over {duration}. Accompanied by occasional palpitations and lightheadedness. Mild ankle swelling noticed. Feels unusually tired.",
            "Experiencing {complaint} since {duration}. Discomfort increases with physical activity. Associated with mild nausea and heaviness in chest. Resting provides some relief.",
        ],
        'high': [
            "Severe {complaint} radiating to left arm and jaw for {duration}. Patient sweating profusely and feeling nauseous. Crushing pressure on chest. History of cardiac issues.",
            "Acute {complaint} with severe crushing sensation for {duration}. Profuse sweating, extreme anxiety, and feeling of impending doom. Pain unrelieved by rest.",
            "Patient reports intense {complaint} since {duration}. Associated with severe breathlessness, cold sweaty skin, and irregular heartbeat. Appears pale and distressed.",
        ],
        'critical': [
            "Patient found with {complaint}, now barely responsive. Severe hypotension, thready pulse, cold clammy skin. Suspected acute myocardial infarction. Immediate resuscitation needed.",
            "Cardiac emergency with {complaint}. Patient collapsed, pulse weak and irregular. Skin ashen gray and diaphoretic. Agonal breathing. Suspected cardiogenic shock.",
            "Critical cardiac event with {complaint}. Patient unresponsive, pulse barely palpable. Blood pressure dangerously low. Signs of poor organ perfusion throughout body.",
        ]
    },
    'Neurological': {
        'low': [
            "Patient complains of mild {complaint} for {duration}. No visual disturbances or nausea. Headache responds to rest and hydration. All limbs moving normally.",
            "Reports intermittent {complaint} since {duration}. No associated vomiting or neck stiffness. Vision and speech are normal. Able to walk without difficulty.",
            "Mild {complaint} present for {duration}. No numbness or tingling in extremities. Patient oriented to time, place, and person. No recent head injury reported.",
        ],
        'moderate': [
            "Patient presents with persistent {complaint} for {duration}. Associated with mild nausea and sensitivity to light. Some difficulty concentrating. Vision slightly blurred.",
            "Reports {complaint} increasing in intensity over {duration}. Accompanied by dizziness and unsteady gait. Mild tingling in one arm reported. Speech remains clear.",
            "Worsening {complaint} since {duration}. Patient reports episodes of confusion and difficulty with balance. Mild neck stiffness present. Vision occasionally doubles.",
        ],
        'high': [
            "Severe {complaint} with sudden onset {duration} ago. Patient reports worst headache of life. Neck rigidity present. Photophobia and vomiting. Suspected meningitis or hemorrhage.",
            "Acute {complaint} with weakness on one side of body since {duration}. Facial drooping noticed on right side. Speech is slurred. Unable to lift right arm. Stroke suspected.",
            "Patient experiencing {complaint} with sudden loss of vision in one eye since {duration}. Severe dizziness and inability to stand. Projectile vomiting. Needs urgent evaluation.",
        ],
        'critical': [
            "Patient found unconscious with history of {complaint}. GCS severely depressed. Fixed dilated pupils bilaterally. Decerebrate posturing observed. Critical neurological emergency.",
            "Status epilepticus with prolonged {complaint}. Continuous seizure activity for over 5 minutes. Patient cyanotic and unresponsive between episodes. Airway at risk.",
            "Critical neurological deterioration with {complaint}. Patient completely unresponsive. No pupillary response to light. Irregular breathing pattern. Signs of brainstem compression.",
        ]
    },
    'Gastrointestinal': {
        'low': [
            "Patient reports mild {complaint} for {duration}. Occasional loose stools but no blood. Able to tolerate oral fluids. Mild tenderness on abdominal examination.",
            "Complains of {complaint} since {duration}. Mild nausea without vomiting. Appetite slightly reduced. No signs of dehydration. Abdomen soft on palpation.",
            "Intermittent {complaint} present for {duration}. Reports mild bloating and gas. No fever or weight loss. Bowel habits slightly irregular but no blood.",
        ],
        'moderate': [
            "Patient presents with {complaint} for {duration}. Multiple episodes of vomiting and watery diarrhea. Showing early signs of dehydration with dry mouth. Moderate abdominal cramping.",
            "Worsening {complaint} over {duration}. Persistent nausea with reduced oral intake. Mild fever present. Abdominal tenderness in lower region. Some signs of dehydration.",
            "Reports {complaint} with increasing severity over {duration}. Unable to keep food down for past day. Moderate abdominal pain comes and goes. Skin turgor slightly reduced.",
        ],
        'high': [
            "Severe {complaint} for {duration}. Vomiting blood-stained material. Abdomen rigid and severely tender. Signs of significant dehydration. Patient appears acutely unwell.",
            "Acute {complaint} with severe cramping and bloody diarrhea since {duration}. More than 10 episodes in past day. Eyes sunken, skin dry. High fever present.",
            "Patient in significant distress with {complaint} for {duration}. Abdomen distended and tense. Severe pain with rebound tenderness. No bowel sounds heard. Suspected obstruction.",
        ],
        'critical': [
            "Acute abdomen with {complaint}. Patient in shock with rapid weak pulse. Abdomen board-like and severely distended. Suspected perforation or internal bleeding. Needs emergency surgery.",
            "Massive gastrointestinal bleeding with {complaint}. Vomiting large amounts of fresh blood. Hemodynamically unstable with plummeting blood pressure. Altered consciousness.",
            "Critical abdominal emergency with {complaint}. Patient barely responsive, abdomen rigid as wood. Signs of peritonitis and septic shock. Multi-organ compromise suspected.",
        ]
    },
    'Infectious': {
        'low': [
            "Patient complains of {complaint} for {duration}. Low grade fever with mild body aches. Eating and drinking adequately. No rash or joint swelling noticed.",
            "Reports {complaint} since {duration}. Mild fatigue and occasional chills. Temperature slightly above normal. No recent travel or known sick contacts.",
            "Mild {complaint} present for {duration}. General feeling of unwellness with slight sore throat. No significant fever. Maintains daily activities with some difficulty.",
        ],
        'moderate': [
            "Patient presents with {complaint} for {duration}. Persistent high fever with significant body aches and chills. Poor appetite and feeling very tired. Some joint pain reported.",
            "Worsening {complaint} over {duration}. Fever spiking especially at night with drenching sweats. Rash noticed on trunk and limbs. Patient confined to bed.",
            "Reports {complaint} with moderate fever since {duration}. Lymph nodes swollen in neck. Red eyes and runny nose. Difficulty eating due to throat pain.",
        ],
        'high': [
            "Severe {complaint} with high persistent fever for {duration}. Petechial rash spreading rapidly. Signs of dehydration despite fluid intake. Patient very lethargic and confused at times.",
            "Acute {complaint} with dangerous fever levels for {duration}. Neck stiffness and severe headache present. Rash does not blanch with pressure. Suspected meningococcal infection.",
            "Patient severely ill with {complaint} for {duration}. High fever not responding to antipyretics. Signs of organ compromise with reduced urine output. Suspected sepsis developing.",
        ],
        'critical': [
            "Septic shock from {complaint}. Patient febrile then suddenly hypothermic. Blood pressure not responding to fluids. Mottled skin with delayed capillary refill. Multi-organ failure developing.",
            "Critical infection with {complaint}. Patient delirious and unresponsive to commands. Disseminated rash with bleeding under skin. Hemodynamically unstable. Suspected severe sepsis.",
            "Life-threatening infection causing {complaint}. Patient in distributive shock with warm peripheries but dangerously low blood pressure. Altered mental status. Needs immediate intensive care.",
        ]
    },
    'Trauma': {
        'low': [
            "Patient sustained minor {complaint} approximately {duration} ago. Small superficial wound with minimal bleeding. Full range of motion preserved. No signs of fracture.",
            "Reports {complaint} from a minor fall {duration} ago. Mild swelling and tenderness at injury site. No deformity visible. Patient able to bear weight normally.",
            "Minor {complaint} occurred {duration} ago. Small bruise and scratches present. No head injury or loss of consciousness. Patient alert and comfortable.",
        ],
        'moderate': [
            "Patient presents with {complaint} sustained {duration} ago. Moderate swelling and pain at injury site. Limited range of motion. Possible fracture needs X-ray evaluation.",
            "Reports {complaint} from a fall {duration} ago. Deep laceration requiring suturing. Moderate blood loss from wound. Pain score moderate. Tetanus status unknown.",
            "Sustained {complaint} during physical activity {duration} ago. Significant swelling and bruising. Unable to move affected limb fully. No neurovascular compromise distally.",
        ],
        'high': [
            "Severe {complaint} from high-impact incident {duration} ago. Obvious deformity of affected limb. Severe pain and inability to bear weight. Open wound with exposed bone visible.",
            "Major {complaint} with significant bleeding for {duration}. Large deep wound with active hemorrhage. Patient pale and tachycardic. Multiple injury sites identified.",
            "Severe {complaint} from road accident {duration} ago. Head injury with brief loss of consciousness. Severe pain in chest and abdomen. Difficulty breathing since injury.",
        ],
        'critical': [
            "Polytrauma from major accident with {complaint}. Patient unresponsive, massive external bleeding. Obvious chest and abdominal injuries. Airway compromised. Needs immediate surgical intervention.",
            "Critical {complaint} with life-threatening hemorrhage. Patient in hypovolemic shock, blood pressure barely measurable. Compound fractures with arterial bleeding. Losing consciousness.",
            "Severe head trauma with {complaint}. GCS critically low. Unequal pupils suggesting brain herniation. Unstable cervical spine suspected. Multiple organ injuries likely.",
        ]
    },
    'Metabolic': {
        'low': [
            "Patient complains of {complaint} for {duration}. Mild increase in thirst and urination. Blood sugar slightly above normal range. No signs of ketosis.",
            "Reports {complaint} since {duration}. General fatigue and mild weight change noticed. No acute distress. Routine monitoring showed slightly abnormal glucose.",
            "Mild {complaint} present for {duration}. Feeling more tired than usual with occasional blurred vision. Diet has been irregular recently. No vomiting or abdominal pain.",
        ],
        'moderate': [
            "Patient presents with {complaint} worsening over {duration}. Significantly increased thirst and urination. Feeling dizzy and nauseous. Blood sugar moderately elevated.",
            "Reports {complaint} for {duration}. Fruity odor on breath noticed. Moderate nausea and abdominal discomfort. Signs of mild dehydration. Known diabetic with poor compliance.",
            "Worsening {complaint} since {duration}. Rapid weight loss and persistent fatigue. Unable to work for several days. Blurred vision and tingling in feet.",
        ],
        'high': [
            "Severe {complaint} for {duration}. Kussmaul breathing pattern observed. Strong fruity breath odor. Severely dehydrated with sunken eyes. Suspected diabetic ketoacidosis.",
            "Acute {complaint} with dangerously low blood sugar found {duration} ago. Patient confused, trembling, and sweating profusely. Unable to take oral glucose. Becoming drowsy.",
            "Patient very unwell with {complaint} for {duration}. Severe vomiting preventing any oral intake. Extremely thirsty but unable to keep fluids down. Rapid deep breathing.",
        ],
        'critical': [
            "Diabetic emergency with {complaint}. Patient found unconscious with severely abnormal blood sugar. Deep rapid breathing, severe dehydration. No response to stimuli. Life-threatening metabolic crisis.",
            "Critical metabolic derangement from {complaint}. Patient in metabolic coma. Blood glucose at dangerous extreme. Signs of cerebral edema developing. Needs immediate intensive management.",
            "Severe hypoglycemic crisis causing {complaint}. Patient having seizures from critically low blood sugar. Unresponsive, diaphoretic. No IV access established. Needs emergency glucose.",
        ]
    },
    'Obstetric': {
        'low': [
            "Pregnant patient reports mild {complaint} for {duration}. Fetal movements felt normally. No vaginal bleeding or fluid leakage. Mild ankle swelling present. Vitals stable.",
            "Complains of {complaint} since {duration}. Mild lower back pain and fatigue. Fetal heart rate within normal limits. No contractions felt. Routine checkup findings.",
            "Reports {complaint} for {duration}. Occasional mild cramping without regular pattern. No bleeding or discharge. Baby moving well. Patient able to eat and drink normally.",
        ],
        'moderate': [
            "Pregnant patient with {complaint} for {duration}. Irregular contractions becoming more frequent. Mild vaginal spotting noticed. Fetal movements slightly reduced. Blood pressure mildly elevated.",
            "Reports {complaint} worsening over {duration}. Persistent headache with mild visual changes. Ankle swelling significantly increased. Protein may be present in urine. Preeclampsia concern.",
            "Presents with {complaint} since {duration}. Regular abdominal tightening every 10 minutes. Possible premature labor if early gestation. Mild bleeding per vaginum.",
        ],
        'high': [
            "Severe {complaint} in third trimester for {duration}. Heavy vaginal bleeding with clots. Severe abdominal pain and uterine tenderness. Suspected placental abruption. Fetal heart rate abnormal.",
            "Acute {complaint} with seizures in pregnant patient since {duration}. Blood pressure dangerously elevated. Visual disturbances and severe headache. Suspected eclampsia. Emergency delivery may be needed.",
            "Pregnant patient with severe {complaint} for {duration}. Profuse vaginal bleeding soaking pads rapidly. Feeling dizzy and faint. Baby not moving. Hemodynamically becoming unstable.",
        ],
        'critical': [
            "Obstetric emergency with {complaint}. Massive hemorrhage, patient in shock. No fetal heart tones detected. Uterus hard and tender. Suspected complete abruption. Needs immediate cesarean.",
            "Critical {complaint} in pregnant patient. Status eclampticus with repeated seizures. Blood pressure at stroke-level. Patient losing consciousness between seizures. Immediate delivery required.",
            "Life-threatening obstetric hemorrhage with {complaint}. Blood loss massive and ongoing. Patient barely conscious, pulse thready. Emergency blood transfusion and surgery needed immediately.",
        ]
    },
    'General': {
        'low': [
            "Patient presents with {complaint} for {duration}. General feeling of unwellness without specific localizing symptoms. Appetite mildly reduced. Sleeping more than usual.",
            "Reports {complaint} since {duration}. Mild body aches and tiredness. No significant fever or weight loss. All systems review otherwise unremarkable. Possibly stress related.",
            "Complains of {complaint} for {duration}. Non-specific discomfort that is hard to localize. Mild fatigue and reduced energy levels. No alarming symptoms identified.",
        ],
        'moderate': [
            "Patient with worsening {complaint} for {duration}. Persistent weakness affecting daily activities. Moderate weight loss noticed over recent weeks. Appetite significantly reduced.",
            "Reports {complaint} progressing over {duration}. Unable to perform routine work. Intermittent low-grade fever. General body pain and malaise. Needs further investigation.",
            "Presents with {complaint} for {duration}. Feeling progressively weaker and more fatigued. Some shortness of breath with exertion. Night sweats reported occasionally.",
        ],
        'high': [
            "Severe {complaint} with marked functional decline over {duration}. Patient bedridden and unable to care for self. Significant weight loss and muscle wasting. Looks acutely unwell.",
            "Acute deterioration with {complaint} since {duration}. Multiple system complaints developing rapidly. Patient confused and lethargic. Skin appears pale and jaundiced.",
            "Patient very ill with {complaint} for {duration}. Severe weakness preventing any movement. Unable to eat or drink for days. Signs of malnutrition and dehydration together.",
        ],
        'critical': [
            "Patient brought in unresponsive with history of {complaint}. Severely cachectic and dehydrated. Multiple organ systems appear compromised. Origin of deterioration unclear. Critical condition.",
            "Found collapsed with {complaint}. Patient in altered mental state, unable to communicate. General condition extremely poor. Signs of systemic failure. Needs immediate multi-system assessment.",
            "Critical presentation with {complaint}. Patient moribund, barely responding to painful stimuli. Generalized edema with ascites. Possible multi-organ dysfunction. Prognosis grave.",
        ]
    }
}

# Duration text variations
duration_texts = {
    'acute': ['a few hours', 'since this morning', 'since last night', 
              'approximately 2 hours', 'about 3 hours', 'since early today'],
    'subacute': ['2 days', 'the past 3 days', 'about 4 days', 'nearly a week',
                 'the last 5 days', '3 to 4 days'],
    'chronic': ['over 2 weeks', 'the past month', 'several weeks', 
                'more than 10 days', 'about 3 weeks', 'over a fortnight']
}

def generate_symptom_text(disease, severity, duration_hours):
    """Generate a natural language symptom description."""
    # Pick chief complaint
    complaint = random.choice(chief_complaints_map[disease]).replace('_', ' ')

    # Pick duration text
    if duration_hours < 12:
        dur_text = random.choice(duration_texts['acute'])
    elif duration_hours < 120:
        dur_text = random.choice(duration_texts['subacute'])
    else:
        dur_text = random.choice(duration_texts['chronic'])

    # Pick template
    templates = symptom_templates[disease][severity]
    template = random.choice(templates)

    return template.format(complaint=complaint, duration=dur_text)

# Generate symptom duration
symptom_durations = []
for sev in severity_levels:
    if sev == 'critical':
        dur = np.random.choice([0.5, 1, 2, 3, 6, 12, 24], p=[0.15, 0.2, 0.2, 0.15, 0.15, 0.1, 0.05])
    elif sev == 'high':
        dur = np.random.choice([1, 3, 6, 12, 24, 48, 72], p=[0.1, 0.15, 0.2, 0.2, 0.2, 0.1, 0.05])
    elif sev == 'moderate':
        dur = np.random.choice([6, 12, 24, 48, 72, 120, 168], p=[0.05, 0.1, 0.2, 0.25, 0.2, 0.15, 0.05])
    else:
        dur = np.random.choice([12, 24, 48, 72, 120, 168, 336, 720], p=[0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05])
    symptom_durations.append(dur)

# Generate chief complaints
chief_complaints = [random.choice(chief_complaints_map[d]) for d in diseases]

# Generate full symptom text
symptom_texts = [
    generate_symptom_text(diseases[i], severity_levels[i], symptom_durations[i])
    for i in range(NUM_RECORDS)
]

# Word count stats
word_counts = [len(t.split()) for t in symptom_texts]
print(f"   ✓ Symptom texts generated")
print(f"   ✓ Word count range: {min(word_counts)} – {max(word_counts)} words")
print(f"   ✓ Average length: {np.mean(word_counts):.0f} words")
print(f"   ✓ Unique chief complaints: {len(set(chief_complaints))}")


# =====================================================================
# MODULE 8: RED FLAGS, CONTEXT & FINAL LABELS
# =====================================================================
print("\n[8/8] Computing red flags, context features, and final labels...")

# Red flag detection
red_flag_vitals = np.array([
    (spo2s[i] < 80) or (bp_sys[i] < 70) or (heart_rates[i] > 170 and ages[i] > 12) or
    (temps[i] > 41.5) or (temps[i] < 34.0) or (resp_rates[i] > 40 and ages[i] > 12) or
    (heart_rates[i] < 35)
    for i in range(NUM_RECORDS)
])

red_flag_keywords = ['unconscious', 'unresponsive', 'cardiac arrest',
                     'not breathing', 'cyanotic', 'hemorrhage',
                     'status epilepticus', 'brainstem', 'herniation',
                     'barely responsive', 'moribund', 'agonal']

red_flag_symptoms = np.array([
    any(kw in symptom_texts[i].lower() for kw in red_flag_keywords)
    for i in range(NUM_RECORDS)
])

# Red flag combos (dangerous symptom+vital combinations - strict)
red_flag_combo = np.array([
    (temps[i] > 39.5 and 'neck stiffness' in symptom_texts[i].lower() and
     consciousness[i] in ['Pain', 'Unresponsive']) or  # meningitis with AMS
    (spo2s[i] < 85 and resp_rates[i] > 30) or  # severe resp failure
    (consciousness[i] == 'Unresponsive' and temps[i] > 40.0)  # septic coma
    for i in range(NUM_RECORDS)
])

any_red_flag = red_flag_vitals | red_flag_symptoms | red_flag_combo
red_flag_count = red_flag_vitals.astype(int) + red_flag_symptoms.astype(int) + red_flag_combo.astype(int)

# Context features
time_of_day = np.random.choice(
    ['Morning', 'Afternoon', 'Evening', 'Night'],
    NUM_RECORDS, p=[0.30, 0.30, 0.25, 0.15]
)

seasons = np.random.choice(
    ['Summer', 'Monsoon', 'Winter', 'Post-Monsoon'],
    NUM_RECORDS, p=[0.25, 0.30, 0.25, 0.20]
)

nearest_hospital_km = np.round(np.random.lognormal(3.2, 0.6, NUM_RECORDS), 1)
nearest_hospital_km = np.clip(nearest_hospital_km, 5, 200)

arrival_modes = []
for sev in severity_levels:
    if sev == 'critical':
        mode = np.random.choice(['Walk-in', 'Carried', 'Ambulance', 'Referred'], p=[0.05, 0.25, 0.50, 0.20])
    elif sev == 'high':
        mode = np.random.choice(['Walk-in', 'Carried', 'Ambulance', 'Referred'], p=[0.15, 0.30, 0.35, 0.20])
    elif sev == 'moderate':
        mode = np.random.choice(['Walk-in', 'Carried', 'Ambulance', 'Referred'], p=[0.50, 0.25, 0.15, 0.10])
    else:
        mode = np.random.choice(['Walk-in', 'Carried', 'Ambulance', 'Referred'], p=[0.80, 0.10, 0.05, 0.05])
    arrival_modes.append(mode)

# =====================================================================
# FINAL TARGET LABELS (clinically derived)
# =====================================================================

# Risk level — based on NEWS score + red flags + severity
risk_levels = []
for i in range(NUM_RECORDS):
    ns = news_scores[i]
    if any_red_flag[i] or ns >= 11:
        risk_levels.append('Critical')
    elif ns >= 7 or (ns >= 5 and any_red_flag[i]):
        risk_levels.append('High')
    elif ns >= 4:
        risk_levels.append('Moderate')
    else:
        risk_levels.append('Low')

# Urgency — based on risk level + additional factors
urgency_levels = []
for i in range(NUM_RECORDS):
    rl = risk_levels[i]
    if rl == 'Critical':
        urgency_levels.append('Emergency')
    elif rl == 'High':
        if symptom_durations[i] < 6 or comorbidity_counts[i] >= 3:
            urgency_levels.append('Emergency')
        else:
            urgency_levels.append('Urgent')
    elif rl == 'Moderate':
        if comorbidity_counts[i] >= 2 or symptom_durations[i] < 6:
            urgency_levels.append('Urgent')
        else:
            urgency_levels.append('Semi-Urgent')
    else:  # Low
        if symptom_durations[i] < 3 or (ages[i] < 1 or ages[i] > 75):
            urgency_levels.append('Semi-Urgent')
        else:
            urgency_levels.append('Non-Urgent')

print(f"   ✓ Red flags triggered: {sum(any_red_flag)} patients ({sum(any_red_flag)/NUM_RECORDS*100:.1f}%)")
print(f"\n   Final Risk Distribution:")
for r in ['Low', 'Moderate', 'High', 'Critical']:
    c = risk_levels.count(r)
    print(f"     {r}: {c} ({c/NUM_RECORDS*100:.1f}%)")
print(f"\n   Urgency Distribution:")
for u in ['Non-Urgent', 'Semi-Urgent', 'Urgent', 'Emergency']:
    c = urgency_levels.count(u)
    print(f"     {u}: {c} ({c/NUM_RECORDS*100:.1f}%)")


# =====================================================================
# ASSEMBLE FINAL DATAFRAME
# =====================================================================
print("\n" + "=" * 60)
print("  Assembling final dataset...")
print("=" * 60)

df = pd.DataFrame({
    # Demographics
    'patient_id': patient_ids,
    'age': ages,
    'age_group': age_groups,
    'gender': genders,
    'weight_kg': weights,
    'is_pregnant': is_pregnant,

    # Core Vitals
    'heart_rate_bpm': heart_rates,
    'bp_systolic_mmhg': bp_sys,
    'bp_diastolic_mmhg': bp_dia,
    'temperature_c': temps,
    'spo2_percent': spo2s,
    'respiratory_rate': resp_rates,
    'blood_glucose_mg': glucoses,
    'pain_score': pain_scores,
    'consciousness_level': consciousness,

    # Derived Scores
    'news_score': news_scores,
    'map_mmhg': map_values,
    'shock_index': shock_indices,
    'pulse_pressure': pulse_pressures,

    # Medical History
    'has_diabetes': has_diabetes,
    'has_hypertension': has_hypertension,
    'has_heart_disease': has_heart_disease,
    'has_asthma_copd': has_asthma_copd,
    'has_immunocompromised': has_immunocompromised,
    'comorbidity_count': comorbidity_counts,

    # Symptom Text Data
    'chief_complaint': chief_complaints,
    'symptom_description': symptom_texts,
    'symptom_duration_hours': symptom_durations,

    # Red Flags
    'red_flag_vitals': red_flag_vitals,
    'red_flag_symptoms': red_flag_symptoms,
    'red_flag_combo': red_flag_combo,
    'any_red_flag': any_red_flag,
    'red_flag_count': red_flag_count,

    # Context
    'time_of_day': time_of_day,
    'season': seasons,
    'nearest_hospital_km': nearest_hospital_km,
    'arrival_mode': arrival_modes,

    # TARGET LABELS
    'risk_level': risk_levels,
    'urgency': urgency_levels,
    'disease_category': diseases,
})

# Save to CSV
output_path = '/home/claude/triage_dataset_10k.csv'
df.to_csv(output_path, index=False)

print(f"\n  ✅ Dataset saved to: {output_path}")
print(f"  ✅ Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\n  Column Summary:")
print(f"  {'─' * 50}")
for col in df.columns:
    dtype = df[col].dtype
    nulls = df[col].isna().sum()
    null_str = f" ({nulls} missing)" if nulls > 0 else ""
    print(f"  {col:<30} {str(dtype):<12}{null_str}")

print(f"\n  {'─' * 50}")
print(f"  Dataset generation complete!")
print(f"  {'─' * 50}")

# Quick sample
print(f"\n  Sample record (Patient P00001):")
print(f"  {'─' * 50}")
sample = df.iloc[0]
for col in ['patient_id', 'age', 'gender', 'heart_rate_bpm', 'bp_systolic_mmhg',
            'spo2_percent', 'news_score', 'chief_complaint', 'risk_level', 'urgency',
            'disease_category']:
    print(f"  {col}: {sample[col]}")
print(f"\n  Symptom text (truncated):")
print(f"  {sample['symptom_description'][:120]}...")
