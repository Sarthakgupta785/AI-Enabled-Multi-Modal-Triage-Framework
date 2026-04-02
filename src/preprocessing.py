import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# =====================================================================
# CONFIGURATION
# =====================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'triage_dataset_10k.csv')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

os.makedirs(PROCESSED_DIR, exist_ok=True)

SEED = 42

print("=" * 60)
print("  PREPROCESSING PIPELINE")
print("=" * 60)



print("\n[1/7] Loading raw dataset...")

df = pd.read_csv(RAW_DATA_PATH)
df_raw = df.copy()  # keep original for missingness flags

print(f"   Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"   Missing values:")
missing = df.isnull().sum()
for col in missing[missing > 0].index:
    print(f"     {col}: {missing[col]} ({missing[col]/len(df)*100:.1f}%)")


# =====================================================================
# STEP 2: HANDLE MISSING VALUES
# =====================================================================
print("\n[2/7] Handling missing values...")

# Create missingness indicator features BEFORE imputing
# (whether a value was missing is itself informative — e.g. glucose not measured
#  means the clinic didn't have the equipment, which signals resource constraints)
df['glucose_was_missing'] = df_raw['blood_glucose_mg'].isna().astype(int)
df['pain_was_missing'] = df_raw['pain_score'].isna().astype(int)

# Impute blood glucose with median (30% missing)
glucose_median = df['blood_glucose_mg'].median()
df['blood_glucose_mg'].fillna(glucose_median, inplace=True)

# Impute pain score with median (10% missing — infants/unconscious)
pain_median = df['pain_score'].median()
df['pain_score'].fillna(pain_median, inplace=True)

print(f"   Blood glucose: imputed {df['glucose_was_missing'].sum()} values with median ({glucose_median})")
print(f"   Pain score: imputed {df['pain_was_missing'].sum()} values with median ({pain_median})")
print(f"   Created 2 missingness indicator features")
remaining = df.isnull().sum().sum()
if remaining > 0:
    print(f"   Remaining missing: {remaining} (will be handled in encoding step)")
else:
    print(f"   Remaining missing: 0 — all clean!")


# =====================================================================
# STEP 3: ENCODE CATEGORICAL FEATURES
# =====================================================================
print("\n[3/7] Encoding categorical features...")

label_encoders = {}

# --- TARGET LABELS ---

# Risk level (ordinal — has a natural order)
risk_order = {'Low': 0, 'Moderate': 1, 'High': 2, 'Critical': 3}
df['risk_level_encoded'] = df['risk_level'].map(risk_order)

# Urgency (ordinal)
urgency_order = {'Non-Urgent': 0, 'Semi-Urgent': 1, 'Urgent': 2, 'Emergency': 3}
df['urgency_encoded'] = df['urgency'].map(urgency_order)

# Disease category (nominal — no natural order)
le_disease = LabelEncoder()
df['disease_category_encoded'] = le_disease.fit_transform(df['disease_category'])
label_encoders['disease_category'] = le_disease

# --- INPUT FEATURES ---

# Gender: binary
df['gender_encoded'] = (df['gender'] == 'M').astype(int)

# Age group: ordinal
age_group_order = {'Neonate': 0, 'Infant': 1, 'Child': 2, 'Adolescent': 3, 'Adult': 4, 'Elderly': 5}
df['age_group_encoded'] = df['age_group'].map(age_group_order)

# Consciousness (AVPU scale): ordinal — Alert is best, Unresponsive is worst
consciousness_order = {'Alert': 0, 'Voice': 1, 'Pain': 2, 'Unresponsive': 3}
df['consciousness_encoded'] = df['consciousness_level'].map(consciousness_order)

# Chief complaint: many categories, use LabelEncoder
le_complaint = LabelEncoder()
df['chief_complaint_encoded'] = le_complaint.fit_transform(df['chief_complaint'])
label_encoders['chief_complaint'] = le_complaint

# Time of day: ordinal
time_order = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
df['time_of_day_encoded'] = df['time_of_day'].map(time_order)

# Season: nominal
le_season = LabelEncoder()
df['season_encoded'] = le_season.fit_transform(df['season'])
label_encoders['season'] = le_season

# Arrival mode: ordinal (severity-based)
arrival_order = {'Walk-in': 0, 'Carried': 1, 'Referred': 2, 'Ambulance': 3}
df['arrival_mode_encoded'] = df['arrival_mode'].map(arrival_order)

# Boolean columns → 0/1 integers
bool_columns = [
    'is_pregnant', 'has_diabetes', 'has_hypertension', 'has_heart_disease',
    'has_asthma_copd', 'has_immunocompromised', 'red_flag_vitals',
    'red_flag_symptoms', 'red_flag_combo', 'any_red_flag'
]
for col in bool_columns:
    df[col] = df[col].astype(int)

print(f"   Encoded targets: risk_level, urgency, disease_category")
print(f"   Encoded {len(bool_columns)} boolean features")
print(f"   Encoded all categorical features")


# =====================================================================
# STEP 4: DEFINE FEATURE GROUPS
# =====================================================================
print("\n[4/7] Organizing feature groups...")

# All features the VITALS MODEL (tabular) will use
vitals_features = [
    # Demographics (5)
    'age', 'gender_encoded', 'weight_kg', 'is_pregnant', 'age_group_encoded',
    # Core vitals (8)
    'heart_rate_bpm', 'bp_systolic_mmhg', 'bp_diastolic_mmhg',
    'temperature_c', 'spo2_percent', 'respiratory_rate',
    'blood_glucose_mg', 'pain_score', 'consciousness_encoded',
    # Derived scores (4)
    'news_score', 'map_mmhg', 'shock_index', 'pulse_pressure',
    # Medical history (6)
    'has_diabetes', 'has_hypertension', 'has_heart_disease',
    'has_asthma_copd', 'has_immunocompromised', 'comorbidity_count',
    # Red flags (5)
    'red_flag_vitals', 'red_flag_symptoms', 'red_flag_combo',
    'any_red_flag', 'red_flag_count',
    # Context (4)
    'time_of_day_encoded', 'season_encoded', 'nearest_hospital_km',
    'arrival_mode_encoded',
    # Missingness indicators (2)
    'glucose_was_missing', 'pain_was_missing',
    # Chief complaint (1)
    'chief_complaint_encoded',
]

# Text feature for NLP MODEL
text_feature = 'symptom_description'

# Targets
target_risk = 'risk_level_encoded'
target_urgency = 'urgency_encoded'
target_disease = 'disease_category_encoded'

print(f"   Vitals model features: {len(vitals_features)} columns")
print(f"   NLP model feature: symptom_description (text)")
print(f"   Targets: risk_level, urgency, disease_category")


# =====================================================================
# STEP 5: SCALE NUMERICAL FEATURES
# =====================================================================
print("\n[5/7] Scaling numerical features...")

# Only scale continuous numerical columns (not encoded categoricals or booleans)
numerical_cols = [
    'age', 'weight_kg', 'heart_rate_bpm', 'bp_systolic_mmhg', 'bp_diastolic_mmhg',
    'temperature_c', 'spo2_percent', 'respiratory_rate', 'blood_glucose_mg',
    'pain_score', 'news_score', 'map_mmhg', 'shock_index', 'pulse_pressure',
    'nearest_hospital_km', 'comorbidity_count', 'red_flag_count'
]

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print(f"   StandardScaler applied to {len(numerical_cols)} numerical columns")
print(f"   Example — heart_rate_bpm: mean={df['heart_rate_bpm'].mean():.4f}, std={df['heart_rate_bpm'].std():.4f}")


# =====================================================================
# STEP 6: TRAIN / VALIDATION / TEST SPLIT (80/10/10)
# =====================================================================
print("\n[6/7] Splitting into train/val/test...")

# Extract arrays
X_vitals = df[vitals_features].values
X_text = df[text_feature].values
y_risk = df[target_risk].values
y_urgency = df[target_urgency].values
y_disease = df[target_disease].values

# Split 1: 80% train, 20% temp (stratified by risk_level)
(X_vitals_train, X_vitals_temp,
 X_text_train, X_text_temp,
 y_risk_train, y_risk_temp,
 y_urgency_train, y_urgency_temp,
 y_disease_train, y_disease_temp) = train_test_split(
    X_vitals, X_text, y_risk, y_urgency, y_disease,
    test_size=0.20, random_state=SEED, stratify=y_risk
)

# Split 2: 50/50 of 20% → 10% val + 10% test
(X_vitals_val, X_vitals_test,
 X_text_val, X_text_test,
 y_risk_val, y_risk_test,
 y_urgency_val, y_urgency_test,
 y_disease_val, y_disease_test) = train_test_split(
    X_vitals_temp, X_text_temp,
    y_risk_temp, y_urgency_temp, y_disease_temp,
    test_size=0.50, random_state=SEED, stratify=y_risk_temp
)

print(f"   Train: {X_vitals_train.shape[0]:,} samples ({X_vitals_train.shape[0]/len(df)*100:.0f}%)")
print(f"   Val:   {X_vitals_val.shape[0]:,} samples ({X_vitals_val.shape[0]/len(df)*100:.0f}%)")
print(f"   Test:  {X_vitals_test.shape[0]:,} samples ({X_vitals_test.shape[0]/len(df)*100:.0f}%)")

# Verify stratification maintained class balance
print(f"\n   Class distribution preserved across splits:")
for name, y in [('Train', y_risk_train), ('Val', y_risk_val), ('Test', y_risk_test)]:
    unique, counts = np.unique(y, return_counts=True)
    pcts = [f"{c/len(y)*100:.1f}%" for c in counts]
    risk_names = ['Low', 'Mod', 'High', 'Crit']
    dist_str = " | ".join([f"{risk_names[int(u)]}:{p}" for u, p in zip(unique, pcts)])
    print(f"     {name:5s}: {dist_str}")


# =====================================================================
# STEP 7: SAVE EVERYTHING
# =====================================================================
print("\n[7/7] Saving processed data...")

# Save numpy arrays
splits = {
    'X_vitals_train': X_vitals_train, 'X_vitals_val': X_vitals_val, 'X_vitals_test': X_vitals_test,
    'X_text_train': X_text_train, 'X_text_val': X_text_val, 'X_text_test': X_text_test,
    'y_risk_train': y_risk_train, 'y_risk_val': y_risk_val, 'y_risk_test': y_risk_test,
    'y_urgency_train': y_urgency_train, 'y_urgency_val': y_urgency_val, 'y_urgency_test': y_urgency_test,
    'y_disease_train': y_disease_train, 'y_disease_val': y_disease_val, 'y_disease_test': y_disease_test,
}

for name, arr in splits.items():
    np.save(os.path.join(PROCESSED_DIR, f'{name}.npy'), arr)

# Save metadata
metadata = {
    'vitals_features': vitals_features,
    'numerical_cols': numerical_cols,
    'risk_order': risk_order,
    'urgency_order': urgency_order,
    'age_group_order': age_group_order,
    'consciousness_order': consciousness_order,
    'time_order': time_order,
    'arrival_order': arrival_order,
    'train_size': int(X_vitals_train.shape[0]),
    'val_size': int(X_vitals_val.shape[0]),
    'test_size': int(X_vitals_test.shape[0]),
    'num_vitals_features': len(vitals_features),
    'glucose_impute_median': float(glucose_median),
    'pain_impute_median': float(pain_median),
}

with open(os.path.join(PROCESSED_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

# Save scaler and encoders (needed for deployment/inference)
with open(os.path.join(PROCESSED_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(PROCESSED_DIR, 'label_encoders.pkl'), 'wb') as f:
    pickle.dump(label_encoders, f)

# Save full processed dataframe (useful for exploration)
df.to_csv(os.path.join(PROCESSED_DIR, 'triage_processed.csv'), index=False)

# Print summary of saved files
print(f"\n   Saved files in data/processed/:")
for f_name in sorted(os.listdir(PROCESSED_DIR)):
    size_kb = os.path.getsize(os.path.join(PROCESSED_DIR, f_name)) / 1024
    print(f"     {f_name:<40} {size_kb:>8.1f} KB")

print(f"\n{'=' * 60}")
print(f"  PREPROCESSING COMPLETE!")
print(f"  Total features for vitals model: {len(vitals_features)}")
print(f"  Total processed files: {len(os.listdir(PROCESSED_DIR))}")
print(f"  Next step: python src/vitals_model.py")
print(f"{'=' * 60}")