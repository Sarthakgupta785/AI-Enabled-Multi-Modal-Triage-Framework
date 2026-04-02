import numpy as np
import os
import json
import re
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  RED FLAG ENGINE — Clinical Safety Override")
print("=" * 60)

BASE_DIR = os.path.dirname(__file__)
REPORT_DIR = os.path.join(BASE_DIR, '..', 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)


# =====================================================================
# RED FLAG RULES DEFINITION
# =====================================================================

class RedFlagEngine:
    """
    Rule-based clinical red flag detection system.
    Overrides ML predictions when critical danger signs are found.
    Can only ESCALATE risk — never downgrade.
    """

    def __init__(self):
        # ---- VITAL SIGN THRESHOLDS ----
        # Based on NEWS protocol critical ranges
        self.vital_rules = {
            'spo2_critical': {'param': 'spo2_percent', 'op': '<', 'value': 85,
                              'message': 'SpO2 critically low (<85%)'},
            'spo2_danger': {'param': 'spo2_percent', 'op': '<', 'value': 90,
                            'message': 'SpO2 dangerously low (<90%)'},
            'bp_shock': {'param': 'bp_systolic_mmhg', 'op': '<', 'value': 80,
                         'message': 'Systolic BP indicates shock (<80 mmHg)'},
            'bp_crisis': {'param': 'bp_systolic_mmhg', 'op': '>', 'value': 200,
                          'message': 'Hypertensive crisis (>200 mmHg)'},
            'hr_critical_high': {'param': 'heart_rate_bpm', 'op': '>', 'value': 160,
                                 'message': 'Heart rate dangerously high (>160 bpm)'},
            'hr_critical_low': {'param': 'heart_rate_bpm', 'op': '<', 'value': 38,
                                'message': 'Heart rate dangerously low (<38 bpm)'},
            'temp_hyperpyrexia': {'param': 'temperature_c', 'op': '>', 'value': 41.0,
                                  'message': 'Hyperpyrexia - dangerously high fever (>41°C)'},
            'temp_hypothermia': {'param': 'temperature_c', 'op': '<', 'value': 34.5,
                                 'message': 'Severe hypothermia (<34.5°C)'},
            'rr_critical_high': {'param': 'respiratory_rate', 'op': '>', 'value': 35,
                                  'message': 'Respiratory rate critically elevated (>35)'},
            'rr_critical_low': {'param': 'respiratory_rate', 'op': '<', 'value': 7,
                                 'message': 'Respiratory rate critically low (<7)'},
            'glucose_hypo': {'param': 'blood_glucose_mg', 'op': '<', 'value': 50,
                             'message': 'Severe hypoglycemia (<50 mg/dL)'},
            'glucose_dka': {'param': 'blood_glucose_mg', 'op': '>', 'value': 400,
                            'message': 'Suspected DKA - glucose >400 mg/dL'},
        }

        # ---- SYMPTOM TEXT DANGER KEYWORDS ----
        # Grouped by severity
        self.critical_keywords = [
            'unconscious', 'unresponsive', 'not breathing', 'cardiac arrest',
            'agonal', 'pulseless', 'cpr', 'resuscitation', 'moribund',
            'respiratory arrest', 'no pulse', 'asystole'
        ]

        self.high_danger_keywords = [
            'seizure', 'convulsion', 'status epilepticus',
            'massive bleeding', 'hemorrhage', 'hemorrhaging',
            'chest pain radiating', 'crushing chest pain',
            'stroke', 'facial drooping', 'slurred speech',
            'anaphylaxis', 'airway obstruction',
            'cyanotic', 'cyanosis', 'blue lips',
            'meningitis', 'neck stiffness',
            'septic shock', 'sepsis',
        ]

        self.warning_keywords = [
            'difficulty breathing', 'breathlessness at rest',
            'vomiting blood', 'blood in stool', 'bloody diarrhea',
            'severe dehydration', 'sunken eyes',
            'worst headache', 'sudden onset headache',
            'chest pain', 'palpitations',
            'confusion', 'disoriented', 'altered mental',
            'high fever', 'not responding to medication',
        ]

        # ---- DANGEROUS COMBINATIONS ----
        self.combo_rules = [
            {
                'name': 'Meningitis suspicion',
                'conditions': {'temp_above': 38.5, 'keywords': ['neck stiffness', 'headache']},
                'message': 'Fever + neck stiffness → suspected meningitis'
            },
            {
                'name': 'Acute MI suspicion',
                'conditions': {'hr_above': 110, 'keywords': ['chest pain']},
                'message': 'Tachycardia + chest pain → suspected MI'
            },
            {
                'name': 'Respiratory failure',
                'conditions': {'spo2_below': 90, 'rr_above': 28},
                'message': 'Low SpO2 + high RR → respiratory failure'
            },
            {
                'name': 'Sepsis suspicion',
                'conditions': {'temp_above': 39.0, 'hr_above': 100, 'consciousness': ['Pain', 'Unresponsive']},
                'message': 'High fever + tachycardia + altered consciousness → sepsis'
            },
            {
                'name': 'Shock',
                'conditions': {'shock_index_above': 1.2},
                'message': 'Shock index >1.2 → hemodynamic instability'
            },
            {
                'name': 'Pediatric emergency',
                'conditions': {'age_below': 1, 'temp_above': 38.0, 'keywords': ['not feeding', 'lethargic', 'floppy']},
                'message': 'Infant with fever + danger signs → pediatric emergency'
            },
        ]

        self.total_checks = 0
        self.total_overrides = 0
        self.override_log = []

    def check_vital_rules(self, patient_data):
        """Check vital signs against critical thresholds."""
        triggered = []

        for rule_name, rule in self.vital_rules.items():
            param = rule['param']
            if param not in patient_data or patient_data[param] is None:
                continue
            if np.isnan(patient_data[param]):
                continue

            value = patient_data[param]
            threshold = rule['value']

            if rule['op'] == '<' and value < threshold:
                triggered.append({
                    'rule': rule_name,
                    'type': 'vital_threshold',
                    'message': rule['message'],
                    'actual_value': float(value),
                    'threshold': threshold,
                })
            elif rule['op'] == '>' and value > threshold:
                triggered.append({
                    'rule': rule_name,
                    'type': 'vital_threshold',
                    'message': rule['message'],
                    'actual_value': float(value),
                    'threshold': threshold,
                })

        return triggered

    def check_symptom_keywords(self, symptom_text):
        """Scan symptom text for danger keywords."""
        triggered = []
        text_lower = str(symptom_text).lower()

        for kw in self.critical_keywords:
            if kw in text_lower:
                triggered.append({
                    'rule': f'keyword_{kw}',
                    'type': 'critical_keyword',
                    'message': f'Critical keyword detected: "{kw}"',
                    'severity': 'critical',
                })

        for kw in self.high_danger_keywords:
            if kw in text_lower:
                triggered.append({
                    'rule': f'keyword_{kw}',
                    'type': 'high_danger_keyword',
                    'message': f'High danger keyword: "{kw}"',
                    'severity': 'high',
                })

        for kw in self.warning_keywords:
            if kw in text_lower:
                triggered.append({
                    'rule': f'keyword_{kw}',
                    'type': 'warning_keyword',
                    'message': f'Warning keyword: "{kw}"',
                    'severity': 'warning',
                })

        return triggered

    def check_combo_rules(self, patient_data, symptom_text):
        """Check for dangerous combinations of symptoms + vitals."""
        triggered = []
        text_lower = str(symptom_text).lower()

        for combo in self.combo_rules:
            conditions = combo['conditions']
            all_met = True

            if 'temp_above' in conditions:
                if patient_data.get('temperature_c', 0) <= conditions['temp_above']:
                    all_met = False
            if 'hr_above' in conditions:
                if patient_data.get('heart_rate_bpm', 0) <= conditions['hr_above']:
                    all_met = False
            if 'spo2_below' in conditions:
                if patient_data.get('spo2_percent', 100) >= conditions['spo2_below']:
                    all_met = False
            if 'rr_above' in conditions:
                if patient_data.get('respiratory_rate', 0) <= conditions['rr_above']:
                    all_met = False
            if 'shock_index_above' in conditions:
                si = patient_data.get('shock_index', 0)
                if si <= conditions['shock_index_above']:
                    all_met = False
            if 'age_below' in conditions:
                if patient_data.get('age', 99) >= conditions['age_below']:
                    all_met = False
            if 'consciousness' in conditions:
                if patient_data.get('consciousness_level', 'Alert') not in conditions['consciousness']:
                    all_met = False
            if 'keywords' in conditions:
                if not any(kw in text_lower for kw in conditions['keywords']):
                    all_met = False

            if all_met:
                triggered.append({
                    'rule': combo['name'],
                    'type': 'combination',
                    'message': combo['message'],
                })

        return triggered

    def evaluate(self, patient_data, symptom_text, ml_risk_pred, ml_urgency_pred):
        """
        Run all red flag checks and override ML predictions if needed.

        Returns:
            dict with final_risk, final_urgency, overridden flag,
            triggered rules, and explanation
        """
        self.total_checks += 1

        # Run all checks
        vital_flags = self.check_vital_rules(patient_data)
        keyword_flags = self.check_symptom_keywords(symptom_text)
        combo_flags = self.check_combo_rules(patient_data, symptom_text)

        all_flags = vital_flags + keyword_flags + combo_flags

        # Determine override level
        has_critical = any(f.get('severity') == 'critical' or f.get('type') == 'critical_keyword' for f in all_flags)
        has_vital_critical = len(vital_flags) > 0
        has_combo = len(combo_flags) > 0
        has_high_danger = any(f.get('severity') == 'high' or f.get('type') == 'high_danger_keyword' for f in all_flags)

        # Start with ML predictions
        final_risk = ml_risk_pred
        final_urgency = ml_urgency_pred
        overridden = False
        override_reason = []

        # Override logic — can only ESCALATE, never downgrade
        if has_critical or (has_vital_critical and has_combo):
            # Force Critical + Emergency
            if final_risk < 3:
                final_risk = 3  # Critical
                overridden = True
                override_reason.append("Critical red flag triggered")
            if final_urgency < 3:
                final_urgency = 3  # Emergency
                overridden = True
                override_reason.append("Escalated to Emergency")

        elif has_vital_critical or has_high_danger or has_combo:
            # Force at least High + Urgent
            if final_risk < 2:
                final_risk = 2  # High
                overridden = True
                override_reason.append("High danger red flag triggered")
            if final_urgency < 2:
                final_urgency = 2  # Urgent
                overridden = True
                override_reason.append("Escalated to Urgent")

        if overridden:
            self.total_overrides += 1

        # Build explanation
        explanation = []
        for flag in all_flags:
            explanation.append(flag['message'])

        result = {
            'final_risk': int(final_risk),
            'final_urgency': int(final_urgency),
            'ml_risk': int(ml_risk_pred),
            'ml_urgency': int(ml_urgency_pred),
            'overridden': overridden,
            'override_reason': override_reason,
            'num_flags_triggered': len(all_flags),
            'flags': all_flags,
            'explanation': explanation,
        }

        if overridden:
            self.override_log.append(result)

        return result

    def get_stats(self):
        """Return engine performance statistics."""
        return {
            'total_checks': self.total_checks,
            'total_overrides': self.total_overrides,
            'override_rate': self.total_overrides / max(self.total_checks, 1),
        }


# =====================================================================
# TEST THE ENGINE ON SAMPLE CASES
# =====================================================================
print("\n[1/3] Testing red flag engine on sample cases...")

engine = RedFlagEngine()

test_cases = [
    {
        'name': 'Healthy adult — routine visit',
        'data': {'heart_rate_bpm': 72, 'bp_systolic_mmhg': 120, 'temperature_c': 36.8,
                 'spo2_percent': 98, 'respiratory_rate': 16, 'blood_glucose_mg': 95,
                 'shock_index': 0.6, 'age': 30, 'consciousness_level': 'Alert'},
        'text': 'Patient complains of mild headache for 2 days. No other symptoms.',
        'ml_risk': 0, 'ml_urgency': 0,
    },
    {
        'name': 'Chest pain + tachycardia — possible MI',
        'data': {'heart_rate_bpm': 125, 'bp_systolic_mmhg': 150, 'temperature_c': 37.0,
                 'spo2_percent': 93, 'respiratory_rate': 22, 'blood_glucose_mg': 130,
                 'shock_index': 0.83, 'age': 55, 'consciousness_level': 'Alert'},
        'text': 'Severe crushing chest pain radiating to left arm. Patient sweating profusely.',
        'ml_risk': 1, 'ml_urgency': 1,  # ML says Moderate — should be overridden!
    },
    {
        'name': 'Respiratory failure — critically low SpO2',
        'data': {'heart_rate_bpm': 140, 'bp_systolic_mmhg': 95, 'temperature_c': 38.5,
                 'spo2_percent': 78, 'respiratory_rate': 38, 'blood_glucose_mg': 110,
                 'shock_index': 1.47, 'age': 65, 'consciousness_level': 'Voice'},
        'text': 'Patient in severe respiratory distress. Difficulty breathing, cyanotic lips.',
        'ml_risk': 2, 'ml_urgency': 2,  # ML says High — should escalate to Critical
    },
    {
        'name': 'Unconscious patient — immediate emergency',
        'data': {'heart_rate_bpm': 45, 'bp_systolic_mmhg': 70, 'temperature_c': 35.0,
                 'spo2_percent': 72, 'respiratory_rate': 6, 'blood_glucose_mg': 40,
                 'shock_index': 0.64, 'age': 70, 'consciousness_level': 'Unresponsive'},
        'text': 'Patient found unconscious. Not breathing properly. No response to stimuli.',
        'ml_risk': 2, 'ml_urgency': 2,  # ML says High — must be Critical/Emergency
    },
    {
        'name': 'Infant with fever — pediatric concern',
        'data': {'heart_rate_bpm': 160, 'bp_systolic_mmhg': 70, 'temperature_c': 39.5,
                 'spo2_percent': 94, 'respiratory_rate': 45, 'blood_glucose_mg': 80,
                 'shock_index': 2.28, 'age': 0.5, 'consciousness_level': 'Voice'},
        'text': 'Infant lethargic and not feeding. High fever for 2 days.',
        'ml_risk': 1, 'ml_urgency': 1,
    },
]

risk_names = ['Low', 'Moderate', 'High', 'Critical']
urgency_names = ['Non-Urgent', 'Semi-Urgent', 'Urgent', 'Emergency']

for case in test_cases:
    result = engine.evaluate(
        case['data'], case['text'], case['ml_risk'], case['ml_urgency']
    )

    print(f"\n   {'─' * 55}")
    print(f"   Case: {case['name']}")
    print(f"   ML predicted:  Risk={risk_names[case['ml_risk']]:>10}, Urgency={urgency_names[case['ml_urgency']]}")
    print(f"   Final output:  Risk={risk_names[result['final_risk']]:>10}, Urgency={urgency_names[result['final_urgency']]}")

    if result['overridden']:
        print(f"   ** OVERRIDDEN ** Reason: {', '.join(result['override_reason'])}")

    if result['explanation']:
        print(f"   Red flags ({result['num_flags_triggered']}):")
        for exp in result['explanation'][:5]:
            print(f"     - {exp}")


# =====================================================================
# RUN ENGINE ON FULL TEST SET
# =====================================================================
print(f"\n\n[2/3] Running engine on full test dataset...")

DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
import pandas as pd

# Load processed data and predictions
df = pd.read_csv(os.path.join(DATA_DIR, 'triage_processed.csv'))

# Load ML predictions from vitals model
y_risk_test = np.load(os.path.join(DATA_DIR, 'y_risk_test.npy'))
vitals_prob = np.load(os.path.join(DATA_DIR, 'vitals_prob_risk_level_test.npy'))
vitals_pred = np.argmax(vitals_prob, axis=1)

urgency_prob = np.load(os.path.join(DATA_DIR, 'vitals_prob_urgency_test.npy'))
urgency_pred = np.argmax(urgency_prob, axis=1)

# We need original (unscaled) vitals for rule checking
df_raw = pd.read_csv(os.path.join(BASE_DIR, '..', 'data', 'triage_dataset_10k.csv'))

# Get test indices (last 1000 based on same split)
from sklearn.model_selection import train_test_split
np.random.seed(42)
indices = np.arange(len(df_raw))
_, temp_idx = train_test_split(indices, test_size=0.20, random_state=42,
                                stratify=df_raw['risk_level'].map({'Low':0,'Moderate':1,'High':2,'Critical':3}))
_, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42,
                                stratify=df_raw.iloc[temp_idx]['risk_level'].map({'Low':0,'Moderate':1,'High':2,'Critical':3}))

df_test = df_raw.iloc[test_idx].reset_index(drop=True)
X_text_test = np.load(os.path.join(DATA_DIR, 'X_text_test.npy'), allow_pickle=True)

# Run engine on each test patient
engine_full = RedFlagEngine()
final_risks = []
final_urgencies = []

for i in range(len(df_test)):
    patient_data = {
        'heart_rate_bpm': df_test.iloc[i]['heart_rate_bpm'],
        'bp_systolic_mmhg': df_test.iloc[i]['bp_systolic_mmhg'],
        'temperature_c': df_test.iloc[i]['temperature_c'],
        'spo2_percent': df_test.iloc[i]['spo2_percent'],
        'respiratory_rate': df_test.iloc[i]['respiratory_rate'],
        'blood_glucose_mg': df_test.iloc[i].get('blood_glucose_mg', 100),
        'shock_index': df_test.iloc[i]['heart_rate_bpm'] / max(df_test.iloc[i]['bp_systolic_mmhg'], 1),
        'age': df_test.iloc[i]['age'],
        'consciousness_level': df_test.iloc[i]['consciousness_level'],
    }

    result = engine_full.evaluate(
        patient_data, str(X_text_test[i]),
        int(vitals_pred[i]), int(urgency_pred[i])
    )

    final_risks.append(result['final_risk'])
    final_urgencies.append(result['final_urgency'])

final_risks = np.array(final_risks)
final_urgencies = np.array(final_urgencies)

stats = engine_full.get_stats()
print(f"   Total patients checked: {stats['total_checks']}")
print(f"   Overrides triggered: {stats['total_overrides']} ({stats['override_rate']*100:.1f}%)")

# Compare ML-only vs ML+RedFlag
from sklearn.metrics import accuracy_score, f1_score

ml_risk_acc = accuracy_score(y_risk_test, vitals_pred)
final_risk_acc = accuracy_score(y_risk_test, final_risks)
ml_risk_f1 = f1_score(y_risk_test, vitals_pred, average='weighted')
final_risk_f1 = f1_score(y_risk_test, final_risks, average='weighted')

print(f"\n   Impact on Risk Level Prediction:")
print(f"   {'Metric':<20} {'ML Only':>10} {'ML + RedFlag':>12} {'Change':>10}")
print(f"   {'─' * 55}")
print(f"   {'Accuracy':<20} {ml_risk_acc:>10.4f} {final_risk_acc:>12.4f} {final_risk_acc-ml_risk_acc:>+10.4f}")
print(f"   {'F1 Score':<20} {ml_risk_f1:>10.4f} {final_risk_f1:>12.4f} {final_risk_f1-ml_risk_f1:>+10.4f}")

# Save final predictions with red flag overrides
np.save(os.path.join(DATA_DIR, 'final_risk_test.npy'), final_risks)
np.save(os.path.join(DATA_DIR, 'final_urgency_test.npy'), final_urgencies)


# =====================================================================
# SAVE ENGINE AND RESULTS
# =====================================================================
print(f"\n\n[3/3] Saving red flag engine...")

import pickle
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
with open(os.path.join(MODEL_DIR, 'red_flag_engine.pkl'), 'wb') as f:
    pickle.dump(engine_full, f)
print(f"   Saved: models/red_flag_engine.pkl")

# Save results
engine_results = {
    'stats': stats,
    'test_cases_results': [
        {'case': tc['name'], 'overridden': True}
        for tc in test_cases
    ],
    'rules_count': {
        'vital_rules': len(engine.vital_rules),
        'critical_keywords': len(engine.critical_keywords),
        'high_danger_keywords': len(engine.high_danger_keywords),
        'warning_keywords': len(engine.warning_keywords),
        'combo_rules': len(engine.combo_rules),
    }
}

with open(os.path.join(REPORT_DIR, 'red_flag_results.json'), 'w') as f:
    json.dump(engine_results, f, indent=2)
print(f"   Saved: reports/red_flag_results.json")

print(f"\n{'=' * 60}")
print(f"  RED FLAG ENGINE COMPLETE!")
print(f"  Rules: {len(engine.vital_rules)} vital + {len(engine.critical_keywords)} critical keywords")
print(f"         + {len(engine.high_danger_keywords)} high danger keywords")
print(f"         + {len(engine.combo_rules)} combination rules")
print(f"  Override rate on test set: {stats['override_rate']*100:.1f}%")
print(f"  Next step: python src/fusion_model.py")
print(f"{'=' * 60}")