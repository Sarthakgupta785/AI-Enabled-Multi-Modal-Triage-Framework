import numpy as np
import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

print("=" * 60)
print("  FUSION MODEL — Multi-Modal Ensemble")
print("=" * 60)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
REPORT_DIR = os.path.join(BASE_DIR, '..', 'reports')

os.makedirs(REPORT_DIR, exist_ok=True)


# =====================================================================
# STEP 1: LOAD ALL PREDICTION PROBABILITIES
# =====================================================================
print("\n[1/5] Loading prediction probabilities from both models...")

targets = ['risk_level', 'urgency', 'disease_category']
splits = ['train', 'val', 'test']

probs = {}
for target in targets:
    probs[target] = {}
    for split in splits:
        probs[target][f'vitals_{split}'] = np.load(
            os.path.join(DATA_DIR, f'vitals_prob_{target}_{split}.npy'))
        probs[target][f'nlp_{split}'] = np.load(
            os.path.join(DATA_DIR, f'nlp_prob_{target}_{split}.npy'))

# Load ground truth — file names use shortened target names
target_file_map = {
    'risk_level': 'risk',
    'urgency': 'urgency',
    'disease_category': 'disease',
}
y_data = {}
for target in targets:
    short = target_file_map[target]
    for split in splits:
        y_data[f'{target}_{split}'] = np.load(
            os.path.join(DATA_DIR, f'y_{short}_{split}.npy'))

print(f"   Loaded vitals + NLP probabilities for all 3 targets")
print(f"   Shapes: vitals={probs['risk_level']['vitals_test'].shape}, nlp={probs['risk_level']['nlp_test'].shape}")


# =====================================================================
# STEP 2: FIND OPTIMAL FUSION WEIGHTS (using validation set)
# =====================================================================
print("\n[2/5] Optimizing fusion weights on validation set...")

optimal_weights = {}

for target in targets:
    vitals_prob_val = probs[target]['vitals_val']
    nlp_prob_val = probs[target]['nlp_val']
    y_val = y_data[f'{target}_val']

    best_f1 = 0
    best_w = 0.5

    # Grid search over weights from 0.0 to 1.0
    for w_vitals in np.arange(0.0, 1.05, 0.05):
        w_nlp = 1.0 - w_vitals

        # Weighted average of probabilities
        fused_prob = w_vitals * vitals_prob_val + w_nlp * nlp_prob_val
        fused_pred = np.argmax(fused_prob, axis=1)

        f1 = f1_score(y_val, fused_pred, average='weighted')

        if f1 > best_f1:
            best_f1 = f1
            best_w = w_vitals

    optimal_weights[target] = {
        'vitals_weight': round(best_w, 2),
        'nlp_weight': round(1.0 - best_w, 2),
        'val_f1': round(best_f1, 4),
    }

    print(f"   {target}:")
    print(f"     Optimal: Vitals={best_w:.2f}, NLP={1-best_w:.2f} → Val F1={best_f1:.4f}")

print(f"\n   Weight interpretation:")
for target, w in optimal_weights.items():
    dominant = "Vitals" if w['vitals_weight'] > w['nlp_weight'] else "NLP"
    print(f"     {target}: {dominant} dominates ({w['vitals_weight']} vs {w['nlp_weight']})")


# =====================================================================
# STEP 3: APPLY FUSION ON TEST SET
# =====================================================================
print("\n[3/5] Applying fusion on test set...")

fusion_results = {}

class_names_map = {
    'risk_level': ['Low', 'Moderate', 'High', 'Critical'],
    'urgency': ['Non-Urgent', 'Semi-Urgent', 'Urgent', 'Emergency'],
    'disease_category': ['Cardiac', 'Gastrointestinal', 'General', 'Infectious',
                         'Metabolic', 'Neurological', 'Obstetric', 'Respiratory', 'Trauma'],
}

for target in targets:
    w = optimal_weights[target]
    vitals_prob_test = probs[target]['vitals_test']
    nlp_prob_test = probs[target]['nlp_test']
    y_test = y_data[f'{target}_test']
    class_names = class_names_map[target]

    # Fuse probabilities
    fused_prob = w['vitals_weight'] * vitals_prob_test + w['nlp_weight'] * nlp_prob_test
    fused_pred = np.argmax(fused_prob, axis=1)

    # Also get individual model predictions for comparison
    vitals_pred = np.argmax(vitals_prob_test, axis=1)
    nlp_pred = np.argmax(nlp_prob_test, axis=1)

    # Calculate all metrics
    vitals_acc = accuracy_score(y_test, vitals_pred)
    nlp_acc = accuracy_score(y_test, nlp_pred)
    fusion_acc = accuracy_score(y_test, fused_pred)

    vitals_f1 = f1_score(y_test, vitals_pred, average='weighted')
    nlp_f1 = f1_score(y_test, nlp_pred, average='weighted')
    fusion_f1 = f1_score(y_test, fused_pred, average='weighted')

    print(f"\n{'─' * 60}")
    print(f"   TARGET: {target.upper()}")
    print(f"{'─' * 60}")
    print(f"   {'Model':<20} {'Accuracy':>10} {'F1 Score':>10}")
    print(f"   {'─' * 42}")
    print(f"   {'Vitals Only':<20} {vitals_acc:>10.4f} {vitals_f1:>10.4f}")
    print(f"   {'NLP Only':<20} {nlp_acc:>10.4f} {nlp_f1:>10.4f}")
    print(f"   {'FUSION':<20} {fusion_acc:>10.4f} {fusion_f1:>10.4f}")

    improvement = fusion_f1 - max(vitals_f1, nlp_f1)
    if improvement > 0:
        print(f"   Fusion IMPROVED over best single model by +{improvement:.4f}")
    else:
        print(f"   Best single model matches/exceeds fusion (delta: {improvement:.4f})")

    # Confusion Matrix for fusion
    cm = confusion_matrix(y_test, fused_pred)
    print(f"\n   Fusion Confusion Matrix:")
    header = " " * 18 + "  ".join([f"{c[:7]:>7}" for c in class_names])
    print(f"   {header}")
    for i, row in enumerate(cm):
        row_str = "  ".join([f"{v:>7}" for v in row])
        print(f"   {class_names[i]:>16} {row_str}")

    # Classification report
    report = classification_report(y_test, fused_pred, target_names=class_names, output_dict=True)

    fusion_results[target] = {
        'vitals_accuracy': float(vitals_acc),
        'nlp_accuracy': float(nlp_acc),
        'fusion_accuracy': float(fusion_acc),
        'vitals_f1': float(vitals_f1),
        'nlp_f1': float(nlp_f1),
        'fusion_f1': float(fusion_f1),
        'weights': w,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
    }

    # Save fusion probabilities
    np.save(os.path.join(DATA_DIR, f'fusion_prob_{target}_test.npy'), fused_prob)
    np.save(os.path.join(DATA_DIR, f'fusion_pred_{target}_test.npy'), fused_pred)


# =====================================================================
# STEP 4: UNDER-TRIAGE SAFETY ANALYSIS
# =====================================================================
print(f"\n\n{'─' * 60}")
print(f"[4/5] Under-Triage Safety Analysis")
print(f"{'─' * 60}")

y_risk_test = y_data['risk_level_test']
risk_fused_pred = np.load(os.path.join(DATA_DIR, 'fusion_pred_risk_level_test.npy'))

risk_names = ['Low', 'Moderate', 'High', 'Critical']

print(f"\n   Under-triage = patient is sicker than predicted")
print(f"   Over-triage = patient is less sick than predicted")
print(f"   Under-triage is DANGEROUS; over-triage is safe (wastes resources)")
print(f"   {'─' * 55}")

total_under = 0
total_over = 0
total_correct = 0

for i in range(len(y_risk_test)):
    if risk_fused_pred[i] < y_risk_test[i]:
        total_under += 1  # DANGEROUS
    elif risk_fused_pred[i] > y_risk_test[i]:
        total_over += 1   # Safe but wasteful
    else:
        total_correct += 1

print(f"   Correctly triaged: {total_correct} ({total_correct/len(y_risk_test)*100:.1f}%)")
print(f"   Over-triaged (safe): {total_over} ({total_over/len(y_risk_test)*100:.1f}%)")
print(f"   UNDER-triaged (dangerous): {total_under} ({total_under/len(y_risk_test)*100:.1f}%)")

# Per-class recall (sensitivity) — most important for Critical
print(f"\n   Per-class sensitivity (recall):")
print(f"   {'Class':<15} {'Correct':>8} {'Total':>8} {'Recall':>8} {'Status':>15}")
print(f"   {'─' * 55}")
cm_risk = confusion_matrix(y_risk_test, risk_fused_pred)
for i, name in enumerate(risk_names):
    total = cm_risk[i].sum()
    correct = cm_risk[i][i]
    recall = correct / max(total, 1)
    status = "EXCELLENT" if recall >= 0.95 else "GOOD" if recall >= 0.85 else "NEEDS WORK"
    print(f"   {name:<15} {correct:>8} {total:>8} {recall:>8.1%} {status:>15}")


# =====================================================================
# STEP 5: SAVE EVERYTHING
# =====================================================================
print(f"\n\n[5/5] Saving fusion pipeline...")

# Save fusion weights and config
fusion_config = {
    'optimal_weights': optimal_weights,
    'fusion_results': fusion_results,
    'safety_analysis': {
        'correct': int(total_correct),
        'over_triaged': int(total_over),
        'under_triaged': int(total_under),
        'under_triage_rate': float(total_under / len(y_risk_test)),
    }
}

with open(os.path.join(MODEL_DIR, 'fusion_config.json'), 'w') as f:
    json.dump(fusion_config, f, indent=2)
print(f"   Saved: models/fusion_config.json")

with open(os.path.join(REPORT_DIR, 'fusion_results.json'), 'w') as f:
    json.dump(fusion_results, f, indent=2)
print(f"   Saved: reports/fusion_results.json")


# =====================================================================
# FINAL SUMMARY — COMPLETE SYSTEM PERFORMANCE
# =====================================================================
print(f"\n\n{'=' * 60}")
print(f"  MULTI-MODAL FUSION COMPLETE!")
print(f"{'=' * 60}")

print(f"\n  Complete System Performance (Test Set):")
print(f"  {'─' * 60}")
print(f"  {'Target':<22} {'Vitals':>8} {'NLP':>8} {'Fusion':>8} {'Weights (V:N)'}")
print(f"  {'─' * 60}")
for target in targets:
    r = fusion_results[target]
    w = optimal_weights[target]
    print(f"  {target:<22} {r['vitals_f1']:>8.4f} {r['nlp_f1']:>8.4f} {r['fusion_f1']:>8.4f} {w['vitals_weight']}:{w['nlp_weight']}")

print(f"\n  Key Insight:")
print(f"  - Vitals model excels at risk/urgency (uses NEWS scores)")
print(f"  - NLP model excels at disease detection (uses symptom text)")
print(f"  - Fusion combines strengths of both models!")
print(f"  - Red flag engine provides safety override layer")

print(f"\n  Safety: Under-triage rate = {total_under/len(y_risk_test)*100:.1f}%")
if total_under / len(y_risk_test) < 0.05:
    print(f"  STATUS: Clinically acceptable (<5% under-triage)")
else:
    print(f"  STATUS: Needs improvement (>5% under-triage)")

print(f"\n  All models saved in models/ folder")
print(f"  All results saved in reports/ folder")
print(f"\n  NEXT: Build the web dashboard demo!")
print(f"{'=' * 60}")