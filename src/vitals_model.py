import numpy as np
import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_score, recall_score
)
from sklearn.model_selection import cross_val_score

print("=" * 60)
print("  VITALS MODEL — Tabular Feature Training")
print("=" * 60)

# =====================================================================
# CONFIGURATION
# =====================================================================
BASE_DIR = os.path.dirname(__file__)

DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
REPORT_DIR = os.path.join(BASE_DIR, '..', 'reports')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

SEED = 42


# =====================================================================
# STEP 1: LOAD PREPROCESSED DATA
# =====================================================================
print("\n[1/6] Loading preprocessed data...")

X_train = np.load(os.path.join(DATA_DIR, 'X_vitals_train.npy'))
X_val = np.load(os.path.join(DATA_DIR, 'X_vitals_val.npy'))
X_test = np.load(os.path.join(DATA_DIR, 'X_vitals_test.npy'))

y_risk_train = np.load(os.path.join(DATA_DIR, 'y_risk_train.npy'))
y_risk_val = np.load(os.path.join(DATA_DIR, 'y_risk_val.npy'))
y_risk_test = np.load(os.path.join(DATA_DIR, 'y_risk_test.npy'))

y_urgency_train = np.load(os.path.join(DATA_DIR, 'y_urgency_train.npy'))
y_urgency_val = np.load(os.path.join(DATA_DIR, 'y_urgency_val.npy'))
y_urgency_test = np.load(os.path.join(DATA_DIR, 'y_urgency_test.npy'))

y_disease_train = np.load(os.path.join(DATA_DIR, 'y_disease_train.npy'))
y_disease_val = np.load(os.path.join(DATA_DIR, 'y_disease_val.npy'))
y_disease_test = np.load(os.path.join(DATA_DIR, 'y_disease_test.npy'))

# Load metadata for feature names
with open(os.path.join(DATA_DIR, 'metadata.json'), 'r') as f:
    metadata = json.load(f)

feature_names = metadata['vitals_features']

print(f"   Train: {X_train.shape[0]:,} samples x {X_train.shape[1]} features")
print(f"   Val:   {X_val.shape[0]:,} samples")
print(f"   Test:  {X_test.shape[0]:,} samples")

# Safety check — clean any remaining NaN values
nan_count = np.isnan(X_train).sum()
if nan_count > 0:
    print(f"   Found {nan_count} NaN values in training data — imputing with 0...")
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    print(f"   NaN values cleaned")


# =====================================================================
# STEP 2: DEFINE MODELS
# =====================================================================
print("\n[2/6] Configuring models...")

# We train 3 separate classifiers — one per target variable
# GradientBoosting for risk & urgency (ordinal, fewer classes)
# RandomForest for disease (nominal, 9 classes, needs balanced weights)

model_configs = {
    'risk_level': {
        'model': HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=5,
            random_state=SEED
        ),
        'y_train': y_risk_train,
        'y_val': y_risk_val,
        'y_test': y_risk_test,
        'classes': ['Low', 'Moderate', 'High', 'Critical'],
    },
    'urgency': {
        'model': HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=5,
            random_state=SEED
        ),
        'y_train': y_urgency_train,
        'y_val': y_urgency_val,
        'y_test': y_urgency_test,
        'classes': ['Non-Urgent', 'Semi-Urgent', 'Urgent', 'Emergency'],
    },
    'disease_category': {
        'model': RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1
        ),
        'y_train': y_disease_train,
        'y_val': y_disease_val,
        'y_test': y_disease_test,
        'classes': ['Cardiac', 'Gastrointestinal', 'General', 'Infectious',
                    'Metabolic', 'Neurological', 'Obstetric', 'Respiratory', 'Trauma'],
    },
}

print(f"   Risk Level:       HistGradientBoosting (200 iters, depth=5, lr=0.1)")
print(f"   Urgency:          HistGradientBoosting (200 iters, depth=5, lr=0.1)")
print(f"   Disease Category: RandomForest (200 trees, depth=12, balanced)")


# =====================================================================
# STEP 3: TRAIN & EVALUATE ALL MODELS
# =====================================================================
print("\n[3/6] Training and evaluating models...")

trained_models = {}
results = {}

for target_name, config in model_configs.items():
    print(f"\n{'─' * 55}")
    print(f"   TARGET: {target_name.upper()}")
    print(f"{'─' * 55}")

    model = config['model']
    y_tr = config['y_train']
    y_v = config['y_val']
    y_te = config['y_test']
    class_names = config['classes']

    # ---- Train ----
    print(f"   Training {type(model).__name__}...")
    model.fit(X_train, y_tr)

    # ---- Predict ----
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    # ---- Prediction Probabilities (for fusion model later) ----
    prob_train = model.predict_proba(X_train)
    prob_val = model.predict_proba(X_val)
    prob_test = model.predict_proba(X_test)

    # ---- Metrics ----
    train_acc = accuracy_score(y_tr, pred_train)
    val_acc = accuracy_score(y_v, pred_val)
    test_acc = accuracy_score(y_te, pred_test)

    train_f1 = f1_score(y_tr, pred_train, average='weighted')
    val_f1 = f1_score(y_v, pred_val, average='weighted')
    test_f1 = f1_score(y_te, pred_test, average='weighted')

    print(f"\n   Results:")
    print(f"   {'Split':<8} {'Accuracy':>10} {'F1 (weighted)':>14}")
    print(f"   {'─' * 35}")
    print(f"   {'Train':<8} {train_acc:>10.4f} {train_f1:>14.4f}")
    print(f"   {'Val':<8} {val_acc:>10.4f} {val_f1:>14.4f}")
    print(f"   {'Test':<8} {test_acc:>10.4f} {test_f1:>14.4f}")

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_te, pred_test)
    print(f"\n   Confusion Matrix (Test Set):")
    max_name_len = max(len(c) for c in class_names)
    header = " " * (max_name_len + 4) + "  ".join([f"{c[:7]:>7}" for c in class_names])
    print(f"   Predicted: {header.strip()}")
    for i, row in enumerate(cm):
        row_str = "  ".join([f"{v:>7}" for v in row])
        print(f"   {class_names[i]:>{max_name_len + 3}} {row_str}")

    # ---- Per-Class Report ----
    print(f"\n   Classification Report (Test Set):")
    report_str = classification_report(y_te, pred_test, target_names=class_names)
    for line in report_str.split('\n'):
        print(f"   {line}")

    report_dict = classification_report(y_te, pred_test, target_names=class_names, output_dict=True)

    # ---- Store Everything ----
    trained_models[target_name] = model
    results[target_name] = {
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'train_f1': float(train_f1),
        'val_f1': float(val_f1),
        'test_f1': float(test_f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': report_dict,
    }

    # ---- Save Probabilities for Fusion ----
    np.save(os.path.join(DATA_DIR, f'vitals_prob_{target_name}_train.npy'), prob_train)
    np.save(os.path.join(DATA_DIR, f'vitals_prob_{target_name}_val.npy'), prob_val)
    np.save(os.path.join(DATA_DIR, f'vitals_prob_{target_name}_test.npy'), prob_test)
    print(f"   Saved prediction probabilities for fusion model")


# =====================================================================
# STEP 4: FEATURE IMPORTANCE (Explainability — key for your project!)
# =====================================================================
print(f"\n\n{'─' * 55}")
print(f"[4/6] Feature Importance Analysis")
print(f"{'─' * 55}")

from sklearn.inspection import permutation_importance

importance_data = {}

for target_name in ['risk_level', 'urgency']:
    model = trained_models[target_name]
    config = model_configs[target_name]

    # Use permutation importance (works with any model)
    print(f"\n   Computing permutation importance for {target_name}...")
    perm_result = permutation_importance(
        model, X_test, config['y_test'],
        n_repeats=10, random_state=SEED, scoring='f1_weighted'
    )
    importances = perm_result.importances_mean
    indices = np.argsort(importances)[::-1]

    print(f"\n   Top 15 features for {target_name.upper()}:")
    print(f"   {'Rank':<5} {'Feature':<32} {'Importance':>10}")
    print(f"   {'─' * 50}")
    for rank, idx in enumerate(indices[:15], 1):
        bar_len = int(importances[idx] * 50)
        bar = '█' * max(bar_len, 0)
        print(f"   {rank:<5} {feature_names[idx]:<32} {importances[idx]:>10.4f} {bar}")

    importance_data[target_name] = {
        'features': [feature_names[i] for i in indices],
        'importances': [float(importances[i]) for i in indices],
    }

# Disease category — RandomForest has feature_importances_ built in
model = trained_models['disease_category']
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print(f"\n   Top 10 features for DISEASE_CATEGORY:")
print(f"   {'Rank':<5} {'Feature':<32} {'Importance':>10}")
print(f"   {'─' * 50}")
for rank, idx in enumerate(indices[:10], 1):
    print(f"   {rank:<5} {feature_names[idx]:<32} {importances[idx]:>10.4f}")

importance_data['disease_category'] = {
    'features': [feature_names[i] for i in indices],
    'importances': [float(importances[i]) for i in indices],
}


# =====================================================================
# STEP 5: CROSS-VALIDATION
# =====================================================================
print(f"\n\n[5/6] Running 5-fold cross-validation...")

X_trainval = np.vstack([X_train, X_val])

for target_name in ['risk_level', 'urgency']:
    config = model_configs[target_name]
    y_trainval = np.concatenate([config['y_train'], config['y_val']])

    cv_model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=5, learning_rate=0.1,
        random_state=SEED
    )
    cv_scores = cross_val_score(cv_model, X_trainval, y_trainval, cv=5, scoring='f1_weighted')

    print(f"   {target_name}: Mean CV F1 = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    results[target_name]['cv_f1_mean'] = float(cv_scores.mean())
    results[target_name]['cv_f1_std'] = float(cv_scores.std())


# =====================================================================
# STEP 6: SAVE EVERYTHING
# =====================================================================
print(f"\n\n[6/6] Saving models and reports...")

# Save trained models
for target_name, model in trained_models.items():
    path = os.path.join(MODEL_DIR, f'vitals_model_{target_name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"   Saved: models/vitals_model_{target_name}.pkl ({size_kb:.1f} KB)")

# Save results
with open(os.path.join(REPORT_DIR, 'vitals_model_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"   Saved: reports/vitals_model_results.json")

# Save feature importance
with open(os.path.join(REPORT_DIR, 'feature_importance.json'), 'w') as f:
    json.dump(importance_data, f, indent=2)
print(f"   Saved: reports/feature_importance.json")


# =====================================================================
# FINAL SUMMARY
# =====================================================================
print(f"\n\n{'=' * 60}")
print(f"  VITALS MODEL TRAINING COMPLETE!")
print(f"{'=' * 60}")

print(f"\n  Test Set Performance:")
print(f"  {'Target':<22} {'Accuracy':>10} {'F1':>10}")
print(f"  {'─' * 45}")
for target_name in ['risk_level', 'urgency', 'disease_category']:
    r = results[target_name]
    print(f"  {target_name:<22} {r['test_accuracy']:>10.4f} {r['test_f1']:>10.4f}")

# Under-triage safety check
print(f"\n  SAFETY CHECK — Under-Triage Analysis:")
print(f"  {'─' * 45}")
cm_risk = np.array(results['risk_level']['confusion_matrix'])
if cm_risk.shape[0] == 4:
    for level, name in [(3, 'Critical'), (2, 'High')]:
        total = cm_risk[level].sum()
        correct = cm_risk[level][level]
        missed = total - correct
        recall = correct / max(total, 1)
        status = "OK" if recall >= 0.85 else "NEEDS IMPROVEMENT"
        print(f"  {name} patients: {correct}/{total} detected ({recall*100:.1f}%) [{status}]")

print(f"\n  Next step: python src/nlp_model.py")
print(f"{'=' * 60}")