"""
nlp_model.py — NLP Model for Symptom Text Classification
=========================================================
Authors: Sarthak Gupta, Dhruv
Guide: Dr R. Madhura

This script:
1. Loads symptom text data (train/val/test)
2. Preprocesses text (lowercase, clean, tokenize)
3. Extracts features using TF-IDF vectorization
4. Trains classifiers for risk, urgency, and disease prediction
5. Evaluates performance with detailed metrics
6. Saves models and TF-IDF vectorizer for deployment
=========================================================
"""

import numpy as np
import os
import json
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

print("=" * 60)
print("  NLP MODEL — Symptom Text Classification")
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
# STEP 1: LOAD TEXT DATA
# =====================================================================
print("\n[1/6] Loading text data...")

X_text_train = np.load(os.path.join(DATA_DIR, 'X_text_train.npy'), allow_pickle=True)
X_text_val = np.load(os.path.join(DATA_DIR, 'X_text_val.npy'), allow_pickle=True)
X_text_test = np.load(os.path.join(DATA_DIR, 'X_text_test.npy'), allow_pickle=True)

y_risk_train = np.load(os.path.join(DATA_DIR, 'y_risk_train.npy'))
y_risk_val = np.load(os.path.join(DATA_DIR, 'y_risk_val.npy'))
y_risk_test = np.load(os.path.join(DATA_DIR, 'y_risk_test.npy'))

y_urgency_train = np.load(os.path.join(DATA_DIR, 'y_urgency_train.npy'))
y_urgency_val = np.load(os.path.join(DATA_DIR, 'y_urgency_val.npy'))
y_urgency_test = np.load(os.path.join(DATA_DIR, 'y_urgency_test.npy'))

y_disease_train = np.load(os.path.join(DATA_DIR, 'y_disease_train.npy'))
y_disease_val = np.load(os.path.join(DATA_DIR, 'y_disease_val.npy'))
y_disease_test = np.load(os.path.join(DATA_DIR, 'y_disease_test.npy'))

print(f"   Train: {len(X_text_train):,} texts")
print(f"   Val:   {len(X_text_val):,} texts")
print(f"   Test:  {len(X_text_test):,} texts")

# Show sample texts
print(f"\n   Sample texts:")
for i in range(3):
    text = str(X_text_train[i])[:100]
    print(f"   [{i+1}] {text}...")


# =====================================================================
# STEP 2: TEXT PREPROCESSING
# =====================================================================
print("\n[2/6] Preprocessing text data...")

def preprocess_text(text):
    """Clean and normalize symptom description text."""
    text = str(text).lower()

    # Remove special characters but keep medical terms
    text = re.sub(r'[^a-z0-9\s\.]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Apply preprocessing
X_text_train_clean = [preprocess_text(t) for t in X_text_train]
X_text_val_clean = [preprocess_text(t) for t in X_text_val]
X_text_test_clean = [preprocess_text(t) for t in X_text_test]

print(f"   Text cleaning applied (lowercase, remove special chars)")
print(f"   Sample cleaned: {X_text_train_clean[0][:80]}...")


# =====================================================================
# STEP 3: TF-IDF FEATURE EXTRACTION
# =====================================================================
print("\n[3/6] Extracting TF-IDF features...")

# TF-IDF captures which words are important in each document
# relative to the entire corpus
tfidf = TfidfVectorizer(
    max_features=5000,       # top 5000 words
    ngram_range=(1, 2),      # unigrams + bigrams (e.g., "chest pain")
    min_df=3,                # word must appear in at least 3 documents
    max_df=0.95,             # ignore words in >95% of documents
    sublinear_tf=True,       # apply log scaling to term frequency
    stop_words='english',    # remove common English words
)

# Fit on training data only (prevent data leakage!)
X_tfidf_train = tfidf.fit_transform(X_text_train_clean)
X_tfidf_val = tfidf.transform(X_text_val_clean)
X_tfidf_test = tfidf.transform(X_text_test_clean)

print(f"   Vocabulary size: {len(tfidf.vocabulary_):,} terms")
print(f"   TF-IDF matrix shape: {X_tfidf_train.shape}")
print(f"   Feature type: sparse matrix (memory efficient)")

# Show top terms
feature_names = tfidf.get_feature_names_out()
print(f"\n   Top 20 TF-IDF terms (by document frequency):")
sorted_indices = np.argsort(tfidf.idf_)
top_terms = [feature_names[i] for i in sorted_indices[:20]]
print(f"   {', '.join(top_terms)}")

# Show some important bigrams
bigrams = [f for f in feature_names if ' ' in f]
print(f"\n   Medical bigrams found: {len(bigrams)}")
print(f"   Examples: {', '.join(bigrams[:15])}")


# =====================================================================
# STEP 4: TRAIN NLP CLASSIFIERS
# =====================================================================
print("\n[4/6] Training NLP classifiers...")

model_configs = {
    'risk_level': {
        'model': LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs',
            
            random_state=SEED
        ),
        'y_train': y_risk_train,
        'y_val': y_risk_val,
        'y_test': y_risk_test,
        'classes': ['Low', 'Moderate', 'High', 'Critical'],
    },
    'urgency': {
        'model': LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs',
            
            random_state=SEED
        ),
        'y_train': y_urgency_train,
        'y_val': y_urgency_val,
        'y_test': y_urgency_test,
        'classes': ['Non-Urgent', 'Semi-Urgent', 'Urgent', 'Emergency'],
    },
    'disease_category': {
        'model': LogisticRegression(
            C=2.0,
            max_iter=1500,
            class_weight='balanced',
            solver='lbfgs',
            
            random_state=SEED
        ),
        'y_train': y_disease_train,
        'y_val': y_disease_val,
        'y_test': y_disease_test,
        'classes': ['Cardiac', 'Gastrointestinal', 'General', 'Infectious',
                    'Metabolic', 'Neurological', 'Obstetric', 'Respiratory', 'Trauma'],
    },
}

print(f"   Risk & Urgency: LogisticRegression (balanced, multinomial)")
print(f"   Disease: LogisticRegression (C=2.0, balanced, multinomial)")

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

    # Train
    print(f"   Training {type(model).__name__}...")
    model.fit(X_tfidf_train, y_tr)

    # Predict
    pred_train = model.predict(X_tfidf_train)
    pred_val = model.predict(X_tfidf_val)
    pred_test = model.predict(X_tfidf_test)

    # Prediction probabilities (for fusion model)
    prob_train = model.predict_proba(X_tfidf_train)
    prob_val = model.predict_proba(X_tfidf_val)
    prob_test = model.predict_proba(X_tfidf_test)

    # Metrics
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

    # Confusion Matrix
    cm = confusion_matrix(y_te, pred_test)
    print(f"\n   Confusion Matrix (Test Set):")
    header = " " * 18 + "  ".join([f"{c[:7]:>7}" for c in class_names])
    print(f"   {header}")
    for i, row in enumerate(cm):
        row_str = "  ".join([f"{v:>7}" for v in row])
        print(f"   {class_names[i]:>16} {row_str}")

    # Classification Report
    print(f"\n   Classification Report (Test):")
    report_str = classification_report(y_te, pred_test, target_names=class_names)
    for line in report_str.split('\n'):
        print(f"   {line}")

    report_dict = classification_report(y_te, pred_test, target_names=class_names, output_dict=True)

    # Store
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

    # Save probabilities for fusion
    np.save(os.path.join(DATA_DIR, f'nlp_prob_{target_name}_train.npy'), prob_train)
    np.save(os.path.join(DATA_DIR, f'nlp_prob_{target_name}_val.npy'), prob_val)
    np.save(os.path.join(DATA_DIR, f'nlp_prob_{target_name}_test.npy'), prob_test)
    print(f"   Saved prediction probabilities for fusion model")


# =====================================================================
# STEP 5: IMPORTANT WORDS ANALYSIS (Explainability)
# =====================================================================
print(f"\n\n{'─' * 55}")
print(f"[5/6] Important Words Analysis (Explainability)")
print(f"{'─' * 55}")

important_words = {}

# For disease category — show top words per disease class
disease_model = trained_models['disease_category']
disease_classes = model_configs['disease_category']['classes']

print(f"\n   Top keywords per disease category:")
print(f"   (Words that most strongly indicate each condition)")
print(f"   {'─' * 55}")

for class_idx, class_name in enumerate(disease_classes):
    # Get coefficients for this class
    coeffs = disease_model.coef_[class_idx]
    top_indices = np.argsort(coeffs)[-8:][::-1]  # top 8 words
    top_words = [(feature_names[i], coeffs[i]) for i in top_indices]

    words_str = ", ".join([f"{w}" for w, c in top_words])
    print(f"   {class_name:<20} {words_str}")

    important_words[class_name] = [
        {'word': w, 'weight': float(c)} for w, c in top_words
    ]

# For risk level — show top danger words
risk_model = trained_models['risk_level']
risk_classes = model_configs['risk_level']['classes']

print(f"\n   Top keywords for CRITICAL risk detection:")
print(f"   {'─' * 55}")
critical_idx = 3  # Critical = class 3
coeffs = risk_model.coef_[critical_idx]
top_indices = np.argsort(coeffs)[-15:][::-1]
for i, idx in enumerate(top_indices, 1):
    bar = '█' * int(coeffs[idx] * 3)
    print(f"   {i:>2}. {feature_names[idx]:<25} {coeffs[idx]:>7.3f} {bar}")

print(f"\n   Top keywords for LOW risk detection:")
print(f"   {'─' * 55}")
low_idx = 0  # Low = class 0
coeffs = risk_model.coef_[low_idx]
top_indices = np.argsort(coeffs)[-10:][::-1]
for i, idx in enumerate(top_indices, 1):
    print(f"   {i:>2}. {feature_names[idx]:<25} {coeffs[idx]:>7.3f}")


# =====================================================================
# STEP 6: SAVE EVERYTHING
# =====================================================================
print(f"\n\n[6/6] Saving models and results...")

# Save TF-IDF vectorizer (critical for deployment!)
with open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(tfidf, f)
size_kb = os.path.getsize(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')) / 1024
print(f"   Saved: models/tfidf_vectorizer.pkl ({size_kb:.1f} KB)")

# Save trained models
for target_name, model in trained_models.items():
    path = os.path.join(MODEL_DIR, f'nlp_model_{target_name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"   Saved: models/nlp_model_{target_name}.pkl ({size_kb:.1f} KB)")

# Save results
with open(os.path.join(REPORT_DIR, 'nlp_model_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"   Saved: reports/nlp_model_results.json")

# Save important words
with open(os.path.join(REPORT_DIR, 'important_words.json'), 'w') as f:
    json.dump(important_words, f, indent=2)
print(f"   Saved: reports/important_words.json")

# Save the preprocessing function reference
nlp_config = {
    'max_features': 5000,
    'ngram_range': [1, 2],
    'min_df': 3,
    'max_df': 0.95,
    'vocabulary_size': len(tfidf.vocabulary_),
    'num_bigrams': len(bigrams),
}
with open(os.path.join(REPORT_DIR, 'nlp_config.json'), 'w') as f:
    json.dump(nlp_config, f, indent=2)
print(f"   Saved: reports/nlp_config.json")


# =====================================================================
# FINAL SUMMARY
# =====================================================================
print(f"\n\n{'=' * 60}")
print(f"  NLP MODEL TRAINING COMPLETE!")
print(f"{'=' * 60}")

print(f"\n  Test Set Performance:")
print(f"  {'Target':<22} {'Accuracy':>10} {'F1':>10}")
print(f"  {'─' * 45}")
for target_name in ['risk_level', 'urgency', 'disease_category']:
    r = results[target_name]
    print(f"  {target_name:<22} {r['test_accuracy']:>10.4f} {r['test_f1']:>10.4f}")

print(f"\n  Comparison — Vitals vs NLP (Text Only):")
print(f"  {'─' * 55}")
print(f"  The NLP model uses ONLY symptom text descriptions.")
print(f"  The vitals model uses ONLY tabular vital signs.")
print(f"  The fusion model (next step) will COMBINE both!")

# Load vitals results for comparison if available
vitals_results_path = os.path.join(REPORT_DIR, 'vitals_model_results.json')
if os.path.exists(vitals_results_path):
    with open(vitals_results_path, 'r') as f:
        vitals_results = json.load(f)

    print(f"\n  {'Target':<22} {'Vitals F1':>10} {'NLP F1':>10} {'Winner':>10}")
    print(f"  {'─' * 55}")
    for target_name in ['risk_level', 'urgency', 'disease_category']:
        v_f1 = vitals_results[target_name]['test_f1']
        n_f1 = results[target_name]['test_f1']
        winner = "Vitals" if v_f1 > n_f1 else "NLP" if n_f1 > v_f1 else "Tie"
        print(f"  {target_name:<22} {v_f1:>10.4f} {n_f1:>10.4f} {winner:>10}")

print(f"\n  Next step: python src/red_flag_engine.py")
print(f"{'=' * 60}")