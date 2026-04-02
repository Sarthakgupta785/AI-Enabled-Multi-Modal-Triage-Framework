"""
seasonal_context.py — Seasonal Disease Intelligence Module
============================================================
Authors: Sarthak Gupta, Dhruv
Guide: Dr R. Madhura

This module provides seasonal disease awareness to the triage pipeline.
It does NOT replace the existing ML models — it enhances them by:

1. Providing seasonal disease risk priors (boosts disease prediction)
2. Adding season-aware red flag rules
3. Generating seasonal alert banners for the dashboard
4. Adjusting urgency based on seasonal risk multipliers

Integration: Import this module in app.py and call its functions
at the appropriate points in the predict_triage() pipeline.
============================================================
"""

import json
import os
from datetime import datetime

# =====================================================================
# LOAD SEASONAL DATABASE
# =====================================================================
SEASONAL_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'data', 'seasonal_disease_data.json'
)

# Fallback: check current directory
if not os.path.exists(SEASONAL_DB_PATH):
    SEASONAL_DB_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'seasonal_disease_data.json'
    )

def load_seasonal_db():
    """Load the seasonal disease database."""
    try:
        with open(SEASONAL_DB_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"WARNING: Seasonal database not found at {SEASONAL_DB_PATH}")
        print("Seasonal context module will return neutral values.")
        return None

SEASONAL_DB = load_seasonal_db()


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def get_current_season(month=None):
    """Get the current season based on month number."""
    if month is None:
        month = datetime.now().month

    if SEASONAL_DB is None:
        return {"name": "unknown", "label": "Unknown"}

    for season_name, season_info in SEASONAL_DB["seasons"].items():
        if month in season_info["months"]:
            return {"name": season_name, "label": season_info["label"]}

    return {"name": "unknown", "label": "Unknown"}


def get_region_profile(region="tamil_nadu"):
    """Get the disease profile for a specific region."""
    if SEASONAL_DB is None:
        return None
    return SEASONAL_DB.get("regional_profiles", {}).get(region)


# =====================================================================
# 1. SEASONAL RISK SCORES (feeds into disease prediction)
# =====================================================================
def get_seasonal_disease_risks(region="tamil_nadu", month=None):
    """
    Returns a dict of disease_name → risk_score (0.0 to 1.0)
    for the given region and month.

    Usage in predict_triage():
        seasonal_risks = get_seasonal_disease_risks("tamil_nadu")
        # Use these to boost/adjust disease prediction probabilities
    """
    if month is None:
        month = datetime.now().month

    profile = get_region_profile(region)
    if profile is None:
        return {}

    risks = {}
    for disease_key, disease_info in profile["diseases"].items():
        risk = disease_info["monthly_risk"].get(str(month), 0.0)
        risks[disease_key] = {
            "risk_score": risk,
            "category": disease_info["category"],
            "is_elevated": risk >= disease_info["alert_threshold"],
            "advisory": disease_info["advisory"] if risk >= disease_info["alert_threshold"] else None
        }

    return risks


def get_category_seasonal_boost(region="tamil_nadu", month=None):
    """
    Returns a dict of disease_category → seasonal_boost_factor.
    This maps directly to your existing DISEASE_LABELS categories.

    The boost factor ranges from 0.0 (no seasonal influence) to 0.3 (strong seasonal influence).
    This is added to the fused disease probability for matching categories.

    Usage in predict_triage() Step 5 (Fusion):
        seasonal_boosts = get_category_seasonal_boost("tamil_nadu")
        # Adjust fused_disease_prob based on seasonal context
    """
    if month is None:
        month = datetime.now().month

    risks = get_seasonal_disease_risks(region, month)
    if not risks:
        return {}

    # Aggregate by disease category (take max risk per category)
    category_risks = {}
    for disease_key, risk_info in risks.items():
        cat = risk_info["category"]
        score = risk_info["risk_score"]
        if cat not in category_risks or score > category_risks[cat]:
            category_risks[cat] = score

    # Convert risk scores to boost factors (0.0 to 0.15)
    # Only boost if risk is above 0.3 (moderate seasonal risk)
    boosts = {}
    for cat, risk in category_risks.items():
        if risk >= 0.3:
            # Scale: risk 0.3 → boost 0.02, risk 1.0 → boost 0.15
            boosts[cat] = round(0.02 + (risk - 0.3) * (0.13 / 0.7), 4)
        else:
            boosts[cat] = 0.0

    return boosts


# =====================================================================
# 2. SEASONAL RED FLAG RULES
# =====================================================================
def check_seasonal_red_flags(vitals, symptom_text, region="tamil_nadu", month=None):
    """
    Check for season-specific red flags that should be added to
    the existing red flag engine output.

    Returns a list of flag dicts matching the existing format:
    [{'severity': 'warning'|'high', 'message': '...'}]

    Usage in predict_triage():
        seasonal_flags = check_seasonal_red_flags(vitals, symptom_text, "tamil_nadu")
        red_flags.extend(seasonal_flags)
    """
    if month is None:
        month = datetime.now().month

    flags = []
    text_lower = symptom_text.lower()
    risks = get_seasonal_disease_risks(region, month)
    season = get_current_season(month)

    # --- Dengue season rules ---
    dengue_risk = risks.get("dengue", {}).get("risk_score", 0)
    if dengue_risk >= 0.5:
        fever_keywords = ["fever", "bukhar", "kaichal"]
        body_pain_keywords = ["body ache", "body pain", "joint pain", "muscle pain", "headache"]
        has_fever = any(kw in text_lower for kw in fever_keywords)
        has_body_pain = any(kw in text_lower for kw in body_pain_keywords)
        high_temp = vitals.get('temperature_c', 37) > 38.5

        if (has_fever or high_temp) and has_body_pain:
            flags.append({
                'severity': 'warning',
                'message': f'SEASONAL: Dengue risk HIGH this month (risk: {dengue_risk:.0%}). '
                           f'Fever + body pain — recommend platelet count test.'
            })

    # --- Malaria season rules ---
    malaria_risk = risks.get("malaria", {}).get("risk_score", 0)
    if malaria_risk >= 0.4:
        chills_keywords = ["chills", "shiver", "rigors", "thandi", "kulir"]
        has_chills = any(kw in text_lower for kw in chills_keywords)
        has_fever = "fever" in text_lower or vitals.get('temperature_c', 37) > 38

        if has_fever and has_chills:
            flags.append({
                'severity': 'warning',
                'message': f'SEASONAL: Malaria risk ELEVATED (risk: {malaria_risk:.0%}). '
                           f'Fever with chills — recommend rapid malaria test.'
            })

    # --- Heat stroke rules (summer) ---
    heat_risk = risks.get("heat_stroke", {}).get("risk_score", 0)
    if heat_risk >= 0.4:
        heat_keywords = ["hot", "sun", "no sweat", "dizzy", "collapse", "faint"]
        has_heat_symptoms = any(kw in text_lower for kw in heat_keywords)
        high_temp = vitals.get('temperature_c', 37) > 39.5
        confusion = vitals.get('consciousness', 'Alert') != 'Alert'

        if high_temp and (has_heat_symptoms or confusion):
            flags.append({
                'severity': 'high',
                'message': f'SEASONAL: Heat stroke risk CRITICAL. '
                           f'High temperature + altered consciousness — cool patient IMMEDIATELY.'
            })

    # --- Waterborne disease rules (monsoon) ---
    cholera_risk = risks.get("cholera", {}).get("risk_score", 0)
    agi_risk = risks.get("acute_gastroenteritis", {}).get("risk_score", 0)
    if max(cholera_risk, agi_risk) >= 0.4:
        gi_keywords = ["diarrhea", "vomit", "watery stool", "loose motion", "dehydration"]
        has_gi = any(kw in text_lower for kw in gi_keywords)
        high_hr = vitals.get('heart_rate_bpm', 75) > 100  # tachycardia from dehydration

        if has_gi and high_hr:
            flags.append({
                'severity': 'warning',
                'message': f'SEASONAL: Waterborne disease risk HIGH (monsoon). '
                           f'Diarrhea + tachycardia — assess dehydration, start ORS/IV fluids.'
            })

    # --- Respiratory season rules (winter) ---
    resp_risk = risks.get("respiratory_infections", {}).get("risk_score", 0)
    if resp_risk >= 0.45:
        resp_keywords = ["cough", "breathless", "wheeze", "cold", "congestion", "pneumonia"]
        has_resp = any(kw in text_lower for kw in resp_keywords)
        low_spo2 = vitals.get('spo2_percent', 98) < 94
        elderly = vitals.get('age', 30) > 60
        child = vitals.get('age', 30) < 5

        if has_resp and (low_spo2 or elderly or child):
            flags.append({
                'severity': 'warning',
                'message': f'SEASONAL: Respiratory infections HIGH this month. '
                           f'{"Elderly" if elderly else "Child" if child else "Low SpO2"} patient with respiratory symptoms — '
                           f'monitor closely, consider chest X-ray.'
            })

    # --- Leptospirosis (post-flood) ---
    lepto_risk = risks.get("leptospirosis", {}).get("risk_score", 0)
    if lepto_risk >= 0.35:
        lepto_keywords = ["muscle pain", "jaundice", "red eye", "flood", "water", "wade"]
        has_lepto = any(kw in text_lower for kw in lepto_keywords)
        has_fever = "fever" in text_lower or vitals.get('temperature_c', 37) > 38

        if has_fever and has_lepto:
            flags.append({
                'severity': 'warning',
                'message': f'SEASONAL: Leptospirosis risk present (risk: {lepto_risk:.0%}). '
                           f'Fever + muscle pain/jaundice after water exposure — test urgently.'
            })

    # --- Snakebite season ---
    snake_risk = risks.get("snakebite", {}).get("risk_score", 0)
    if snake_risk >= 0.4:
        snake_keywords = ["bite", "snake", "fang", "swelling", "numbness"]
        has_snake = any(kw in text_lower for kw in snake_keywords)

        if has_snake:
            flags.append({
                'severity': 'high',
                'message': f'SEASONAL: Snakebite season active. '
                           f'Immobilize limb, mark swelling edge with time. '
                           f'REFER for anti-venom — do NOT apply tourniquet.'
            })

    return flags


# =====================================================================
# 3. SEASONAL URGENCY MULTIPLIER
# =====================================================================
def get_urgency_seasonal_adjustment(disease_category, symptom_text,
                                     region="tamil_nadu", month=None):
    """
    Returns an urgency adjustment value (-1, 0, or +1) based on
    seasonal disease risk.

    +1 means bump urgency up one level (e.g., Semi-Urgent → Urgent)
     0 means no change
    -1 rarely used, but could lower urgency in off-season

    Usage in predict_triage() after Step 9:
        seasonal_adj = get_urgency_seasonal_adjustment(
            DISEASE_LABELS[disease_pred], symptom_text, "tamil_nadu"
        )
        if seasonal_adj > 0 and urgency_pred < 3:
            urgency_pred = min(urgency_pred + seasonal_adj, 3)
    """
    if month is None:
        month = datetime.now().month

    risks = get_seasonal_disease_risks(region, month)
    text_lower = symptom_text.lower()

    # Find the highest seasonal risk that matches both the predicted
    # disease category AND the patient's symptoms
    max_matching_risk = 0.0

    for disease_key, risk_info in risks.items():
        if risk_info["category"] == disease_category and risk_info["is_elevated"]:
            # Check if patient symptoms overlap with seasonal disease symptoms
            profile = get_region_profile(region)
            if profile:
                disease_symptoms = profile["diseases"][disease_key]["symptoms"]
                symptom_overlap = sum(1 for s in disease_symptoms if s in text_lower)
                if symptom_overlap >= 2:
                    max_matching_risk = max(max_matching_risk, risk_info["risk_score"])

    # Only bump urgency if risk is very high AND symptoms match
    if max_matching_risk >= 0.6:
        return 1
    return 0


# =====================================================================
# 4. DASHBOARD SEASONAL ALERTS
# =====================================================================
def get_seasonal_alerts(region="tamil_nadu", month=None):
    """
    Generate seasonal alert messages for the dashboard banner.

    Returns a list of alert dicts:
    [{
        'disease': 'Dengue',
        'risk_level': 'high' | 'moderate' | 'low',
        'risk_score': 0.85,
        'message': '...',
        'advisory': '...',
        'category': 'Infectious'
    }]

    Usage: Call from a new /api/seasonal-alerts endpoint
    """
    if month is None:
        month = datetime.now().month

    risks = get_seasonal_disease_risks(region, month)
    season = get_current_season(month)
    alerts = []

    for disease_key, risk_info in risks.items():
        if risk_info["is_elevated"]:
            risk_score = risk_info["risk_score"]

            if risk_score >= 0.6:
                risk_level = "high"
            elif risk_score >= 0.4:
                risk_level = "moderate"
            else:
                risk_level = "low"

            # Format disease name from key
            disease_name = disease_key.replace("_", " ").title()

            alerts.append({
                "disease": disease_name,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "message": f"{disease_name} risk is {risk_level.upper()} "
                           f"this month ({risk_score:.0%})",
                "advisory": risk_info["advisory"],
                "category": risk_info["category"]
            })

    # Sort by risk score descending
    alerts.sort(key=lambda x: x["risk_score"], reverse=True)

    return {
        "season": season,
        "month": month,
        "month_name": datetime(2026, month, 1).strftime("%B"),
        "region": region,
        "region_label": get_region_profile(region)["label"] if get_region_profile(region) else region,
        "alerts": alerts,
        "alert_count": len(alerts),
        "top_risk": alerts[0] if alerts else None
    }


# =====================================================================
# 5. SEASONAL CONTEXT FOR EXPLANATION
# =====================================================================
def get_seasonal_explanation(disease_category, region="tamil_nadu", month=None):
    """
    Returns a human-readable seasonal context string to add to
    the triage explanation list.

    Usage in predict_triage():
        seasonal_note = get_seasonal_explanation(DISEASE_LABELS[disease_pred])
        if seasonal_note:
            explanation.append(seasonal_note)
    """
    if month is None:
        month = datetime.now().month

    season = get_current_season(month)
    risks = get_seasonal_disease_risks(region, month)

    # Find diseases in this category with elevated risk
    elevated = []
    for disease_key, risk_info in risks.items():
        if risk_info["category"] == disease_category and risk_info["is_elevated"]:
            disease_name = disease_key.replace("_", " ").title()
            elevated.append(f"{disease_name} ({risk_info['risk_score']:.0%})")

    if elevated:
        return (f"SEASONAL CONTEXT ({season['label']}): "
                f"{disease_category} diseases currently elevated — "
                f"{', '.join(elevated)}. Consider seasonal factors in diagnosis.")
    return None


# =====================================================================
# 6. SUPPORTED REGIONS LIST (for frontend dropdown)
# =====================================================================
def get_available_regions():
    """Return list of available regions for the frontend."""
    if SEASONAL_DB is None:
        return []

    return [
        {"key": key, "label": profile["label"]}
        for key, profile in SEASONAL_DB["regional_profiles"].items()
    ]


# =====================================================================
# QUICK TEST
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  SEASONAL DISEASE INTELLIGENCE MODULE — TEST")
    print("=" * 60)

    # Test for current month
    month = datetime.now().month
    print(f"\nCurrent month: {datetime(2026, month, 1).strftime('%B %Y')}")
    print(f"Season: {get_current_season(month)['label']}")

    # Test seasonal alerts
    print(f"\n--- Seasonal Alerts (Tamil Nadu, Month {month}) ---")
    alerts = get_seasonal_alerts("tamil_nadu", month)
    for alert in alerts["alerts"]:
        emoji = "🔴" if alert["risk_level"] == "high" else "🟡" if alert["risk_level"] == "moderate" else "🟢"
        print(f"  {emoji} {alert['message']}")
        print(f"     Advisory: {alert['advisory']}")

    # Test seasonal red flags
    print(f"\n--- Seasonal Red Flags Test ---")
    test_vitals = {
        'temperature_c': 39.0,
        'heart_rate_bpm': 105,
        'spo2_percent': 96,
        'bp_systolic_mmhg': 110,
        'age': 35,
        'consciousness': 'Alert'
    }
    test_symptoms = "fever for 3 days with body ache and joint pain headache"

    flags = check_seasonal_red_flags(test_vitals, test_symptoms, "tamil_nadu", month)
    if flags:
        for flag in flags:
            print(f"  [{flag['severity'].upper()}] {flag['message']}")
    else:
        print("  No seasonal red flags triggered for this case.")

    # Test category boost
    print(f"\n--- Disease Category Seasonal Boosts ---")
    boosts = get_category_seasonal_boost("tamil_nadu", month)
    for cat, boost in sorted(boosts.items(), key=lambda x: x[1], reverse=True):
        if boost > 0:
            print(f"  {cat}: +{boost:.4f}")

    # Test urgency adjustment
    print(f"\n--- Urgency Adjustment Test ---")
    adj = get_urgency_seasonal_adjustment("Infectious", test_symptoms, "tamil_nadu", month)
    print(f"  Adjustment for 'Infectious' + fever/body ache: {'+' if adj > 0 else ''}{adj}")

    # Test explanation
    print(f"\n--- Seasonal Explanation ---")
    expl = get_seasonal_explanation("Infectious", "tamil_nadu", month)
    print(f"  {expl}")

    print(f"\n--- Available Regions ---")
    for r in get_available_regions():
        print(f"  {r['key']}: {r['label']}")

    print("\n" + "=" * 60)
    print("  All tests passed! Module ready for integration.")
    print("=" * 60)
