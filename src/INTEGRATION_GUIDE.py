"""
INTEGRATION GUIDE — How to wire seasonal_context.py into app.py
================================================================
Authors: Sarthak Gupta, Dhruv

This file shows the EXACT code changes needed in your app.py
to integrate the seasonal disease intelligence module.

Follow these 6 steps in order.
================================================================
"""

# ================================================================
# STEP 1: Add import at the top of app.py (after line 18)
# ================================================================
# Add this line alongside your other imports:

from seasonal_context import (
    get_seasonal_disease_risks,
    get_category_seasonal_boost,
    check_seasonal_red_flags,
    get_urgency_seasonal_adjustment,
    get_seasonal_alerts,
    get_seasonal_explanation,
    get_available_regions
)


# ================================================================
# STEP 2: Add region parameter to predict endpoint (line ~597)
# ================================================================
# In the predict() route, extract the region from the request:

# FIND THIS (around line 629):
#     symptom_text = data.get('symptoms', 'Patient reports feeling unwell')
#     na_fields = data.get('na_fields', [])
#
# ADD AFTER IT:
#     region = data.get('region', 'tamil_nadu')  # Default to Tamil Nadu


# ================================================================
# STEP 3: Integrate into predict_triage() function
# ================================================================
# You need to modify predict_triage() to accept and use the region.
# Change the function signature:

# CHANGE FROM:
#     def predict_triage(vitals, symptom_text):
#
# CHANGE TO:
#     def predict_triage(vitals, symptom_text, region="tamil_nadu"):

# Then add these integration points inside predict_triage():

# ---------- AFTER Step 5 (Fusion, around line 328) ----------
# Add seasonal boost to disease probabilities:

"""
    # --- Step 5b: Apply seasonal disease boost ---
    seasonal_boosts = get_category_seasonal_boost(region)
    for i, disease_name in DISEASE_LABELS.items():
        boost = seasonal_boosts.get(disease_name, 0.0)
        if boost > 0 and i < len(fused_disease_prob):
            fused_disease_prob[i] += boost

    # Re-normalize so probabilities still sum to 1.0
    prob_sum = fused_disease_prob.sum()
    if prob_sum > 0:
        fused_disease_prob = fused_disease_prob / prob_sum
"""

# ---------- AFTER Step 8 (Red flags, around line 270) ----------
# Add seasonal red flags to existing flags:

"""
    # --- Step 8b: Seasonal red flags ---
    seasonal_flags = check_seasonal_red_flags(vitals, symptom_text, region)
    red_flags.extend(seasonal_flags)

    # Re-check has_critical/has_high after adding seasonal flags
    has_critical = any(f['severity'] == 'critical' for f in red_flags)
    has_high = any(f['severity'] == 'high' for f in red_flags)
"""

# ---------- AFTER Step 9 (Confidence escalation, around line 365) ----------
# Add seasonal urgency adjustment:

"""
    # --- Step 9b: Seasonal urgency adjustment ---
    seasonal_urgency_adj = get_urgency_seasonal_adjustment(
        DISEASE_LABELS[disease_pred], symptom_text, region
    )
    if seasonal_urgency_adj > 0 and urgency_pred < 3:
        urgency_pred = min(urgency_pred + seasonal_urgency_adj, 3)
        explanation.append(
            f"SEASONAL ADJUSTMENT: Urgency elevated due to high {DISEASE_LABELS[disease_pred]} "
            f"season risk in this region"
        )
"""

# ---------- AFTER Step 12 (Explanation, around line 464) ----------
# Add seasonal context to explanation:

"""
    # --- Step 12b: Seasonal explanation ---
    seasonal_note = get_seasonal_explanation(DISEASE_LABELS[disease_pred], region)
    if seasonal_note:
        explanation.append(seasonal_note)
"""

# ---------- In the return dict (around line 466) ----------
# Add seasonal data to the response:

"""
    # Add these keys to the return dict:

    # === NEW: Seasonal Context ===
    'seasonal_alerts': get_seasonal_alerts(region),
    'seasonal_region': region,
"""


# ================================================================
# STEP 4: Update the predict() route call (line ~632)
# ================================================================

# CHANGE FROM:
#     result = predict_triage(vitals, symptom_text)
#
# CHANGE TO:
#     result = predict_triage(vitals, symptom_text, region)


# ================================================================
# STEP 5: Add new API endpoint for seasonal alerts (before the RUN section)
# ================================================================
# Add this new route to app.py:

"""
@app.route('/api/seasonal-alerts', methods=['GET'])
def seasonal_alerts():
    \"\"\"Get current seasonal disease alerts for a region.\"\"\"
    region = request.args.get('region', 'tamil_nadu')
    month = request.args.get('month', None)
    if month:
        month = int(month)
    alerts = get_seasonal_alerts(region, month)
    return jsonify(alerts)


@app.route('/api/regions', methods=['GET'])
def regions():
    \"\"\"Get list of available regions.\"\"\"
    return jsonify(get_available_regions())
"""


# ================================================================
# STEP 6: Frontend Integration (index.html dashboard)
# ================================================================
# Add a region selector dropdown and seasonal alert banner.
# Here's the JavaScript to add to your dashboard page:

"""
<!-- Add to the dashboard form (index.html), before the Submit button -->
<div class="form-group">
    <label>Region</label>
    <select id="region" name="region">
        <option value="tamil_nadu" selected>Tamil Nadu</option>
        <option value="south_india">South India (KA, KL, AP, TS)</option>
        <option value="north_india">North India (UP, Bihar, MP, Delhi)</option>
        <option value="west_india">West India (MH, GJ, Goa)</option>
        <option value="northeast_india">Northeast India</option>
    </select>
</div>

<!-- Add a seasonal alert banner div at the top of the results section -->
<div id="seasonal-banner" style="display:none;"></div>

<script>
// Fetch and display seasonal alerts on page load
async function loadSeasonalAlerts() {
    const region = document.getElementById('region').value;
    try {
        const res = await fetch(`/api/seasonal-alerts?region=${region}`);
        const data = await res.json();
        const banner = document.getElementById('seasonal-banner');

        if (data.alerts && data.alerts.length > 0) {
            let html = `
                <div style="background: #FEF3C7; border-left: 4px solid #F59E0B;
                            padding: 12px 16px; margin-bottom: 16px; border-radius: 4px;">
                    <strong>Seasonal Alert — ${data.month_name} (${data.season.label})</strong>
                    <p style="margin: 4px 0; font-size: 14px; color: #92400E;">
                        Region: ${data.region_label}
                    </p>
                    <ul style="margin: 8px 0; padding-left: 20px;">
            `;
            data.alerts.forEach(alert => {
                const color = alert.risk_level === 'high' ? '#DC2626' :
                              alert.risk_level === 'moderate' ? '#F59E0B' : '#22C55E';
                html += `
                    <li style="margin: 4px 0; font-size: 13px;">
                        <span style="color: ${color}; font-weight: bold;">
                            ${alert.risk_level.toUpperCase()}
                        </span>
                        — ${alert.message}
                    </li>
                `;
            });
            html += '</ul></div>';
            banner.innerHTML = html;
            banner.style.display = 'block';
        }
    } catch (err) {
        console.log('Seasonal alerts not available:', err);
    }
}

// Load alerts when page loads and when region changes
document.addEventListener('DOMContentLoaded', loadSeasonalAlerts);
document.getElementById('region').addEventListener('change', loadSeasonalAlerts);

// Include region in the triage submission
// In your existing form submit handler, add:
// formData.region = document.getElementById('region').value;
</script>
"""


# ================================================================
# FILE PLACEMENT
# ================================================================
"""
Place the files in your project directory like this:

project/
├── app/
│   ├── app.py                    (modify as shown above)
│   ├── seasonal_context.py       (NEW — place here)
│   ├── templates/
│   │   ├── welcome.html
│   │   ├── index.html            (modify to add region selector + banner)
│   │   └── chatbot.html
│   └── static/
├── data/
│   ├── processed/
│   │   ├── scaler.pkl
│   │   ├── label_encoders.pkl
│   │   └── metadata.json
│   └── seasonal_disease_data.json  (NEW — place here)
├── models/
│   ├── vitals_model_risk_level.pkl
│   ├── nlp_model_risk_level.pkl
│   └── ... (other model files)
"""


# ================================================================
# QUICK VERIFICATION
# ================================================================
"""
After integration, verify with:

1. Start the app: python app/app.py
2. Open browser → http://localhost:5001/dashboard
3. Check that the seasonal alert banner appears
4. Submit a triage assessment and verify:
   - Seasonal red flags appear in the output
   - Seasonal context appears in the explanation section
   - Disease predictions are influenced by seasonal data
5. Test the API: http://localhost:5001/api/seasonal-alerts?region=tamil_nadu
6. Try different months: http://localhost:5001/api/seasonal-alerts?region=tamil_nadu&month=10
"""
