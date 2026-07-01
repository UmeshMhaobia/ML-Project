from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# ── Load model ──────────────────────────────────────────────────────────────
# Place your trained model file as  model.pkl  in this same folder.
# To save from notebook: import joblib; joblib.dump(rf, 'model.pkl')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'heart_model.pkl')
model = None

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print("⚠️  model.pkl not found — running in DEMO mode (returns dummy prediction)")

# ── Encoding maps (match notebook LabelEncoder alphabetical order) ───────────
GENDER_MAP         = {'Female': 0, 'Male': 1}
REGION_MAP         = {'East': 0, 'North': 1, 'South': 2, 'West': 3}
MARITAL_MAP        = {'Divorced': 0, 'Married': 1, 'Single': 2}
BMI_MAP            = {'Normal': 0, 'Obese': 1, 'Overweight': 2, 'Underweight': 3}
EMPLOYMENT_MAP     = {'Employed': 0, 'Self-Employed': 1, 'Student': 2, 'Unemployed': 3}
SMOKING_MAP        = {'No': 0, 'Yes': 1}
INCOME_LEVEL_MAP   = {'High': 0, 'Low': 1, 'Medium': 2}
MEDICAL_MAP        = {'Diabetes': 0, 'Heart Disease': 1, 'Hypertension': 2, 'None': 3}
INSURANCE_PLAN_MAP = {'Basic': 0, 'Gold': 1, 'Premium': 2, 'Silver': 3}
PHYSICAL_MAP       = {'High': 0, 'Low': 1, 'Medium': 2}  # alphabetical LE
STRESS_MAP         = {'High': 0, 'Low': 1, 'Medium': 2}  # alphabetical LE

# Lifestyle risk score (from notebook custom map)
PHYSICAL_RISK = {'High': 0, 'Medium': 1, 'Low': 4}
STRESS_RISK   = {'Low': 0, 'Medium': 1, 'High': 4}

# ── Column order (confirmed from notebook test_data) ─────────────────────────
# [Age, Gender, Region, Marital_status, Number_Of_Dependants, BMI_Category,
#  Employment_Status, Smoking_Status, Income_Level, Medical History,
#  Insurance_Plan, Income_Lakhs, Lifestyle_Risk_Score,
#  Physical_Activity, Stress_Level]


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        age          = int(data['age'])
        gender       = data['gender']
        region       = data['region']
        marital      = data['marital_status']
        dependants   = int(data['dependants'])
        bmi          = data['bmi_category']
        employment   = data['employment_status']
        smoking      = data['smoking_status']
        income_level = data['income_level']
        medical      = data['medical_history']
        ins_plan     = data['insurance_plan']
        income_lakhs = float(data['income_lakhs'])
        physical     = data['physical_activity']
        stress       = data['stress_level']

        # Lifestyle risk score
        lrs = PHYSICAL_RISK[physical] + STRESS_RISK[stress]

        # Build feature vector (exact column order)
        features = [
            age,
            GENDER_MAP[gender],
            REGION_MAP[region],
            MARITAL_MAP[marital],
            dependants,
            BMI_MAP[bmi],
            EMPLOYMENT_MAP[employment],
            SMOKING_MAP[smoking],
            INCOME_LEVEL_MAP[income_level],
            MEDICAL_MAP[medical],
            INSURANCE_PLAN_MAP[ins_plan],
            income_lakhs,
            lrs,
            PHYSICAL_MAP[physical],
            STRESS_MAP[stress],
        ]

        X = np.array([features])

        if model is not None:
            prediction = float(model.predict(X)[0])
        else:
            # Demo mode: simple heuristic so UI still works
            base = {'Basic': 5000, 'Silver': 8000, 'Gold': 12000, 'Premium': 18000}[ins_plan]
            age_load = base * (0.5 if age >= 60 else 0.3 if age >= 45 else 0.1 if age >= 30 else 0)
            health = (0.4 if smoking == 'Yes' else 0) + \
                     (0.25 if bmi == 'Obese' else 0.1 if bmi == 'Overweight' else 0) + \
                     (0.35 if medical == 'Heart Disease' else 0.25 if medical == 'Diabetes' else 0.15 if medical == 'Hypertension' else 0)
            prediction = (base + age_load + base * health) * {'Basic':1.0,'Silver':1.2,'Gold':1.5,'Premium':2.0}[ins_plan]

        return jsonify({
            'success': True,
            'annual_premium': round(prediction, 2),
            'monthly_premium': round(prediction / 12, 2),
            'lifestyle_risk_score': lrs,
            'features_used': features,
            'model_mode': 'live' if model is not None else 'demo'
        })

    except KeyError as e:
        return jsonify({'success': False, 'error': f'Invalid value: {e}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
