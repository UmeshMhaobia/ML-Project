# Insurance Premium Predictor

## Folder structure
```
insurance_app/
├── app.py              ← Flask backend
├── requirements.txt    ← Python dependencies
├── model.pkl           ← YOUR trained model (add this!)
└── static/
    └── index.html      ← Frontend UI
```

## Step 1 — Save your model from the notebook

Add this at the END of your notebook (after training):

```python
import joblib
joblib.dump(rf, 'model.pkl')   # rf = RandomForestRegressor
# OR for Linear Regression:
# joblib.dump(model, 'model.pkl')
```

Copy the generated `model.pkl` into the `insurance_app/` folder.

## Step 2 — Install dependencies

```bash
pip install flask flask-cors numpy scikit-learn joblib
```

## Step 3 — Run the backend

```bash
cd insurance_app
python app.py
```

Server starts at: http://localhost:5000

## Step 4 — Open the frontend

Open http://localhost:5000 in your browser.

---

## Column order used (confirmed from notebook test_data)

| Position | Column              | Encoding                                |
|----------|---------------------|-----------------------------------------|
| 0        | Age                 | numeric                                 |
| 1        | Gender              | Female=0, Male=1                        |
| 2        | Region              | East=0, North=1, South=2, West=3        |
| 3        | Marital_status      | Divorced=0, Married=1, Single=2         |
| 4        | Number_Of_Dependants| numeric                                 |
| 5        | BMI_Category        | Normal=0, Obese=1, Overweight=2, Under=3|
| 6        | Employment_Status   | Employed=0, Self=1, Student=2, Unemp=3  |
| 7        | Smoking_Status      | No=0, Yes=1                             |
| 8        | Income_Level        | High=0, Low=1, Medium=2                 |
| 9        | Medical History     | Diabetes=0, HeartDisease=1, Hyp=2, None=3|
| 10       | Insurance_Plan      | Basic=0, Gold=1, Premium=2, Silver=3    |
| 11       | Income_Lakhs        | numeric                                 |
| 12       | Lifestyle_Risk_Score| numeric (physical_map + stress_map)     |
| 13       | Physical_Activity   | High=0, Low=1, Medium=2 (alphabetical)  |
| 14       | Stress_Level        | High=0, Low=1, Medium=2 (alphabetical)  |

## Note on demo mode

If `model.pkl` is missing, the app runs in **demo mode** with a
rule-based premium estimate. The UI shows "⚠ Demo mode" badge.
