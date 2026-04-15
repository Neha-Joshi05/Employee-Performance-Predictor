# src/predict.py
# Predicts performance for new employees using the trained model

import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------------
# STEP 1: Load all saved models and encoders
# -------------------------------------------------------
model          = joblib.load('models/best_model.pkl')
encoders       = joblib.load('models/label_encoders.pkl')
target_encoder = joblib.load('models/target_encoder.pkl')
scaler         = joblib.load('models/scaler.pkl')
feature_names  = joblib.load('models/feature_names.pkl')

print("✅ All models and encoders loaded\n")

# -------------------------------------------------------
# STEP 2: Define new employees to predict
# -------------------------------------------------------
# You can change these values to test different employees!
new_employees = [
    {
        'name'                : 'Rahul Sharma',
        'age'                 : 30,
        'gender'              : 'Male',
        'education'           : 'Master',
        'department'          : 'Engineering',
        'years_at_company'    : 5,
        'monthly_salary'      : 85000,
        'training_hours'      : 70,
        'projects_completed'  : 12,
        'overtime_hours'      : 10,
        'satisfaction_score'  : 4.2,
        'manager_rating'      : 4.5,
        'attendance_rate'     : 95.0,
        'last_promotion_years': 1,
    },
    {
        'name'                : 'Priya Mehta',
        'age'                 : 45,
        'gender'              : 'Female',
        'education'           : 'Bachelor',
        'department'          : 'Sales',
        'years_at_company'    : 15,
        'monthly_salary'      : 42000,
        'training_hours'      : 10,
        'projects_completed'  : 3,
        'overtime_hours'      : 45,
        'satisfaction_score'  : 1.8,
        'manager_rating'      : 2.1,
        'attendance_rate'     : 67.0,
        'last_promotion_years': 8,
    },
    {
        'name'                : 'Aman Verma',
        'age'                 : 35,
        'gender'              : 'Male',
        'education'           : 'PhD',
        'department'          : 'Finance',
        'years_at_company'    : 8,
        'monthly_salary'      : 95000,
        'training_hours'      : 50,
        'projects_completed'  : 9,
        'overtime_hours'      : 20,
        'satisfaction_score'  : 3.5,
        'manager_rating'      : 3.8,
        'attendance_rate'     : 88.0,
        'last_promotion_years': 3,
    },
]

# -------------------------------------------------------
# STEP 3: Preprocess + Predict for each employee
# -------------------------------------------------------
def preprocess_employee(emp):
    """Encode, engineer features, and scale one employee record."""
    df = pd.DataFrame([emp])

    # Encode categoricals
    for col in ['gender', 'education', 'department']:
        df[col + '_encoded'] = encoders[col].transform(df[col])

    # Feature engineering (same as training)
    df['productivity_index'] = (
        df['projects_completed'] / (df['years_at_company'] + 1)
    ).round(3)
    df['engagement_score'] = (
        (df['satisfaction_score'] + df['manager_rating']) / 2
    ).round(3)
    df['overwork_flag'] = (
        (df['overtime_hours'] > 30) & (df['satisfaction_score'] < 3)
    ).astype(int)

    # Base feature columns (same order as training)
    base_cols = [
        'age', 'years_at_company', 'monthly_salary', 'training_hours',
        'projects_completed', 'overtime_hours', 'satisfaction_score',
        'manager_rating', 'attendance_rate', 'last_promotion_years',
        'gender_encoded', 'education_encoded', 'department_encoded'
    ]
    X_base = df[base_cols]

    # Scale base features
    X_base_scaled = pd.DataFrame(
        scaler.transform(X_base), columns=base_cols
    )

    # Scale engineered features
    from sklearn.preprocessing import StandardScaler
    eng_cols = ['productivity_index', 'engagement_score', 'overwork_flag']
    scaler_eng = StandardScaler()

    # Load training data to fit scaler on same distribution
    train_raw = pd.read_csv('outputs/cleaned_data.csv')
    train_raw['productivity_index'] = (
        train_raw['projects_completed'] / (train_raw['years_at_company'] + 1)
    ).round(3)
    train_raw['engagement_score'] = (
        (train_raw['satisfaction_score'] + train_raw['manager_rating']) / 2
    ).round(3)
    train_raw['overwork_flag'] = (
        (train_raw['overtime_hours'] > 30) & (train_raw['satisfaction_score'] < 3)
    ).astype(int)

    scaler_eng.fit(train_raw[eng_cols])
    X_eng_scaled = pd.DataFrame(
        scaler_eng.transform(df[eng_cols]), columns=eng_cols
    )

    # Combine all features
    X_final = pd.concat([X_base_scaled, X_eng_scaled], axis=1)
    return X_final


def predict_employee(emp):
    """Run full prediction pipeline for one employee."""
    X = preprocess_employee(emp)
    pred_encoded = model.predict(X)[0]
    pred_label   = target_encoder.inverse_transform([pred_encoded])[0]
    probabilities = model.predict_proba(X)[0]
    classes       = target_encoder.classes_

    prob_dict = {cls: round(prob * 100, 1)
                 for cls, prob in zip(classes, probabilities)}

    return pred_label, prob_dict


# -------------------------------------------------------
# STEP 4: Display Results
# -------------------------------------------------------
ICONS = {'High': '🟢', 'Medium': '🟡', 'Low': '🔴'}
HR_ADVICE = {
    'High'  : 'Consider for promotion or leadership roles.',
    'Medium': 'Provide mentoring and additional training.',
    'Low'   : 'Urgent intervention needed — counseling & PIP.',
}

print("=" * 58)
print("       🤖 EMPLOYEE PERFORMANCE PREDICTION RESULTS")
print("=" * 58)

results = []

for emp in new_employees:
    label, probs = predict_employee(emp)
    icon = ICONS[label]

    print(f"\n👤 Employee  : {emp['name']}")
    print(f"   Dept      : {emp['department']} | Age: {emp['age']}")
    print(f"   Salary    : ₹{emp['monthly_salary']:,} | Training: {emp['training_hours']} hrs")
    print(f"   Rating    : {emp['manager_rating']} | Satisfaction: {emp['satisfaction_score']}")
    print(f"\n   {icon} Predicted Performance : {label.upper()}")
    print(f"   📊 Confidence Scores:")
    for cls, prob in sorted(probs.items(), key=lambda x: -x[1]):
        bar = '█' * int(prob / 5)
        print(f"      {cls:<8}: {prob:5.1f}%  {bar}")
    print(f"\n   💼 HR Recommendation: {HR_ADVICE[label]}")
    print("-" * 58)

    results.append({
        'Name'           : emp['name'],
        'Department'     : emp['department'],
        'Predicted Label': label,
        'High %'         : probs.get('High', 0),
        'Medium %'       : probs.get('Medium', 0),
        'Low %'          : probs.get('Low', 0),
        'HR Action'      : HR_ADVICE[label],
    })

# -------------------------------------------------------
# STEP 5: Save prediction results to CSV
# -------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv('outputs/predictions.csv', index=False)
print("\n✅ Predictions saved to outputs/predictions.csv")
print("\n🎯 Prediction System Complete!")