# src/preprocess.py
# Cleans and preprocesses the HR dataset for ML model training

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# --- Step 1: Load Data ---
df = pd.read_csv('data/employee_data.csv')
print("✅ Data Loaded")
print(f"Shape: {df.shape}")

# --- Step 2: Check for Missing Values ---
print("\n🔍 Missing Values:")
print(df.isnull().sum())

# --- Step 3: Check for Duplicates ---
dupes = df.duplicated().sum()
print(f"\n🔍 Duplicate Rows: {dupes}")
df.drop_duplicates(inplace=True)

# --- Step 4: Basic Stats ---
print("\n📊 Basic Statistics:")
print(df.describe().round(2))

# --- Step 5: Encode Categorical Columns ---
# Columns to encode: gender, education, department
categorical_cols = ['gender', 'education', 'department']

encoders = {}  # save encoders for later use in prediction

for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"✅ Encoded: {col} → {col}_encoded | Classes: {list(le.classes_)}")

# Save encoders
joblib.dump(encoders, 'models/label_encoders.pkl')
print("\n✅ Label encoders saved to models/label_encoders.pkl")

# --- Step 6: Encode Target Label ---
target_encoder = LabelEncoder()
df['performance_encoded'] = target_encoder.fit_transform(df['performance_label'])
joblib.dump(target_encoder, 'models/target_encoder.pkl')
print(f"✅ Target encoded | Classes: {list(target_encoder.classes_)}")
# High=0, Low=1, Medium=2 (alphabetical order by LabelEncoder)

# --- Step 7: Select Features for ML ---
feature_cols = [
    'age',
    'years_at_company',
    'monthly_salary',
    'training_hours',
    'projects_completed',
    'overtime_hours',
    'satisfaction_score',
    'manager_rating',
    'attendance_rate',
    'last_promotion_years',
    'gender_encoded',
    'education_encoded',
    'department_encoded'
]

X = df[feature_cols]
y = df['performance_encoded']

# --- Step 8: Scale Numerical Features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Scaler saved to models/scaler.pkl")

# --- Step 9: Save Cleaned Data ---
df.to_csv('outputs/cleaned_data.csv', index=False)
X_scaled.to_csv('outputs/X_scaled.csv', index=False)
y.to_csv('outputs/y_target.csv', index=False)

print("\n✅ Cleaned data saved to outputs/")
print(f"\nFinal Feature Matrix Shape: {X_scaled.shape}")
print(f"Target Distribution:\n{y.value_counts()}")
print("\n🎯 Preprocessing Complete!")