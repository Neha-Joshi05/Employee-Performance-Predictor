# data/generate_data.py
# Generates a realistic synthetic HR dataset for 1000 employees

import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

n = 1000  # number of employees

# --- Generate raw features ---
departments = ['Sales', 'Engineering', 'HR', 'Marketing', 'Finance', 'Operations']
genders = ['Male', 'Female']
education_levels = ['Bachelor', 'Master', 'PhD', 'Diploma']

data = {
    'employee_id': [f'EMP{1000+i}' for i in range(n)],
    'age': np.random.randint(22, 58, n),
    'gender': np.random.choice(genders, n),
    'education': np.random.choice(education_levels, n, p=[0.5, 0.3, 0.1, 0.1]),
    'department': np.random.choice(departments, n),
    'years_at_company': np.random.randint(1, 20, n),
    'monthly_salary': np.random.randint(30000, 120000, n),
    'training_hours': np.random.randint(0, 100, n),
    'projects_completed': np.random.randint(1, 20, n),
    'overtime_hours': np.random.randint(0, 50, n),
    'satisfaction_score': np.round(np.random.uniform(1.0, 5.0, n), 1),
    'manager_rating': np.round(np.random.uniform(1.0, 5.0, n), 1),
    'attendance_rate': np.round(np.random.uniform(60.0, 100.0, n), 1),
    'last_promotion_years': np.random.randint(0, 10, n),
}

df = pd.DataFrame(data)

# --- Create realistic performance score (target variable) ---
# Higher score = better performance. Logic mirrors real HR thinking.
score = (
    df['manager_rating'] * 20 +
    df['projects_completed'] * 2 +
    df['training_hours'] * 0.3 +
    df['satisfaction_score'] * 5 +
    df['attendance_rate'] * 0.3 -
    df['overtime_hours'] * 0.2 -
    df['last_promotion_years'] * 1.5 +
    np.random.normal(0, 5, n)  # noise
)

# Normalize score to 0-100
score = (score - score.min()) / (score.max() - score.min()) * 100
df['performance_score'] = np.round(score, 2)

# --- Create performance label (classification target) ---
def label_performance(s):
    if s >= 75:
        return 'High'
    elif s >= 50:
        return 'Medium'
    else:
        return 'Low'

df['performance_label'] = df['performance_score'].apply(label_performance)

# --- Save dataset ---
os.makedirs('data', exist_ok=True)
df.to_csv('data/employee_data.csv', index=False)

print("✅ Dataset created successfully!")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nPerformance Label Distribution:")
print(df['performance_label'].value_counts())