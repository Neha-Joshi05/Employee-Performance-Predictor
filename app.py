# app.py
# Streamlit Web Dashboard for Employee Performance Predictor

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------
# Page Config
# -------------------------------------------------------
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="🏢",
    layout="wide"
)

# -------------------------------------------------------
# Load Models
# -------------------------------------------------------
@st.cache_resource
def load_models():
    model          = joblib.load('models/best_model.pkl')
    encoders       = joblib.load('models/label_encoders.pkl')
    target_encoder = joblib.load('models/target_encoder.pkl')
    scaler         = joblib.load('models/scaler.pkl')
    return model, encoders, target_encoder, scaler

@st.cache_data
def load_data():
    return pd.read_csv('outputs/cleaned_data.csv')

model, encoders, target_encoder, scaler = load_models()
df = load_data()

# -------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/performance-macbook.png", width=80)
st.sidebar.title("🏢 HR Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "📊 Data Overview",
    "📈 EDA Dashboard",
    "🤖 Predict Performance",
    "💡 HR Insights"
])

# -------------------------------------------------------
# PAGE 1: Home
# -------------------------------------------------------
if page == "🏠 Home":
    st.title("🏢 Employee Performance Predictor")
    st.markdown("### AI-powered HR Analytics System")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Employees", len(df))
    col2.metric("🟢 High Performers",
                int((df['performance_label'] == 'High').sum()))
    col3.metric("🟡 Medium Performers",
                int((df['performance_label'] == 'Medium').sum()))
    col4.metric("🔴 Low Performers",
                int((df['performance_label'] == 'Low').sum()))

    st.markdown("---")
    st.markdown("""
    ### 🎯 What this system does
    This dashboard helps HR teams and managers:
    - ✅ **Predict** employee performance using Machine Learning
    - ✅ **Analyze** workforce trends with interactive charts
    - ✅ **Identify** high performers and at-risk employees
    - ✅ **Generate** data-driven HR recommendations

    ### 🔬 Tech Stack
    | Component | Technology |
    |-----------|------------|
    | Language | Python 3.11 |
    | ML Model | Random Forest Classifier |
    | Data | Synthetic HR Dataset (1000 employees) |
    | Dashboard | Streamlit |
    | Libraries | Pandas, Scikit-learn, Seaborn |

    ### 📁 Project Workflow
Data Generation → Preprocessing → EDA → Feature Engineering
     → Model Training → Evaluation → Prediction → Dashboard

""")

# -------------------------------------------------------
# PAGE 2: Data Overview
# -------------------------------------------------------
elif page == "📊 Data Overview":
    st.title("📊 Dataset Overview")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Raw Data Sample")
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        st.subheader("📐 Dataset Shape")
        st.info(f"Rows: **{df.shape[0]}** | Columns: **{df.shape[1]}**")
        st.subheader("🔍 Column Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values.astype(str)
        })
        st.dataframe(dtype_df, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe().round(2), use_container_width=True)

# -------------------------------------------------------
# PAGE 3: EDA Dashboard
# -------------------------------------------------------
elif page == "📈 EDA Dashboard":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("---")

    PALETTE = {'High': '#2ecc71', 'Medium': '#f39c12', 'Low': '#e74c3c'}
    order = ['High', 'Medium', 'Low']

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Performance Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df['performance_label'].value_counts().reindex(order)
        ax.bar(counts.index, counts.values,
               color=[PALETTE[k] for k in order], edgecolor='black')
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Department vs Performance")
        fig, ax = plt.subplots(figsize=(6, 4))
        dept_perf = df.groupby(
            ['department', 'performance_label']
        ).size().unstack(fill_value=0)[order]
        dept_perf.plot(kind='bar', stacked=True, ax=ax,
                       color=[PALETTE[k] for k in order], edgecolor='black')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        ax.legend(title='Performance')
        st.pyplot(fig)
        plt.close()

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Training Hours by Performance")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='performance_label', y='training_hours',
                    order=order, palette=PALETTE, ax=ax)
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("Manager Rating by Performance")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.violinplot(data=df, x='performance_label', y='manager_rating',
                       order=order, palette=PALETTE, ax=ax, inner='quartile')
        st.pyplot(fig)
        plt.close()

    # Heatmap
    st.subheader("🔥 Correlation Heatmap")
    num_cols = ['age', 'monthly_salary', 'training_hours', 'projects_completed',
                'overtime_hours', 'satisfaction_score', 'manager_rating',
                'attendance_rate', 'performance_score']
    fig, ax = plt.subplots(figsize=(10, 6))
    mask = np.triu(np.ones_like(df[num_cols].corr(), dtype=bool))
    sns.heatmap(df[num_cols].corr(), mask=mask, annot=True, fmt='.2f',
                cmap='RdYlGn', ax=ax, linewidths=0.5)
    st.pyplot(fig)
    plt.close()

# -------------------------------------------------------
# PAGE 4: Predict Performance
# -------------------------------------------------------
elif page == "🤖 Predict Performance":
    st.title("🤖 Predict Employee Performance")
    st.markdown("Fill in the employee details below:")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Personal Info")
        name        = st.text_input("Employee Name", "Rahul Sharma")
        age         = st.slider("Age", 22, 60, 30)
        gender      = st.selectbox("Gender", ["Male", "Female"])
        education   = st.selectbox("Education",
                                   ["Bachelor", "Master", "PhD", "Diploma"])

    with col2:
        st.subheader("🏢 Work Info")
        department        = st.selectbox("Department",
                            ["Engineering", "Sales", "HR",
                             "Marketing", "Finance", "Operations"])
        years_at_company  = st.slider("Years at Company", 1, 20, 5)
        monthly_salary    = st.number_input("Monthly Salary (₹)",
                            30000, 120000, 75000, step=5000)
        last_promotion    = st.slider("Years Since Last Promotion", 0, 10, 2)

    with col3:
        st.subheader("📈 Performance Metrics")
        training_hours      = st.slider("Training Hours", 0, 100, 50)
        projects_completed  = st.slider("Projects Completed", 1, 20, 8)
        overtime_hours      = st.slider("Overtime Hours", 0, 50, 10)
        satisfaction_score  = st.slider("Satisfaction Score", 1.0, 5.0, 3.5)
        manager_rating      = st.slider("Manager Rating", 1.0, 5.0, 3.8)
        attendance_rate     = st.slider("Attendance Rate (%)", 60.0, 100.0, 90.0)

    st.markdown("---")

    if st.button("🚀 Predict Performance", use_container_width=True):
        # Preprocess
        emp = {
            'gender': gender, 'education': education,
            'department': department
        }

        base = {
            'age': age, 'years_at_company': years_at_company,
            'monthly_salary': monthly_salary, 'training_hours': training_hours,
            'projects_completed': projects_completed,
            'overtime_hours': overtime_hours,
            'satisfaction_score': satisfaction_score,
            'manager_rating': manager_rating,
            'attendance_rate': attendance_rate,
            'last_promotion_years': last_promotion,
            'gender_encoded'    : encoders['gender'].transform([gender])[0],
            'education_encoded' : encoders['education'].transform([education])[0],
            'department_encoded': encoders['department'].transform([department])[0],
        }

        base_cols = [
            'age', 'years_at_company', 'monthly_salary', 'training_hours',
            'projects_completed', 'overtime_hours', 'satisfaction_score',
            'manager_rating', 'attendance_rate', 'last_promotion_years',
            'gender_encoded', 'education_encoded', 'department_encoded'
        ]

        X_base = pd.DataFrame([base])[base_cols]
        X_base_scaled = pd.DataFrame(scaler.transform(X_base), columns=base_cols)

        # Engineered features
        prod_idx   = projects_completed / (years_at_company + 1)
        eng_score  = (satisfaction_score + manager_rating) / 2
        overwork   = int(overtime_hours > 30 and satisfaction_score < 3)

        train_raw = df.copy()
        train_raw['productivity_index'] = (
            train_raw['projects_completed'] / (train_raw['years_at_company'] + 1)
        )
        train_raw['engagement_score'] = (
            (train_raw['satisfaction_score'] + train_raw['manager_rating']) / 2
        )
        train_raw['overwork_flag'] = (
            (train_raw['overtime_hours'] > 30) &
            (train_raw['satisfaction_score'] < 3)
        ).astype(int)

        eng_cols = ['productivity_index', 'engagement_score', 'overwork_flag']
        sc2 = StandardScaler()
        sc2.fit(train_raw[eng_cols])
        X_eng = sc2.transform([[prod_idx, eng_score, overwork]])
        X_eng = pd.DataFrame(X_eng, columns=eng_cols)

        X_final = pd.concat([X_base_scaled, X_eng], axis=1)

        # Predict
        pred_enc   = model.predict(X_final)[0]
        pred_label = target_encoder.inverse_transform([pred_enc])[0]
        probs      = model.predict_proba(X_final)[0]
        classes    = target_encoder.classes_
        prob_dict  = {c: round(p * 100, 1) for c, p in zip(classes, probs)}

        # Display result
        COLOR = {'High': '🟢', 'Medium': '🟡', 'Low': '🔴'}
        ADVICE = {
            'High'  : '⭐ Consider for promotion or leadership role.',
            'Medium': '📚 Provide mentoring and additional training.',
            'Low'   : '🚨 Urgent intervention needed — counseling & PIP.'
        }

        st.markdown("---")
        st.subheader(f"Results for: **{name}**")

        r1, r2, r3 = st.columns(3)
        r1.metric("Prediction", f"{COLOR[pred_label]} {pred_label}")
        r2.metric("Top Confidence",
                  f"{max(prob_dict.values())}%")
        r3.metric("HR Action", ADVICE[pred_label])

        st.markdown("#### 📊 Confidence Breakdown")
        for cls in ['High', 'Medium', 'Low']:
            st.progress(int(prob_dict[cls]),
                        text=f"{cls}: {prob_dict[cls]}%")

# -------------------------------------------------------
# PAGE 5: HR Insights
# -------------------------------------------------------
elif page == "💡 HR Insights":
    st.title("💡 HR Insights & Recommendations")
    st.markdown("---")

    order = ['High', 'Medium', 'Low']
    PALETTE = {'High': '#2ecc71', 'Medium': '#f39c12', 'Low': '#e74c3c'}

    for label in order:
        subset = df[df['performance_label'] == label]
        icon = {'High': '🟢', 'Medium': '🟡', 'Low': '🔴'}[label]
        with st.expander(f"{icon} {label} Performers — {len(subset)} employees"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Manager Rating",
                      f"{subset['manager_rating'].mean():.2f}")
            c2.metric("Avg Training Hours",
                      f"{subset['training_hours'].mean():.1f}")
            c3.metric("Avg Satisfaction",
                      f"{subset['satisfaction_score'].mean():.2f}")
            c4.metric("Avg Projects",
                      f"{subset['projects_completed'].mean():.1f}")

    st.markdown("---")
    st.subheader("🏆 Top 10 High Performers")
    top = df[df['performance_label'] == 'High'].nlargest(10, 'performance_score')[
        ['employee_id', 'department', 'performance_score',
         'manager_rating', 'projects_completed']
    ]
    st.dataframe(top, use_container_width=True)

    st.subheader("⚠️ Bottom 10 At-Risk Employees")
    bottom = df[df['performance_label'] == 'Low'].nsmallest(10, 'performance_score')[
        ['employee_id', 'department', 'performance_score',
         'satisfaction_score', 'attendance_rate']
    ]
    st.dataframe(bottom, use_container_width=True)