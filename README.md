# 🏢 Employee Performance Predictor using Data Analytics

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![ML](https://img.shields.io/badge/ML-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> An AI-powered HR Analytics system that predicts employee performance
> using Machine Learning and presents insights via an interactive dashboard.

---

## 🎯 Project Overview

Companies struggle to identify high-performing and at-risk employees
before it's too late. This system uses **Data Analytics + Machine Learning**
to predict employee performance levels (High / Medium / Low) and provide
actionable HR recommendations.

**Built for:** Placement portfolio | Internship projects | Data Science showcase

---

## 🖥️ Dashboard Preview

| Home | Prediction | EDA |
|------|-----------|-----|
| Metrics & Overview | Live Prediction | Charts & Heatmaps |

---

## 🔬 Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11 |
| Dashboard | Streamlit |
| ML Model | Random Forest Classifier |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model Saving | Joblib |
| Dataset | Synthetic HR Data (1000 employees) |

---

## 📁 Project Structure
Employee-Performance-Predictor/
│
├── data/
│   └── employee_data.csv        # Synthetic HR dataset
│
├── src/
│   ├── preprocess.py            # Data cleaning & encoding
│   ├── eda.py                   # EDA charts generation
│   └── train_model.py           # Model training & evaluation
│
├── models/
│   ├── best_model.pkl           # Trained Random Forest
│   ├── scaler.pkl               # Feature scaler
│   ├── label_encoders.pkl       # Categorical encoders
│   └── target_encoder.pkl       # Target label encoder
│
├── outputs/
│   ├── cleaned_data.csv         # Preprocessed dataset
│   └── predictions.csv          # Sample predictions
│
├── images/                      # All EDA charts (11 plots)
│
├── app.py                       # Streamlit dashboard
├── predict.py                   # CLI prediction script
├── main.py                      # Run full pipeline
├── requirements.txt             # Dependencies
└── README.md
---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Employee-Performance-Predictor.git
cd Employee-Performance-Predictor

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run full pipeline
python main.py

# 5. Launch dashboard
streamlit run app.py
```

---

## 🚀 How to Run

### Option A — Full Pipeline (recommended first time)
```bash
python main.py
```
This runs: data generation → preprocessing → EDA → model training → prediction

### Option B — Dashboard only (after pipeline runs once)
```bash
streamlit run app.py
```

### Option C — Predict single employee
```bash
python predict.py
```

---

## 📊 Model Performance

| Model | Test Accuracy | CV Accuracy |
|-------|--------------|-------------|
| Logistic Regression | ~74% | ~73% |
| **Random Forest** | **~90%** | **~89%** |

✅ **Best Model: Random Forest Classifier**

---

## 🧠 Key Features

- 📊 **11 EDA visualizations** — distributions, heatmaps, boxplots, violin plots
- 🤖 **ML Pipeline** — encoding, feature engineering, scaling, training
- 🎯 **3 engineered features** — productivity index, engagement score, overwork flag
- 💡 **HR Recommendations** — automated advice per prediction
- 🖥️ **5-page Streamlit dashboard** — fully interactive

---

## 💼 HR Use Cases

| Prediction | HR Action |
|-----------|-----------|
| 🟢 High | Promotion / Leadership track |
| 🟡 Medium | Mentoring + Training plan |
| 🔴 Low | PIP + Counseling intervention |

---

## 🎓 What I Learned

- End-to-end ML project lifecycle
- Feature engineering for HR domain
- Model evaluation with cross-validation
- Building interactive dashboards with Streamlit
- Professional project structuring for GitHub

---

## 👤 Author

**Your Name**
- GitHub: [@your_username](https://github.com/your_username)
- LinkedIn: [your_linkedin](https://linkedin.com/in/your_linkedin)

---

⭐ **Star this repo if you found it useful!**