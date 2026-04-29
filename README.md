# 🛡️ ChurnShield — Bank Attrition Intelligence Platform
### **Demo :** https://customer-churn-prediction-2fpdi4kmo5a4djcfmx8kes.streamlit.app/

## Overview
A full-stack ML-powered Streamlit application for predicting and reducing bank customer churn.

## Features
- 📊 Executive Dashboard with live KPIs
- 🔍 Automated Data Quality Report
- 📈 Deep Exploratory Data Analysis (demographic, financial, behavioral)
- 🤖 ML Model Comparison (Random Forest, Gradient Boosting, Logistic Regression) — AUC 0.97
- 🎯 At-Risk Customer Identification with downloadable lists
- 🔄 Returning Customer Scoring & Strategy
- 📣 Full Marketing Campaign Plan with KPI Tracker
- 💡 AI Advisor powered by **Groq Free API** (llama3-8b-8192)

## Quick Start (Local)
```bash
pip install -r requirements.txt
python model_trainer.py   # Train models (runs automatically on first launch too)
streamlit run app.py
```

## Deploy to Streamlit Cloud (Free)
1. Push this folder to a GitHub repository
2. Go to share.streamlit.io
3. Connect your repo and set `app.py` as the main file
4. Add `GROQ_API_KEY` in Streamlit Cloud secrets (optional)

## AI Advisor Setup (Free)
1. Visit https://console.groq.com
2. Create free account → Generate API key
3. Paste key in the app's "API Configuration" section
4. No credit card required!

## Alternative Free LLM (HuggingFace)
Replace the Groq endpoint in app.py with:
```python
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}
```

## Dataset
`credit_card_customers.csv` — 10,127 bank customers with 22 features
- Demographics: age, gender, education, marital status, income
- Financial: credit limit, revolving balance, transaction amounts
- Behavioral: months inactive, contacts, transaction counts, Q4/Q1 changes

## Model Performance
| Model | AUC-ROC |
|-------|---------|
| Gradient Boosting | **0.9714** |
| Random Forest | 0.9646 |
| Logistic Regression | 0.9263 |

## Tech Stack
- **ML**: scikit-learn, imbalanced-learn (SMOTE), XGBoost
- **Visualization**: Plotly
- **App**: Streamlit
- **AI**: Groq API (free) / HuggingFace Inference API (free)
