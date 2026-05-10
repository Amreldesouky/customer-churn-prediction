import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ── Column name mapping: your actual columns ──────────────────────────────────
# Customer_Number, Attrition_Flag, Age, Date_of_birth, Gender, Dependent_count,
# Education_Level, Marital_Status, Income_Category, Card_Category,
# Months_on_book, Total_Relationship_Count, Months_Inactive_12_mon,
# Contacts_Count_12_mon, Credit_Limit, Total_Revolving_Bal, Avg_Open_To_Buy,
# Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Trans_Ct,
# Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio
# ─────────────────────────────────────────────────────────────────────────────

def detect_and_load(path):
    """Load CSV or Excel automatically."""
    if path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return df

def load_and_preprocess(path):
    df = detect_and_load(path)

    # ── Data quality report ───────────────────────────────────────────────────
    quality_report = {
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': int(df.duplicated().sum()),
        'unknown_counts': {}
    }
    for col in df.select_dtypes(include='object').columns:
        unk = int((df[col].astype(str).str.strip().str.lower() == 'unknown').sum())
        if unk > 0:
            quality_report['unknown_counts'][col] = unk

    # ── Target ───────────────────────────────────────────────────────────────
    df['Target'] = (df['Attrition_Flag'] == 'Attrited Customer').astype(int)

    # ── Feature engineering ───────────────────────────────────────────────────
    df['Utilization_x_Revolving']  = df['Avg_Utilization_Ratio'] * df['Total_Revolving_Bal']
    df['Trans_Amt_per_Trans']       = df['Total_Trans_Amt'] / (df['Total_Trans_Ct'] + 1)
    df['Inactivity_Contact_Ratio']  = df['Months_Inactive_12_mon'] / (df['Contacts_Count_12_mon'] + 1)
    df['Credit_Used_Pct']           = df['Total_Revolving_Bal'] / (df['Credit_Limit'] + 1)

    # ── Encode categoricals ───────────────────────────────────────────────────
    cat_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str).str.strip())
        encoders[col] = le

    feature_cols = [
        'Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Utilization_x_Revolving', 'Trans_Amt_per_Trans',
        'Inactivity_Contact_Ratio', 'Credit_Used_Pct',
        'Gender_enc', 'Education_Level_enc', 'Marital_Status_enc',
        'Income_Category_enc', 'Card_Category_enc'
    ]

    X = df[feature_cols]
    y = df['Target']
    return df, X, y, encoders, feature_cols, quality_report


def train_models(path=None):
    import os
    if path is None:
        path = os.environ.get('CHURN_DATA_PATH', 'credit-card_customers.xlsx')
    df, X, y, encoders, feature_cols, quality_report = load_and_preprocess(path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_res)
    X_test_sc  = scaler.transform(X_test)

    models = {
        'Random Forest':      RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting':  GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    }

    results, trained_models = {}, {}
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_sc, y_train_res)
            y_pred = model.predict(X_test_sc)
            y_prob = model.predict_proba(X_test_sc)[:, 1]
        else:
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        auc    = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm     = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)

        results[name] = {
            'auc': auc, 'report': report, 'cm': cm.tolist(),
            'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
            'precision': prec.tolist(), 'recall': rec.tolist()
        }
        trained_models[name] = model
        print(f"{name}: AUC={auc:.4f}")

    best_model  = trained_models['Gradient Boosting']
    importances = pd.DataFrame({
        'feature':    feature_cols,
        'importance': trained_models['Random Forest'].feature_importances_
    }).sort_values('importance', ascending=False)

    # ── Score full dataset ────────────────────────────────────────────────────
    df['churn_prob'] = best_model.predict_proba(X)[:, 1]
    df['churn_pred'] = best_model.predict(X)

    # ── Returning customers: attrited with highest historical engagement ───────
    attrited = df[df['Attrition_Flag'] == 'Attrited Customer'].copy()
    returning = attrited.sort_values('Total_Trans_Ct', ascending=False).head(200)

    artifacts = {
        'best_model':          best_model,
        'scaler':              scaler,
        'encoders':            encoders,
        'feature_cols':        feature_cols,
        'results':             results,
        'importances':         importances,
        'quality_report':      quality_report,
        'df_with_scores':      df,
        'returning_customers': returning
    }
    with open('C:\\Users\\Amr\\Downloads\\New folder (2)\\model_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)

    print("✅ Artifacts saved.")
    return artifacts


if __name__ == '__main__':
    train_models()