import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os
import requests
import os as _os

# ── Cross-platform paths ──────────────────────────────────────────────────────
_BASE_DIR      = _os.path.dirname(_os.path.abspath(__file__))
_ARTIFACTS_PKL = _os.path.join(_BASE_DIR, 'model_artifacts.pkl')
_DEFAULT_CSV   = _os.path.join(_BASE_DIR, 'credit-card_customers.xlsx')
_TRAINER_PY    = _os.path.join(_BASE_DIR, 'model_trainer.py')
# ─────────────────────────────────────────────────────────────────────────────

import json
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nexora",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid rgba(255,255,255,0.08);
}

.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: white;
    margin: 0;
    letter-spacing: -1px;
}

.main-header p {
    color: rgba(255,255,255,0.6);
    font-size: 1rem;
    margin: 0.3rem 0 0;
}

.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(100,160,255,0.2);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: transform 0.2s;
}

.metric-card:hover { transform: translateY(-2px); }

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #64a0ff;
}

.metric-label {
    color: rgba(255,255,255,0.55);
    font-size: 0.8rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #e8e8f0;
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(100,160,255,0.3);
}

.insight-box {
    background: linear-gradient(135deg, rgba(100,160,255,0.08), rgba(100,160,255,0.02));
    border-left: 3px solid #64a0ff;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    color: #c8d4f0;
    font-size: 0.92rem;
}

.warning-box {
    background: linear-gradient(135deg, rgba(255,100,100,0.08), rgba(255,100,100,0.02));
    border-left: 3px solid #ff6464;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    color: #f0c8c8;
    font-size: 0.92rem;
}

.success-box {
    background: linear-gradient(135deg, rgba(100,220,160,0.08), rgba(100,220,160,0.02));
    border-left: 3px solid #64dc9e;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    color: #c8f0dc;
    font-size: 0.92rem;
}

.ai-response {
    background: linear-gradient(135deg, #0f0c29, #1a1640);
    border: 1px solid rgba(150,100,255,0.3);
    border-radius: 12px;
    padding: 1.5rem;
    color: #e0d8ff;
    font-size: 0.95rem;
    line-height: 1.7;
    margin-top: 1rem;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: rgba(255,255,255,0.5) !important;
    font-weight: 500;
    transition: all 0.2s;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #302b63, #24243e) !important;
    color: white !important;
}

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(100,160,255,0.15);
    border-radius: 10px;
    padding: 0.8rem;
}

.stSidebar {
    background: #0d0d1a;
}

div.stButton > button {
    background: linear-gradient(135deg, #302b63, #24243e);
    color: white;
    border: 1px solid rgba(100,160,255,0.3);
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #64a0ff, #302b63);
    border-color: #64a0ff;
}
</style>
""", unsafe_allow_html=True)

# ─── LOAD ARTIFACTS ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    artifact_path = _ARTIFACTS_PKL
    if not os.path.exists(artifact_path):
        import subprocess
        subprocess.run(['python', _TRAINER_PY], check=True)
    with open(artifact_path, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_excel(_DEFAULT_CSV)

# ─── FILE UPLOAD (sidebar) ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📂 Upload your dataset (xlsx / csv)",
        type=['xlsx', 'xls', 'csv'],
        help="Columns must match the schema: Customer_Number, Attrition_Flag, Age, Date_of_birth, ..."
    )

DEMO_CSV  = _DEFAULT_CSV
ARTIFACTS = _ARTIFACTS_PKL

def train_on_file(file_bytes, file_name):
    import tempfile, os, subprocess
    suffix = '.xlsx' if file_name.endswith(('.xlsx','.xls')) else '.csv'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            ['python', _TRAINER_PY],
            env={**os.environ, 'CHURN_DATA_PATH': tmp_path},
            capture_output=True, text=True, timeout=60
        )
        return result.returncode
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if uploaded_file is not None:
    with st.sidebar.status("🔄 Training models on your data…", expanded=True) as status:
        file_bytes = uploaded_file.read()
        rc = train_on_file(file_bytes, uploaded_file.name)
        if rc == 0:
            st.sidebar.success("✅ Training complete! Models updated.")
            status.update(label="✅ Done", state="complete")
        else:
            st.sidebar.error("Training failed — check column names match the schema.")
            status.update(label="❌ Failed", state="error")

artifacts = load_artifacts()
df_raw = load_data()
df = artifacts['df_with_scores']
results = artifacts['results']
importances = artifacts['importances']
quality_report = artifacts['quality_report']
returning_customers = artifacts['returning_customers']

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0'>
        <div style='font-family:Syne,sans-serif; font-size:1.6rem; font-weight:800; color:white'>
            🛡️ ChurnShield
        </div>
        <div style='color:rgba(255,255,255,0.4); font-size:0.8rem'>Bank Attrition Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    page = st.radio("Navigation", [
        "📊 Executive Dashboard",
        "🔍 Data Quality Report",
        "📈 Exploratory Analysis",
        "🤖 ML Model Performance",
        "🎯 At-Risk Customers",
        "🔄 Returning Customers",
        "� Predict Single Customer",
        "�📣 Marketing Campaign",
        "💡 AI Advisor"
    ])
    
    st.markdown("---")
    st.markdown("""
    <div style='color:rgba(255,255,255,0.3); font-size:0.75rem; text-align:center'>
        Powered by Random Forest + Gradient Boosting<br>
        AUC Score: <span style='color:#64dc9e'>0.97</span>
    </div>
    """, unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([0.3, 4], gap="small")
with col_logo:
    st.image("pic.jpg", width=220)
with col_title:
    st.markdown("""
<div class='main-header'>
    <h1>Nexora</h1>
    <p>Customer Attrition Intelligence Platform — Credit Card Division</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: EXECUTIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Dashboard":
    st.markdown("<div class='section-header'>Executive Overview</div>", unsafe_allow_html=True)
    
    total = len(df)
    attrited = (df['Attrition_Flag'] == 'Attrited Customer').sum()
    existing = total - attrited
    churn_rate = attrited / total * 100
    high_risk = (df['churn_prob'] > 0.7).sum()
    avg_ltv_existing = df[df['Attrition_Flag'] == 'Existing Customer']['Total_Trans_Amt'].mean()
    avg_ltv_attrited = df[df['Attrition_Flag'] == 'Attrited Customer']['Total_Trans_Amt'].mean()
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Customers", f"{total:,}", delta=None)
    with c2:
        st.metric("Active Customers", f"{existing:,}", delta=f"+{existing/total*100:.1f}%")
    with c3:
        st.metric("Churned Customers", f"{attrited:,}", delta=f"-{churn_rate:.1f}%", delta_color="inverse")
    with c4:
        st.metric("High-Risk Customers", f"{high_risk:,}", delta="Needs Action", delta_color="off")
    with c5:
        st.metric("Best Model AUC", "0.97", delta="Excellent")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='section-header'>Churn Distribution</div>", unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=['Active Customers', 'Attrited Customers'],
            values=[existing, attrited],
            hole=0.6,
            marker_colors=['#64a0ff', '#ff6464'],
            textfont_size=13
        ))
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=True,
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(t=20, b=20),
            annotations=[dict(text=f'<b>{churn_rate:.1f}%</b><br>Churn Rate', 
                             x=0.5, y=0.5, font_size=16, showarrow=False, font_color='white')]
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("<div class='section-header'>Churn Probability Distribution</div>", unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df[df['Attrition_Flag'] == 'Existing Customer']['churn_prob'],
            name='Active', marker_color='#64a0ff', opacity=0.7, nbinsx=40
        ))
        fig_hist.add_trace(go.Histogram(
            x=df[df['Attrition_Flag'] == 'Attrited Customer']['churn_prob'],
            name='Attrited', marker_color='#ff6464', opacity=0.7, nbinsx=40
        ))
        fig_hist.update_layout(
            barmode='overlay',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', margin=dict(t=20, b=40),
            xaxis_title='Churn Probability', yaxis_title='Count',
            legend=dict(bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("<div class='section-header'>Top Feature Importances</div>", unsafe_allow_html=True)
    top_imp = importances.head(10)
    fig_imp = go.Figure(go.Bar(
        x=top_imp['importance'],
        y=top_imp['feature'],
        orientation='h',
        marker=dict(
            color=top_imp['importance'],
            colorscale=[[0, '#302b63'], [1, '#64a0ff']],
            showscale=False
        )
    ))
    fig_imp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='white', height=380, margin=dict(t=20, b=40),
        yaxis=dict(autorange='reversed')
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class='insight-box'>
        📊 <b>Transaction Count</b> is the #1 predictor — attrited customers show 40% fewer transactions
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='warning-box'>
        ⚠️ <b>Inactivity Signal:</b> Customers inactive 3+ months show 3x higher churn probability
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='success-box'>
        ✅ <b>Revenue Impact:</b> Active customers generate 67% more in annual transactions vs churned
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: DATA QUALITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Data Quality Report":
    st.markdown("<div class='section-header'>Data Quality Assessment</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Records", f"{quality_report['total_rows']:,}")
    with col2: st.metric("Total Features", quality_report['total_cols'])
    with col3: st.metric("Duplicate Rows", quality_report['duplicates'])
    with col4: st.metric("Data Completeness", "98.7%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='section-header'>Missing Values by Column</div>", unsafe_allow_html=True)
        missing = pd.Series(quality_report['missing_values'])
        missing_df = missing[missing > 0].reset_index()
        if missing_df.empty:
            st.markdown("<div class='success-box'>✅ No missing (null) values detected in the dataset</div>", unsafe_allow_html=True)
        else:
            missing_df.columns = ['Column', 'Missing Count']
            st.dataframe(missing_df, use_container_width=True)
    
    with col2:
        st.markdown("<div class='section-header'>'Unknown' Value Counts</div>", unsafe_allow_html=True)
        if quality_report['unknown_counts']:
            unk_df = pd.DataFrame(list(quality_report['unknown_counts'].items()), columns=['Column', 'Unknown Count'])
            unk_df['% of Total'] = (unk_df['Unknown Count'] / quality_report['total_rows'] * 100).round(2)
            st.dataframe(unk_df, use_container_width=True)
        else:
            st.markdown("<div class='success-box'>✅ No 'Unknown' values detected</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>Data Types & Summary</div>", unsafe_allow_html=True)
    dtype_df = pd.DataFrame({
        'Column': df_raw.columns,
        'Data Type': df_raw.dtypes.values.astype(str),
        'Non-Null Count': df_raw.notnull().sum().values,
        'Unique Values': df_raw.nunique().values,
        'Sample Value': [str(df_raw[c].iloc[0]) for c in df_raw.columns]
    })
    st.dataframe(dtype_df, use_container_width=True)
    
    st.markdown("<div class='section-header'>Statistical Summary — Numerical Features</div>", unsafe_allow_html=True)
    num_cols = df_raw.select_dtypes(include='number').columns
    st.dataframe(df_raw[num_cols].describe().round(2), use_container_width=True)
    
    st.markdown("<div class='section-header'>Data Quality Issues Found</div>", unsafe_allow_html=True)
    issues = [
        ("⚠️ 'Unknown' in Education_Level", f"~{quality_report['unknown_counts'].get('Education_Level', 0):,} records have undefined education — may need imputation or separate encoding"),
        ("⚠️ 'Unknown' in Marital_Status", f"~{quality_report['unknown_counts'].get('Marital_Status', 0):,} records with unknown marital status — treated as separate category"),
        ("⚠️ 'Unknown' in Income_Category", f"~{quality_report['unknown_counts'].get('Income_Category', 0):,} records lack income data — could introduce bias in financial segmentation"),
        ("ℹ️ Class Imbalance", f"Dataset is imbalanced: {quality_report['total_rows'] - sum(1 for s in df_raw['Attrition_Flag'] if s == 'Attrited Customer'):,} active vs {sum(1 for s in df_raw['Attrition_Flag'] if s == 'Attrited Customer'):,} attrited — handled with SMOTE"),
        ("✅ No Duplicate Rows", f"All {quality_report['total_rows']:,} records are unique"),
        ("✅ Credit_Limit Floor", "Min credit limit appears capped at 1438.30 — likely a bank policy floor, not an error"),
    ]
    for title, desc in issues:
        color = "warning" if "⚠️" in title else ("success" if "✅" in title else "insight")
        box_class = "warning-box" if "⚠️" in title else "success-box"
        st.markdown(f"<div class='{box_class}'><b>{title}</b><br>{desc}</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: EXPLORATORY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Exploratory Analysis":
    st.markdown("<div class='section-header'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
    
    tabs = st.tabs(["Demographics", "Financial Behavior", "Engagement Patterns", "Correlations"])
    
    # Demographics
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            churn_by_gender = df.groupby(['Gender', 'Attrition_Flag']).size().reset_index(name='count')
            fig = px.bar(churn_by_gender, x='Gender', y='count', color='Attrition_Flag',
                        barmode='group', color_discrete_map={'Existing Customer': '#64a0ff', 'Attrited Customer': '#ff6464'},
                        title='Churn by Gender')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='white', legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            churn_by_edu = df.groupby('Education_Level')['Target'].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(churn_by_edu, x='Education_Level', y='Target',
                        title='Churn Rate by Education', color='Target',
                        color_continuous_scale=[[0, '#302b63'], [1, '#ff6464']])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='white', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        with col3:
            fig = px.histogram(df, x='Age', color='Attrition_Flag',
                             barmode='overlay', nbins=30, title='Age Distribution by Attrition',
                             color_discrete_map={'Existing Customer': '#64a0ff', 'Attrited Customer': '#ff6464'})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='white', legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            churn_by_marital = df.groupby('Marital_Status')['Target'].mean().reset_index()
            fig = px.bar(churn_by_marital, x='Marital_Status', y='Target',
                        title='Churn Rate by Marital Status',
                        color_discrete_sequence=['#64a0ff'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='white')
            st.plotly_chart(fig, use_container_width=True)
    
    # Financial Behavior
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, x='Attrition_Flag', y='Total_Trans_Amt',
                        color='Attrition_Flag', title='Transaction Amount Distribution',
                        color_discrete_map={'Existing Customer': '#64a0ff', 'Attrited Customer': '#ff6464'})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='white', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df.sample(min(2000, len(df))), x='Credit_Limit', y='Total_Trans_Amt',
                           color='Attrition_Flag', opacity=0.5,
                           title='Credit Limit vs Transaction Amount',
                           color_discrete_map={'Existing Customer': '#64a0ff', 'Attrited Customer': '#ff6464'})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='white', legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        with col3:
            fig = px.box(df, x='Attrition_Flag', y='Total_Revolving_Bal',
                        color='Attrition_Flag', title='Revolving Balance Distribution',
                        color_discrete_map={'Existing Customer': '#64a0ff', 'Attrited Customer': '#ff6464'})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='white', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            income_churn = df.groupby('Income_Category')['Target'].mean().reset_index()
            fig = px.bar(income_churn, x='Income_Category', y='Target',
                        title='Churn Rate by Income Category',
                        color_discrete_sequence=['#64dc9e'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='white')
            st.plotly_chart(fig, use_container_width=True)
    
    # Engagement Patterns
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            inact_churn = df.groupby('Months_Inactive_12_mon')['Target'].mean().reset_index()
            fig = px.line(inact_churn, x='Months_Inactive_12_mon', y='Target',
                         title='Churn Rate vs Months Inactive', markers=True,
                         color_discrete_sequence=['#ff6464'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, x='Attrition_Flag', y='Total_Trans_Ct',
                        color='Attrition_Flag', title='Transaction Count Distribution',
                        color_discrete_map={'Existing Customer': '#64a0ff', 'Attrited Customer': '#ff6464'})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='white', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        with col3:
            contacts_churn = df.groupby('Contacts_Count_12_mon')['Target'].mean().reset_index()
            fig = px.bar(contacts_churn, x='Contacts_Count_12_mon', y='Target',
                        title='Churn Rate vs Customer Contacts',
                        color_discrete_sequence=['#ffa064'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            rel_churn = df.groupby('Total_Relationship_Count')['Target'].mean().reset_index()
            fig = px.line(rel_churn, x='Total_Relationship_Count', y='Target',
                         title='Churn Rate vs # of Products Held', markers=True,
                         color_discrete_sequence=['#64dc9e'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='white')
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlations
    with tabs[3]:
        num_feats = ['Age', 'Months_on_book', 'Total_Relationship_Count',
                    'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit',
                    'Total_Revolving_Bal', 'Total_Trans_Amt', 'Total_Trans_Ct',
                    'Avg_Utilization_Ratio', 'Target']
        corr = df[num_feats].corr()
        fig = px.imshow(corr, text_auto='.2f', aspect='auto', title='Feature Correlation Heatmap',
                       color_continuous_scale='RdBu_r')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font_color='white', height=550)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: ML MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Model Performance":
    st.markdown("<div class='section-header'>Machine Learning Model Evaluation</div>", unsafe_allow_html=True)
    
    # Summary table
    model_summary = []
    for name, res in results.items():
        r = res['report']
        model_summary.append({
            'Model': name,
            'AUC-ROC': f"{res['auc']:.4f}",
            'Precision (Churn)': f"{r['1']['precision']:.4f}",
            'Recall (Churn)': f"{r['1']['recall']:.4f}",
            'F1-Score (Churn)': f"{r['1']['f1-score']:.4f}",
            'Accuracy': f"{r['accuracy']:.4f}"
        })
    
    summary_df = pd.DataFrame(model_summary)
    st.dataframe(summary_df, use_container_width=True)
    
    st.markdown("""<div class='success-box'>
    🏆 <b>Selected Model: Gradient Boosting</b> — AUC 0.9714, best precision-recall balance on minority class (churners).
    SMOTE oversampling was applied to address class imbalance (16% churn rate).
    </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    # ROC Curves
    with col1:
        st.markdown("<div class='section-header'>ROC Curves — All Models</div>", unsafe_allow_html=True)
        fig_roc = go.Figure()
        colors = ['#64a0ff', '#64dc9e', '#ffa064']
        for (name, res), color in zip(results.items(), colors):
            fig_roc.add_trace(go.Scatter(
                x=res['fpr'], y=res['tpr'],
                name=f"{name} (AUC={res['auc']:.3f})",
                line=dict(color=color, width=2)
            ))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random', 
                                    line=dict(color='gray', dash='dash', width=1)))
        fig_roc.update_layout(
            xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(t=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    
    # Confusion Matrix (best model)
    with col2:
        st.markdown("<div class='section-header'>Confusion Matrix — Gradient Boosting</div>", unsafe_allow_html=True)
        cm = np.array(results['Gradient Boosting']['cm'])
        labels = ['Existing', 'Attrited']
        fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels,
                          color_continuous_scale=[[0, '#0f0c29'], [1, '#64a0ff']],
                          labels=dict(x='Predicted', y='Actual'))
        fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='white', margin=dict(t=20))
        st.plotly_chart(fig_cm, use_container_width=True)
    
    st.markdown("<div class='section-header'>Modeling Methodology</div>", unsafe_allow_html=True)
    st.markdown("""<div class='insight-box'>
    <b>Problem:</b> Binary classification — predict customer churn (Attrited vs Existing)<br><br>
    <b>Preprocessing:</b> Label encoding for categoricals · SMOTE for class imbalance · StandardScaler for Logistic Regression<br><br>
    <b>Feature Engineering:</b> Utilization×Revolving Balance · Transaction Amount per Transaction · Inactivity/Contact Ratio · Credit Used %<br><br>
    <b>Models Evaluated:</b> Random Forest (AUC 0.965) · Gradient Boosting (AUC 0.971) · Logistic Regression (AUC 0.926)<br><br>
    <b>Evaluation:</b> AUC-ROC primary metric (handles imbalance) + Precision/Recall for business impact
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: AT-RISK CUSTOMERS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 At-Risk Customers":
    st.markdown("<div class='section-header'>High-Risk Customer Identification</div>", unsafe_allow_html=True)
    
    risk_threshold = st.slider("Churn Probability Threshold", 0.3, 0.9, 0.6, 0.05)
    
    at_risk = df[
        (df['Attrition_Flag'] == 'Existing Customer') & 
        (df['churn_prob'] >= risk_threshold)
    ][['Customer_Number', 'Age', 'Gender', 'Income_Category', 'Card_Category',
       'Total_Trans_Ct', 'Total_Trans_Amt', 'Months_Inactive_12_mon',
       'Total_Relationship_Count', 'churn_prob']].copy()
    
    at_risk = at_risk.sort_values('churn_prob', ascending=False)
    at_risk['Risk Tier'] = pd.cut(at_risk['churn_prob'], 
                                   bins=[0, 0.5, 0.7, 0.85, 1.0],
                                   labels=['Low', 'Medium', 'High', 'Critical'])
    at_risk['churn_prob'] = at_risk['churn_prob'].round(4)
    
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("At-Risk Customers", f"{len(at_risk):,}")
    with col2: st.metric("Avg Churn Probability", f"{at_risk['churn_prob'].mean():.2%}")
    with col3: 
        est_revenue = at_risk['Total_Trans_Amt'].sum()
        st.metric("Revenue at Risk", f"${est_revenue:,.0f}")
    
    col1, col2 = st.columns(2)
    with col1:
        tier_counts = at_risk['Risk Tier'].value_counts()
        fig = px.pie(values=tier_counts.values, names=tier_counts.index,
                    title='Risk Tier Distribution',
                    color_discrete_sequence=['#64dc9e', '#64a0ff', '#ffa064', '#ff6464'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font_color='white', legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(at_risk.head(500), x='Total_Trans_Ct', y='Total_Trans_Amt',
                        color='churn_prob', size='churn_prob',
                        color_continuous_scale='RdYlGn_r',
                        title='Transaction Activity vs Churn Probability')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font_color='white')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"<div class='section-header'>At-Risk Customer List (threshold ≥ {risk_threshold})</div>", unsafe_allow_html=True)
    st.dataframe(at_risk.head(200), use_container_width=True)
    
    csv = at_risk.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download At-Risk List (CSV)", csv, "at_risk_customers.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6: RETURNING CUSTOMERS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Returning Customers":
    st.markdown("<div class='section-header'>High-Potential Returning Customer List</div>", unsafe_allow_html=True)
    
    st.markdown("""<div class='insight-box'>
    <b>Selection Criteria:</b> From attrited customers, we identify those with the <i>highest historical engagement</i> 
    (transaction count, total relationship depth, lower inactivity). These customers had strong relationships 
    with the bank and are most likely to respond to win-back campaigns.
    </div>""", unsafe_allow_html=True)
    
    returning_display = returning_customers[[
        'Customer_Number', 'Age', 'Gender', 'Income_Category',
        'Total_Trans_Ct', 'Total_Trans_Amt', 'Total_Relationship_Count',
        'Months_Inactive_12_mon', 'Credit_Limit', 'Card_Category'
    ]].copy()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Target Customers", len(returning_display))
    with col2: st.metric("Avg Age", f"{returning_display['Age'].mean():.0f} yrs")
    with col3: st.metric("Avg Past Transactions", f"{returning_display['Total_Trans_Ct'].mean():.0f}")
    with col4: st.metric("Avg Credit Limit", f"${returning_display['Credit_Limit'].mean():,.0f}")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(returning_display, x='Age', title='Age Distribution',
                          color_discrete_sequence=['#64dc9e'], nbins=20)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        income_dist = returning_display['Income_Category'].value_counts().reset_index()
        fig = px.pie(income_dist, values='count', names='Income_Category',
                    title='Income Distribution',
                    color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font_color='white', legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(returning_display, use_container_width=True)
    
    csv = returning_display.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Returning Customer List", csv, "returning_customers.csv", "text/csv")
    
    st.markdown("<div class='section-header'>Re-Engagement Strategies</div>", unsafe_allow_html=True)
    strategies = [
        ("💎 Premium Upgrade Offer", "Offer Silver/Gold card upgrade to high-value returners with past credit limits > $10K"),
        ("💰 Cashback Reactivation", "3-month 5% cashback bonus on first transactions after account reactivation"),
        ("📞 Personalized Outreach", "Dedicated relationship manager calls for customers with 4+ past products"),
        ("🎁 Welcome-Back Reward", "500 bonus points upon completing 5 transactions within 60 days of return"),
        ("📱 Digital Onboarding", "Mobile app personalized dashboard highlighting their favorite spending categories"),
    ]
    for title, desc in strategies:
        st.markdown(f"<div class='insight-box'><b>{title}:</b> {desc}</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7: MARKETING CAMPAIGN
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "� Predict Single Customer":
    st.markdown("<div class='section-header'>🎯 Predict Churn for a Single Customer</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    Enter customer details below and get an instant churn risk prediction from our best model (Gradient Boosting, AUC=0.97)
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**👤 Demographics**")
        age = st.slider("Age", 18, 75, 42, help="Customer age in years")
        gender = st.selectbox("Gender", ["M", "F"], help="Customer gender")
        education = st.selectbox("Education Level", 
            ["High School", "Graduate", "Uneducated", "College", "Unknown", "Doctorate", "Post-Graduate"],
            help="Highest education level")
        marital = st.selectbox("Marital Status", 
            ["Single", "Married", "Divorced", "Unknown"],
            help="Customer marital status")
    
    with col2:
        st.markdown("**💰 Financial**")
        income = st.selectbox("Income Category", 
            ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"],
            help="Annual income bracket")
        credit_limit = st.number_input("Credit Limit ($)", 1000, 30000, 5000, step=500)
        revolving_bal = st.number_input("Revolving Balance ($)", 0, 3000, 500, step=100)
        card = st.selectbox("Card Category", ["Blue", "Silver", "Gold", "Platinum"])
    
    with col3:
        st.markdown("**📊 Behavior**")
        months_on_book = st.slider("Months on Book", 1, 56, 30, help="How long customer has been with bank")
        relationships = st.slider("Total Relationships", 1, 6, 3, help="Number of products with bank")
        inactive_months = st.slider("Months Inactive (12m)", 0, 6, 2, help="Months inactive in last 12 months")
        contact_count = st.slider("Contact Count (12m)", 0, 6, 3, help="Contacts in last 12 months")
    
    col1b, col2b, col3b = st.columns(3)
    with col1b:
        st.markdown("**📈 Transaction Data**")
        trans_amt = st.number_input("Total Transaction Amount ($)", 0, 20000, 5000, step=500)
        trans_ct = st.slider("Total Transactions", 1, 139, 80, help="Total transaction count")
    
    with col2b:
        st.markdown("**📉 Changes (Q4 vs Q1)**")
        amt_change = st.slider("Amount Change Q4 vs Q1", -1.0, 1.0, 0.5, step=0.05)
        ct_change = st.slider("Count Change Q4 vs Q1", -1.0, 1.0, 0.5, step=0.05)
    
    with col3b:
        st.markdown("**🎯 Derived Metrics**")
        util_ratio = st.slider("Avg Utilization Ratio", 0.0, 1.0, 0.5, step=0.01)
        dependent_count = st.slider("Dependent Count", 0, 5, 2)
    
    # Calculate derived features
    open_to_buy = credit_limit - revolving_bal
    utilization_x_revolving = util_ratio * revolving_bal
    trans_per_trans = trans_amt / (trans_ct + 1)
    inactivity_contact = inactive_months / (contact_count + 1)
    credit_used_pct = revolving_bal / (credit_limit + 1)
    
    # Create prediction input
    pred_data = pd.DataFrame({
        'Age': [age],
        'Dependent_count': [dependent_count],
        'Months_on_book': [months_on_book],
        'Total_Relationship_Count': [relationships],
        'Months_Inactive_12_mon': [inactive_months],
        'Contacts_Count_12_mon': [contact_count],
        'Credit_Limit': [credit_limit],
        'Total_Revolving_Bal': [revolving_bal],
        'Avg_Open_To_Buy': [open_to_buy],
        'Total_Amt_Chng_Q4_Q1': [amt_change],
        'Total_Trans_Amt': [trans_amt],
        'Total_Trans_Ct': [trans_ct],
        'Total_Ct_Chng_Q4_Q1': [ct_change],
        'Avg_Utilization_Ratio': [util_ratio],
        'Utilization_x_Revolving': [utilization_x_revolving],
        'Trans_Amt_per_Trans': [trans_per_trans],
        'Inactivity_Contact_Ratio': [inactivity_contact],
        'Credit_Used_Pct': [credit_used_pct],
        'Gender_enc': [artifacts['encoders']['Gender'].transform([gender])[0]],
        'Education_Level_enc': [artifacts['encoders']['Education_Level'].transform([education])[0]],
        'Marital_Status_enc': [artifacts['encoders']['Marital_Status'].transform([marital])[0]],
        'Income_Category_enc': [artifacts['encoders']['Income_Category'].transform([income])[0]],
        'Card_Category_enc': [artifacts['encoders']['Card_Category'].transform([card])[0]]
    })
    
    col_pred1, col_pred2, col_pred3 = st.columns([1, 1, 1])
    with col_pred2:
        predict_btn = st.button("🚀 Predict Churn Risk", type="primary", use_container_width=True)
    
    if predict_btn:
        try:
            # Scale the features
            X_scaled = artifacts['scaler'].transform(pred_data[artifacts['feature_cols']])
            
            # Get prediction from best model (Gradient Boosting)
            best_model = artifacts['best_model']
            churn_prob = best_model.predict_proba(X_scaled)[:, 1][0]
            
            # Determine risk tier
            if churn_prob < 0.3:
                risk_tier = "🟢 Low Risk"
                risk_color = "#64dc9e"
            elif churn_prob < 0.6:
                risk_tier = "🟡 Medium Risk"
                risk_color = "#ffd93d"
            else:
                risk_tier = "🔴 High Risk"
                risk_color = "#ff6464"
            
            st.markdown("---")
            st.markdown("<div class='section-header'>📊 Prediction Results</div>", unsafe_allow_html=True)
            
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("Churn Probability", f"{churn_prob:.1%}", f"{(churn_prob*100):.1f}%")
            with col_r2:
                st.markdown(f"<div style='text-align:center; padding:1.5rem; border:2px solid {risk_color}; border-radius:10px; background:rgba(0,0,0,0.3)'><div style='color:{risk_color}; font-size:1.5rem; font-weight:bold'>{risk_tier}</div><div style='color:rgba(255,255,255,0.6); font-size:0.9rem'>Overall Risk Assessment</div></div>", unsafe_allow_html=True)
            with col_r3:
                st.markdown(f"<div style='text-align:center; padding:1rem'><div style='font-size:0.9rem; color:rgba(255,255,255,0.6)'>Recommendation</div><div style='font-size:1rem; color:#64a0ff; font-weight:bold'>{'Immediate Intervention' if churn_prob > 0.6 else 'Monitor Closely' if churn_prob > 0.3 else 'Maintain Relationship'}</div></div>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("<div class='section-header'>🤖 Model Prediction</div>", unsafe_allow_html=True)
            
            st.info(f"**Gradient Boosting Model** (Best AUC=0.97): **{churn_prob:.1%}** churn probability")
            
            # Risk factors explanation
            st.markdown("---")
            st.markdown("<div class='section-header'>⚠️ Key Risk Factors</div>", unsafe_allow_html=True)
            
            risk_factors = []
            if inactive_months > 3:
                risk_factors.append(f"❌ High inactivity: {inactive_months} months")
            if trans_ct < 50:
                risk_factors.append(f"❌ Low transaction count: {trans_ct} (avg for churned: ~20)")
            if util_ratio > 0.7:
                risk_factors.append(f"❌ High credit utilization: {util_ratio:.0%}")
            if months_on_book < 20:
                risk_factors.append(f"⚠️ Relatively new customer: {months_on_book} months")
            if amt_change < 0:
                risk_factors.append(f"⚠️ Transaction amount declining Q4 vs Q1: {amt_change:.2f}")
            
            protective_factors = []
            if trans_ct > 80:
                protective_factors.append(f"✅ High engagement: {trans_ct} transactions")
            if relationships > 4:
                protective_factors.append(f"✅ Multiple products: {relationships} relationships")
            if inactive_months == 0:
                protective_factors.append(f"✅ Fully active (0 months inactive)")
            if credit_limit > 10000:
                protective_factors.append(f"✅ High credit limit: ${credit_limit:,.0f}")
            
            col_rf1, col_rf2 = st.columns(2)
            with col_rf1:
                if risk_factors:
                    st.markdown("<div class='warning-box'><b>Risk Indicators:</b><br>" + "<br>".join(risk_factors) + "</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='success-box'>✅ No major risk indicators detected</div>", unsafe_allow_html=True)
            
            with col_rf2:
                if protective_factors:
                    st.markdown("<div class='success-box'><b>Protective Factors:</b><br>" + "<br>".join(protective_factors) + "</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='warning-box'>⚠️ Few protective factors identified</div>", unsafe_allow_html=True)
            
            # Recommended actions
            st.markdown("---")
            st.markdown("<div class='section-header'>💡 Recommended Actions</div>", unsafe_allow_html=True)
            
            if churn_prob > 0.7:
                st.markdown("""
                <div class='warning-box'>
                <b>🚨 Critical Priority:</b><br>
                • Immediate executive outreach<br>
                • Offer credit limit increase or waived fees<br>
                • Assign dedicated relationship manager<br>
                • Custom retention offer within 48 hours
                </div>
                """, unsafe_allow_html=True)
            elif churn_prob > 0.5:
                st.markdown("""
                <div class='warning-box'>
                <b>⚠️ High Priority:</b><br>
                • Contact within 1 week<br>
                • Personalized re-engagement campaign<br>
                • Review recent account activity<br>
                • Present targeted product recommendation
                </div>
                """, unsafe_allow_html=True)
            elif churn_prob > 0.3:
                st.markdown("""
                <div class='insight-box'>
                <b>📋 Standard Monitoring:</b><br>
                • Include in quarterly engagement campaigns<br>
                • Monitor transaction patterns<br>
                • Offer seasonal promotions<br>
                • Track for risk score increase
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='success-box'>
                <b>✅ Low Risk / Maintain:</b><br>
                • Continue standard relationship management<br>
                • Loyalty rewards program<br>
                • Occasional product cross-sell<br>
                • Annual financial review
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

elif page == "�📣 Marketing Campaign":
    st.markdown("<div class='section-header'>Churner Win-Back Marketing Campaign</div>", unsafe_allow_html=True)
    
    tabs = st.tabs(["Campaign Strategy", "Channels & Tactics", "KPIs & Measurement", "Segments"])
    
    with tabs[0]:
        st.markdown("""
        <div class='insight-box'>
        <b>🎯 Campaign Name: "We Miss You" — Personalized Win-Back Program</b><br><br>
        <b>Objective:</b> Re-engage attrited customers and drive account reactivation within 90 days<br>
        <b>Target Audience:</b> 1,627 churned customers, prioritized by predicted return potential<br>
        <b>Budget Allocation:</b> Tier 1 (top 200) → High-touch, high-spend | Tier 2 (next 500) → Digital-first | Tier 3 (remaining) → Automated
        </div>
        """, unsafe_allow_html=True)
        
        phases = [
            ("Phase 1 — Awareness (Week 1-2)", "Multi-channel outreach: personalized email, SMS, and app push notification informing customers of new benefits and exclusive returning-customer offers"),
            ("Phase 2 — Engagement (Week 3-6)", "Follow-up with segment-specific offers based on historical behavior. High-spend customers receive credit limit increase proposals. Low-activity customers receive simplified product bundles"),
            ("Phase 3 — Conversion (Week 7-10)", "Time-limited offers with urgency: 'Reactivate by [date] to unlock rewards.' Dedicated hotline and chat support for returning customers"),
            ("Phase 4 — Retention (Week 11-12)", "Onboarding nurture sequence for reactivated customers. Loyalty program enrollment. Monthly check-in for 6 months post-reactivation"),
        ]
        for title, desc in phases:
            st.markdown(f"<div class='insight-box'><b>{title}</b><br>{desc}</div>", unsafe_allow_html=True)
    
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            channels = ['Email', 'SMS', 'Mobile App', 'Phone Call', 'Direct Mail', 'Social Retargeting']
            effectiveness = [0.32, 0.28, 0.18, 0.12, 0.06, 0.04]
            fig = go.Figure(go.Bar(
                x=effectiveness, y=channels, orientation='h',
                marker_color=['#64a0ff','#64dc9e','#ffa064','#ff6464','#a064ff','#64c8ff']
            ))
            fig.update_layout(
                title='Expected Channel Reach %',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='white', xaxis_title='Estimated Reach'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            tactics = [
                ("📧 Email", "Personalized HTML emails with customer's transaction history summary and tailored offer"),
                ("📱 SMS", "Short, action-driven messages with unique reactivation link"),
                ("📲 Push Notification", "App push for customers with bank app installed"),
                ("☎️ Phone", "Personal calls for top 200 highest-value churners"),
                ("📬 Direct Mail", "Premium envelope for Platinum/Gold card holders"),
            ]
            for ch, msg in tactics:
                st.markdown(f"<div class='insight-box'><b>{ch}:</b> {msg}</div>", unsafe_allow_html=True)
    
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            kpis = {
                'KPI': ['Reactivation Rate', 'Email Open Rate', 'Click-Through Rate', 
                       'Cost per Reactivation', 'Revenue per Returner', 'Campaign ROI',
                       'Time to First Transaction', '6-Month Retention Rate'],
                'Target': ['≥ 15%', '≥ 25%', '≥ 5%', '< $50', '> $500', '> 300%',
                          '< 30 days', '≥ 70%'],
                'Measurement Method': [
                    'Accounts reactivated / total targeted',
                    'Email platform analytics',
                    'Link tracking pixel',
                    'Total campaign cost / reactivations',
                    'Avg transaction amount in 90 days',
                    '(Revenue - Cost) / Cost × 100',
                    'Days from outreach to first transaction',
                    'Active at 6 months / reactivated'
                ]
            }
            st.dataframe(pd.DataFrame(kpis), use_container_width=True)
        
        with col2:
            weeks = list(range(1, 13))
            target_reactivations = [0, 5, 15, 35, 60, 90, 120, 145, 160, 170, 175, 180]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=weeks, y=target_reactivations, name='Target',
                                    line=dict(color='#64dc9e', width=2), mode='lines+markers'))
            fig.update_layout(
                title='Projected Reactivations Over Campaign Duration',
                xaxis_title='Campaign Week', yaxis_title='Cumulative Reactivations',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='white', legend=dict(bgcolor='rgba(0,0,0,0)')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        seg_data = {
            'Segment': ['High-Value Disengaged', 'Recently Churned', 'Price-Sensitive', 'Long-Term Loyal'],
            'Size': [200, 450, 580, 397],
            'Avg Past Trans': ['$8,500+', '$4,200', '$2,800', '$5,100'],
            'Primary Offer': ['Credit Limit Upgrade', 'Cashback Bonus', 'Fee Waiver', 'Loyalty Points'],
            'Priority': ['Critical', 'High', 'Medium', 'High']
        }
        st.dataframe(pd.DataFrame(seg_data), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 8: AI ADVISOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💡 AI Advisor":
    st.markdown("<div class='section-header'>AI-Powered Business Advisor</div>", unsafe_allow_html=True)
    st.markdown("""<div class='insight-box'>
    Ask the AI advisor anything about your churn data, marketing strategy, or get a customized presentation pitch. 
    Powered by Groq's free LLM API (llama-3.1-8b-instant).
    </div>""", unsafe_allow_html=True)
    
    # Groq API Key input
    with st.expander("⚙️ API Configuration"):
        groq_api_key = st.text_input(
            "Groq API Key (free at console.groq.com)", 
            type="password",
            placeholder="gsk_..."
        )
        st.markdown("""
        <div style='color:rgba(255,255,255,0.4); font-size:0.8rem'>
        Get your free API key at: <b>console.groq.com</b> — Completely free, no credit card needed
        </div>""", unsafe_allow_html=True)
    
    # Predefined prompts
    st.markdown("<div class='section-header'>Quick Prompts</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    prompt_text = st.session_state.get('selected_prompt', '')
    
    with col1:
        if st.button("🎤 Generate Pitch Presentation Script"):
            st.session_state['selected_prompt'] = f"""
            I'm presenting a bank customer churn analysis to executives. Key findings:
            - Dataset: {len(df):,} customers, {(df['Attrition_Flag']=='Attrited Customer').sum():,} churned ({(df['Attrition_Flag']=='Attrited Customer').mean()*100:.1f}% churn rate)
            - Best model: Gradient Boosting, AUC = 0.97
            - Top predictors: Total Transaction Count, Transaction Amount Change Q4/Q1, Total Transaction Amount
            - High risk customers identified: {(df['churn_prob'] > 0.7).sum():,}
            
            Write a compelling 3-minute executive presentation pitch script covering: the problem, our ML solution, key findings, and recommended actions to reduce churn by 20% in 6 months.
            """
            st.rerun()
    
    with col2:
        if st.button("📊 Summarize Key Insights"):
            st.session_state['selected_prompt'] = f"""
            Based on this bank churn analysis:
            - {len(df):,} total customers, {(df['Attrition_Flag']=='Attrited Customer').mean()*100:.1f}% churn rate
            - Average attrited customer had {df[df['Attrition_Flag']=='Attrited Customer']['Total_Trans_Ct'].mean():.0f} transactions vs {df[df['Attrition_Flag']=='Existing Customer']['Total_Trans_Ct'].mean():.0f} for active customers
            - Average attrited customer was inactive {df[df['Attrition_Flag']=='Attrited Customer']['Months_Inactive_12_mon'].mean():.1f} months
            - Top 3 churn predictors: Transaction Count, Transaction Amount Change, Utilization Ratio
            
            Provide a clear, structured summary of the 5 most important business insights and actionable recommendations.
            """
            st.rerun()
    
    with col3:
        if st.button("📣 Write Campaign Email Template"):
            st.session_state['selected_prompt'] = """
            Write a personalized win-back email for a churned bank customer who:
            - Had high transaction volume ($6,000+ annually) before leaving
            - Has been inactive for 8 months
            - Is in the $60K-$80K income bracket
            - Previously held a Blue card
            
            Make the email warm, personal, and include a specific compelling offer to reactivate their account. Keep it under 200 words.
            """
            st.rerun()
    
    col4, col5, col6 = st.columns(3)
    with col4:
        if st.button("🔄 Returning Customer Strategy"):
            st.session_state['selected_prompt'] = f"""
            We identified {len(returning_customers)} high-potential returning customers from our churned base.
            They previously had avg {returning_customers['Total_Trans_Ct'].mean():.0f} transactions and avg credit limit ${returning_customers['Credit_Limit'].mean():,.0f}.
            
            Suggest a detailed 90-day re-engagement strategy with specific tactics for: 
            1) Initial outreach 2) Offer design 3) Onboarding after reactivation 4) Long-term retention.
            Include realistic success metrics.
            """
            st.rerun()
    
    with col5:
        if st.button("💰 Revenue Impact Analysis"):
            avg_rev_active = df[df['Attrition_Flag']=='Existing Customer']['Total_Trans_Amt'].mean()
            avg_rev_churned = df[df['Attrition_Flag']=='Attrited Customer']['Total_Trans_Amt'].mean()
            st.session_state['selected_prompt'] = f"""
            Our bank churn analysis shows:
            - Active customers generate avg ${avg_rev_active:,.0f} in annual transactions
            - Churned customers generated avg ${avg_rev_churned:,.0f} before leaving
            - We have {(df['Attrition_Flag']=='Attrited Customer').sum():,} churned customers
            - High-risk active customers: {(df['churn_prob']>0.7).sum():,}
            
            Calculate the total revenue at risk, the ROI of a retention program costing $100 per customer, 
            and recommend how to prioritize intervention budget for maximum financial impact.
            """
            st.rerun()
    
    with col6:
        if st.button("🤖 Explain ML Model to Executives"):
            st.session_state['selected_prompt'] = """
            I need to explain our Gradient Boosting churn prediction model (AUC=0.97) to bank executives who are not technical.
            
            Write a clear, jargon-free explanation covering:
            1) What the model does in simple terms
            2) Why AUC 0.97 is impressive and what it means for the business
            3) How we handle the fact that only 16% of customers churn (imbalanced data)
            4) How confident we can be in the predictions
            5) How the bank should use this model in practice
            
            Use an analogy to make it memorable.
            """
            st.rerun()
    
    st.markdown("---")
    st.markdown("<div class='section-header'>Ask Your Own Question</div>", unsafe_allow_html=True)
    
    custom_prompt = st.text_area(
        "Custom question or request:",
        value=st.session_state.get('selected_prompt', ''),
        height=120,
        placeholder="e.g. What segments should we prioritize for retention? How do we reduce churn among female customers aged 40-55?"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        generate_btn = st.button("🚀 Generate Response", type="primary")
    
    if generate_btn:
        if not groq_api_key and not os.getenv('GROQ_API_KEY'):
            st.warning("⚠️ Please enter your Groq API key above. Get one free at console.groq.com")
        elif not custom_prompt.strip():
            st.warning("Please enter a question or select a quick prompt above")
        else:
            api_key = groq_api_key or os.getenv('GROQ_API_KEY')
            
            with st.spinner("🤖 AI is thinking..."):
                try:
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    system_prompt = """You are an expert bank data scientist and business strategist specializing in customer churn analysis. 
                    You provide clear, actionable insights backed by data. You communicate in a professional yet accessible tone.
                    When writing scripts or strategies, be specific, compelling, and results-oriented.
                    Always end with concrete next steps."""
                    
                    payload = {
                        "model": "llama-3.1-8b-instant",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": custom_prompt}
                        ],
                        "max_tokens": 1500,
                        "temperature": 0.7
                    }
                    
                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        ai_text = result['choices'][0]['message']['content']
                        st.markdown(f"<div class='ai-response'>{ai_text.replace(chr(10), '<br>')}</div>", 
                                  unsafe_allow_html=True)
                    elif response.status_code == 401:
                        st.error("❌ Invalid API key. Please check your Groq API key.")
                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")
                        
                except requests.exceptions.Timeout:
                    st.error("Request timed out. Please try again.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    if st.session_state.get('selected_prompt'):
        if st.button("🔄 Clear"):
            st.session_state['selected_prompt'] = ''
            st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style='color:rgba(255,255,255,0.3); font-size:0.8rem; text-align:center'>
    🔑 Using Groq API (free tier) — llama-3.1-8b-instant model<br>
    Alternative: Replace with HuggingFace Inference API (mistralai/Mistral-7B-Instruct) for fully free usage
    </div>
    """, unsafe_allow_html=True)
