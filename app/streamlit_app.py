import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

"""
ASD Early Detection System — Professional Web Application
Features: TabM-PLE Neural Network (70 Features), SHAP Explainability,
M-CHAT-R Clinical Screening, Multi-role Authentication (Patient & Admin),
and Admin Management Portal.
"""

import os, json, warnings, gc
import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use('Agg')
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from tabm_ple_model import TabMPLE
    MODEL_CLASS = TabMPLE
    ENHANCED = True
except Exception:
    ENHANCED = False

warnings.filterwarnings('ignore')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
def rpath(*p): return os.path.join(ROOT, *p)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ASD Detection — Clinical Intelligence Portal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Modern CSS Design System ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Background & layout */
.stApp {
    background-color: #070E1A !important;
    color: #F1F5F9 !important;
}
.main {
    background-color: #070E1A !important;
}
.block-container {
    max-width: 1400px;
    padding: 1.5rem 2.5rem 4rem;
}

/* Streamlit Button Styling Fix (High Contrast & Custom Dark Buttons) */
div.stButton > button {
    background-color: #1E293B !important;
    color: #F8FAFC !important;
    border: 1px solid #475569 !important;
    border-radius: 12px !important;
    padding: 0.55rem 1.1rem !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
    transition: all 0.2s ease-in-out !important;
}
div.stButton > button p, div.stButton > button span, div.stButton > button div {
    color: #F8FAFC !important;
}
div.stButton > button:hover {
    background-color: #334155 !important;
    color: #FFFFFF !important;
    border-color: #60A5FA !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(59, 130, 246, 0.25) !important;
}
div.stButton > button[kind="primary"], 
div.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid #3B82F6 !important;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4) !important;
}
div.stButton > button[kind="primary"] p, 
div.stButton > button[kind="primary"] span, 
div.stButton > button[data-testid="stBaseButton-primary"] p,
div.stButton > button[data-testid="stBaseButton-primary"] span {
    color: #FFFFFF !important;
}
div.stButton > button[kind="primary"]:hover, 
div.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
    color: #FFFFFF !important;
    border-color: #60A5FA !important;
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5) !important;
}

/* Streamlit Tabs Styling Fix (High Contrast Active/Inactive Tabs) */
button[data-baseweb="tab"] {
    background-color: transparent !important;
    color: #94A3B8 !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    padding: 0.75rem 1.25rem !important;
    border-radius: 10px 10px 0 0 !important;
    border: 0 !important;
}
button[data-baseweb="tab"] p, button[data-baseweb="tab"] div, button[data-baseweb="tab"] span {
    color: #94A3B8 !important;
}
button[data-baseweb="tab"]:hover {
    color: #F8FAFC !important;
    background-color: rgba(255, 255, 255, 0.05) !important;
}
button[data-baseweb="tab"]:hover p, button[data-baseweb="tab"]:hover div, button[data-baseweb="tab"]:hover span {
    color: #F8FAFC !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #60A5FA !important;
    border-bottom: 3px solid #3B82F6 !important;
    background-color: rgba(59, 130, 246, 0.12) !important;
}
button[data-baseweb="tab"][aria-selected="true"] p, 
button[data-baseweb="tab"][aria-selected="true"] div, 
button[data-baseweb="tab"][aria-selected="true"] span {
    color: #60A5FA !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: #3B82F6 !important;
}
div[data-baseweb="tab-border"] {
    background-color: #1E293B !important;
}

/* Form Inputs, Textboxes, and Widgets */
label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
    color: #CBD5E1 !important;
    font-weight: 600 !important;
}

/* Top Header Bar */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.5rem;
    background: linear-gradient(135deg, #0F172A, #1E293B);
    border: 1px solid #334155;
    border-radius: 16px;
    margin-bottom: 1.8rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
.brand-group {
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.brand-icon {
    display: grid;
    place-items: center;
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    color: #FFFFFF;
    font-size: 1.4rem;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
}
.brand-title {
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #FFFFFF, #94A3B8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.brand-sub {
    font-size: 0.8rem;
    color: #64748B;
    margin: 0;
}
.user-badge {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: #0F172A;
    border: 1px solid #334155;
    border-radius: 999px;
    padding: 0.4rem 1rem;
}
.user-name {
    font-size: 0.9rem;
    font-weight: 600;
    color: #E2E8F0;
}
.role-pill-patient {
    background: rgba(59, 130, 246, 0.15);
    color: #60A5FA;
    border: 1px solid rgba(59, 130, 246, 0.3);
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.15rem 0.65rem;
    border-radius: 999px;
    text-transform: uppercase;
}
.role-pill-admin {
    background: rgba(236, 72, 153, 0.15);
    color: #F472B6;
    border: 1px solid rgba(236, 72, 153, 0.3);
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.15rem 0.65rem;
    border-radius: 999px;
    text-transform: uppercase;
}

/* Authentication Page Styles */
.auth-hero-container {
    background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%);
    border: 1px solid #312E81;
    border-radius: 24px;
    padding: 3rem 2.5rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 20px 50px rgba(0,0,0,0.4);
}
.auth-kicker {
    color: #818CF8;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-size: 0.8rem;
    font-weight: 800;
    margin-bottom: 0.75rem;
}
.auth-headline {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.15;
    background: linear-gradient(135deg, #FFFFFF 0%, #C7D2FE 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1.2rem;
}
.auth-desc {
    color: #94A3B8;
    font-size: 1.05rem;
    line-height: 1.75;
    margin-bottom: 2rem;
}
.feature-pill-group {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
}
.feature-pill {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #E2E8F0;
    padding: 0.4rem 0.9rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
}

.auth-form-card {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 24px;
    padding: 2.5rem;
    box-shadow: 0 20px 50px rgba(0,0,0,0.5);
}
.auth-form-title {
    font-size: 1.75rem;
    font-weight: 800;
    color: #F8FAFC;
    margin-bottom: 0.3rem;
}
.auth-form-sub {
    font-size: 0.9rem;
    color: #64748B;
    margin-bottom: 1.8rem;
}

/* Dashboard Hero */
.hero {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 12px 35px rgba(0,0,0,0.3);
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60A5FA, #A78BFA, #F472B6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
}
.hero-sub {
    color: #94A3B8;
    font-size: 0.98rem;
    margin: 0;
}

/* Admin Banner & Cards */
.admin-banner {
    background: linear-gradient(135deg, #1E1B4B, #0F172A);
    border: 1px solid #4338CA;
    border-radius: 18px;
    padding: 1.5rem 2rem;
    color: white;
    margin-bottom: 1.8rem;
    box-shadow: 0 10px 30px rgba(67, 56, 202, 0.2);
}
.kpi-card {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 1.3rem;
    text-align: center;
    transition: all 0.25s ease;
}
.kpi-card:hover {
    border-color: #3B82F6;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
}
.kpi-val {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
}
.kpi-label {
    color: #64748B;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 0.3rem 0 0 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Risk result cards */
.risk-high {
    background: linear-gradient(135deg, #2D1515, #1E0A0A);
    border: 2px solid #EF4444;
    border-radius: 20px;
    padding: 1.8rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(239, 68, 68, 0.2);
}
.risk-med {
    background: linear-gradient(135deg, #2D2210, #1E1608);
    border: 2px solid #F59E0B;
    border-radius: 20px;
    padding: 1.8rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(245, 158, 11, 0.2);
}
.risk-low {
    background: linear-gradient(135deg, #0D2B1E, #071D14);
    border: 2px solid #10B981;
    border-radius: 20px;
    padding: 1.8rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
}
.risk-label {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
}

.doc-card {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 14px;
    padding: 1.2rem;
    margin-top: 0.8rem;
    transition: all 0.2s;
}
.doc-card:hover {
    border-color: #3B82F6;
}
.q-box {
    background: #0F172A;
    border-left: 4px solid #3B82F6;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.8rem;
}
.q-box b, .q-box, .q-box p, .q-box div {
    color: #F8FAFC !important;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #334155, transparent);
    margin: 2rem 0;
}
.feature-group {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.group-title {
    color: #60A5FA;
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.9rem;
}

/* Patient Inspector Card */
.patient-inspector {
    background: #0F172A;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 1.8rem;
    margin-top: 1.2rem;
}

/* Hide Streamlit element defaults */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Load resources ────────────────────────────────────────────
@st.cache_resource
def load_enhanced_model():
    with open(rpath("results", "model_config_enhanced.json")) as f:
        cfg = json.load(f)
    m = TabMPLE.build(n_num=cfg['n_num'], n_bin=cfg['n_bin'],
                    n_bins=cfg['n_bins'], hidden_dim=cfg['hidden_dim'],
                    n_layers=cfg['n_layers'], k=cfg['k'], dropout=0.0)
    m.load_state_dict(torch.load(rpath("models", "tabm_ple_best.pth"), map_location="cpu"))
    m.eval()
    return m, cfg

@st.cache_resource
def load_prep():
    return joblib.load(rpath("models", "enhanced_preprocessor.pkl"))

@st.cache_data
def load_feat_meta():
    try:
        with open(rpath("dataset", "feature_metadata.json")) as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_data
def load_metrics():
    try:
        with open(rpath("results", "metrics_enhanced.json")) as f:
            return json.load(f)
    except Exception:
        try:
            with open(rpath("results", "metrics.json")) as f:
                return json.load(f)
        except Exception:
            return {}

model_ok = False
try:
    model, cfg = load_enhanced_model()
    prep = load_prep()
    scaler = prep['scaler']
    feat_names = prep['features']
    num_idx = prep['num_idx']
    bin_idx = prep['bin_idx']
    model_ok = True
except Exception as e:
    feat_names = []; num_idx = []; bin_idx = []
    st.warning(f"Enhanced PyTorch model not loaded: {e}")

feat_meta = load_feat_meta()
metrics = load_metrics()
SCREENING_THRESHOLD = float(metrics.get("Classification_Threshold", 0.35))

# ── Feature groups for 70 clinical inputs ─────────────────────
FEATURE_GROUPS = {
    "👤 Demographics": {
        "SC_AGE_YEARS": ("Age (Years)", "slider", 0, 100, 8),
        "SC_SEX":       ("Sex", "radio", {1: "Male", 0: "Female"}, 1),
        "SC_RACE_R":    ("Race", "select", {1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "Other"}, 1),
        "SC_HISPANIC_R":("Hispanic?", "radio", {1: "Yes", 0: "No"}, 0),
        "A1_GRADE":     ("Parent Education", "select", {1: "< High School", 2: "High School", 3: "Some College", 4: "Bachelor+"}, 3),
    },
    "🏥 Medical Conditions": {
        "ALLERGIES":  ("Allergies Diagnosed?", "radio", {1: "Yes", 0: "No"}, 0),
        "DIABETES":   ("Diabetes?", "radio", {1: "Yes", 0: "No"}, 0),
        "HEART":      ("Heart Condition?", "radio", {1: "Yes", 0: "No"}, 0),
        "DOWNSYN":    ("Down Syndrome?", "radio", {1: "Yes", 0: "No"}, 0),
        "CYSTFIB":    ("Cystic Fibrosis?", "radio", {1: "Yes", 0: "No"}, 0),
        "BLINDNESS":  ("Blindness?", "radio", {1: "Yes", 0: "No"}, 0),
        "K2Q40A":     ("Epilepsy/Seizures?", "radio", {1: "Yes", 0: "No"}, 0),
        "HEADACHE":   ("Chronic Headaches?", "radio", {1: "Yes", 0: "No"}, 0),
        "AUTOIMMUNE": ("Autoimmune Condition?", "radio", {1: "Yes", 0: "No"}, 0),
        "BREATHING":  ("Breathing Problem?", "radio", {1: "Yes", 0: "No"}, 0),
        "STOMACH":    ("Stomach/Digestive?", "radio", {1: "Yes", 0: "No"}, 0),
        "BLOOD":      ("Blood Disorder?", "radio", {1: "Yes", 0: "No"}, 0),
        "FASD":       ("Fetal Alcohol Spectrum?", "radio", {1: "Yes", 0: "No"}, 0),
        "OVERWEIGHT": ("Overweight?", "radio", {1: "Yes", 0: "No"}, 0),
        "K2Q42A":     ("Brain Injury?", "radio", {1: "Yes", 0: "No"}, 0),
    },
    "🧩 Comorbidities": {
        "K2Q31A": ("ADHD Diagnosed?", "radio", {1: "Yes", 0: "No"}, 0),
        "K2Q32A": ("Depression?", "radio", {1: "Yes", 0: "No"}, 0),
        "K2Q33A": ("Anxiety?", "radio", {1: "Yes", 0: "No"}, 0),
        "K2Q36A": ("Intellectual Disability?", "radio", {1: "Yes", 0: "No"}, 0),
        "K2Q37A": ("Tourette Syndrome?", "radio", {1: "Yes", 0: "No"}, 0),
        "K2Q60A": ("Behavioral/Conduct Problems?", "radio", {1: "Yes", 0: "No"}, 0),
    },
    "🗣️ Language Development": {
        "ONEWORD":    ("Said first word by 12 months?", "radio", {1: "Yes", 0: "No"}, 1),
        "TWOWORDS":   ("Said two-word phrases by 16 months?", "radio", {1: "Yes", 0: "No"}, 1),
        "THREEWORDS": ("Said 3-word sentences by 24 months?", "radio", {1: "Yes", 0: "No"}, 1),
        "ASKQUESTION":("Asks simple questions?", "radio", {1: "Yes", 0: "No"}, 1),
        "ASKQUESTION2":("Asks complex questions?", "radio", {1: "Yes", 0: "No"}, 1),
        "TELLSTORY":  ("Tells stories or describes events?", "radio", {1: "Yes", 0: "No"}, 1),
        "UNDERSTAND": ("Understands what you say?", "radio", {1: "Yes", 0: "No"}, 1),
        "DIRECTIONS": ("Follows 2-step directions?", "radio", {1: "Yes", 0: "No"}, 1),
        "POINT":      ("Points to show you things?", "radio", {1: "Yes", 0: "No"}, 1),
    },
    "🎯 Motor & Cognitive": {
        "BOUNCEABALL": ("Can bounce/catch a ball?", "radio", {1: "Yes", 0: "No"}, 1),
        "DRAWACIRCLE": ("Can draw a circle?", "radio", {1: "Yes", 0: "No"}, 1),
        "DRAWAPERSON": ("Can draw a person?", "radio", {1: "Yes", 0: "No"}, 1),
        "RECOGBEGIN":  ("Recognizes beginning sounds?", "radio", {1: "Yes", 0: "No"}, 1),
        "SAMESOUND":   ("Identifies rhyming words?", "radio", {1: "Yes", 0: "No"}, 1),
        "WRITENAME":   ("Writes own name?", "radio", {1: "Yes", 0: "No"}, 1),
        "FOCUSON":     ("Stays focused on tasks?", "radio", {1: "Yes", 0: "No"}, 1),
        "READONEDIGIT":("Reads single-digit numbers?", "radio", {1: "Yes", 0: "No"}, 1),
        "SIMPLEADDITION":("Does simple addition?", "radio", {1: "Yes", 0: "No"}, 1),
    },
    "👥 Social & Behavioral": {
        "TEMPER_R":   ("Has severe temper tantrums?", "radio", {1: "Yes", 0: "No"}, 0),
        "PLAYWELL":   ("Plays well with other children?", "radio", {1: "Yes", 0: "No"}, 1),
        "DISTRACTED": ("Easily distracted?", "radio", {1: "Yes", 0: "No"}, 0),
        "HURTSAD":    ("Often feels hurt or sad?", "radio", {1: "Yes", 0: "No"}, 0),
        "CALMDOWN_R": ("Calms down when upset?", "radio", {1: "Yes", 0: "No"}, 1),
        "WAITFORTURN":("Waits for turn in games?", "radio", {1: "Yes", 0: "No"}, 1),
        "HARDWORK":   ("Works hard at tasks?", "radio", {1: "Yes", 0: "No"}, 1),
        "SHARETOYS":  ("Shares toys with others?", "radio", {1: "Yes", 0: "No"}, 1),
        "MAKEFRIEND": ("Makes friends easily?", "radio", {1: "Yes", 0: "No"}, 1),
        "TALKABOUT":  ("Talks about feelings?", "radio", {1: "Yes", 0: "No"}, 1),
        "K7Q30":      ("Feels safe in neighborhood?", "radio", {1: "Yes", 0: "No"}, 1),
        "K7Q31":      ("Community members help each other?", "radio", {1: "Yes", 0: "No"}, 1),
    },
}

# ── M-CHAT-R questions ────────────────────────────────────────
MCHAT = [
    ("If you point at something, does your child look at it?", "No"),
    ("Have you ever wondered if your child might be deaf?", "Yes"),
    ("Does your child play pretend (e.g., drink from empty cup)?", "No"),
    ("Does your child like climbing on things?", "No"),
    ("Does your child make unusual finger movements near eyes?", "Yes"),
    ("Does your child point to ask for something?", "No"),
    ("Does your child point to show you something interesting?", "No"),
    ("Is your child interested in other children?", "No"),
    ("Does your child show you things by holding them up?", "No"),
    ("Does your child respond to his/her name?", "No"),
    ("When you smile, does your child smile back?", "No"),
    ("Does your child get upset by everyday noises?", "Yes"),
    ("Does your child walk?", "No"),
    ("Does your child look you in the eye while talking?", "No"),
    ("Does your child try to copy what you do?", "No"),
    ("If you look at something, does your child look too?", "No"),
    ("Does your child try to get you to watch him/her?", "No"),
    ("Does your child understand simple instructions?", "No"),
    ("Does your child look at your face to see how you feel?", "No"),
    ("Does your child like movement activities (swinging)?", "No"),
]

# ── Doctor recommendation engine ──────────────────────────────
def get_recommendation(ai_risk, mchat_score=None, mchat_done=False):
    combined = (0.6 * ai_risk + 0.4 * (mchat_score / 20 * 100)) if mchat_done and mchat_score is not None else ai_risk
    if combined >= 60:
        return {
            "level": "HIGH", "color": "#EF4444", "emoji": "🚨",
            "combined": round(combined, 1),
            "urgency": "URGENT — Visit specialist within 1–2 weeks",
            "docs": [
                ("🧠 Pediatric Neurologist", "Evaluates brain development & neurological conditions"),
                ("👨‍⚕️ Child Psychiatrist", "Diagnoses ASD and developmental disorders"),
                ("🏥 Developmental Pediatrician", "Specialises in developmental delays"),
                ("🗣️ Speech-Language Pathologist", "Assesses communication development"),
            ],
            "hospitals": ["NIMHANS, Bangalore", "AIIMS, New Delhi", "Child Dev. Centre, JIPMER, Puducherry",
                         "Institute of Child Health, Chennai", "Kokilaben Hospital, Mumbai"],
            "tips": ["Do NOT wait — early intervention before age 5 gives the best outcomes",
                    "Record your child's behavior on video (helps doctors enormously)",
                    "Contact the nearest government medical college immediately",
                    "Ask for a full developmental / ASD evaluation"]
        }
    elif combined >= 30:
        return {
            "level": "MODERATE", "color": "#F59E0B", "emoji": "⚠️",
            "combined": round(combined, 1),
            "urgency": "Schedule appointment within 1 month",
            "docs": [
                ("🏥 Developmental Pediatrician", "First point of contact for developmental concerns"),
                ("👨‍⚕️ Child Psychologist", "Behavioral assessment and early intervention"),
                ("🗣️ Speech Therapist", "Check communication milestones"),
            ],
            "hospitals": ["Local Govt District Hospital — Pediatrics dept.",
                         "Any accredited child development centre in your city"],
            "tips": ["Monitor milestones closely over the next 2–4 weeks",
                    "Encourage eye contact, play, and name-response at home",
                    "Discuss with your regular pediatrician first"]
        }
    else:
        return {
            "level": "LOW", "color": "#10B981", "emoji": "✅",
            "combined": round(combined, 1),
            "urgency": "Continue routine pediatric checkups",
            "docs": [("👶 Regular Pediatrician", "Continue routine developmental checkups")],
            "hospitals": ["Your regular child doctor / family physician",
                         "Local Primary Health Centre (PHC)"],
            "tips": ["Continue regular developmental monitoring",
                    "Encourage reading, social play, and eye contact",
                    "Re-screen at 18 and 24 months"]
        }

# ── Patient Storage & Auth Functions ─────────────────────────
PATIENTS_FILE = rpath("patients_data.json")

def load_patients():
    try:
        if os.path.exists(PATIENTS_FILE):
            with open(PATIENTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}

def save_patients(patients):
    with open(PATIENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(patients, f, indent=2)

def clear_current_eval_state():
    st.session_state["ai_risk"] = None
    st.session_state["mchat_score"] = None
    st.session_state["mchat_done"] = False
    st.session_state["inputs"] = {}
    for i in range(len(MCHAT)):
        st.session_state.pop(f"mchat_{i}", None)

def register_patient(username, password, full_name=""):
    username = username.strip()
    if not username:
        return False, "Please enter a patient username."
    if len(password) < 4:
        return False, "Password must be at least 4 characters long."
    patients = load_patients()
    if username in patients or username == "admin":
        return False, f"Username '{username}' already exists. Please choose another username or log in."
    
    patients[username] = {
        "password": password,
        "full_name": full_name or username,
        "role": "patient",
        "registered_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "history": []
    }
    save_patients(patients)
    st.session_state["patient_username"] = username
    st.session_state["current_user_role"] = "patient"
    clear_current_eval_state()
    return True, f"Patient account for '{username}' created successfully."

def login_patient(username, password):
    username = username.strip()
    if username == "admin" and password == "admin123":
        st.session_state["patient_username"] = "admin"
        st.session_state["current_user_role"] = "admin"
        clear_current_eval_state()
        return True, "Admin authentication successful."
    
    patients = load_patients()
    if username not in patients:
        return False, "User account not found. Please check credentials or sign up."
    if patients[username].get("password") != password:
        return False, "Incorrect password. Please try again."
    
    st.session_state["patient_username"] = username
    st.session_state["current_user_role"] = patients[username].get("role", "patient")
    clear_current_eval_state()
    return True, "Login successful."

def delete_patient(username):
    patients = load_patients()
    if username in patients:
        del patients[username]
        save_patients(patients)
        return True
    return False

def clear_patient_history(username):
    patients = load_patients()
    if username in patients:
        patients[username]["history"] = []
        if "latest_result" in patients[username]:
            del patients[username]["latest_result"]
        save_patients(patients)
        return True
    return False

def save_patient_result(result_dict):
    username = st.session_state.get("patient_username")
    if not username or username == "admin":
        return
    patients = load_patients()
    patient = patients.setdefault(username, {"password": "", "history": []})
    patient.setdefault("history", [])
    record = dict(result_dict)
    record["saved_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    patient["history"].append(record)
    patient["latest_result"] = record
    save_patients(patients)

def get_current_patient_data():
    username = st.session_state.get("patient_username")
    if not username:
        return None
    patients = load_patients()
    return patients.get(username)

def get_current_patient_history():
    patient = get_current_patient_data()
    if not patient:
        return []
    return patient.get("history", [])

def get_all_patient_records():
    patients = load_patients()
    rows = []
    for username, data in patients.items():
        if username == "admin":
            continue
        history = data.get("history", [])
        latest = history[-1] if history else {}
        rows.append({
            "Username": username,
            "Full Name": data.get("full_name", username),
            "Role": data.get("role", "patient"),
            "Total Assessments": len(history),
            "Latest AI Risk": f"{latest.get('ai_risk')}%" if latest.get('ai_risk') is not None else "N/A",
            "Latest Combined": f"{latest.get('combined')}%" if latest.get('combined') is not None else "N/A",
            "Risk Level": latest.get("level", "Unscreened"),
            "Last Updated": latest.get("saved_at", "—"),
            "Urgency": latest.get("urgency", "—"),
            "_raw_level": latest.get("level", "UNSCREENED"),
            "_raw_combined": latest.get("combined", 0.0)
        })
    return rows

# ── Session state defaults ───────────────────────────────────
for key, default in [
    ("ai_risk", None), ("mchat_score", None), ("mchat_done", False),
    ("inputs", {}), ("patient_username", None), ("current_user_role", None),
    ("auth_mode", "login"), ("page", "🔍 Clinical Input")
]:
    if key not in st.session_state:
        st.session_state[key] = default

logged_in = st.session_state["patient_username"] is not None
role = st.session_state.get("current_user_role")

# ── TOPBAR HEADER ─────────────────────────────────────────────
role_badge = f'<span class="role-pill-admin">ADMINISTRATOR</span>' if role == "admin" else f'<span class="role-pill-patient">PATIENT</span>' if logged_in else ''
user_info = f'<div class="user-badge"><span class="user-name">👤 {st.session_state["patient_username"]}</span>{role_badge}</div>' if logged_in else ''

st.markdown(f"""
<div class="topbar">
    <div class="brand-group">
        <div class="brand-icon">🧠</div>
        <div>
            <h1 class="brand-title">ASD Clinical Detection Portal</h1>
            <p class="brand-sub">TabM PyTorch Neural Network & SHAP Clinical Screening</p>
        </div>
    </div>
    {user_info}
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# AUTHENTICATION ROUTER (LOGIN & SIGNUP PAGES)
# ═══════════════════════════════════════════════════════════════
if not logged_in:
    auth_col1, auth_col2 = st.columns([1.1, 1], gap="large")
    
    with auth_col1:
        st.markdown("""
        <div class="auth-hero-container">
            <div class="auth-kicker">Clinical Intelligence Platform</div>
            <h1 class="auth-headline">Early ASD Screening with Explainable AI</h1>
            <p class="auth-desc">
                An advanced pediatric developmental assessment portal powered by TabM ensemble deep learning models, M-CHAT-R clinical protocol, and SHAP explainability.
            </p>
            <div class="feature-pill-group">
                <span class="feature-pill">⚡ 70 Clinical Features</span>
                <span class="feature-pill">🧠 TabM Deep Neural Net</span>
                <span class="feature-pill">📊 SHAP Interpretability</span>
                <span class="feature-pill">📋 M-CHAT-R Standard</span>
                <span class="feature-pill">🔒 Secure Data Handling</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with auth_col2:
        # Auth Toggle Buttons
        t1, t2 = st.columns(2)
        with t1:
            if st.button("🔑 Login", type="primary" if st.session_state["auth_mode"] == "login" else "secondary", use_container_width=True):
                st.session_state["auth_mode"] = "login"
                st.rerun()
        with t2:
            if st.button("📝 Sign Up", type="primary" if st.session_state["auth_mode"] == "signup" else "secondary", use_container_width=True):
                st.session_state["auth_mode"] = "signup"
                st.rerun()
        
        st.write("")
        
        if st.session_state["auth_mode"] == "login":
            st.markdown("""
            <div class="auth-form-card">
                <h2 class="auth-form-title">Welcome Back</h2>
                <p class="auth-form-sub">Sign in to access your screening workspace or admin portal</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("login_form"):
                login_user = st.text_input("Username", placeholder="e.g. Nandhini or admin")
                login_pass = st.text_input("Password", type="password", placeholder="Enter your password")
                submit_login = st.form_submit_button("Sign In to Workspace", type="primary", use_container_width=True)
                
                if submit_login:
                    ok, msg = login_patient(login_user, login_pass)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

            d1, d2 = st.columns(2)
            with d1:
                if st.button("👤 Patient Login", use_container_width=True):
                    register_patient("DemoPatient", "demo123", "Demo Patient")
                    login_patient("DemoPatient", "demo123")
                    st.rerun()
            with d2:
                if st.button("👑 Admin Login", use_container_width=True):
                    login_patient("admin", "admin123")
                    st.rerun()

        else: # SIGNUP
            st.markdown("""
            <div class="auth-form-card">
                <h2 class="auth-form-title">Create Account</h2>
                <p class="auth-form-sub">Register a new patient account for confidential ASD screening</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("signup_form"):
                reg_name = st.text_input("Full Name / Child Name", placeholder="e.g. John Doe")
                reg_user = st.text_input("Create Patient Username", placeholder="e.g. johndoe")
                reg_pass = st.text_input("Create Password", type="password", placeholder="Min. 4 characters")
                reg_confirm = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
                submit_signup = st.form_submit_button("Register Patient Account", type="primary", use_container_width=True)
                
                if submit_signup:
                    if not reg_user or not reg_pass:
                        st.error("Please fill in all required fields.")
                    elif reg_pass != reg_confirm:
                        st.error("Passwords do not match. Please verify your password.")
                    else:
                        ok, msg = register_patient(reg_user, reg_pass, reg_name)
                        if ok:
                            st.success(msg)
                            st.session_state["auth_mode"] = "login"
                            st.rerun()
                        else:
                            st.error(msg)
                            
    st.stop()

# ═══════════════════════════════════════════════════════════════
# MAIN LOGGED IN NAVIGATION
# ═══════════════════════════════════════════════════════════════
nav_items = ["🔍 Clinical Input", "📋 M-CHAT-R Quiz", "🩺 Result & Doctor", "📊 Dashboard", "📈 SHAP Explorer"]
if role == "admin":
    nav_items.insert(0, "👑 Admin Panel")

nav_cols = st.columns(len(nav_items) + 1)
for col, item in zip(nav_cols, nav_items):
    with col:
        is_active = (st.session_state.get("page") == item)
        if st.button(item, type="primary" if is_active else "secondary", use_container_width=True):
            st.session_state["page"] = item
            st.rerun()

with nav_cols[-1]:
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state["patient_username"] = None
        st.session_state["current_user_role"] = None
        clear_current_eval_state()
        st.session_state["auth_mode"] = "login"
        st.rerun()

page = st.session_state.get("page", nav_items[0])

# ═══════════════════════════════════════════════════════════════
# PAGE 0 — 👑 ADMIN MANAGEMENT PORTAL
# ═══════════════════════════════════════════════════════════════
if "Admin Panel" in page and role == "admin":
    st.markdown("""
    <div class="admin-banner">
        <h2 style="margin:0 0 0.4rem 0;">👑 Admin Management & Patient Records Portal</h2>
        <small style="color:#C7D2FE;">Comprehensive view of patient assessments, risk breakdown, and system diagnostics.</small>
    </div>
    """, unsafe_allow_html=True)
    
    rows = get_all_patient_records()
    total_patients = len(rows)
    high_count = sum(1 for r in rows if r["_raw_level"] == "HIGH")
    med_count = sum(1 for r in rows if r["_raw_level"] == "MODERATE")
    low_count = sum(1 for r in rows if r["_raw_level"] == "LOW")
    total_screenings = sum(r["Total Assessments"] for r in rows)
    avg_score = round(np.mean([r["_raw_combined"] for r in rows if r["_raw_combined"] > 0]), 1) if rows else 0.0

    # Admin KPI Row
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(f'<div class="kpi-card"><p class="kpi-val" style="color:#60A5FA;">{total_patients}</p><p class="kpi-label">Patients</p></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card"><p class="kpi-val" style="color:#EF4444;">{high_count}</p><p class="kpi-label">High Risk</p></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card"><p class="kpi-val" style="color:#F59E0B;">{med_count}</p><p class="kpi-label">Mod Risk</p></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-card"><p class="kpi-val" style="color:#10B981;">{low_count}</p><p class="kpi-label">Low Risk</p></div>', unsafe_allow_html=True)
    k5.markdown(f'<div class="kpi-card"><p class="kpi-val" style="color:#A78BFA;">{total_screenings}</p><p class="kpi-label">Screenings</p></div>', unsafe_allow_html=True)
    k6.markdown(f'<div class="kpi-card"><p class="kpi-val" style="color:#F472B6;">{avg_score}%</p><p class="kpi-label">Avg Risk</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    admin_tabs = st.tabs(["📋 Patient Directory & Screening Records", "➕ Create Patient Account", "⚙️ System & Model Specs"])

    with admin_tabs[0]:
        st.markdown("### Patient Screening Directory")
        
        c_search, c_filter = st.columns([2, 1])
        with c_search:
            search_query = st.text_input("🔍 Search patient by username or full name", placeholder="Type username...").strip().lower()
        with c_filter:
            risk_filter = st.selectbox("Filter by Risk Level", ["All Risk Levels", "🚨 High Risk", "⚠️ Moderate Risk", "✅ Low Risk", "Unscreened"])

        filtered_rows = rows
        if search_query:
            filtered_rows = [r for r in filtered_rows if search_query in r["Username"].lower() or search_query in r["Full Name"].lower()]
        if risk_filter == "🚨 High Risk":
            filtered_rows = [r for r in filtered_rows if r["_raw_level"] == "HIGH"]
        elif risk_filter == "⚠️ Moderate Risk":
            filtered_rows = [r for r in filtered_rows if r["_raw_level"] == "MODERATE"]
        elif risk_filter == "✅ Low Risk":
            filtered_rows = [r for r in filtered_rows if r["_raw_level"] == "LOW"]
        elif risk_filter == "Unscreened":
            filtered_rows = [r for r in filtered_rows if r["Total Assessments"] == 0]

        if not filtered_rows:
            st.info("No matching patient records found.")
        else:
            display_df = pd.DataFrame(filtered_rows)[["Username", "Full Name", "Total Assessments", "Latest AI Risk", "Latest Combined", "Risk Level", "Last Updated", "Urgency"]]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("### 🔎 Patient History Inspector & Actions")
            selected_patient = st.selectbox("Select Patient to Inspect", options=[r["Username"] for r in filtered_rows])
            
            p_data = load_patients().get(selected_patient, {})
            p_history = p_data.get("history", [])
            
            st.markdown(f'<div class="patient-inspector">', unsafe_allow_html=True)
            ic1, ic2 = st.columns([2, 1])
            with ic1:
                st.markdown(f"#### 👤 Patient: **{selected_patient}** ({p_data.get('full_name', selected_patient)})")
                st.write(f"**Registered At:** {p_data.get('registered_at', 'N/A')}")
                st.write(f"**Total Screenings Performed:** {len(p_history)}")
            with ic2:
                btn_clear, btn_del = st.columns(2)
                with btn_clear:
                    if st.button("🧹 Clear History", key=f"clr_{selected_patient}"):
                        clear_patient_history(selected_patient)
                        st.success(f"History cleared for {selected_patient}")
                        st.rerun()
                with btn_del:
                    if st.button("🗑️ Delete User", key=f"del_{selected_patient}"):
                        delete_patient(selected_patient)
                        st.success(f"User {selected_patient} deleted.")
                        st.rerun()
            
            if p_history:
                latest = p_history[-1]
                st.markdown("##### Latest Assessment Summary")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("AI Model Risk", f"{latest.get('ai_risk')}%")
                m2.metric("M-CHAT Score", f"{latest.get('mchat_score')}/20" if latest.get('mchat_score') is not None else "N/A")
                m3.metric("Combined Score", f"{latest.get('combined')}%")
                m4.metric("Risk Level", latest.get('level'))
                
                if latest.get("docs"):
                    st.markdown("**Assigned Specialists:**")
                    for d_name, d_desc in latest["docs"]:
                        st.markdown(f"- **{d_name}**: {d_desc}")
                
                st.markdown("##### Assessment History Timeline")
                hist_df = pd.DataFrame(p_history)[["saved_at", "ai_risk", "mchat_score", "combined", "level", "urgency"]]
                st.dataframe(hist_df, use_container_width=True, hide_index=True)
            else:
                st.info("This patient has not completed any screening assessment yet.")
            st.markdown('</div>', unsafe_allow_html=True)

    with admin_tabs[1]:
        st.markdown("### Add New Patient Account")
        with st.form("admin_add_patient"):
            adm_name = st.text_input("Full Name")
            adm_user = st.text_input("Username")
            adm_pass = st.text_input("Password", type="password")
            btn_create = st.form_submit_button("Create Patient Account", type="primary")
            if btn_create:
                ok, msg = register_patient(adm_user, adm_pass, adm_name)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    with admin_tabs[2]:
        st.markdown("### System & Model Diagnostics")
        st.write(f"**TabM PyTorch Model Status:** {'✅ Ready' if model_ok else '❌ Not Loaded'}")
        st.write(f"**Input Vector Features:** {len(feat_names)} features")
        st.write(f"**Classification Threshold:** {SCREENING_THRESHOLD:.2f}")
        if metrics:
            st.json(metrics)

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — CLINICAL INPUT (70 features)
# ═══════════════════════════════════════════════════════════════
elif "Clinical Input" in page:
    st.markdown("""
    <div class="hero">
        <h1 class="hero-title">Step 1: Clinical Feature Assessment</h1>
        <p class="hero-sub">Complete the 70 clinical demographic, medical, developmental, and behavioral parameters derived from NSCH 2023.</p>
    </div>
    """, unsafe_allow_html=True)

    inputs = {}
    for group_name, fields in FEATURE_GROUPS.items():
        with st.expander(group_name, expanded=(group_name == "👤 Demographics")):
            st.markdown(f'<div class="group-title">{group_name}</div>', unsafe_allow_html=True)
            n_cols = 2 if len(fields) > 4 else 1
            cols = st.columns(n_cols)
            for ci, (feat, (label, ftype, *opts)) in enumerate(fields.items()):
                col = cols[ci % n_cols]
                with col:
                    if ftype == "slider":
                        mn, mx, dflt = opts
                        inputs[feat] = st.slider(label, mn, mx, dflt, key=f"f_{feat}")
                    elif ftype == "radio":
                        opt_dict, dflt = opts
                        inputs[feat] = st.radio(label, list(opt_dict.keys()),
                            format_func=lambda x, od=opt_dict: od[x],
                            index=list(opt_dict.keys()).index(dflt),
                            horizontal=True, key=f"f_{feat}")
                    elif ftype == "select":
                        opt_dict, dflt = opts
                        sel = st.selectbox(label, list(opt_dict.keys()),
                            format_func=lambda x, od=opt_dict: od[x],
                            index=list(opt_dict.keys()).index(dflt),
                            key=f"f_{feat}")
                        inputs[feat] = sel

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    predict_btn = st.button("🔮 Run AI Neural Network Prediction (TabM-PLE)", type="primary", use_container_width=True)

    if predict_btn:
        if not model_ok:
            st.error("Model state not loaded. Ensure PyTorch tabm model exists in models/ directory.")
        else:
            x_raw = np.array([[inputs.get(f, 0) for f in feat_names]], dtype=np.float32)
            x_sc = x_raw.copy()
            x_sc[:, num_idx] = scaler.transform(x_raw[:, num_idx])

            xn = torch.tensor(x_sc[:, num_idx])
            xb = torch.tensor(x_sc[:, bin_idx])
            with torch.no_grad():
                prob = torch.sigmoid(model(xn, xb)).item()
            predicted_class = int(prob >= SCREENING_THRESHOLD)
            risk = round(prob * 100, 1)
            st.session_state.ai_risk = risk
            st.session_state.inputs = inputs
            st.session_state.mchat_score = None
            st.session_state.mchat_done = False
            for i in range(len(MCHAT)):
                st.session_state.pop(f"mchat_{i}", None)

            if st.session_state.get("current_user_role") == "patient":
                save_patient_result({
                    "ai_risk": risk,
                    "mchat_score": None,
                    "mchat_done": False,
                    "combined": risk,
                    "level": "HIGH" if risk >= 60 else "MODERATE" if risk >= 30 else "LOW",
                    "urgency": "AI-only screening result",
                    "recommendation": "Await M-CHAT-R follow-up for final combined score.",
                })

            if risk >= 60:
                st.markdown(f'<div class="risk-high"><p class="risk-label" style="color:#EF4444;">🚨 HIGH RISK ASSESSMENT</p><p style="color:#94A3B8;font-size:1.1rem;">TabM Neural Network Risk Score: <b style="color:#EF4444;font-size:2.2rem;">{risk}%</b></p></div>', unsafe_allow_html=True)
            elif risk >= 30:
                st.markdown(f'<div class="risk-med"><p class="risk-label" style="color:#F59E0B;">⚠️ MODERATE RISK ASSESSMENT</p><p style="color:#94A3B8;font-size:1.1rem;">TabM Neural Network Risk Score: <b style="color:#F59E0B;font-size:2.2rem;">{risk}%</b></p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low"><p class="risk-label" style="color:#10B981;">✅ LOW RISK ASSESSMENT</p><p style="color:#94A3B8;font-size:1.1rem;">TabM Neural Network Risk Score: <b style="color:#10B981;font-size:2.2rem;">{risk}%</b></p></div>', unsafe_allow_html=True)

            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=risk,
                title={"text": "TabM AI Risk Gauge", "font": {"color": "#94A3B8", "size": 14}},
                gauge={"axis": {"range": [0, 100], "tickcolor": "#94A3B8"},
                       "bar": {"color": "#EF4444" if risk >= 60 else "#F59E0B" if risk >= 30 else "#10B981"},
                       "bgcolor": "#1A2235", "bordercolor": "#1E293B",
                       "steps": [{"range": [0, 30], "color": "#0D2B1E"},
                                {"range": [30, 60], "color": "#2B2112"},
                                {"range": [60, 100], "color": "#2D1515"}],
                       "threshold": {"line": {"color": "white", "width": 3}, "thickness": .8,
                                     "value": SCREENING_THRESHOLD * 100}},
                number={"suffix": "%", "font": {"color": "#F1F5F9", "size": 32}},
            ))
            gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="#0F172A", font_color="#F1F5F9")
            st.plotly_chart(gauge, use_container_width=True)
            st.success("✅ Step 1 Complete! Now proceed to **Step 2: M-CHAT-R Quiz**.")

    elif st.session_state.ai_risk is not None:
        st.metric("Current AI Risk Score", f"{st.session_state.ai_risk}%")

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — M-CHAT-R
# ═══════════════════════════════════════════════════════════════
elif "M-CHAT-R" in page:
    st.markdown("""
    <div class="hero">
        <h1 class="hero-title">Step 2: M-CHAT-R Clinical Questionnaire</h1>
        <p class="hero-sub">The Modified Checklist for Autism in Toddlers (Revised) 20-question pediatric screening protocol.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.ai_risk is None:
        st.warning("⚠️ Please complete Step 1 first to establish the baseline AI risk score.")

    answers = []
    for i, (q, risk_ans) in enumerate(MCHAT):
        st.markdown(f'<div class="q-box"><b>Q{i+1}.</b> {q}</div>', unsafe_allow_html=True)
        ans = st.radio("", ["Yes", "No"], horizontal=True, key=f"mchat_{i}", label_visibility="collapsed")
        answers.append(ans)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if st.button("📊 Calculate M-CHAT-R Score", type="primary", use_container_width=True):
        score = sum(1 for (q, ra), a in zip(MCHAT, answers) if a == ra)
        st.session_state.mchat_score = score
        st.session_state.mchat_done = True

        if score <= 2: color = "#10B981"; label = "LOW RISK"
        elif score <= 7: color = "#F59E0B"; label = "MEDIUM RISK"
        else: color = "#EF4444"; label = "HIGH RISK"

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"""<div style='text-align:center;background:#0F172A;
            border:2px solid {color};border-radius:18px;padding:1.8rem;'>
            <p style='font-size:3.2rem;font-weight:800;color:{color};margin:0;'>{score}/20</p>
            <p style='color:{color};font-weight:700;font-size:1.1rem;margin:0;'>{label}</p>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
| Score Range | Clinical Category | Recommended Action |
|:---|:---|:---|
| **0 – 2** | 🟢 Low Risk | Continue routine developmental monitoring |
| **3 – 7** | 🟡 Medium Risk | Administer follow-up screening & consult pediatrician |
| **8 – 20** | 🔴 High Risk | Immediate referral for comprehensive evaluation |

**Your M-CHAT Score: `{score}/20` ({label})**
""")
        st.success("✅ M-CHAT-R complete! Proceed to **Step 3: Result & Doctor Recommendation**.")

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — COMBINED RESULT + DOCTOR
# ═══════════════════════════════════════════════════════════════
elif "Result & Doctor" in page:
    st.markdown("""
    <div class="hero">
        <h1 class="hero-title">Step 3: Combined Risk Assessment & Doctor Referral</h1>
        <p class="hero-sub">Unified risk calculation blending TabM PyTorch Neural Network prediction and M-CHAT-R clinical checklist.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.ai_risk is None:
        st.error("❌ Please complete Step 1 first.")
        st.stop()

    rec = get_recommendation(st.session_state.ai_risk, st.session_state.mchat_score, st.session_state.mchat_done)
    if st.session_state.get("current_user_role") == "patient":
        save_patient_result({
            "ai_risk": st.session_state.ai_risk,
            "mchat_score": st.session_state.mchat_score if st.session_state.mchat_done else None,
            "mchat_done": st.session_state.mchat_done,
            "combined": rec["combined"],
            "level": rec["level"],
            "urgency": rec["urgency"],
            "docs": rec["docs"],
            "hospitals": rec["hospitals"],
            "tips": rec["tips"],
        })

    box = {"HIGH": "risk-high", "MODERATE": "risk-med", "LOW": "risk-low"}[rec["level"]]

    st.markdown(f"""<div class="{box}">
    <p class="risk-label" style="color:{rec['color']};">{rec['emoji']} {rec['level']} RISK</p>
    <p style="color:#94A3B8;font-size:1.1rem;margin:.4rem 0;">
      Combined Multi-Modal Risk: <b style="color:{rec['color']};font-size:2.2rem;">{rec['combined']}%</b></p>
    <p style="color:{rec['color']};font-weight:700;font-size:1.05rem;">{rec['urgency']}</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("🤖 AI Risk (TabM Neural Net)", f"{st.session_state.ai_risk}%")
    if st.session_state.mchat_done:
        c2.metric("📋 M-CHAT-R Protocol", f"{st.session_state.mchat_score}/20")
        c3.metric("🔀 Final Combined Risk", f"{rec['combined']}%")
    else:
        c2.info("M-CHAT-R questionnaire pending")

    # Bar chart
    src = ["AI (TabM-PLE)"]
    vals = [st.session_state.ai_risk]
    if st.session_state.mchat_done:
        src.append("M-CHAT-R Clinical")
        vals.append(round(st.session_state.mchat_score / 20 * 100, 1))
    src.append("Combined Score")
    vals.append(rec['combined'])
    clrs = ["#EF4444" if v >= 60 else "#F59E0B" if v >= 30 else "#10B981" for v in vals]
    
    fig = go.Figure(go.Bar(x=src, y=vals, marker_color=clrs, marker_line_color="white",
                         marker_line_width=1.5, text=[f"{v}%" for v in vals], textposition="outside"))
    fig.add_hline(y=30, line_dash="dash", line_color="#F59E0B", annotation_text="Moderate threshold")
    fig.add_hline(y=60, line_dash="dash", line_color="#EF4444", annotation_text="High threshold")
    fig.update_layout(height=280, paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
                      font_color="#F1F5F9", yaxis_range=[0, 115], showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 👨‍⚕️ Recommended Specialists")
    for icon_name, why in rec["docs"]:
        st.markdown(f"""<div class="doc-card">
        <h4 style="color:#60A5FA;margin:0;">{icon_name}</h4>
        <p style="color:#94A3B8;margin:.3rem 0 0 0;">{why}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🏥 Accredited Healthcare Centers")
    for i, h in enumerate(rec["hospitals"], 1):
        st.markdown(f"**{i}.** {h}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🏠 Recommended Action Plan")
    for t in rec["tips"]:
        st.markdown(f"✅ {t}")

    # Report Download
    report = f"""
ASD CLINICAL DETECTION & ASSESSMENT REPORT
===================================================
Patient Username: {st.session_state.get('patient_username')}
Assessment Date:  {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Architecture:     TabM Neural Network (70 Features) + M-CHAT-R + SHAP

SCORES & METRICS:
  AI Model (TabM-PLE) : {st.session_state.ai_risk}%
  M-CHAT-R Checklist  : {f'{st.session_state.mchat_score}/20' if st.session_state.mchat_done else 'Not completed'}
  Combined Risk Score : {rec['combined']}%

CLINICAL CLASSIFICATION: {rec['level']} RISK
ACTION STATUS:          {rec['urgency']}

RECOMMENDED SPECIALISTS:
{chr(10).join([f'  - {n}: {w}' for n, w in rec['docs']])}

ACCREDITED HOSPITALS:
{chr(10).join([f'  {i+1}. {h}' for i, h in enumerate(rec['hospitals'])])}

ACTION PLAN:
{chr(10).join([f'  - {t}' for t in rec['tips']])}

===================================================
DISCLAIMER: This report is generated by an AI screening system for informational purposes only.
It is not a formal medical diagnosis. Consult a licensed pediatrician or neurologist.
===================================================
"""
    st.download_button("📥 Download Official Screening Report (.txt)", data=report.encode(),
                       file_name=f"ASD_Screening_Report_{st.session_state.get('patient_username')}.txt",
                       mime="text/plain", use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — DASHBOARD
# ═══════════════════════════════════════════════════════════════
elif "Dashboard" in page:
    username = st.session_state.get("patient_username")
    st.markdown(f"## 📊 Patient Workspace & Analytics — {username}")

    history = get_current_patient_history()
    if not history:
        st.info("No saved assessment history available for this patient yet.")
    else:
        latest = history[-1]
        st.markdown("### Latest Screening Metrics")
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Latest AI Risk", f"{latest.get('ai_risk')}%" if latest.get('ai_risk') is not None else "N/A")
        dc2.metric("Latest Combined Risk", f"{latest.get('combined')}%" if latest.get('combined') is not None else "N/A")
        dc3.metric("Risk Level", latest.get("level", "N/A"))

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### Model Diagnostic Specs")
    if metrics:
        kpis = [("ROC-AUC", "#3B82F6"), ("F1-Score", "#10B981"),
                ("Recall(Sens)", "#8B5CF6"), ("Accuracy", "#F59E0B"), ("MCC", "#14B8A6")]
        cols = st.columns(5)
        for col, (k, c) in zip(cols, kpis):
            v = metrics.get(k, "N/A")
            col.markdown(f"""<div class="kpi-card">
            <p class="kpi-val" style="color:{c};">{v}</p>
            <p class="kpi-label">{k}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### Model Upgrade Baseline Comparison")
    comp_data = {
        "Metric": ["ROC-AUC", "Recall", "F1-Score", "Model Parameters", "Feature Vector"],
        "Baseline (7 Features)": ["0.6648", "0.6816", "0.1172", "11,849", "7"],
        "Enhanced TabM-PLE (70 Features)": [
            str(metrics.get("ROC-AUC", "0.9270")),
            str(metrics.get("Recall(Sens)", "—")),
            str(metrics.get("F1-Score", "—")),
            "359,441", "70"
        ],
    }
    comp_df = pd.DataFrame(comp_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    plots = {
        "Training Curves": "results/plots/30_tabmple_training_curves.png",
        "ROC Curve (Enhanced)": "results/plots/32_enh_roc_curve.png",
        "Confusion Matrix": "results/plots/31_enh_confusion_matrix.png",
        "PR Curve": "results/plots/33_enh_pr_curve.png",
        "SHAP Importance": "results/plots/35_enh_shap_bar.png",
        "SHAP Beeswarm": "results/plots/34_enh_shap_beeswarm.png",
    }
    keys = list(plots.keys())
    for i in range(0, len(keys), 2):
        c1, c2 = st.columns(2)
        for col, idx in [(c1, i), (c2, i + 1)]:
            if idx < len(keys):
                fp = rpath(plots[keys[idx]])
                if os.path.exists(fp):
                    col.markdown(f"**{keys[idx]}**")
                    col.image(fp, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 5 — SHAP EXPLORER
# ═══════════════════════════════════════════════════════════════
elif "SHAP Explorer" in page:
    username = st.session_state.get("patient_username")
    st.markdown(f"## 📈 SHAP Feature Explainability — {username}")

    st.markdown("### Model Explainability Visualizations")
    tabs = st.tabs(["Beeswarm Plot (Top 20 Features)", "Feature Importance Bar Chart", "Waterfall Plot — ASD Prediction"])
    shap_files = [
        "results/plots/34_enh_shap_beeswarm.png",
        "results/plots/35_enh_shap_bar.png",
        "results/plots/36_enh_shap_waterfall_asd.png",
    ]
    for tab, fp in zip(tabs, shap_files):
        with tab:
            full = rpath(fp)
            if os.path.exists(full):
                st.image(full, use_container_width=True)
            else:
                st.info("Run `python src/07b_shap_enhanced.py` to generate this plot.")

    si = rpath("results/shap_feature_importance_enhanced.csv")
    if os.path.exists(si):
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### Top 20 SHAP Feature Importances")
        df_si = pd.read_csv(si)
        st.dataframe(df_si[['Rank', 'Feature', 'Label', 'Mean|SHAP|']].head(20),
                     use_container_width=True, hide_index=True)

# ── FOOTER ────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#64748B;font-size:0.82rem;padding:0.8rem 0;">
    🧠 <b>ASD Early Detection System v3.0</b> — TabM PyTorch Neural Network + SHAP + M-CHAT-R<br>
    <span style="color:#EF4444;">⚠️ Screening Tool Only — Not a clinical medical diagnosis. Always consult a qualified specialist.</span>
</div>
""", unsafe_allow_html=True)
