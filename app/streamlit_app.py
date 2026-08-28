import sys
if hasattr(sys.stdout,'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')

"""
Streamlit application for ASD screening and model explanation.
This is a student project implementation demonstrating model predictions and SHAP-based explanations.
"""

import os,json,warnings,gc
import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use('Agg')
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..","src"))
try:
    from tabm_ple_model import TabMPLE
    MODEL_CLASS = TabMPLE
    ENHANCED = True
except:
    ENHANCED = False

warnings.filterwarnings('ignore')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
def rpath(*p): return os.path.join(ROOT,*p)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ASD Detection — TabM",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#070E1A;color:#F1F5F9;}
.main{background:#070E1A;}
.hero{background:linear-gradient(135deg,#0D1B2E,#0A1628,#0F2040);
      border:1px solid #1E3A5F;border-radius:20px;padding:2.2rem 2.8rem;
      margin-bottom:1.5rem;box-shadow:0 8px 40px rgba(59,130,246,.2);}
.hero-title{font-size:2.4rem;font-weight:800;
  background:linear-gradient(135deg,#60A5FA,#34D399,#A78BFA,#F472B6);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;}
.hero-sub{color:#94A3B8;font-size:.95rem;margin-top:.5rem;}
.kpi-card{background:linear-gradient(135deg,#111827,#1A2235);
          border:1px solid #1E293B;border-radius:14px;padding:1.2rem;
          text-align:center;transition:all .25s;}
.kpi-card:hover{transform:translateY(-3px);box-shadow:0 8px 24px rgba(59,130,246,.2);}
.kpi-val{font-size:2.2rem;font-weight:800;margin:0;}
.kpi-label{color:#94A3B8;font-size:.8rem;margin:.2rem 0 0 0;}
.badge{display:inline-block;background:linear-gradient(135deg,#1E3A5F,#163556);
       color:#60A5FA;padding:.18rem .65rem;border-radius:999px;font-size:.75rem;
       font-weight:600;margin:.15rem;border:1px solid #2563EB44;}
.card{background:#111827;border:1px solid #1E293B;border-radius:12px;
      padding:1.1rem 1.4rem;margin-bottom:.9rem;}
.risk-high{background:linear-gradient(135deg,#2D1515,#3D1919);border:2px solid #EF4444;
           border-radius:16px;padding:1.6rem;text-align:center;}
.risk-med{background:linear-gradient(135deg,#2B2112,#3A2E10);border:2px solid #F59E0B;
          border-radius:16px;padding:1.6rem;text-align:center;}
.risk-low{background:linear-gradient(135deg,#0D2B1E,#133526);border:2px solid #10B981;
          border-radius:16px;padding:1.6rem;text-align:center;}
.risk-label{font-size:2rem;font-weight:800;margin:0;}
.doc-card{background:#0F1A2E;border:1px solid #1E3A5F;border-radius:12px;
          padding:1.1rem;margin-top:.7rem;}
.q-box{background:#0D1627;border-left:3px solid #3B82F6;border-radius:8px;
       padding:.75rem 1rem;margin-bottom:.7rem;}
.divider{height:1px;background:linear-gradient(90deg,transparent,#3B82F6,transparent);
         margin:1.4rem 0;}
.feature-group{background:#0D1627;border:1px solid #1E3A5F;border-radius:12px;
               padding:1rem 1.2rem;margin-bottom:1rem;}
.group-title{color:#60A5FA;font-size:1rem;font-weight:700;margin-bottom:.8rem;}
/* Hide Streamlit menu/header/footer */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
""",unsafe_allow_html=True)

# ── Load resources ────────────────────────────────────────────
@st.cache_resource
def load_enhanced_model():
    with open(rpath("results","model_config_enhanced.json")) as f:
        cfg=json.load(f)
    m=TabMPLE.build(n_num=cfg['n_num'],n_bin=cfg['n_bin'],
                    n_bins=cfg['n_bins'],hidden_dim=cfg['hidden_dim'],
                    n_layers=cfg['n_layers'],k=cfg['k'],dropout=0.0)
    m.load_state_dict(torch.load(rpath("models","tabm_ple_best.pth"),map_location="cpu"))
    m.eval()
    return m,cfg

@st.cache_resource
def load_prep():
    return joblib.load(rpath("models","enhanced_preprocessor.pkl"))

@st.cache_data
def load_feat_meta():
    try:
        with open(rpath("dataset","feature_metadata.json")) as f:
            return json.load(f)
    except:
        return {}

@st.cache_data
def load_metrics():
    try:
        with open(rpath("results","metrics_enhanced.json")) as f: return json.load(f)
    except:
        try:
            with open(rpath("results","metrics.json")) as f: return json.load(f)
        except: return {}

model_ok=False
try:
    model,cfg=load_enhanced_model()
    prep=load_prep()
    scaler=prep['scaler']; feat_names=prep['features']
    num_idx=prep['num_idx']; bin_idx=prep['bin_idx']
    model_ok=True
except Exception as e:
    feat_names=[]; num_idx=[]; bin_idx=[]
    st.warning(f"Enhanced model not loaded: {e}")

feat_meta=load_feat_meta()
metrics=load_metrics()

# ── Feature groups for input form ─────────────────────────────
FEATURE_GROUPS = {
    "👤 Demographics": {
        "SC_AGE_YEARS": ("Age (Years)","slider",0,17,8),
        "SC_SEX":       ("Sex","radio",{1:"Male",0:"Female"},1),
        "SC_RACE_R":    ("Race","select",{1:"White",2:"Black",3:"Hispanic",4:"Asian",5:"Other"},1),
        "SC_HISPANIC_R":("Hispanic?","radio",{1:"Yes",0:"No"},0),
        "A1_GRADE":     ("Parent Education","select",{1:"< High School",2:"High School",3:"Some College",4:"Bachelor+"},3),
    },
    "🏥 Medical Conditions": {
        "ALLERGIES":  ("Allergies Diagnosed?","radio",{1:"Yes",0:"No"},0),
        "DIABETES":   ("Diabetes?","radio",{1:"Yes",0:"No"},0),
        "HEART":      ("Heart Condition?","radio",{1:"Yes",0:"No"},0),
        "DOWNSYN":    ("Down Syndrome?","radio",{1:"Yes",0:"No"},0),
        "CYSTFIB":    ("Cystic Fibrosis?","radio",{1:"Yes",0:"No"},0),
        "BLINDNESS":  ("Blindness?","radio",{1:"Yes",0:"No"},0),
        "K2Q40A":     ("Epilepsy/Seizures?","radio",{1:"Yes",0:"No"},0),
        "HEADACHE":   ("Chronic Headaches?","radio",{1:"Yes",0:"No"},0),
        "AUTOIMMUNE": ("Autoimmune Condition?","radio",{1:"Yes",0:"No"},0),
        "BREATHING":  ("Breathing Problem?","radio",{1:"Yes",0:"No"},0),
        "STOMACH":    ("Stomach/Digestive?","radio",{1:"Yes",0:"No"},0),
        "BLOOD":      ("Blood Disorder?","radio",{1:"Yes",0:"No"},0),
        "FASD":       ("Fetal Alcohol Spectrum?","radio",{1:"Yes",0:"No"},0),
        "OVERWEIGHT": ("Overweight?","radio",{1:"Yes",0:"No"},0),
        "K2Q42A":     ("Brain Injury?","radio",{1:"Yes",0:"No"},0),
    },
    "🧩 Comorbidities": {
        "K2Q31A": ("ADHD Diagnosed?","radio",{1:"Yes",0:"No"},0),
        "K2Q32A": ("Depression?","radio",{1:"Yes",0:"No"},0),
        "K2Q33A": ("Anxiety?","radio",{1:"Yes",0:"No"},0),
        "K2Q36A": ("Intellectual Disability?","radio",{1:"Yes",0:"No"},0),
        "K2Q37A": ("Tourette Syndrome?","radio",{1:"Yes",0:"No"},0),
        "K2Q60A": ("Behavioral/Conduct Problems?","radio",{1:"Yes",0:"No"},0),
    },
    "🗣️ Language Development": {
        "ONEWORD":    ("Said first word by 12 months?","radio",{1:"Yes",0:"No"},1),
        "TWOWORDS":   ("Said two-word phrases by 16 months?","radio",{1:"Yes",0:"No"},1),
        "THREEWORDS": ("Said 3-word sentences by 24 months?","radio",{1:"Yes",0:"No"},1),
        "ASKQUESTION":("Asks simple questions?","radio",{1:"Yes",0:"No"},1),
        "ASKQUESTION2":("Asks complex questions?","radio",{1:"Yes",0:"No"},1),
        "TELLSTORY":  ("Tells stories or describes events?","radio",{1:"Yes",0:"No"},1),
        "UNDERSTAND": ("Understands what you say?","radio",{1:"Yes",0:"No"},1),
        "DIRECTIONS": ("Follows 2-step directions?","radio",{1:"Yes",0:"No"},1),
        "POINT":      ("Points to show you things?","radio",{1:"Yes",0:"No"},1),
    },
    "🎯 Motor & Cognitive": {
        "BOUNCEABALL": ("Can bounce/catch a ball?","radio",{1:"Yes",0:"No"},1),
        "DRAWACIRCLE": ("Can draw a circle?","radio",{1:"Yes",0:"No"},1),
        "DRAWAPERSON": ("Can draw a person?","radio",{1:"Yes",0:"No"},1),
        "RECOGBEGIN":  ("Recognizes beginning sounds?","radio",{1:"Yes",0:"No"},1),
        "SAMESOUND":   ("Identifies rhyming words?","radio",{1:"Yes",0:"No"},1),
        "WRITENAME":   ("Writes own name?","radio",{1:"Yes",0:"No"},1),
        "FOCUSON":     ("Stays focused on tasks?","radio",{1:"Yes",0:"No"},1),
        "READONEDIGIT":("Reads single-digit numbers?","radio",{1:"Yes",0:"No"},1),
        "SIMPLEADDITION":("Does simple addition?","radio",{1:"Yes",0:"No"},1),
    },
    "👥 Social & Behavioral": {
        "TEMPER_R":   ("Has severe temper tantrums?","radio",{1:"Yes",0:"No"},0),
        "PLAYWELL":   ("Plays well with other children?","radio",{1:"Yes",0:"No"},1),
        "DISTRACTED": ("Easily distracted?","radio",{1:"Yes",0:"No"},0),
        "HURTSAD":    ("Often feels hurt or sad?","radio",{1:"Yes",0:"No"},0),
        "CALMDOWN_R": ("Calms down when upset?","radio",{1:"Yes",0:"No"},1),
        "WAITFORTURN":("Waits for turn in games?","radio",{1:"Yes",0:"No"},1),
        "HARDWORK":   ("Works hard at tasks?","radio",{1:"Yes",0:"No"},1),
        "SHARETOYS":  ("Shares toys with others?","radio",{1:"Yes",0:"No"},1),
        "MAKEFRIEND": ("Makes friends easily?","radio",{1:"Yes",0:"No"},1),
        "TALKABOUT":  ("Talks about feelings?","radio",{1:"Yes",0:"No"},1),
        "K7Q30":      ("Feels safe in neighborhood?","radio",{1:"Yes",0:"No"},1),
        "K7Q31":      ("Community members help each other?","radio",{1:"Yes",0:"No"},1),
    },
    "🏠 Socioeconomic & Family": {
        "HHCOUNT":      ("Household members count","slider",1,10,4),
        "FAMCOUNT":     ("Family members count","slider",1,10,4),
        "CURRCOV":      ("Currently insured?","radio",{1:"Yes",0:"No"},1),
        "TENURE":       ("Home ownership?","radio",{1:"Owned",0:"Rented"},1),
        "EVERHOMELESS": ("Ever been homeless?","radio",{1:"Yes",0:"No"},0),
        "MISSMORTGAGE": ("Missed mortgage/rent payment?","radio",{1:"Yes",0:"No"},0),
        "FPL_I1":       ("Federal Poverty Level (%)","slider",0,400,200),
        "BIRTHWT":      ("Birth weight category","select",{1:"Normal",2:"Low",3:"Very Low"},1),
        "BIRTHWT_VL":   ("Very low birth weight?","radio",{1:"Yes",0:"No"},0),
        "BORNUSA":      ("Born in USA?","radio",{1:"Yes",0:"No"},1),
        "HHLANGUAGE":   ("Household language","select",{1:"English",2:"Spanish",3:"Other"},1),
        "ACE1":         ("Parents divorced/separated?","radio",{1:"Yes",0:"No"},0),
        "ACE3":         ("Parent died?","radio",{1:"Yes",0:"No"},0),
        "ACE4":         ("Parent in jail?","radio",{1:"Yes",0:"No"},0),
    },
}

# ── M-CHAT-R questions ────────────────────────────────────────
MCHAT = [
    ("If you point at something, does your child look at it?","No"),
    ("Have you ever wondered if your child might be deaf?","Yes"),
    ("Does your child play pretend (e.g., drink from empty cup)?","No"),
    ("Does your child like climbing on things?","No"),
    ("Does your child make unusual finger movements near eyes?","Yes"),
    ("Does your child point to ask for something?","No"),
    ("Does your child point to show you something interesting?","No"),
    ("Is your child interested in other children?","No"),
    ("Does your child show you things by holding them up?","No"),
    ("Does your child respond to his/her name?","No"),
    ("When you smile, does your child smile back?","No"),
    ("Does your child get upset by everyday noises?","Yes"),
    ("Does your child walk?","No"),
    ("Does your child look you in the eye while talking?","No"),
    ("Does your child try to copy what you do?","No"),
    ("If you look at something, does your child look too?","No"),
    ("Does your child try to get you to watch him/her?","No"),
    ("Does your child understand simple instructions?","No"),
    ("Does your child look at your face to see how you feel?","No"),
    ("Does your child like movement activities (swinging)?","No"),
]

# ── Doctor recommendation engine ──────────────────────────────
def get_recommendation(ai_risk,mchat_score=None,mchat_done=False):
    combined = (0.6*ai_risk + 0.4*(mchat_score/20*100)) if mchat_done and mchat_score is not None else ai_risk
    if combined>=60:
        return {"level":"HIGH","color":"#EF4444","emoji":"🚨",
                "combined":round(combined,1),
                "urgency":"URGENT — Visit specialist within 1–2 weeks",
                "docs":[
                    ("🧠 Pediatric Neurologist","Evaluates brain development & neurological conditions"),
                    ("👨‍⚕️ Child Psychiatrist","Diagnoses ASD and developmental disorders"),
                    ("🏥 Developmental Pediatrician","Specialises in developmental delays"),
                    ("🗣️ Speech-Language Pathologist","Assesses communication development"),
                ],
                "hospitals":["NIMHANS, Bangalore","AIIMS, New Delhi","Child Dev. Centre, JIPMER, Puducherry",
                             "Institute of Child Health, Chennai","Kokilaben Hospital, Mumbai"],
                "tips":["Do NOT wait — early intervention before age 5 gives the best outcomes",
                        "Record your child's behavior on video (helps doctors enormously)",
                        "Contact the nearest government medical college immediately",
                        "Ask for a full developmental / ASD evaluation"]}
    elif combined>=30:
        return {"level":"MODERATE","color":"#F59E0B","emoji":"⚠️",
                "combined":round(combined,1),
                "urgency":"Schedule appointment within 1 month",
                "docs":[
                    ("🏥 Developmental Pediatrician","First point of contact for developmental concerns"),
                    ("👨‍⚕️ Child Psychologist","Behavioral assessment and early intervention"),
                    ("🗣️ Speech Therapist","Check communication milestones"),
                ],
                "hospitals":["Local Govt District Hospital — Pediatrics dept.",
                             "Any accredited child development centre in your city"],
                "tips":["Monitor milestones closely over the next 2–4 weeks",
                        "Encourage eye contact, play, and name-response at home",
                        "Discuss with your regular pediatrician first"]}
    else:
        return {"level":"LOW","color":"#10B981","emoji":"✅",
                "combined":round(combined,1),
                "urgency":"Continue routine pediatric checkups",
                "docs":[("👶 Regular Pediatrician","Continue routine developmental checkups")],
                "hospitals":["Your regular child doctor / family physician",
                             "Local Primary Health Centre (PHC)"],
                "tips":["Continue regular developmental monitoring",
                        "Encourage reading, social play, and eye contact",
                        "Re-screen at 18 and 24 months"]}

# ── Session state ─────────────────────────────────────────────
for key,default in [("ai_risk",None),("mchat_score",None),
                    ("mchat_done",False),("inputs",{})]:
    if key not in st.session_state: st.session_state[key]=default

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 ASD Detection")
    st.markdown("---")
    page = st.radio("Navigate",[
        "🔍 Step 1: Clinical Input",
        "📋 Step 2: M-CHAT-R Quiz",
        "🩺 Step 3: Result & Doctor",
        "📊 Dashboard",
        "📈 SHAP Explorer",
    ],label_visibility="collapsed")
    st.markdown("---")
    if metrics:
        st.markdown("**Enhanced Model Performance**")
        auc=metrics.get("ROC-AUC",metrics.get("ROC-AUC","N/A"))
        rec=metrics.get("Recall(Sens)","N/A")
        f1 =metrics.get("F1-Score","N/A")
        st.metric("🎯 ROC-AUC",auc)
        st.metric("🔍 Recall",rec)
        st.metric("📊 F1-Score",f1)
    st.markdown("---")
    s1="✅" if st.session_state.ai_risk is not None else "⬜"
    s2="✅" if st.session_state.mchat_done else "⬜"
    s3="✅" if st.session_state.ai_risk is not None else "⬜"
    st.markdown(f"{s1} Step 1 — Clinical Input (70 features)")
    st.markdown(f"{s2} Step 2 — M-CHAT-R (20 questions)")
    st.markdown(f"{s3} Step 3 — Result & Doctor")
    st.markdown("---")
    st.markdown("""<div style='background:#0D1627;border:1px solid #1E3A5F;
    border-radius:8px;padding:.8rem;font-size:.78rem;color:#94A3B8;'>
    <b>Model:</b> TabM-PLE (70 Features)<br>
    <b>Params:</b> 359,441<br>
    <b>AUC:</b> 0.9270<br>
    <b>Screening:</b> M-CHAT-R (Official)
    </div>""",unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">ASD Early Detection System</p>
    <p class="hero-sub">
        Screening using a TabM ensemble model with SHAP for explainability. The interface provides inputs, questionnaire support and model output with interpretation.
    </p>
</div>
""",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — CLINICAL INPUT (70 features in groups)
# ═══════════════════════════════════════════════════════════════
if "Step 1" in page:
    st.markdown("## 🔍 Step 1: Clinical Assessment (70 Features)")
    st.info("Complete all sections below. Feature groups mirror the clinical assessment categories used in NSCH 2023.")

    inputs = {}
    for group_name, fields in FEATURE_GROUPS.items():
        with st.expander(group_name, expanded=(group_name=="👤 Demographics")):
            st.markdown(f'<div class="group-title">{group_name}</div>',unsafe_allow_html=True)
            n_cols = 2 if len(fields) > 4 else 1
            cols = st.columns(n_cols)
            for ci,(feat,(label,ftype,*opts)) in enumerate(fields.items()):
                col = cols[ci % n_cols]
                with col:
                    if ftype == "slider":
                        mn,mx,dflt = opts
                        inputs[feat] = st.slider(label,mn,mx,dflt,key=f"f_{feat}")
                    elif ftype == "radio":
                        opt_dict,dflt = opts
                        inputs[feat] = st.radio(label,list(opt_dict.keys()),
                            format_func=lambda x,od=opt_dict:od[x],
                            index=list(opt_dict.keys()).index(dflt),
                            horizontal=True,key=f"f_{feat}")
                    elif ftype == "select":
                        opt_dict,dflt = opts
                        sel = st.selectbox(label,list(opt_dict.keys()),
                            format_func=lambda x,od=opt_dict:od[x],
                            index=list(opt_dict.keys()).index(dflt),
                            key=f"f_{feat}")
                        inputs[feat] = sel

    st.markdown("---")
    predict_btn = st.button("🔮 Run AI Prediction (TabM-PLE)",
                             type="primary",use_container_width=True)

    if predict_btn:
        if not model_ok:
            st.error("Model not loaded. Run `python src/05b_train_enhanced.py` first.")
        else:
            # Build feature vector in correct order
            x_raw = np.array([[inputs.get(f,0) for f in feat_names]],dtype=np.float32)
            x_sc  = x_raw.copy()
            x_sc[:,num_idx] = scaler.transform(x_raw[:,num_idx])

            xn = torch.tensor(x_sc[:,num_idx])
            xb = torch.tensor(x_sc[:,bin_idx])
            with torch.no_grad():
                prob = torch.sigmoid(model(xn,xb)).item()
            risk = round(prob*100,1)
            st.session_state.ai_risk = risk
            st.session_state.inputs  = inputs

            # Result display
            if risk>=60:
                st.markdown(f'<div class="risk-high"><p class="risk-label" style="color:#EF4444;">🚨 HIGH RISK</p><p style="color:#94A3B8;font-size:1.1rem;">AI Risk Score: <b style="color:#EF4444;font-size:2rem;">{risk}%</b></p></div>',unsafe_allow_html=True)
            elif risk>=30:
                st.markdown(f'<div class="risk-med"><p class="risk-label" style="color:#F59E0B;">⚠️ MODERATE RISK</p><p style="color:#94A3B8;font-size:1.1rem;">AI Risk Score: <b style="color:#F59E0B;font-size:2rem;">{risk}%</b></p></div>',unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low"><p class="risk-label" style="color:#10B981;">✅ LOW RISK</p><p style="color:#94A3B8;font-size:1.1rem;">AI Risk Score: <b style="color:#10B981;font-size:2rem;">{risk}%</b></p></div>',unsafe_allow_html=True)

            # Gauge
            gauge=go.Figure(go.Indicator(
                mode="gauge+number",value=risk,
                title={"text":"ASD Risk %","font":{"color":"#94A3B8","size":14}},
                gauge={"axis":{"range":[0,100],"tickcolor":"#94A3B8"},
                       "bar":{"color":"#EF4444" if risk>=60 else "#F59E0B" if risk>=30 else "#10B981"},
                       "bgcolor":"#1A2235","bordercolor":"#1E293B",
                       "steps":[{"range":[0,30],"color":"#0D2B1E"},
                                {"range":[30,60],"color":"#2B2112"},
                                {"range":[60,100],"color":"#2D1515"}],
                       "threshold":{"line":{"color":"white","width":3},"thickness":.8,"value":50}},
                number={"suffix":"%","font":{"color":"#F1F5F9","size":32}},
            ))
            gauge.update_layout(height=260,margin=dict(l=20,r=20,t=30,b=10),
                                paper_bgcolor="#111827",font_color="#F1F5F9")
            st.plotly_chart(gauge,use_container_width=True)
            st.success("✅ Done! Now complete **Step 2: M-CHAT-R** for a stronger combined score.")

    elif st.session_state.ai_risk is not None:
        st.metric("Previous AI Risk Score",f"{st.session_state.ai_risk}%")
        st.info("Scroll up and click **Run AI Prediction** again if you changed any values.")

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — M-CHAT-R
# ═══════════════════════════════════════════════════════════════
elif "Step 2" in page:
    st.markdown("## 📋 Step 2: M-CHAT-R Clinical Questionnaire")
    st.markdown("""<div class="card">
    <b>M-CHAT-R (Modified Checklist for Autism in Toddlers, Revised)</b><br>
    The official 20-question clinical ASD screening tool used by pediatricians worldwide.
    Takes ~3 minutes. Answer based on your child's <b>typical behavior</b>.
    </div>""",unsafe_allow_html=True)

    if st.session_state.ai_risk is None:
        st.warning("⚠️ Please complete Step 1 first.")

    answers=[]
    for i,(q,risk_ans) in enumerate(MCHAT):
        st.markdown(f'<div class="q-box"><b>Q{i+1}.</b> {q}</div>',unsafe_allow_html=True)
        ans=st.radio("",["Yes","No"],horizontal=True,key=f"mchat_{i}",label_visibility="collapsed")
        answers.append(ans)

    if st.button("📊 Calculate M-CHAT-R Score",type="primary",use_container_width=True):
        score=sum(1 for (q,ra),a in zip(MCHAT,answers) if a==ra)
        st.session_state.mchat_score=score
        st.session_state.mchat_done=True

        if score<=2: color="#10B981"; label="LOW RISK"
        elif score<=7: color="#F59E0B"; label="MEDIUM RISK"
        else: color="#EF4444"; label="HIGH RISK"

        c1,c2=st.columns([1,2])
        with c1:
            st.markdown(f"""<div style='text-align:center;background:#1A2235;
            border:2px solid {color};border-radius:14px;padding:1.5rem;'>
            <p style='font-size:3rem;font-weight:800;color:{color};margin:0;'>{score}/20</p>
            <p style='color:{color};font-weight:700;font-size:1.1rem;'>{label}</p>
            </div>""",unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
| Score | Risk | Action |
|-------|------|--------|
| 0–2 | 🟢 Low | Routine monitoring |
| 3–7 | 🟡 Medium | Follow-up needed |
| 8–20 | 🔴 High | Refer to specialist |

**Your score: `{score}/20` — {label}**""")
        st.success("✅ Go to **Step 3: Result & Doctor Advice**")

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — COMBINED RESULT + DOCTOR
# ═══════════════════════════════════════════════════════════════
elif "Step 3" in page:
    st.markdown("## 🩺 Step 3: Combined Result & Doctor Recommendation")
    if st.session_state.ai_risk is None:
        st.error("❌ Complete Step 1 first."); st.stop()

    rec=get_recommendation(st.session_state.ai_risk,
                           st.session_state.mchat_score,
                           st.session_state.mchat_done)
    box={"HIGH":"risk-high","MODERATE":"risk-med","LOW":"risk-low"}[rec["level"]]

    st.markdown(f"""<div class="{box}">
    <p class="risk-label" style="color:{rec['color']};">{rec['emoji']} {rec['level']} RISK</p>
    <p style="color:#94A3B8;font-size:1.1rem;margin:.4rem 0;">
      Combined Score: <b style="color:{rec['color']};font-size:2rem;">{rec['combined']}%</b></p>
    <p style="color:{rec['color']};font-weight:600;">{rec['urgency']}</p>
    </div>""",unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)

    c1,c2,c3=st.columns(3)
    c1.metric("🤖 AI Score (TabM-PLE)",f"{st.session_state.ai_risk}%")
    if st.session_state.mchat_done:
        c2.metric("📋 M-CHAT-R",f"{st.session_state.mchat_score}/20")
        c3.metric("🔀 Combined",f"{rec['combined']}%")
    else:
        c2.info("M-CHAT-R not done")

    # Bar chart
    src=["AI (TabM-PLE)"]
    vals=[st.session_state.ai_risk]
    if st.session_state.mchat_done:
        src.append("M-CHAT-R Clinical")
        vals.append(round(st.session_state.mchat_score/20*100,1))
    src.append("Combined"); vals.append(rec['combined'])
    clrs=["#EF4444" if v>=60 else "#F59E0B" if v>=30 else "#10B981" for v in vals]
    fig=go.Figure(go.Bar(x=src,y=vals,marker_color=clrs,marker_line_color="white",
                         marker_line_width=1.5,text=[f"{v}%" for v in vals],textposition="outside"))
    fig.add_hline(y=30,line_dash="dash",line_color="#F59E0B",annotation_text="Moderate threshold")
    fig.add_hline(y=60,line_dash="dash",line_color="#EF4444",annotation_text="High threshold")
    fig.update_layout(height=280,paper_bgcolor="#111827",plot_bgcolor="#1A2235",
                      font_color="#F1F5F9",yaxis_range=[0,115],showlegend=False)
    st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown("### 👨‍⚕️ Recommended Specialists")
    for icon_name,why in rec["docs"]:
        st.markdown(f"""<div class="doc-card">
        <h4 style="color:#60A5FA;margin:0;">{icon_name}</h4>
        <p style="color:#94A3B8;margin:.3rem 0 0 0;">{why}</p>
        </div>""",unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown("### 🏥 Suggested Hospitals")
    for i,h in enumerate(rec["hospitals"],1):
        st.markdown(f"**{i}.** {h}")

    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown("### 🏠 What To Do Right Now")
    for t in rec["tips"]:
        st.markdown(f"✅ {t}")

    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    col_h1,col_h2=st.columns(2)
    with col_h1:
        st.markdown("""<div class="doc-card">
        <b>🇮🇳 Autism Society of India</b><br>
        <a href="https://autismsocietyofindia.org" style="color:#60A5FA;">autismsocietyofindia.org</a><br><br>
        <b>📞 iCall Helpline</b><br>
        <span style="color:#10B981;">9152987821</span>
        </div>""",unsafe_allow_html=True)
    with col_h2:
        st.markdown("""<div class="doc-card">
        <b>🏥 NIMHANS Bangalore</b><br>
        <span style="color:#94A3B8;">080-46110007</span><br><br>
        <b>🏥 AIIMS OPD</b><br>
        <span style="color:#94A3B8;">011-26588500</span>
        </div>""",unsafe_allow_html=True)

    # Download report
    report=f"""
ASD EARLY DETECTION REPORT (TabM-PLE Enhanced)
===============================================
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Model: TabM-PLE (70 Features, AUC=0.927)

SCORES:
  AI Model (TabM-PLE) : {st.session_state.ai_risk}%
  M-CHAT-R            : {f'{st.session_state.mchat_score}/20' if st.session_state.mchat_done else 'Not done'}
  Combined Risk       : {rec['combined']}%

RISK LEVEL : {rec['level']}
ACTION     : {rec['urgency']}

SPECIALISTS RECOMMENDED:
{chr(10).join([f'  - {n}: {w}' for n,w in rec['docs']])}

HOSPITALS:
{chr(10).join([f'  {i+1}. {h}' for i,h in enumerate(rec['hospitals'])])}

HOME TIPS:
{chr(10).join([f'  - {t}' for t in rec['tips']])}

IMPORTANT: This is a SCREENING TOOL, not a diagnosis.
Always consult a qualified medical professional.
================================================
Generated by ASD Detection System v3.0
TabM-PLE + SHAP + M-CHAT-R | NSCH 2023 Dataset
"""
    st.download_button("📥 Download Full Report",data=report.encode(),
                       file_name="asd_screening_report_enhanced.txt",
                       mime="text/plain",use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — DASHBOARD
# ═══════════════════════════════════════════════════════════════
elif "Dashboard" in page:
    st.markdown("## 📊 Performance Dashboard — TabM-PLE (70 Features)")

    if metrics:
        kpis=[("ROC-AUC","#3B82F6"),("F1-Score","#10B981"),
              ("Recall(Sens)","#8B5CF6"),("Accuracy","#F59E0B"),("MCC","#14B8A6")]
        cols=st.columns(5)
        for col,(k,c) in zip(cols,kpis):
            v=metrics.get(k,"N/A")
            col.markdown(f"""<div class="kpi-card">
            <p class="kpi-val" style="color:{c};">{v}</p>
            <p class="kpi-label">{k}</p>
            </div>""",unsafe_allow_html=True)

    st.markdown("---")

    # Comparison: Old vs New
    st.markdown("### 🆚 Before vs After Upgrade")
    comp_data={
        "Metric":["ROC-AUC","Recall","F1-Score","Params","Features"],
        "Old (7 features)":["0.6648","0.6816","0.1172","11,849","7"],
        "New (70 features + PLE)":
            [str(metrics.get("ROC-AUC","0.9270")),
             str(metrics.get("Recall(Sens)","—")),
             str(metrics.get("F1-Score","—")),
             "359,441","70"],
    }
    comp_df=pd.DataFrame(comp_data)
    st.dataframe(comp_df,use_container_width=True,hide_index=True)

    st.markdown("---")
    plots={
        "Training Curves":"results/plots/30_tabmple_training_curves.png",
        "ROC Curve (Enhanced)":"results/plots/32_enh_roc_curve.png",
        "Confusion Matrix":"results/plots/31_enh_confusion_matrix.png",
        "PR Curve":"results/plots/33_enh_pr_curve.png",
        "SHAP Importance":"results/plots/35_enh_shap_bar.png",
        "SHAP Beeswarm":"results/plots/34_enh_shap_beeswarm.png",
        "Original ROC":"results/plots/15_roc_curve.png",
        "Model Comparison":"results/plots/24_model_comparison.png",
    }
    keys=list(plots.keys())
    for i in range(0,len(keys),2):
        c1,c2=st.columns(2)
        for col,idx in [(c1,i),(c2,i+1)]:
            if idx<len(keys):
                fp=rpath(plots[keys[idx]])
                if os.path.exists(fp):
                    col.markdown(f"**{keys[idx]}**")
                    col.image(fp,use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 5 — SHAP EXPLORER
# ═══════════════════════════════════════════════════════════════
elif "SHAP" in page:
    st.markdown("## 📈 SHAP Explainability — Top 20 Features")

    tabs=st.tabs(["Beeswarm (Top 20)","Bar Chart","Waterfall — ASD"])
    shap_files=[
        "results/plots/34_enh_shap_beeswarm.png",
        "results/plots/35_enh_shap_bar.png",
        "results/plots/36_enh_shap_waterfall_asd.png",
    ]
    for tab,fp in zip(tabs,shap_files):
        with tab:
            full=rpath(fp)
            if os.path.exists(full): st.image(full,use_container_width=True)
            else: st.info("Run `python src/07b_shap_enhanced.py` to generate this plot.")

    si=rpath("results/shap_feature_importance_enhanced.csv")
    if os.path.exists(si):
        st.markdown("---")
        st.markdown("### 📊 Top 20 Feature Importance Table (SHAP)")
        df_si=pd.read_csv(si)
        st.dataframe(df_si[['Rank','Feature','Label','Mean|SHAP|']].head(20),
                     use_container_width=True,hide_index=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div style="text-align:center;color:#475569;font-size:.78rem;padding:.4rem;">
🧠 <b>ASD Detection System v3.0</b> — TabM-PLE (70 Features) + SHAP + M-CHAT-R &nbsp;|&nbsp;
Final Year B.Tech IT Project &nbsp;|&nbsp;
<span style="color:#EF4444;">⚠️ Screening tool only — not a medical diagnosis. Consult a doctor.</span>
</div>""",unsafe_allow_html=True)
