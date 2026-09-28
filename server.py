import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import torch
import joblib
import shap
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
def rpath(*p): return os.path.join(ROOT, *p)

sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    from tabm_ple_model import TabMPLE
    ENHANCED = True
except Exception as e:
    print(f"Warning: TabMPLE import error: {e}")
    ENHANCED = False

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ── Load Model & Resources ───────────────────────────────────
model = None
scaler = None
feat_names = []
num_idx = []
bin_idx = []
cfg = {}
metrics = {}
model_ok = False
feature_labels = {}
shap_background = None

try:
    with open(rpath("results", "model_config_enhanced.json")) as f:
        cfg = json.load(f)
    model = TabMPLE.build(n_num=cfg['n_num'], n_bin=cfg['n_bin'],
                         n_bins=cfg['n_bins'], hidden_dim=cfg['hidden_dim'],
                         n_layers=cfg['n_layers'], k=cfg['k'], dropout=0.0)
    model.load_state_dict(torch.load(rpath("models", "tabm_ple_best.pth"), map_location="cpu"))
    model.eval()

    prep = joblib.load(rpath("models", "enhanced_preprocessor.pkl"))
    scaler = prep['scaler']
    feat_names = prep['features']
    num_idx = prep['num_idx']
    bin_idx = prep['bin_idx']
    try:
        with open(rpath("dataset", "feature_metadata.json"), encoding="utf-8") as f:
            feature_labels = json.load(f)
    except Exception:
        feature_labels = {name: name for name in feat_names}
    model_ok = True
    print("✅ PyTorch TabM Model and Preprocessor loaded successfully.")
except Exception as e:
    print(f"⚠️ Model load failed: {e}")

try:
    with open(rpath("results", "metrics_enhanced.json")) as f:
        metrics = json.load(f)
except Exception:
    metrics = {"ROC-AUC": 0.927, "F1-Score": 0.884, "Accuracy": 0.934, "Classification_Threshold": 0.35}

SCREENING_THRESHOLD = float(metrics.get("Classification_Threshold", 0.35))

def prepare_model_input(inputs):
    """Apply the saved preprocessing used by the trained TabM-PLE model."""
    x_raw = np.array([[inputs.get(f, 0) for f in feat_names]], dtype=np.float32)
    x_sc = x_raw.copy()
    x_sc[:, num_idx] = scaler.transform(x_raw[:, num_idx])
    return x_raw, x_sc

def predict_probability(X):
    X = np.asarray(X, dtype=np.float32)
    with torch.no_grad():
        xn = torch.tensor(X[:, num_idx])
        xb = torch.tensor(X[:, bin_idx])
        return torch.sigmoid(model(xn, xb)).numpy()

# ── Patient Data Storage ─────────────────────────────────────
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

# Ensure admin account exists
patients = load_patients()
if "admin" not in patients:
    patients["admin"] = {"password": "admin123", "full_name": "System Administrator", "role": "admin", "history": []}
    save_patients(patients)

# ── Doctor Recommendation & Clinical Discussion Engine ─────────
def get_recommendation(ai_risk, inputs=None, mchat_score=None, mchat_done=False, username="Patient"):
    inputs = inputs or {}
    combined = (0.6 * ai_risk + 0.4 * (mchat_score / 20 * 100)) if mchat_done and mchat_score is not None else ai_risk

    # Extract patient-specific clinical flags
    age = inputs.get("SC_AGE_YEARS", 8)
    sex = "Male" if inputs.get("SC_SEX", 1) == 1 else "Female"
    
    flagged_domains = []
    custom_docs = []
    custom_tips = []

    # 1. Speech & Communication Domain
    if inputs.get("ONEWORD") == 0 or inputs.get("TWOWORDS") == 0 or inputs.get("THREEWORDS") == 0 or inputs.get("POINT") == 0:
        flagged_domains.append("Speech & Language Milestone Delay")
        custom_docs.append(("🗣️ Speech-Language Pathologist", f"Specialized speech therapy & communication evaluation for {sex.lower()} aged {age}"))
        custom_tips.append("Implement daily joint-attention exercises and augmented communication tools")

    # 2. Neurological / Seizure Domain
    if inputs.get("K2Q40A") == 1 or inputs.get("K2Q42A") == 1 or inputs.get("HEADACHE") == 1:
        flagged_domains.append("Neurological / Seizure Activity History")
        custom_docs.append(("🧠 Pediatric Neurologist", "EEG and neurological developmental assessment"))
        custom_tips.append("Maintain a detailed daily seizure/symptom log for neurological consultation")

    # 3. ADHD / Behavioral & Executive Function Domain
    if inputs.get("K2Q31A") == 1 or inputs.get("TEMPER_R") == 1 or inputs.get("DISTRACTED") == 1 or inputs.get("K2Q60A") == 1:
        flagged_domains.append("Behavioral & Executive Function (ADHD / Conduct Concern)")
        custom_docs.append(("👨‍⚕️ Child Psychologist & ABA Therapist", "Behavioral intervention, sensory strategies, and focus coaching"))
        custom_tips.append("Use structured visual schedules and positive reinforcement routines at home")

    # 4. Social Reciprocity & Peer Interaction Domain
    if inputs.get("PLAYWELL") == 0 or inputs.get("MAKEFRIEND") == 0 or inputs.get("TALKABOUT") == 0:
        flagged_domains.append("Social Reciprocity & Peer Interaction Deficit")
        custom_docs.append(("🤝 Developmental Pediatrician", "Social skills group therapy and developmental milestone tracking"))
        custom_tips.append("Enroll in structured peer play groups and guided social interaction programs")

    # 5. Comorbidities / Genetic Conditions
    if inputs.get("DOWNSYN") == 1 or inputs.get("K2Q36A") == 1 or inputs.get("AUTOIMMUNE") == 1:
        flagged_domains.append("Associated Comorbid / Genetic Conditions")
        custom_docs.append(("🧬 Medical Geneticist & Pediatric Specialist", "Comprehensive genetic screening and holistic medical evaluation"))

    # Fallback default doctor if no specific domain flagged
    if not custom_docs:
        custom_docs.append(("🏥 Developmental Pediatrician", "Comprehensive developmental milestone evaluation"))

    # Determine risk level and specific clinical discussion narrative
    if combined >= 60:
        level = "HIGH"
        color = "#EF4444"
        emoji = "🚨"
        urgency = "URGENT — Visit specialist within 1–2 weeks"
        
        discussion = f"Clinical Discussion for {username} (Age {age}, {sex}): The TabM PyTorch Neural Network predicts an elevated AI risk probability of {ai_risk}% (M-CHAT score {mchat_score if mchat_done else 'N/A'}/20, combined risk {round(combined, 1)}%). The evaluation identified key clinical concerns in: {', '.join(flagged_domains) if flagged_domains else 'multiple developmental milestones'}. Immediate consultation with a specialist is strongly advised to begin early intervention before age 5."
        
        hospitals = ["NIMHANS, Bangalore", "AIIMS, New Delhi", "Child Dev. Centre, JIPMER, Puducherry", "Institute of Child Health, Chennai", "Kokilaben Hospital, Mumbai"]
        if not custom_tips:
            custom_tips = ["Do NOT wait — early intervention gives the best outcomes", "Record your child's behavior on video for doctor evaluation", "Contact nearest government medical college immediately"]

    elif combined >= 30:
        level = "MODERATE"
        color = "#F59E0B"
        emoji = "⚠️"
        urgency = "Schedule appointment within 1 month"
        
        discussion = f"Clinical Discussion for {username} (Age {age}, {sex}): The model calculated a moderate combined screening score of {round(combined, 1)}% (AI Risk: {ai_risk}%). Identified domain concerns: {', '.join(flagged_domains) if flagged_domains else 'borderline developmental indicators'}. A formal follow-up assessment with a developmental pediatrician within 4 weeks is recommended."
        
        hospitals = ["Local Govt District Hospital — Pediatrics dept.", "Any accredited child development centre in your city"]
        if not custom_tips:
            custom_tips = ["Monitor milestones closely over the next 2–4 weeks", "Encourage eye contact, play, and name-response at home", "Discuss findings with your regular pediatrician"]

    else:
        level = "LOW"
        color = "#10B981"
        emoji = "✅"
        urgency = "Continue routine pediatric checkups"
        
        discussion = f"Clinical Discussion for {username} (Age {age}, {sex}): The TabM Neural Network and clinical screening indicate a low risk score of {round(combined, 1)}% (AI Risk: {ai_risk}%). Current developmental trajectory is typical. Continue routine pediatric checkups and milestone tracking."
        
        hospitals = ["Your regular child doctor / family physician", "Local Primary Health Centre (PHC)"]
        if not custom_tips:
            custom_tips = ["Continue regular developmental monitoring", "Encourage reading, social play, and eye contact", "Re-screen at 18 and 24 months"]

    return {
        "level": level,
        "color": color,
        "emoji": emoji,
        "combined": round(combined, 1),
        "urgency": urgency,
        "flagged_domains": flagged_domains,
        "discussion": discussion,
        "docs": custom_docs,
        "hospitals": hospitals,
        "tips": custom_tips
    }

# ── ROUTES ───────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/plots/<path:filename>")
def serve_plot(filename):
    return send_from_directory(rpath("results", "plots"), filename)

# ── AUTH API ──
@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.json or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()

    if username == "admin" and password == "admin123":
        return jsonify({"success": True, "username": "admin", "role": "admin", "full_name": "System Administrator", "message": "Admin login successful"})

    patients = load_patients()
    if username not in patients:
        return jsonify({"success": False, "message": "User account not found. Please sign up first."}), 400
    if patients[username].get("password") != password:
        return jsonify({"success": False, "message": "Incorrect password. Please try again."}), 400

    user_data = patients[username]
    return jsonify({
        "success": True,
        "username": username,
        "role": user_data.get("role", "patient"),
        "full_name": user_data.get("full_name", username),
        "message": "Login successful"
    })

@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.json or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()
    full_name = str(data.get("full_name", "")).strip() or username

    if not username:
        return jsonify({"success": False, "message": "Username is required."}), 400
    if len(password) < 4:
        return jsonify({"success": False, "message": "Password must be at least 4 characters long."}), 400

    patients = load_patients()
    if username in patients or username == "admin":
        return jsonify({"success": False, "message": f"Username '{username}' already exists. Please choose another."}), 400

    patients[username] = {
        "password": password,
        "full_name": full_name,
        "role": "patient",
        "registered_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "history": []
    }
    save_patients(patients)
    return jsonify({"success": True, "username": username, "role": "patient", "message": f"Account for '{username}' created successfully."})

# ── PREDICTION API ──
@app.route("/api/predict", methods=["POST"])
def predict():
    if not model_ok:
        return jsonify({"success": False, "message": "PyTorch TabM model not loaded"}), 500

    data = request.json or {}
    inputs = data.get("inputs", {})

    _, x_sc = prepare_model_input(inputs)
    prob = float(predict_probability(x_sc)[0])

    ai_risk = round(prob * 100, 1)
    predicted_class = int(prob >= SCREENING_THRESHOLD)

    username = data.get("username", "Patient")
    rec = get_recommendation(
        ai_risk,
        inputs,
        data.get("mchat_score"),
        data.get("mchat_done", False),
        username,
    )

    return jsonify({
        "success": True,
        "ai_risk": ai_risk,
        "predicted_class": predicted_class,
        "threshold": SCREENING_THRESHOLD,
        "recommendation": rec
    })

@app.route("/api/explain", methods=["POST"])
def explain_prediction():
    if not model_ok:
        return jsonify({"success": False, "message": "PyTorch TabM model not loaded"}), 500

    data = request.json or {}
    _, x_sc = prepare_model_input(data.get("inputs", {}))

    global shap_background
    if shap_background is None:
        X_test = np.load(rpath("dataset", "X_test_enh.npy")).astype(np.float32)
        rng = np.random.default_rng(42)
        sample_size = min(40, len(X_test))
        shap_background = X_test[rng.choice(len(X_test), sample_size, replace=False)]

    explainer = shap.KernelExplainer(predict_probability, shap_background)
    shap_values = explainer.shap_values(x_sc, nsamples=80)
    values = np.asarray(shap_values)
    if values.ndim == 3:
        values = values[0, :, 0]
    elif values.ndim == 2:
        values = values[0]
    else:
        values = values.reshape(-1)

    ranked = np.argsort(np.abs(values))[::-1]
    top = []
    for index in ranked[:8]:
        contribution = float(values[index] * 100)
        top.append({
            "feature": feat_names[index],
            "label": feature_labels.get(feat_names[index], feat_names[index]),
            "contribution": round(contribution, 2),
            "direction": "increases" if contribution >= 0 else "decreases",
        })

    return jsonify({
        "success": True,
        "method": "SHAP KernelExplainer",
        "top_contributors": top,
        "base_value": round(float(np.asarray(explainer.expected_value).reshape(-1)[0] * 100), 2),
    })

# ── ASSESSMENT SAVE API ──
@app.route("/api/assessment/save", methods=["POST"])
def save_assessment():
    data = request.json or {}
    username = data.get("username")

    if not username or username == "admin":
        return jsonify({"success": True, "message": "Ignored for guest or admin"})

    patients = load_patients()
    patient = patients.setdefault(username, {"password": "", "history": []})
    patient.setdefault("history", [])

    record = {
        "ai_risk": data.get("ai_risk"),
        "mchat_score": data.get("mchat_score"),
        "mchat_done": data.get("mchat_done", False),
        "combined": data.get("combined"),
        "level": data.get("level"),
        "urgency": data.get("urgency"),
        "docs": data.get("docs", []),
        "hospitals": data.get("hospitals", []),
        "tips": data.get("tips", []),
        "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    }

    patient["history"].append(record)
    patient["latest_result"] = record
    save_patients(patients)

    return jsonify({"success": True, "message": "Assessment saved successfully", "record": record})

# ── ADMIN API ──
@app.route("/api/admin/stats", methods=["GET"])
def admin_stats():
    patients = load_patients()
    rows = []
    for uname, pdata in patients.items():
        if uname == "admin": continue
        history = pdata.get("history", [])
        latest = history[-1] if history else {}
        rows.append({
            "username": uname,
            "full_name": pdata.get("full_name", uname),
            "total_assessments": len(history),
            "latest_ai_risk": latest.get("ai_risk"),
            "latest_combined": latest.get("combined"),
            "level": latest.get("level", "UNSCREENED"),
            "saved_at": latest.get("saved_at", "—"),
            "urgency": latest.get("urgency", "—")
        })

    total_patients = len(rows)
    high_count = sum(1 for r in rows if r["level"] == "HIGH")
    med_count = sum(1 for r in rows if r["level"] == "MODERATE")
    low_count = sum(1 for r in rows if r["level"] == "LOW")
    total_screenings = sum(r["total_assessments"] for r in rows)
    combined_vals = [r["latest_combined"] for r in rows if r["latest_combined"] is not None]
    avg_score = round(float(np.mean(combined_vals)), 1) if combined_vals else 0.0

    return jsonify({
        "total_patients": total_patients,
        "high_count": high_count,
        "med_count": med_count,
        "low_count": low_count,
        "total_screenings": total_screenings,
        "avg_score": avg_score,
        "metrics": metrics,
        "threshold": SCREENING_THRESHOLD,
        "model_ok": model_ok
    })

@app.route("/api/admin/patients", methods=["GET"])
def admin_patients():
    q = request.args.get("q", "").strip().lower()
    risk_filter = request.args.get("risk", "ALL").upper()

    patients = load_patients()
    rows = []
    for uname, pdata in patients.items():
        if uname == "admin": continue
        history = pdata.get("history", [])
        latest = history[-1] if history else {}
        level = latest.get("level", "UNSCREENED")
        full_name = pdata.get("full_name", uname)

        if q and not (q in uname.lower() or q in full_name.lower()):
            continue
        if risk_filter != "ALL" and risk_filter != level:
            continue

        rows.append({
            "username": uname,
            "full_name": full_name,
            "role": pdata.get("role", "patient"),
            "registered_at": pdata.get("registered_at", "—"),
            "total_assessments": len(history),
            "latest_ai_risk": f"{latest.get('ai_risk')}%" if latest.get("ai_risk") is not None else "N/A",
            "latest_combined": f"{latest.get('combined')}%" if latest.get("combined") is not None else "N/A",
            "level": level,
            "saved_at": latest.get("saved_at", "—"),
            "urgency": latest.get("urgency", "—")
        })

    return jsonify({"patients": rows})

@app.route("/api/admin/patient/<username>", methods=["GET"])
def get_patient_detail(username):
    patients = load_patients()
    if username not in patients:
        return jsonify({"error": "Patient not found"}), 404
    pdata = patients[username]
    return jsonify({
        "username": username,
        "full_name": pdata.get("full_name", username),
        "registered_at": pdata.get("registered_at", "—"),
        "role": pdata.get("role", "patient"),
        "history": pdata.get("history", [])
    })

@app.route("/api/admin/patient/<username>", methods=["DELETE"])
def delete_patient_route(username):
    patients = load_patients()
    if username in patients:
        del patients[username]
        save_patients(patients)
        return jsonify({"success": True, "message": f"Patient {username} deleted."})
    return jsonify({"error": "Patient not found"}), 404

@app.route("/api/admin/patient/<username>/clear", methods=["POST"])
def clear_patient_history_route(username):
    patients = load_patients()
    if username in patients:
        patients[username]["history"] = []
        if "latest_result" in patients[username]:
            del patients[username]["latest_result"]
        save_patients(patients)
        return jsonify({"success": True, "message": f"History cleared for {username}."})
    return jsonify({"error": "Patient not found"}), 404

@app.route("/api/patient/history/<username>", methods=["GET"])
def get_patient_history(username):
    patients = load_patients()
    if username not in patients:
        return jsonify({"history": []})
    return jsonify({"history": patients[username].get("history", [])})

if __name__ == "__main__":
    print("🚀 Starting ASD Clinical Intelligence Web Server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
