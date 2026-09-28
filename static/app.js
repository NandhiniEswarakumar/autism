/* ASD Clinical Intelligence Portal — Frontend Application Logic */

// Global State
let currentUser = null;
let currentRole = null;
let currentAuthMode = 'login';
let currentActiveView = 'view-step1';

let patientInputs = {};
let aiRiskScore = null;
let mchatAnswers = Array(20).fill('Yes');
let mchatScore = null;
let mchatDone = false;
let latestRecommendation = null;

// Feature Groups Definition (70 Clinical Parameters)
const FEATURE_GROUPS = {
    "👤 Demographics": {
        "SC_AGE_YEARS": ["Age (Years)", "slider", [0, 100], 8],
        "SC_SEX": ["Sex", "radio", {1: "Male", 0: "Female"}, 1],
        "SC_RACE_R": ["Race", "select", {1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "Other"}, 1],
        "SC_HISPANIC_R": ["Hispanic?", "radio", {1: "Yes", 0: "No"}, 0],
        "A1_GRADE": ["Parent Education", "select", {1: "< High School", 2: "High School", 3: "Some College", 4: "Bachelor+"}, 3],
    },
    "🏥 Medical Conditions": {
        "ALLERGIES": ["Allergies Diagnosed?", "radio", {1: "Yes", 0: "No"}, 0],
        "DIABETES": ["Diabetes?", "radio", {1: "Yes", 0: "No"}, 0],
        "HEART": ["Heart Condition?", "radio", {1: "Yes", 0: "No"}, 0],
        "DOWNSYN": ["Down Syndrome?", "radio", {1: "Yes", 0: "No"}, 0],
        "CYSTFIB": ["Cystic Fibrosis?", "radio", {1: "Yes", 0: "No"}, 0],
        "BLINDNESS": ["Blindness?", "radio", {1: "Yes", 0: "No"}, 0],
        "K2Q40A": ["Epilepsy/Seizures?", "radio", {1: "Yes", 0: "No"}, 0],
        "HEADACHE": ["Chronic Headaches?", "radio", {1: "Yes", 0: "No"}, 0],
        "AUTOIMMUNE": ["Autoimmune Condition?", "radio", {1: "Yes", 0: "No"}, 0],
        "BREATHING": ["Breathing Problem?", "radio", {1: "Yes", 0: "No"}, 0],
        "STOMACH": ["Stomach/Digestive?", "radio", {1: "Yes", 0: "No"}, 0],
        "BLOOD": ["Blood Disorder?", "radio", {1: "Yes", 0: "No"}, 0],
        "FASD": ["Fetal Alcohol Spectrum?", "radio", {1: "Yes", 0: "No"}, 0],
        "OVERWEIGHT": ["Overweight?", "radio", {1: "Yes", 0: "No"}, 0],
        "K2Q42A": ["Brain Injury?", "radio", {1: "Yes", 0: "No"}, 0],
    },
    "🧩 Comorbidities": {
        "K2Q31A": ["ADHD Diagnosed?", "radio", {1: "Yes", 0: "No"}, 0],
        "K2Q32A": ["Depression?", "radio", {1: "Yes", 0: "No"}, 0],
        "K2Q33A": ["Anxiety?", "radio", {1: "Yes", 0: "No"}, 0],
        "K2Q36A": ["Intellectual Disability?", "radio", {1: "Yes", 0: "No"}, 0],
        "K2Q37A": ["Tourette Syndrome?", "radio", {1: "Yes", 0: "No"}, 0],
        "K2Q60A": ["Behavioral/Conduct Problems?", "radio", {1: "Yes", 0: "No"}, 0],
    },
    "🗣️ Language Development": {
        "ONEWORD": ["Said first word by 12 months?", "radio", {1: "Yes", 0: "No"}, 1],
        "TWOWORDS": ["Said two-word phrases by 16 months?", "radio", {1: "Yes", 0: "No"}, 1],
        "THREEWORDS": ["Said 3-word sentences by 24 months?", "radio", {1: "Yes", 0: "No"}, 1],
        "ASKQUESTION": ["Asks simple questions?", "radio", {1: "Yes", 0: "No"}, 1],
        "ASKQUESTION2": ["Asks complex questions?", "radio", {1: "Yes", 0: "No"}, 1],
        "TELLSTORY": ["Tells stories or describes events?", "radio", {1: "Yes", 0: "No"}, 1],
        "UNDERSTAND": ["Understands what you say?", "radio", {1: "Yes", 0: "No"}, 1],
        "DIRECTIONS": ["Follows 2-step directions?", "radio", {1: "Yes", 0: "No"}, 1],
        "POINT": ["Points to show you things?", "radio", {1: "Yes", 0: "No"}, 1],
    },
    "🎯 Motor & Cognitive": {
        "BOUNCEABALL": ["Can bounce/catch a ball?", "radio", {1: "Yes", 0: "No"}, 1],
        "DRAWACIRCLE": ["Can draw a circle?", "radio", {1: "Yes", 0: "No"}, 1],
        "DRAWAPERSON": ["Can draw a person?", "radio", {1: "Yes", 0: "No"}, 1],
        "RECOGBEGIN": ["Recognizes beginning sounds?", "radio", {1: "Yes", 0: "No"}, 1],
        "SAMESOUND": ["Identifies rhyming words?", "radio", {1: "Yes", 0: "No"}, 1],
        "WRITENAME": ["Writes own name?", "radio", {1: "Yes", 0: "No"}, 1],
        "FOCUSON": ["Stays focused on tasks?", "radio", {1: "Yes", 0: "No"}, 1],
        "READONEDIGIT": ["Reads single-digit numbers?", "radio", {1: "Yes", 0: "No"}, 1],
        "SIMPLEADDITION": ["Does simple addition?", "radio", {1: "Yes", 0: "No"}, 1],
    },
    "👥 Social & Behavioral": {
        "TEMPER_R": ["Has severe temper tantrums?", "radio", {1: "Yes", 0: "No"}, 0],
        "PLAYWELL": ["Plays well with other children?", "radio", {1: "Yes", 0: "No"}, 1],
        "DISTRACTED": ["Easily distracted?", "radio", {1: "Yes", 0: "No"}, 0],
        "HURTSAD": ["Often feels hurt or sad?", "radio", {1: "Yes", 0: "No"}, 0],
        "CALMDOWN_R": ["Calms down when upset?", "radio", {1: "Yes", 0: "No"}, 1],
        "WAITFORTURN": ["Waits for turn in games?", "radio", {1: "Yes", 0: "No"}, 1],
        "HARDWORK": ["Works hard at tasks?", "radio", {1: "Yes", 0: "No"}, 1],
        "SHARETOYS": ["Shares toys with others?", "radio", {1: "Yes", 0: "No"}, 1],
        "MAKEFRIEND": ["Makes friends easily?", "radio", {1: "Yes", 0: "No"}, 1],
        "TALKABOUT": ["Talks about feelings?", "radio", {1: "Yes", 0: "No"}, 1],
        "K7Q30": ["Feels safe in neighborhood?", "radio", {1: "Yes", 0: "No"}, 1],
        "K7Q31": ["Community members help each other?", "radio", {1: "Yes", 0: "No"}, 1],
    }
};

const MCHAT_QUESTIONS = [
    ["If you point at something, does your child look at it?", "No"],
    ["Have you ever wondered if your child might be deaf?", "Yes"],
    ["Does your child play pretend (e.g., drink from empty cup)?", "No"],
    ["Does your child like climbing on things?", "No"],
    ["Does your child make unusual finger movements near eyes?", "Yes"],
    ["Does your child point to ask for something?", "No"],
    ["Does your child point to show you something interesting?", "No"],
    ["Is your child interested in other children?", "No"],
    ["Does your child show you things by holding them up?", "No"],
    ["Does your child respond to his/her name?", "No"],
    ["When you smile, does your child smile back?", "No"],
    ["Does your child get upset by everyday noises?", "Yes"],
    ["Does your child walk?", "No"],
    ["Does your child look you in the eye while talking?", "No"],
    ["Does your child try to copy what you do?", "No"],
    ["If you look at something, does your child look too?", "No"],
    ["Does your child try to get you to watch him/her?", "No"],
    ["Does your child understand simple instructions?", "No"],
    ["Does your child look at your face to see how you feel?", "No"],
    ["Does your child like movement activities (swinging)?", "No"],
];

// DOM Initialization
document.addEventListener('DOMContentLoaded', () => {
    initAuthToggles();
    initLoginForm();
    initSignupForm();
    initDemoLoginButtons();
    initNavButtons();
    renderClinicalForm();
    renderMchatForm();
    initAdminSubtabs();
    initAdminSearchFilter();
    
    // Check session
    const savedUser = localStorage.getItem('asd_username');
    const savedRole = localStorage.getItem('asd_role');
    if (savedUser && savedRole) {
        setUserSession(savedUser, savedRole, localStorage.getItem('asd_fullname') || savedUser);
    }
});

// AUTH TOGGLES
function initAuthToggles() {
    const btnLogin = document.getElementById('btn-toggle-login');
    const btnSignup = document.getElementById('btn-toggle-signup');
    const boxLogin = document.getElementById('form-box-login');
    const boxSignup = document.getElementById('form-box-signup');

    btnLogin.addEventListener('click', () => {
        btnLogin.classList.add('active');
        btnSignup.classList.remove('active');
        boxLogin.style.display = 'block';
        boxSignup.style.display = 'none';
        currentAuthMode = 'login';
    });

    btnSignup.addEventListener('click', () => {
        btnSignup.classList.add('active');
        btnLogin.classList.remove('active');
        boxSignup.style.display = 'block';
        boxLogin.style.display = 'none';
        currentAuthMode = 'signup';
    });
}

// LOGIN FORM
function initLoginForm() {
    const form = document.getElementById('form-login');
    const alertBox = document.getElementById('login-alert');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        alertBox.style.display = 'none';
        
        const username = document.getElementById('login-username').value.trim();
        const password = document.getElementById('login-password').value.trim();

        try {
            const res = await fetch('/api/auth/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({username, password})
            });
            const data = await res.json();
            
            if (data.success) {
                setUserSession(data.username, data.role, data.full_name);
            } else {
                alertBox.textContent = data.message || 'Login failed';
                alertBox.style.display = 'block';
            }
        } catch (err) {
            alertBox.textContent = 'Server connection error. Is server.py running?';
            alertBox.style.display = 'block';
        }
    });
}

// SIGNUP FORM
function initSignupForm() {
    const form = document.getElementById('form-signup');
    const alertBox = document.getElementById('signup-alert');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        alertBox.style.display = 'none';

        const full_name = document.getElementById('signup-fullname').value.trim();
        const username = document.getElementById('signup-username').value.trim();
        const password = document.getElementById('signup-password').value.trim();
        const confirm = document.getElementById('signup-confirm').value.trim();

        if (password !== confirm) {
            alertBox.textContent = 'Passwords do not match. Please verify.';
            alertBox.style.display = 'block';
            return;
        }

        try {
            const res = await fetch('/api/auth/signup', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({username, password, full_name})
            });
            const data = await res.json();

            if (data.success) {
                alert('Account created successfully! Please log in.');
                document.getElementById('btn-toggle-login').click();
                document.getElementById('login-username').value = username;
            } else {
                alertBox.textContent = data.message || 'Registration failed';
                alertBox.style.display = 'block';
            }
        } catch (err) {
            alertBox.textContent = 'Server connection error.';
            alertBox.style.display = 'block';
        }
    });
}

// LOGIN SHORTCUTS
function initDemoLoginButtons() {
    document.getElementById('btn-demo-patient').addEventListener('click', async () => {
        try {
            await fetch('/api/auth/signup', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({username: 'DemoPatient', password: 'demo123', full_name: 'Demo Patient'})
            });
        } catch (e) {}
        setUserSession('DemoPatient', 'patient', 'Demo Patient');
    });

    document.getElementById('btn-demo-admin').addEventListener('click', () => {
        setUserSession('admin', 'admin', 'System Administrator');
    });
}

// SESSION MANAGEMENT
function setUserSession(username, role, fullName) {
    currentUser = username;
    currentRole = role;
    
    localStorage.setItem('asd_username', username);
    localStorage.setItem('asd_role', role);
    localStorage.setItem('asd_fullname', fullName || username);

    document.getElementById('header-user-name').textContent = `👤 ${fullName || username}`;
    const rolePill = document.getElementById('header-role-pill');
    rolePill.textContent = role.toUpperCase();
    rolePill.className = (role === 'admin') ? 'role-pill-admin' : 'role-pill-patient';
    
    document.getElementById('user-header-profile').style.display = 'flex';
    document.getElementById('view-auth').style.display = 'none';
    document.getElementById('workspace-container').style.display = 'block';

    const navAdmin = document.getElementById('nav-btn-admin');
    if (role === 'admin') {
        navAdmin.style.display = 'inline-flex';
        switchView('view-admin');
    } else {
        navAdmin.style.display = 'none';
        switchView('view-step1');
    }

    loadDashboardData();
}

// LOGOUT
document.getElementById('btn-header-logout').addEventListener('click', () => {
    currentUser = null;
    currentRole = null;
    localStorage.removeItem('asd_username');
    localStorage.removeItem('asd_role');
    localStorage.removeItem('asd_fullname');

    document.getElementById('user-header-profile').style.display = 'none';
    document.getElementById('workspace-container').style.display = 'none';
    document.getElementById('view-auth').style.display = 'block';
});

// NAVIGATION TABS ROUTER
function initNavButtons() {
    const navBtns = document.querySelectorAll('.nav-btn');
    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            navBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const targetView = btn.getAttribute('data-view');
            switchView(targetView);
        });
    });
}

function switchView(viewId) {
    currentActiveView = viewId;
    const views = document.querySelectorAll('.workspace-view');
    views.forEach(v => v.style.display = 'none');
    
    const target = document.getElementById(viewId);
    if (target) target.style.display = 'block';

    if (viewId === 'view-admin') loadAdminData();
    if (viewId === 'view-dashboard') loadDashboardData();
    if (viewId === 'view-shap') loadPersonalShapView();
}

// RENDER STEP 1: CLINICAL FORM (70 FEATURES)
function renderClinicalForm() {
    const container = document.getElementById('clinical-form-container');
    container.innerHTML = '';

    for (const [groupName, fields] of Object.entries(FEATURE_GROUPS)) {
        const groupCard = document.createElement('div');
        groupCard.className = 'feature-group-card';
        
        const title = document.createElement('div');
        title.className = 'group-title';
        title.textContent = groupName;
        groupCard.appendChild(title);

        const grid = document.createElement('div');
        grid.className = 'fields-grid';

        for (const [featKey, spec] of Object.entries(fields)) {
            const [label, ftype, opts, dflt] = spec;
            const fieldBox = document.createElement('div');
            fieldBox.className = 'form-group';

            const lbl = document.createElement('label');
            lbl.textContent = label;
            fieldBox.appendChild(lbl);

            if (ftype === 'slider') {
                const [min, max] = opts;
                const slider = document.createElement('input');
                slider.type = 'range';
                slider.min = min;
                slider.max = max;
                slider.value = dflt;
                slider.className = 'form-control';
                slider.id = `f_${featKey}`;
                
                const valDisplay = document.createElement('span');
                valDisplay.style.fontSize = '0.9rem';
                valDisplay.style.color = '#60A5FA';
                valDisplay.textContent = ` ${dflt}`;
                
                slider.addEventListener('input', () => {
                    valDisplay.textContent = ` ${slider.value}`;
                    patientInputs[featKey] = parseFloat(slider.value);
                });

                patientInputs[featKey] = dflt;
                fieldBox.appendChild(slider);
                fieldBox.appendChild(valDisplay);
            } 
            else if (ftype === 'radio') {
                const pillGroup = document.createElement('div');
                pillGroup.className = 'radio-pill-group';

                for (const [val, optText] of Object.entries(opts)) {
                    const pill = document.createElement('div');
                    pill.className = `radio-pill ${parseInt(val) === dflt ? 'active' : ''}`;
                    pill.textContent = optText;
                    
                    pill.addEventListener('click', () => {
                        pillGroup.querySelectorAll('.radio-pill').forEach(p => p.classList.remove('active'));
                        pill.classList.add('active');
                        patientInputs[featKey] = parseInt(val);
                    });

                    pillGroup.appendChild(pill);
                }

                patientInputs[featKey] = dflt;
                fieldBox.appendChild(pillGroup);
            }
            else if (ftype === 'select') {
                const sel = document.createElement('select');
                sel.className = 'form-control';
                sel.id = `f_${featKey}`;

                for (const [val, optText] of Object.entries(opts)) {
                    const opt = document.createElement('option');
                    opt.value = val;
                    opt.textContent = optText;
                    if (parseInt(val) === dflt) opt.selected = true;
                    sel.appendChild(opt);
                }

                sel.addEventListener('change', () => {
                    patientInputs[featKey] = parseInt(sel.value);
                });

                patientInputs[featKey] = dflt;
                fieldBox.appendChild(sel);
            }

            grid.appendChild(fieldBox);
        }

        groupCard.appendChild(grid);
        container.appendChild(groupCard);
    }

    // RUN AI PREDICTION BUTTON
    document.getElementById('btn-run-ai-prediction').addEventListener('click', runAiPrediction);
}

function renderShapExplanation(explanation) {
    if (!explanation || !explanation.success || !explanation.top_contributors) {
        return '<p class="explanation-note">Personalized SHAP explanation is currently unavailable. The prediction above is still available for clinical review.</p>';
    }

    const contributors = explanation.top_contributors.map(item => {
        const color = item.contribution >= 0 ? '#FCA5A5' : '#86EFAC';
        const sign = item.contribution >= 0 ? '+' : '';
        return `<li><span>${item.label}</span><b style="color:${color};">${sign}${item.contribution}%</b></li>`;
    }).join('');

    return `
        <div class="shap-patient-explanation">
            <h3>🧠 Why did the AI make this prediction?</h3>
            <p>SHAP explains how each patient feature moved the AI prediction relative to the model baseline. Positive values increased the estimated ASD risk; negative values reduced it.</p>
            <ul>${contributors}</ul>
            <p class="explanation-note">Method: ${explanation.method}. SHAP contributions describe model behavior; they do not prove that a feature caused ASD.</p>
        </div>
    `;
}

async function loadPersonalShapView() {
    const container = document.getElementById('personal-shap-content');
    if (!container) return;

    container.innerHTML = '<div class="alert alert-success">⏳ Calculating this patient\'s personalized SHAP explanation...</div>';
    try {
        const res = await fetch('/api/explain', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({inputs: patientInputs})
        });
        const explanation = await res.json();
        if (!explanation.success) {
            container.innerHTML = `<div class="alert alert-danger">${explanation.message || 'Personalized explanation unavailable.'}</div>`;
            return;
        }
        container.innerHTML = `
            <h2 class="personal-shap-title">Personalized Explanation for ${currentUser || 'Current Patient'}</h2>
            <p class="explanation-intro">These contributions are recalculated from this patient's current clinical answers. They will differ when the patient's answers differ.</p>
            ${renderShapExplanation(explanation)}
        `;
    } catch (error) {
        container.innerHTML = '<div class="alert alert-danger">Personalized SHAP explanation is currently unavailable.</div>';
    }
}

// RUN AI PREDICTION ENDPOINT
async function runAiPrediction() {
    const resultBox = document.getElementById('step1-result-box');
    resultBox.style.display = 'block';
    resultBox.innerHTML = '<div class="alert alert-success">⏳ Running PyTorch TabM Ensemble Model Inference...</div>';

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                inputs: patientInputs,
                mchat_score: mchatScore,
                mchat_done: mchatDone
            })
        });
        const data = await res.json();

        if (data.success) {
            aiRiskScore = data.ai_risk;
            latestRecommendation = data.recommendation;

            resultBox.innerHTML = '<div class="alert alert-success">✅ Prediction complete. Calculating personalized SHAP explanation...</div>';
            let explanation = null;
            try {
                const explanationRes = await fetch('/api/explain', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({inputs: patientInputs})
                });
                explanation = await explanationRes.json();
            } catch (explanationError) {}
            latestRecommendation.shap_explanation = explanation;

            saveCurrentAssessment();

            const level = data.recommendation.level;
            const color = data.recommendation.color;

            resultBox.innerHTML = `
                <div class="risk-${level.toLowerCase()}">
                    <p class="risk-label" style="color:${color};">${data.recommendation.emoji} ${level} RISK ASSESSMENT</p>
                    <p style="font-size:1.1rem;color:#94A3B8;margin-top:0.4rem;">
                        TabM Neural Network Risk Probability: <b style="color:${color};font-size:2.2rem;">${aiRiskScore}%</b>
                    </p>
                    <p style="color:${color};font-weight:700;">${data.recommendation.urgency}</p>
                </div>
                ${renderShapExplanation(explanation)}
                <div class="alert alert-success" style="margin-top:1.2rem;">
                    ✅ Step 1 Complete! Now proceed to <b>Step 2: M-CHAT-R Quiz</b> for a combined clinical score.
                </div>
            `;
        } else {
            resultBox.innerHTML = `<div class="alert alert-danger">Error: ${data.message}</div>`;
        }
    } catch (err) {
        resultBox.innerHTML = '<div class="alert alert-danger">Server error running prediction.</div>';
    }
}

// RENDER STEP 2: M-CHAT-R QUESTIONNAIRE
function renderMchatForm() {
    const container = document.getElementById('mchat-questions-container');
    container.innerHTML = '';

    MCHAT_QUESTIONS.forEach(([qText, riskAns], idx) => {
        const qCard = document.createElement('div');
        qCard.className = 'q-card';

        const textDiv = document.createElement('div');
        textDiv.innerHTML = `<b>Q${idx + 1}.</b> ${qText}`;
        qCard.appendChild(textDiv);

        const pillGroup = document.createElement('div');
        pillGroup.className = 'radio-pill-group';

        ['Yes', 'No'].forEach(choice => {
            const pill = document.createElement('div');
            pill.className = `radio-pill ${choice === 'Yes' ? 'active' : ''}`;
            pill.textContent = choice;

            pill.addEventListener('click', () => {
                pillGroup.querySelectorAll('.radio-pill').forEach(p => p.classList.remove('active'));
                pill.classList.add('active');
                mchatAnswers[idx] = choice;
            });

            pillGroup.appendChild(pill);
        });

        qCard.appendChild(pillGroup);
        container.appendChild(qCard);
    });

    document.getElementById('btn-calc-mchat').addEventListener('click', calculateMchatScore);
}

// CALCULATE M-CHAT SCORE
function calculateMchatScore() {
    let score = 0;
    MCHAT_QUESTIONS.forEach(([qText, riskAns], idx) => {
        if (mchatAnswers[idx] === riskAns) score++;
    });

    mchatScore = score;
    mchatDone = true;

    const resultBox = document.getElementById('step2-result-box');
    resultBox.style.display = 'block';

    let color = '#10B981', label = 'LOW RISK';
    if (score >= 8) { color = '#EF4444'; label = 'HIGH RISK'; }
    else if (score >= 3) { color = '#F59E0B'; label = 'MEDIUM RISK'; }

    resultBox.innerHTML = `
        <div style="display:flex;gap:2rem;align-items:center;background:#0F172A;border:2px solid ${color};border-radius:18px;padding:1.8rem;margin-top:1rem;">
            <div style="text-align:center;min-width:140px;">
                <p style="font-size:3.5rem;font-weight:800;color:${color};margin:0;">${score}/20</p>
                <p style="color:${color};font-weight:700;font-size:1.1rem;margin:0;">${label}</p>
            </div>
            <div>
                <h3>M-CHAT-R Screening Result: ${score}/20</h3>
                <p style="color:#94A3B8;margin-top:0.4rem;">The score is calculated based on clinical risk indicators. Proceed to Step 3 for your personalized combined recommendation.</p>
            </div>
        </div>
        <div class="alert alert-success" style="margin-top:1.2rem;">
            ✅ M-CHAT-R complete! Click <b>Step 3: Result & Doctor</b> in the navigation bar.
        </div>
    `;

    if (aiRiskScore !== null) {
        runAiPrediction();
    }
}

// SAVE ASSESSMENT TO BACKEND
async function saveCurrentAssessment() {
    if (!currentUser || currentUser === 'admin' || !latestRecommendation) return;

    try {
        await fetch('/api/assessment/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                username: currentUser,
                ai_risk: aiRiskScore,
                mchat_score: mchatScore,
                mchat_done: mchatDone,
                combined: latestRecommendation.combined,
                level: latestRecommendation.level,
                urgency: latestRecommendation.urgency,
                docs: latestRecommendation.docs,
                hospitals: latestRecommendation.hospitals,
                tips: latestRecommendation.tips
            })
        });
        renderStep3View();
    } catch (e) {}
}

// RENDER STEP 3: RESULT & DOCTOR VIEW
function renderStep3View() {
    const container = document.getElementById('step3-content');
    if (!aiRiskScore && !latestRecommendation) {
        container.innerHTML = '<div class="alert alert-danger">❌ Please complete Step 1 (Clinical Assessment) first.</div>';
        return;
    }

    const rec = latestRecommendation;
    const levelClass = `risk-${rec.level.toLowerCase()}`;

    let flaggedHtml = (rec.flagged_domains && rec.flagged_domains.length > 0)
        ? rec.flagged_domains.map(d => `<span class="feature-pill" style="background:rgba(239, 68, 68, 0.15);color:#FCA5A5;border:1px solid rgba(239, 68, 68, 0.3);">⚠️ ${d}</span>`).join(' ')
        : '<span class="feature-pill" style="background:rgba(16, 185, 129, 0.15);color:#6EE7B7;border:1px solid rgba(16, 185, 129, 0.3);">✅ Typical Developmental Trajectory</span>';

    let docsHtml = rec.docs.map(([docName, desc]) => `
        <div class="doc-card">
            <h4 style="color:#60A5FA;margin:0;">${docName}</h4>
            <p style="color:#94A3B8;margin:.3rem 0 0 0;">${desc}</p>
        </div>
    `).join('');

    let hospHtml = rec.hospitals.map((h, i) => `<li><b>${i+1}.</b> ${h}</li>`).join('');
    let tipsHtml = rec.tips.map(t => `<li>✅ ${t}</li>`).join('');
    const mchatPercent = mchatDone ? (mchatScore / 20 * 100) : null;
    const scoreExplanation = mchatDone
        ? `(${aiRiskScore}% × 60%) + (${mchatScore}/20 = ${mchatPercent.toFixed(1)}% × 40%) = ${rec.combined}%`
        : `${aiRiskScore}% AI screening result (M-CHAT-R not completed)`;
    const explanationDomains = rec.flagged_domains && rec.flagged_domains.length > 0
        ? rec.flagged_domains.map(domain => `<li>⚠️ ${domain}</li>`).join('')
        : '<li>✅ No predefined risk domains were flagged by the clinical answers.</li>';
    const shapHtml = renderShapExplanation(rec.shap_explanation);
    const highRiskExplanation = rec.level === 'HIGH' ? `
        <div class="high-risk-explanation">
            <h3>🚨 Why is this result HIGH risk?</h3>
            <p>This screening result crossed the high-risk threshold because the AI prediction, the M-CHAT-R responses, or both identified multiple developmental or behavioral indicators. The specific indicators found in this assessment are:</p>
            <ul class="explanation-list">${explanationDomains}</ul>
            <p><b>What this can mean:</b> These findings may be associated with developmental, communication, social-interaction, behavioral, or medical concerns that need professional assessment. They do not confirm autism or any other disease by themselves.</p>
            <p><b>What is most urgent:</b> A high score does not mean that one disease is more dangerous than another. It means the child should receive a timely evaluation by a qualified developmental pediatrician or appropriate specialist. Neurological symptoms, loss of previously gained skills, seizures, or immediate safety concerns should be discussed with a doctor promptly.</p>
            <p class="explanation-note"><b>Next step:</b> Share this result and the completed answers with a qualified healthcare professional. Do not use this screening score alone to make a diagnosis or treatment decision.</p>
        </div>
    ` : '';

    container.innerHTML = `
        <div class="${levelClass}">
            <p class="risk-label" style="color:${rec.color};">${rec.emoji} ${rec.level} RISK ASSESSMENT</p>
            <p style="color:#94A3B8;font-size:1.1rem;margin:0.4rem 0;">
                Combined Multi-Modal Risk Score: <b style="color:${rec.color};font-size:2.4rem;">${rec.combined}%</b>
            </p>
            <p style="color:${rec.color};font-weight:700;font-size:1.1rem;">${rec.urgency}</p>
        </div>

        <!-- PERSONALIZED CLINICAL DISCUSSION -->
        <div class="card-box" style="margin-top:1.5rem;background:#0F172A;border:1px solid #334155;border-radius:18px;padding:1.8rem;">
            <h3 style="color:#60A5FA;margin-bottom:0.8rem;">📋 Person-Specific Clinical Discussion Narrative</h3>
            <p style="color:#E2E8F0;font-size:1.02rem;line-height:1.7;">${rec.discussion || 'Individual clinical findings calculated based on 70 feature parameters.'}</p>
            <div style="margin-top:1.2rem;">
                <b style="color:#CBD5E1;display:block;margin-bottom:0.5rem;">Identified Risk Flag Domains for this Patient:</b>
                <div style="display:flex;flex-wrap:wrap;gap:0.5rem;">${flaggedHtml}</div>
            </div>
        </div>

        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:2rem 0;">
            <div class="kpi-card"><p class="kpi-val text-blue">${aiRiskScore}%</p><p class="kpi-label">AI Risk (TabM Net)</p></div>
            <div class="kpi-card"><p class="kpi-val text-amber">${mchatDone ? mchatScore + '/20' : 'Pending'}</p><p class="kpi-label">M-CHAT Checklist</p></div>
            <div class="kpi-card"><p class="kpi-val text-purple">${rec.combined}%</p><p class="kpi-label">Final Combined Risk</p></div>
        </div>

        <div class="card-box explanation-card">
            <h3>🔎 Why did the system give this result?</h3>
            <p class="explanation-intro">The final percentage combines the TabM-PLE AI prediction with the M-CHAT-R clinical checklist. This shows the calculation used for this screening:</p>
            <div class="score-formula">${scoreExplanation}</div>
            <div class="explanation-columns">
                <div>
                    <h4>Clinical indicators considered</h4>
                    <ul class="explanation-list">${explanationDomains}</ul>
                </div>
                <div>
                    <h4>How to interpret it</h4>
                    <p class="explanation-copy">The AI score is a probability learned from the 70 clinical features. The M-CHAT-R score reflects the number of risk-indicator answers out of 20 questions. The final category is based on the combined score: below 30% is low, 30% to below 60% is moderate, and 60% or above is high risk.</p>
                </div>
            </div>
            <p class="explanation-note">The SHAP Explorer provides the model's global feature-importance plots. This result is a screening aid, not a medical diagnosis; a qualified clinician must make the final assessment.</p>
        </div>

        ${highRiskExplanation}

        ${shapHtml}

        <h3 style="margin-top:2rem;">👨‍⚕️ Targeted Specialist Referrals (Based on Patient Flags)</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">${docsHtml}</div>

        <h3 style="margin-top:2rem;">🏥 Accredited Healthcare Centers</h3>
        <ul style="list-style:none;padding:0;line-height:2;">${hospHtml}</ul>

        <h3 style="margin-top:2rem;">🏠 Personalized Action Plan</h3>
        <ul style="list-style:none;padding:0;line-height:2;">${tipsHtml}</ul>

        <div style="margin-top:2.5rem;">
            <button id="btn-download-report" class="btn btn-primary btn-block btn-lg">📥 Download Official Personalized Screening Report (.txt)</button>
        </div>
    `;

    document.getElementById('btn-download-report').addEventListener('click', downloadReport);
}

// REPORT DOWNLOAD BUILDER
function downloadReport() {
    if (!latestRecommendation) return;
    const rec = latestRecommendation;

    const reportText = `
ASD CLINICAL DETECTION & ASSESSMENT REPORT
===================================================
Patient Username: ${currentUser || 'Guest'}
Assessment Date:  ${new Date().toLocaleString()}
Architecture:     TabM PyTorch Neural Network + M-CHAT-R + SHAP

SCORES & METRICS:
  AI Model (TabM-PLE) : ${aiRiskScore}%
  M-CHAT-R Checklist  : ${mchatDone ? mchatScore + '/20' : 'Not completed'}
  Combined Risk Score : ${rec.combined}%

CLINICAL CLASSIFICATION: ${rec.level} RISK
ACTION STATUS:          ${rec.urgency}

RECOMMENDED SPECIALISTS:
${rec.docs.map(([n, w]) => `  - ${n}: ${w}`).join('\n')}

ACCREDITED HOSPITALS:
${rec.hospitals.map((h, i) => `  ${i+1}. ${h}`).join('\n')}

ACTION PLAN:
${rec.tips.map(t => `  - ${t}`).join('\n')}

===================================================
DISCLAIMER: Generated by an AI screening system for informational purposes only.
Not a clinical diagnosis. Always consult a licensed pediatrician.
===================================================
`;

    const blob = new Blob([reportText], {type: 'text/plain'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ASD_Screening_Report_${currentUser || 'Patient'}.txt`;
    a.click();
    URL.revokeObjectURL(url);
}

// LOAD DASHBOARD DATA
async function loadDashboardData() {
    if (!currentUser) return;

    try {
        const res = await fetch(`/api/patient/history/${currentUser}`);
        const data = await res.json();
        const history = data.history || [];
        const container = document.getElementById('dashboard-content');

        if (history.length === 0) {
            container.innerHTML = `
                <div class="alert alert-success">
                    No history saved yet. Complete Step 1 (Clinical Assessment) to view your personalized dashboard timeline.
                </div>
            `;
            return;
        }

        const latest = history[history.length - 1];
        let historyRows = history.map((h, idx) => `
            <tr>
                <td>#${idx + 1}</td>
                <td>${h.saved_at}</td>
                <td>${h.ai_risk}%</td>
                <td>${h.mchat_score !== null ? h.mchat_score + '/20' : 'N/A'}</td>
                <td><b>${h.combined}%</b></td>
                <td><span class="badge-risk badge-${h.level.toLowerCase().slice(0,3)}">${h.level}</span></td>
            </tr>
        `).join('');

        container.innerHTML = `
            <h3>Latest Assessment Result</h3>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:2rem;">
                <div class="kpi-card"><p class="kpi-val text-blue">${latest.ai_risk}%</p><p class="kpi-label">Latest AI Risk</p></div>
                <div class="kpi-card"><p class="kpi-val text-purple">${latest.combined}%</p><p class="kpi-label">Latest Combined Score</p></div>
                <div class="kpi-card"><p class="kpi-val text-green">${latest.level}</p><p class="kpi-label">Risk Level</p></div>
            </div>

            <h3>Screening History Timeline</h3>
            <div class="table-responsive">
                <table class="table-custom">
                    <thead>
                        <tr>
                            <th>Assessment</th>
                            <th>Date</th>
                            <th>AI Risk</th>
                            <th>M-CHAT</th>
                            <th>Combined Score</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>${historyRows}</tbody>
                </table>
            </div>
        `;
    } catch (e) {}
}

// ADMIN SUBTABS ROUTER
function initAdminSubtabs() {
    const subBtns = document.querySelectorAll('.subtab-btn');
    subBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            subBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            document.getElementById('admin-subtab-directory').style.display = 'none';
            document.getElementById('admin-subtab-create').style.display = 'none';
            document.getElementById('admin-subtab-specs').style.display = 'none';

            if (btn.id === 'subtab-btn-directory') document.getElementById('admin-subtab-directory').style.display = 'block';
            if (btn.id === 'subtab-btn-create') document.getElementById('admin-subtab-create').style.display = 'block';
            if (btn.id === 'subtab-btn-specs') document.getElementById('admin-subtab-specs').style.display = 'block';
        });
    });

    // Admin Create Form
    document.getElementById('form-admin-add-patient').addEventListener('submit', async (e) => {
        e.preventDefault();
        const alertBox = document.getElementById('admin-create-alert');
        alertBox.style.display = 'none';

        const full_name = document.getElementById('admin-add-fullname').value.trim();
        const username = document.getElementById('admin-add-username').value.trim();
        const password = document.getElementById('admin-add-password').value.trim();

        try {
            const res = await fetch('/api/auth/signup', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({username, password, full_name})
            });
            const data = await res.json();
            if (data.success) {
                alert(`Patient '${username}' registered successfully!`);
                document.getElementById('form-admin-add-patient').reset();
                document.getElementById('subtab-btn-directory').click();
                loadAdminData();
            } else {
                alertBox.textContent = data.message;
                alertBox.style.display = 'block';
            }
        } catch (err) {}
    });
}

// ADMIN SEARCH & FILTER
function initAdminSearchFilter() {
    document.getElementById('admin-search-input').addEventListener('input', loadAdminPatientsTable);
    document.getElementById('admin-risk-select').addEventListener('change', loadAdminPatientsTable);
}

// LOAD ADMIN DATA
async function loadAdminData() {
    try {
        const statsRes = await fetch('/api/admin/stats');
        const stats = await statsRes.json();

        document.getElementById('kpi-total-patients').textContent = stats.total_patients;
        document.getElementById('kpi-high-risk').textContent = stats.high_count;
        document.getElementById('kpi-mod-risk').textContent = stats.med_count;
        document.getElementById('kpi-low-risk').textContent = stats.low_count;
        document.getElementById('kpi-total-screenings').textContent = stats.total_screenings;
        document.getElementById('kpi-avg-score').textContent = stats.avg_score + '%';

        loadAdminPatientsTable();
    } catch (e) {}
}

async function loadAdminPatientsTable() {
    const q = document.getElementById('admin-search-input').value.trim();
    const risk = document.getElementById('admin-risk-select').value;

    try {
        const res = await fetch(`/api/admin/patients?q=${encodeURIComponent(q)}&risk=${encodeURIComponent(risk)}`);
        const data = await res.json();
        const tbody = document.getElementById('table-patients-body');

        if (!data.patients || data.patients.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-center">No matching patient records found.</td></tr>';
            return;
        }

        tbody.innerHTML = data.patients.map(p => `
            <tr>
                <td><b>${p.username}</b></td>
                <td>${p.full_name}</td>
                <td>${p.total_assessments}</td>
                <td>${p.latest_ai_risk}</td>
                <td><b>${p.latest_combined}</b></td>
                <td><span class="badge-risk badge-${p.level.toLowerCase().slice(0,3)}">${p.level}</span></td>
                <td>${p.saved_at}</td>
                <td>
                    <button class="btn btn-secondary btn-sm" onclick="inspectPatient('${p.username}')">🔎 Inspect</button>
                </td>
            </tr>
        `).join('');
    } catch (e) {}
}

// INSPECT PATIENT DETAILS DRAWER
async function inspectPatient(username) {
    const card = document.getElementById('patient-inspector-card');
    card.style.display = 'block';
    card.scrollIntoView({behavior: 'smooth'});

    try {
        const res = await fetch(`/api/admin/patient/${username}`);
        const data = await res.json();

        document.getElementById('inspect-patient-title').textContent = `👤 Patient: ${data.username} (${data.full_name})`;
        document.getElementById('inspect-patient-sub').textContent = `Registered At: ${data.registered_at} | Total Screenings: ${data.history.length}`;

        const history = data.history || [];
        const metricsBox = document.getElementById('inspect-latest-summary');
        
        if (history.length > 0) {
            const latest = history[history.length - 1];
            metricsBox.innerHTML = `
                <div class="kpi-card"><p class="kpi-val text-blue">${latest.ai_risk}%</p><p class="kpi-label">Latest AI Score</p></div>
                <div class="kpi-card"><p class="kpi-val text-amber">${latest.mchat_score !== null ? latest.mchat_score + '/20' : 'N/A'}</p><p class="kpi-label">M-CHAT Score</p></div>
                <div class="kpi-card"><p class="kpi-val text-purple">${latest.combined}%</p><p class="kpi-label">Combined Score</p></div>
                <div class="kpi-card"><p class="kpi-val text-green">${latest.level}</p><p class="kpi-label">Risk Level</p></div>
            `;
        } else {
            metricsBox.innerHTML = '<div class="alert alert-success">This patient has not completed any screening assessment yet.</div>';
        }

        const tbody = document.getElementById('table-patient-history-body');
        if (history.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center">No screening history.</td></tr>';
        } else {
            tbody.innerHTML = history.map(h => `
                <tr>
                    <td>${h.saved_at}</td>
                    <td>${h.ai_risk}%</td>
                    <td>${h.mchat_score !== null ? h.mchat_score + '/20' : 'N/A'}</td>
                    <td><b>${h.combined}%</b></td>
                    <td><span class="badge-risk badge-${h.level.toLowerCase().slice(0,3)}">${h.level}</span></td>
                    <td>${h.urgency}</td>
                </tr>
            `).join('');
        }

        // Action buttons
        document.getElementById('btn-inspect-clear').onclick = async () => {
            if (confirm(`Are you sure you want to clear screening history for ${username}?`)) {
                await fetch(`/api/admin/patient/${username}/clear`, {method: 'POST'});
                inspectPatient(username);
                loadAdminData();
            }
        };

        document.getElementById('btn-inspect-delete').onclick = async () => {
            if (confirm(`Are you sure you want to delete patient account '${username}'?`)) {
                await fetch(`/api/admin/patient/${username}`, {method: 'DELETE'});
                card.style.display = 'none';
                loadAdminData();
            }
        };

    } catch (e) {}
}
