import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Checker",
    page_icon="🩺",
    layout="centered"
)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("best_diabetes_model.pkl", "rb") as f:
        return pickle.load(f)

payload      = load_model()
model        = payload["model"]
scaler       = payload["scaler"]
encoders     = payload["encoders"]
feature_cols = payload["feature_cols"]
best_cv      = payload["best_cv"]

# ── Symptom definitions ──────────────────────────────────────
# Maps plain-language label → internal feature name
# Only symptoms a non-medical person can self-identify are shown.
# Hidden (kept as No by default): partial paresis, Genital thrush, Polyphagia
SYMPTOMS = {
    "Urinating much more than usual":          "Polyuria",
    "Feeling unusually thirsty all the time":  "Polydipsia",
    "Sudden unexplained weight loss":          "sudden weight loss",
    "Feeling weak or tired most of the time":  "weakness",
    "Blurred or unclear vision":               "visual blurring",
    "Skin itching (not related to a rash)":    "Itching",
    "Mood swings or feeling easily irritated": "Irritability",
    "Wounds or cuts taking a long time to heal": "delayed healing",
    "Muscle stiffness or cramps":              "muscle stiffness",
    "Noticeable hair loss":                    "Alopecia",
    "Overweight or obese":                     "Obesity",
}

# Features hidden from UI — default to "No"
HIDDEN_DEFAULTS = {
    "Polyphagia":    "No",
    "Genital thrush":"No",
    "partial paresis":"No",
}

def get_risk_level(prob):
    if prob >= 0.75: return "High",    "#ef4444", "🔴"
    if prob >= 0.50: return "Medium",  "#f59e0b", "🟡"
    if prob >= 0.25: return "Low",     "#3b82f6", "🔵"
    return                  "Very Low","#22c55e", "🟢"

def encode_and_scale(raw: dict) -> np.ndarray:
    row = {}
    for feat in feature_cols:
        val = raw[feat]
        if feat in encoders:
            row[feat] = encoders[feat].transform([str(val)])[0]
        else:
            row[feat] = val
    arr = np.array([list(row.values())])
    return scaler.transform(arr)


# ── App ──────────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Checker")
st.write(
    "Answer a few simple questions to get an early-stage diabetes risk estimate. "
    "This takes about 30 seconds."
)
st.caption("⚠️ This is an educational tool, not a medical diagnosis.")
st.divider()

# ════════════════════════════════════════
# SECTION 1 — Basic Info
# ════════════════════════════════════════
st.subheader("Step 1 — Basic Information")

col1, col2 = st.columns(2)
with col1:
    age    = st.slider("How old are you?", 1, 100, 35)
with col2:
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

st.divider()

# ════════════════════════════════════════
# SECTION 2 — Symptoms
# ════════════════════════════════════════
st.subheader("Step 2 — Symptoms")
st.write("Tick any symptoms you have been experiencing recently:")

selected = st.multiselect(
    label="Select all that apply:",
    options=list(SYMPTOMS.keys()),
    placeholder="Choose symptoms...",
)

st.divider()

# ── Predict button ───────────────────────────────────────────
if st.button("Check My Risk", use_container_width=True, type="primary"):

    # Build full feature dict
    raw = {"Age": age, "Gender": gender}

    # Symptoms shown in UI
    for label, feat in SYMPTOMS.items():
        raw[feat] = "Yes" if label in selected else "No"

    # Hidden features default to No
    for feat, val in HIDDEN_DEFAULTS.items():
        raw[feat] = val

    # Encode + scale + predict
    input_scaled = encode_and_scale(raw)
    prediction   = model.predict(input_scaled)[0]
    prob         = model.predict_proba(input_scaled)[0][1]
    risk, color, icon = get_risk_level(prob)
    result_label = "Diabetes Risk Detected" if prediction == 1 else "No Diabetes Risk Detected"

    st.subheader("Your Results")

    # ── Result cards ─────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Result", result_label)
    with c2:
        st.metric("Risk Probability", f"{prob:.1%}")
    with c3:
        st.metric("Risk Level", f"{icon} {risk}")

    # ── Message ──────────────────────────────────────────────
    messages = {
        "High":     ("error",   "Your responses suggest a **high risk** of early-stage diabetes. Please consult a doctor as soon as possible."),
        "Medium":   ("warning", "Your responses suggest a **moderate risk**. Consider speaking with a healthcare provider and monitoring your symptoms."),
        "Low":      ("info",    "Your responses suggest a **low risk**. Maintain a healthy diet and lifestyle."),
        "Very Low": ("success", "Your responses suggest a **very low risk**. Keep up your healthy habits!"),
    }
    msg_type, msg_text = messages[risk]
    getattr(st, msg_type)(msg_text)

    # ── Gauge ─────────────────────────────────────────────────
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 30}},
        title={"text": "Diabetes Risk Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  25], "color": "#dcfce7"},
                {"range": [25, 50], "color": "#dbeafe"},
                {"range": [50, 75], "color": "#fef9c3"},
                {"range": [75, 100],"color": "#fee2e2"},
            ],
        }
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── Symptoms summary ──────────────────────────────────────
    if selected:
        with st.expander("Symptoms you reported"):
            for s in selected:
                st.write(f"• {s}")
    else:
        st.info("You didn't report any symptoms. If you are experiencing any, try again and select them.")

    # ── About the model ───────────────────────────────────────
    with st.expander("About this model"):
        st.markdown(f"""
        | | |
        |---|---|
        | **Algorithm** | Support Vector Machine (SVM) |
        | **Kernel** | RBF · C=3 |
        | **Dataset** | UCI Early Stage Diabetes Risk · 520 patients |
        | **Features** | 16 clinical symptoms |
        | **Test Accuracy** | 96.2% |
        | **5-Fold CV** | {best_cv:.1%} |
        | **Train / Test split** | 80% / 20% |

        Model trained using GridSearchCV over C=[1,2,3] and kernel=[rbf, linear].
        Features preprocessed with LabelEncoder and StandardScaler.
        """)

st.divider()
st.caption(
    "⚠️ This application is for educational purposes only and is not a substitute "
    "for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider."
)
