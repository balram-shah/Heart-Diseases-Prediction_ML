
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Heart Disease ML – Interactive Report", page_icon="❤️", layout="wide")

# ------------------------------
# Helpers
# ------------------------------
@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path)
    # try to normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    # map common variations to standard names
    rename_map = {
        "num": "target",
        "output": "target",
        "target_variable": "target",
        "restingbp": "trestbps",
        "resting_blood_pressure": "trestbps",
        "maxhr": "thalach",
        "max_heart_rate": "thalach",
        "fasting_blood_sugar": "fbs",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df

def split_Xy(df):
    target_col = "target" if "target" in df.columns else None
    if target_col is None:
        raise ValueError("Could not find target column. Expected a column named 'target'.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def scale_fit_transform(X_train, X_test):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return scaler, X_train_sc, X_test_sc

def metrics_dict(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }

def plot_confusion_matrix(cm, labels):
    z = cm
    x = labels
    y = labels
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, text=z, texttemplate="%{text}", colorscale="Blues"))
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", height=420)
    return fig

def roc_curve_fig(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc_val:.2f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=420)
    return fig, auc_val

# ------------------------------
# Load Data & Optional Model
# ------------------------------
DATA_PATH = "heart.csv"
MODEL_CANDIDATES = [
    "log,model_joblib_heart",   # user-uploaded name (contains comma)
    "model_heart.joblib",
    "heart_model.joblib",
]

st.title("❤️ Heart Disease Prediction – Interactive ML Report")
st.caption("Dataset insights • Model training & evaluation • Feature importance • Try-it-yourself predictions")

with st.sidebar:
    st.header("⚙️ Controls")
    data_path = st.text_input("Dataset path", value=DATA_PATH)
    df = load_dataset(data_path)
    st.success(f"Loaded dataset with shape {df.shape}")
    show_raw = st.checkbox("Show raw data head()", value=False)
    model_path = None
    for cand in MODEL_CANDIDATES:
        if os.path.exists(cand):
            model_path = cand
            break
    use_saved_model = st.checkbox("Use saved model if available", value=model_path is not None)
    st.write("Detected model file:" if model_path else "No saved model found.")
    if model_path:
        st.code(model_path)

if show_raw:
    st.subheader("Raw Preview")
    st.dataframe(df.head())

# ------------------------------
# Dataset Overview
# ------------------------------
st.header("📊 Dataset Overview")
left, right = st.columns([1.2, 1])
with left:
    st.write(f"**Rows:** {df.shape[0]}  •  **Columns:** {df.shape[1]}")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
with right:
    missing = df.isna().sum().reset_index().rename(columns={"index": "feature", 0: "missing"})
    missing["missing"] = df.isna().sum().values
    fig_miss = px.bar(missing, x="feature", y="missing", title="Missing values by feature")
    st.plotly_chart(fig_miss, use_container_width=True)

# ------------------------------
# EDA
# ------------------------------
st.header("🔎 Exploratory Data Analysis")

# If target present, show distributions conditional on target
if "target" in df.columns:
    col_options = [c for c in df.columns if c != "target"]
    sel = st.multiselect("Select up to 3 features to view distributions", options=col_options, default=col_options[:3], max_selections=3)
    for c in sel:
        try:
            fig = px.histogram(df, x=c, color="target", barmode="overlay", marginal="box", nbins=30, title=f"{c} by target")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Could not plot {c}: {e}")
else:
    st.warning("No 'target' column detected. Add/rename target to enable class-conditional EDA.")

# Correlation heatmap
try:
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig_corr = px.imshow(corr, text_auto=False, title="Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)
except Exception as e:
    st.info(f"Correlation heatmap skipped: {e}")

# ------------------------------
# Train / Evaluate
# ------------------------------
st.header("🤖 Model Training & Evaluation")

X, y = split_Xy(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

scaler, X_train_sc, X_test_sc = scale_fit_transform(X_train, X_test)

loaded_model = None
loaded_scaler = None

if use_saved_model and model_path:
    try:
        loaded = joblib.load(model_path)
        # support dict bundle: {"model": ..., "scaler": ...}
        if isinstance(loaded, dict) and "model" in loaded:
            loaded_model = loaded.get("model", None)
            loaded_scaler = loaded.get("scaler", None)
        else:
            loaded_model = loaded
        st.success("Loaded saved model successfully.")
    except Exception as e:
        st.warning(f"Could not load saved model: {e}")

# Train fresh models if no usable saved model
trained_models = {}
reports = {}

def train_and_eval(model, Xtr, Xte, ytr, yte, name, scaled=False):
    model.fit(Xtr, ytr)
    y_pred = model.predict(Xte)
    try:
        y_prob = model.predict_proba(Xte)[:,1]
    except Exception:
        # fallback if predict_proba not available
        y_prob = getattr(model, "decision_function", lambda X: np.zeros(len(X)))(Xte)
        # scale to 0..1
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)

    cm = confusion_matrix(yte, y_pred)
    fig_cm = plot_confusion_matrix(cm, ["No Disease", "Disease"])

    roc_fig, auc_val = roc_curve_fig(yte, y_prob)
    metrics = metrics_dict(yte, y_pred)
    metrics["ROC-AUC"] = auc_val

    return {
        "model": model,
        "metrics": metrics,
        "cm_fig": fig_cm,
        "roc_fig": roc_fig,
        "y_prob": y_prob
    }

if loaded_model is not None:
    # use scaler if bundled, else use computed scaler
    scaler_for_loaded = loaded_scaler if loaded_scaler is not None else scaler
    Xtr_loaded = scaler_for_loaded.transform(X_train)
    Xte_loaded = scaler_for_loaded.transform(X_test)
    res_loaded = train_and_eval(loaded_model, Xtr_loaded, Xte_loaded, y_train, y_test, "LoadedModel", scaled=True)
    trained_models["Loaded Model"] = res_loaded
else:
    # Train Logistic Regression and Random Forest
    lr = LogisticRegression(max_iter=2000)
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    res_lr = train_and_eval(lr, X_train_sc, X_test_sc, y_train, y_test, "Logistic Regression", scaled=True)
    res_rf = train_and_eval(rf, X_train_sc, X_test_sc, y_train, y_test, "Random Forest", scaled=True)
    trained_models["Logistic Regression"] = res_lr
    trained_models["Random Forest"] = res_rf

# Show results
tabs = st.tabs(list(trained_models.keys()))
for tab, (name, res) in zip(tabs, trained_models.items()):
    with tab:
        st.subheader(f"Results: {name}")
        st.json({k: (float(v) if hasattr(v, "__float__") else v) for k, v in res["metrics"].items()})
        st.plotly_chart(res["cm_fig"], use_container_width=True)
        st.plotly_chart(res["roc_fig"], use_container_width=True)

# Choose best by ROC-AUC
best_name = max(trained_models, key=lambda k: trained_models[k]["metrics"]["ROC-AUC"])
best = trained_models[best_name]
st.success(f"🏆 Best model: {best_name}  •  ROC-AUC: {best['metrics']['ROC-AUC']:.3f}")

# ------------------------------
# Feature Importance
# ------------------------------
st.header("🌟 Feature Importance / Effects")

current_model = best["model"]
try:
    # Try model-native importance first
    if hasattr(current_model, "feature_importances_"):
        importances = current_model.feature_importances_
        fi = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values("importance", ascending=False)
    elif hasattr(current_model, "coef_"):
        coefs = np.abs(current_model.coef_).ravel()
        fi = pd.DataFrame({"feature": X.columns, "importance": coefs}).sort_values("importance", ascending=False)
    else:
        raise AttributeError("No native feature importance")
    fig_fi = px.bar(fi.head(12), x="feature", y="importance", title=f"Top Features – {best_name}")
    st.plotly_chart(fig_fi, use_container_width=True)
except Exception as e:
    st.info(f"Native feature importance unavailable ({e}). Using permutation importance...")
    try:
        # permutation on test set
        Xte_used = X_test_sc
        r = permutation_importance(current_model, Xte_used, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        fi = pd.DataFrame({"feature": X.columns, "importance": r.importances_mean}).sort_values("importance", ascending=False)
        fig_fi = px.bar(fi.head(12), x="feature", y="importance", title=f"Top Features (Permutation) – {best_name}")
        st.plotly_chart(fig_fi, use_container_width=True)
    except Exception as e2:
        st.warning(f"Permutation importance failed: {e2}")

# ------------------------------
# Prediction Playground
# ------------------------------
st.header("🧑‍⚕️ Try Your Own Inputs")

def render_form():
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 20, 90, 50)
        sex = st.selectbox("Sex (1=Male, 0=Female)", [1,0])
        cp = st.selectbox("Chest Pain Type (0–3)", [0,1,2,3])
        trestbps = st.slider("Resting BP (trestbps)", 80, 220, 120)
        chol = st.slider("Cholesterol (chol)", 100, 650, 240)
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 (fbs)", [0,1])
        restecg = st.selectbox("Resting ECG (0–2)", [0,1,2])
        thalach = st.slider("Max Heart Rate (thalach)", 70, 230, 150)
        exang = st.selectbox("Exercise Induced Angina (exang)", [0,1])
    with col3:
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
        slope = st.selectbox("Slope (0–2)", [0,1,2])
        ca = st.slider("Major Vessels (ca)", 0, 4, 0)
        thal = st.selectbox("Thal (1=Normal, 2=Fixed, 3=Reversible)", [1,2,3])

    data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    return pd.DataFrame([data])

# Align form features with training X
form_df = render_form()
missing_in_form = [c for c in X.columns if c not in form_df.columns]
for col in missing_in_form:
    form_df[col] = 0  # default filler for unexpected columns
form_df = form_df[X.columns]  # reorder

# Use scaler corresponding to best model training branch
scaler_for_best = None
if best_name == "Loaded Model" and loaded_scaler is not None:
    scaler_for_best = loaded_scaler
else:
    scaler_for_best = scaler
form_scaled = scaler_for_best.transform(form_df)

proba = None
try:
    proba = current_model.predict_proba(form_scaled)[0,1]
except Exception:
    # decision_function fallback
    s = getattr(current_model, "decision_function", lambda X: np.zeros(len(X)))(form_scaled)
    proba = (s - s.min())/(s.max()-s.min()+1e-9)
    proba = float(proba[0])

pred = int(proba >= 0.5)
colL, colR = st.columns([1,1])
with colL:
    st.metric("Predicted Class", "Disease" if pred==1 else "No Disease")
with colR:
    st.metric("Predicted Probability (Disease)", f"{proba:.3f}")

# Gauge-like bar
fig_g = go.Figure(go.Indicator(
    mode="gauge+number",
    value=proba*100,
    title={'text': "Risk Probability (%)"},
    gauge={'axis': {'range': [0, 100]}}
))
st.plotly_chart(fig_g, use_container_width=True)

st.caption("⚠️ Educational tool only. Not medical advice.")
