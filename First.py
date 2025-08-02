
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import shap
try:
    import matplotlib.pyplot as plt
except ImportError:
    st.error("matplotlib is not installed. Please install it with 'pip install matplotlib' and restart the app.")
    st.stop()

# --- UI Setup ---

# --- Enhanced UI Setup ---
st.set_page_config(page_title="", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for background and widgets

st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #FFFDE4 0%, #0052D4 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #FFFDE4 0%, #0052D4 100%);
    }
    .sidebar .sidebar-content {
        background: #FCE38A;
        color: #0052D4;
    }
    .stButton>button {
        background-color: #F38181;
        color: white;
        border-radius: 10px;
        font-size: 20px;
        padding: 10px 28px;
        border: none;
        box-shadow: 2px 2px 8px #A8E063;
    }
    .stNumberInput>div>input {
        background-color: #EAFFD0;
        color: #0052D4;
        border-radius: 8px;
        font-size: 18px;
    }
    .stSelectbox>div>div {
        background-color: #FCE38A;
        color: #0052D4;
        border-radius: 8px;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
<h1 style='text-align:center; font-size:56px; font-family:Arial Black; font-weight:900; letter-spacing:2px; color:#D500F9; text-shadow: 2px 2px 8px #FCE38A;'>AI Health Agent 🩺</h1>
<h2 style='text-align:center; font-size:32px; font-family:Arial Black; font-weight:900; color:#0052D4; margin-top:-20px; text-shadow: 1px 1px 6px #F38181;'>Predict your risk for <span style='color:#D500F9;'>Diabetes</span>, <span style='color:#00B8D4;'>Heart Disease</span> & <span style='color:#43A047;'>Hypertension</span></h2>
""", unsafe_allow_html=True)


# Display local image in the UI
from PIL import Image
img_path = r"c:\Users\NEEL\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\TempState\5516ADB142FCB18A017C72602ABBDB6D\WhatsApp Image 2025-08-02 at 12.31.51_bfddeb15.jpg"
try:
    image = Image.open(img_path)
    st.image(image, width=120)
except Exception as e:
    st.warning("Could not load image. Please check the path.")

# --- Data Loading (placeholder) ---
@st.cache_data
def load_data():
    # Replace with actual dataset paths or URLs
    # Example: Pima Indians Diabetes Dataset
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# --- Data Preprocessing ---
def preprocess_data(df):
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    scaler = StandardScaler()
    X = df_imputed.drop('Outcome', axis=1)
    y = df_imputed['Outcome']
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss')
}

selected_model = st.sidebar.selectbox("Choose Model", list(models.keys()))
model = models[selected_model]
model.fit(X_train, y_train)

# --- Prediction Form ---

st.markdown("<h4 style='color:#D500F9;font-size:22px;'>Enter your health details:</h4>", unsafe_allow_html=True)
user_input = {}
cols = st.columns(2)
for idx, col in enumerate(df.columns[:-1]):
    with cols[idx % 2]:
        val = st.number_input(f"{col}", min_value=float(df[col].min()), max_value=float(df[col].max()), value=float(df[col].mean()), key=col)
        user_input[col] = val

if st.button("Predict Risk"):
    input_df = pd.DataFrame([user_input])
    input_scaled = StandardScaler().fit(X_train).transform(input_df)
    risk = model.predict_proba(input_scaled)[0][1]
    st.success(f"Predicted Risk Score: {risk:.2f}")
    # --- Health Advice ---
    if risk < 0.3:
        st.info("Low risk. Maintain healthy habits!")
    elif risk < 0.7:
        st.warning("Moderate risk. Consider lifestyle improvements.")
    else:
        st.error("High risk. Consult a healthcare professional.")

    # --- Explainability ---
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(input_scaled)
    st.markdown("<h5 style='color:#FF6F61;'>Feature Impact:</h5>", unsafe_allow_html=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_df, show=False, plot_type="bar")
    st.pyplot(fig)

st.sidebar.markdown("<hr>")
st.sidebar.markdown("<b>About:</b> This demo predicts health risks using open datasets and ML models. For educational use only.")
