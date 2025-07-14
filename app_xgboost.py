import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
from fpdf import FPDF
import io
import tempfile

# === 🧠 Feature Engineering ===
def engineer_features(volume, circumference, kvp):
    vol_kvp = volume * kvp
    circ_kvp = circumference * kvp
    vol_by_circ = volume / (circumference + 1e-6)
    log_volume = np.log(volume + 1)
    is_low_kvp = int(kvp == 80)
    return np.array([[volume, circumference, kvp,
                      vol_kvp, circ_kvp, vol_by_circ,
                      log_volume, is_low_kvp]])

feature_names = [
    'volume', 'circumference', 'kVp',
    'vol_kvp', 'circ_kvp', 'vol_by_circ',
    'log_volume', 'is_low_kvp'
]

# === 📋 App Layout ===
st.set_page_config(page_title="CT Predictor", layout="wide")
st.title("🧠 ImageIQ-CT")
st.markdown("Predict **SNR & CNR** from abdominal CT scan parameters using machine learning.")

# === 🧭 SIDEBAR INPUTS
with st.sidebar:
    st.header("🛠 Input Parameters")
    volume = st.number_input("Abdominal Volume (cm³)", min_value=0.0, format="%.2f")
    circumference = st.number_input("Abdominal Circumference (cm)", min_value=0.0, format="%.2f")
    kvp = st.selectbox("Tube Voltage (kVp)", [80, 120])
    run_pred = st.button("🔍 Run Prediction")

# === STATE PLACEHOLDERS
input_scaled = None
model_used = None
prediction = None

# === 🔍 MAIN PREDICTION LOGIC
if run_pred:
    input_data = engineer_features(volume, circumference, kvp)

    if kvp == 80:
        model_used = joblib.load("xgb_model_80.pkl")
        scaler = joblib.load("scaler_80.pkl")
    else:
        model_used = joblib.load("xgb_model_120.pkl")
        scaler = joblib.load("scaler_120.pkl")

    input_scaled = scaler.transform(input_data)
    prediction = model_used.predict(input_scaled)
    snr, cnr = prediction[0]

    # === Display Prediction
    col1, col2 = st.columns(2)
    col1.metric("📊 Predicted SNR", f"{snr:.2f}")
    col2.metric("📊 Predicted CNR", f"{cnr:.2f}")

    # === Simulated Evaluation Metrics (Optional)
    #y_fake = np.array([[snr + np.random.uniform(-1, 1), cnr + np.random.uniform(-1, 1)]])
    #mae = mean_absolute_error(y_fake, prediction)
    #rmse = np.sqrt(mean_squared_error(y_fake, prediction))
    #r2 = r2_score(y_fake, prediction)

    st.markdown("---")
    st.subheader("📈 Evaluation Metrics (Sampled)")
    col3, col4, col5 = st.columns(3)
    #col3.metric("📉 MAE", f"{mae:.2f}")
    #col4.metric("📉 RMSE", f"{rmse:.2f}")
    #col5.metric("🧠 R²", f"{r2:.2f}")

    # === Plot Tabs
    st.markdown("---")
    tabs = st.tabs(["📊 Prediction Plot", "🧠 Feature Importance", "📌 SHAP Explanation"])

    # === PREDICTION PLOT
    with tabs[0]:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        metrics = ['SNR', 'CNR']
        values = [snr, cnr]
        palette = sns.color_palette("coolwarm", len(metrics))
        bars = ax.barh(metrics, values, color=palette)
        ax.set_xlim(0, max(values) * 1.2)
        ax.set_xlabel("Predicted Value")
        ax.set_title(f"Predicted SNR and CNR at {kvp} kVp")
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}', va='center', fontsize=10, color='black')
        sns.despine(left=True)
        st.pyplot(fig)

    # === FEATURE IMPORTANCE
    with tabs[1]:
        booster = model_used.estimators_[0].get_booster()
        importance_dict = booster.get_score(importance_type='weight')
        feature_map = {f'f{i}': name for i, name in enumerate(feature_names)}
        importance_named = {feature_map[k]: v for k, v in importance_dict.items() if k in feature_map}
        sorted_items = sorted(importance_named.items(), key=lambda item: item[1], reverse=True)
        labels, values = zip(*sorted_items)

        fig_imp, ax_imp = plt.subplots()
        bars = ax_imp.barh(labels, values, color='cadetblue')
        ax_imp.set_title('Feature Importance')
        ax_imp.set_xlabel('Relative Importance')
        ax_imp.invert_yaxis()
        for bar in bars:
            width = bar.get_width()
            ax_imp.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.1f}', va='center')
        st.pyplot(fig_imp)

    # === SHAP WATERFALL PLOT
    with tabs[2]:
        explainer = shap.Explainer(model_used.estimators_[0])
        shap_values = explainer(input_scaled)
        shap.plots.waterfall(shap_values[0], show=False)
        fig_shap = plt.gcf()
        st.pyplot(fig_shap)

    # === EXPORT TO PDF
    st.markdown("---")
    st.subheader("📥 Export Report")

    # Save prediction plot to image
    img_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    fig.savefig(img_path, bbox_inches="tight")
    plt.close(fig)

    # Build PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pdf.set_title("CT SNR/CNR Prediction Report")
    pdf.cell(200, 10, txt="CT SNR/CNR Prediction Report", ln=True, align='C')
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Date/Time: {timestamp}", ln=True)
    pdf.ln(3)
    pdf.cell(200, 10, txt=f"Tube Voltage (kVp): {kvp}", ln=True)
    pdf.cell(200, 10, txt=f"Abdominal Volume: {volume:.2f} cm³", ln=True)
    pdf.cell(200, 10, txt=f"Abdominal Circumference: {circumference:.2f} cm", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt=f"Predicted SNR: {snr:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Predicted CNR: {cnr:.2f}", ln=True)
    #pdf.ln(5)
    #pdf.set_font("Arial", "", 12)
    #pdf.cell(200, 10, txt=f"MAE: {mae:.2f}", ln=True)
    #pdf.cell(200, 10, txt=f"RMSE: {rmse:.2f}", ln=True)
    #pdf.cell(200, 10, txt=f"R² Score: {r2:.2f}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Prediction Plot:", ln=True)
    pdf.image(img_path, x=15, w=180)

    # Footer
    pdf.set_y(-40)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(200, 10, txt="Generated by CT Image Quality App", ln=True, align="C")
    pdf.cell(200, 10, txt="Radiation Labs by Mamman", ln=True, align="C")
    pdf.cell(200, 10, txt="Signature: ___________________________", ln=True, align="C")

    # Output to memory
    pdf_buffer = io.BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    pdf_buffer = io.BytesIO(pdf_bytes)

    # Download button
    st.download_button(
        label="📥 Download Full PDF Report",
        data=pdf_buffer,
        file_name=f"CT_Report_{kvp}kVp_{timestamp.replace(' ', '_')}.pdf",
        mime="application/pdf"
    )
