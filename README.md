# ImageIQ CT 

This Streamlit app predicts **SNR and CNR** in abdominal CT using volume, circumference, and kVp as input.

### 🔍 Features
- Separate models for 80 and 120 kVp
- SHAP & feature importance visualizations
- PDF report download with embedded plots
- Dashboard-style user interface

### 📦 Run locally
```bash
pip install -r requirements.txt
streamlit run app_xgboost.py
