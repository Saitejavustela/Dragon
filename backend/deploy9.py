#psad Record
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import pandas as pd
import numpy as np
import joblib
import io
import re
import uvicorn

app = FastAPI()

# ✅ Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load the trained model and scaler
MODEL_PATH = "lasso_model.pkl"
SCALER_PATH = "scaler_pdas.pkl"
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Prostate Cancer PSA Density Model Loaded Successfully!")

# ✅ Define feature order
feature_order = ["PSA_Level", "Prostate_Volume", "Age"]

# ✅ Feature extraction function
def extract_pdf_data(pdf_bytes):
    extracted_data = {}
    
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

    if not text:
        return None

    # ✅ Define patterns for feature extraction
    patterns = {
        "PSA_Level": r"PSA Level[:\s]*([\d\.]+)\s*ng/mL",
        "Prostate_Volume": r"Prostate Volume[:\s]*([\d\.]+)\s*mL",
        "Age": r"Age[:\s]*(\d+)\s*years|(\d+)-year-old",
    }

    # ✅ Extract feature values
    for feature, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if feature == "Age":
                value = match.group(1) or match.group(2)  # Handle both age formats
            else:
                value = match.group(1)
            extracted_data[feature] = float(value)
        else:
            extracted_data[feature] = None

    # Check for missing features
    missing_features = [f for f in feature_order if extracted_data[f] is None]
    if missing_features:
        return None  # Return None if any required feature is missing

    # ✅ Convert to DataFrame and scale
    feature_values = [extracted_data[f] for f in feature_order]
    df = pd.DataFrame([feature_values], columns=feature_order)
    df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_order)

    return df_scaled

# ✅ Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    
    # Extract and scale features
    data = extract_pdf_data(pdf_bytes)
    if data is None:
        return {"error": "Invalid PDF format or missing required data (PSA_Level, Prostate_Volume, Age)."}

    # Make prediction
    predicted_psa_density = model.predict(data)[0]
    calculated_psa_density = data["PSA_Level"][0] / data["Prostate_Volume"][0]  # For comparison

    # ✅ Provide suggestions based on PSA density
    if predicted_psa_density >= 0.15:
        risk_level = "Elevated"
        suggestions = [
            "Consult a urologist for further evaluation.",
            "Consider a prostate biopsy if not already done.",
            "Monitor PSA levels every 6 months.",
        ]
    elif 0.10 <= predicted_psa_density < 0.15:
        risk_level = "Moderate"
        suggestions = [
            "Schedule a follow-up PSA test in 6-12 months.",
            "Discuss with your doctor about prostate health.",
            "Maintain a healthy lifestyle to reduce risk.",
        ]
    else:
        risk_level = "Low"
        suggestions = [
            "Continue regular prostate health screenings.",
            "Maintain a balanced diet and exercise routine.",
        ]

    # ✅ Dynamically generate suggestion list
    suggestion_text = "\n".join(f"- {s}" for s in suggestions)

    # ✅ Format everything as a single string
    result_text = f"""
    Predicted PSA Density: {predicted_psa_density:.4f} ng/mL²
    Calculated PSA Density: {calculated_psa_density:.4f} ng/mL²
    Risk Level: {risk_level}
    Suggestions:
    {suggestion_text}
    """

    return {"prediction": result_text.strip()}

# ✅ Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8009)