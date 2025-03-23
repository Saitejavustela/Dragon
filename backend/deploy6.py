from fastapi import FastAPI, File, UploadFile
import pdfplumber
import re
import pandas as pd
import joblib
import numpy as np
import io
from fastapi.middleware.cors import CORSMiddleware
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

# ✅ Load the trained model
MODEL_PATH = "diabetes_model.pkl"
model = joblib.load(MODEL_PATH)
print("✅ Diabetes Model Loaded Successfully!")

# ✅ Get feature order from model
feature_order = model.feature_names_in_

# ✅ Feature extraction function
def extract_pdf_data(pdf_bytes):
    extracted_data = {}
    
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

    if not text:
        return None

    # ✅ Define patterns for feature extraction
    patterns = {
        "BMI": r"BMI:\s*(\d+\.\d+|\d+)",
        "HighBP": r"High Blood Pressure:\s*(Yes|No|1|0)",
        "Age": r"Age:\s*(\d+)",
    }

    # ✅ Extract feature values
    for feature, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            if feature == "HighBP" and value in ["Yes", "No"]:
                extracted_data[feature] = 1 if value == "Yes" else 0
            else:
                extracted_data[feature] = float(value)
        else:
            extracted_data[feature] = 0  # Default if missing

    # ✅ Ensure all features exist
    feature_values = [extracted_data.get(f, 0) for f in feature_order]

    # ✅ Convert to DataFrame
    df = pd.DataFrame([feature_values], columns=feature_order)

    return df

# ✅ Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    
    # Extract features
    data = extract_pdf_data(pdf_bytes)
    if data is None:
        return {"error": "Invalid PDF format. Could not extract data."}

    # Make probability prediction
    probabilities = model.predict_proba(data)[0]
    diabetes_risk_percentage = probabilities[1] * 100  # Probability of diabetes

    # ✅ Provide suggestions
    if diabetes_risk_percentage >= 70:
        risk_level = "High"
        suggestions = [
            "Significantly reduce sugar and refined carbs.",
            "Engage in at least 30-45 minutes of exercise daily.",
            "Check blood sugar levels frequently.",
            "Consult a doctor immediately."
        ]
    elif 30 <= diabetes_risk_percentage < 70:
        risk_level = "Moderate"
        suggestions = [
            "Limit sugary foods, increase fiber intake.",
            "Aim for 30 minutes of exercise most days.",
            "Consider periodic blood sugar checks.",
            "Schedule a checkup with a healthcare provider."
        ]
    else:
        risk_level = "Low"
        suggestions = [
            "Continue regular physical activity.",
            "Maintain a balanced diet.",
            "Annual health screenings can help detect early changes."
        ]

    # ✅ Dynamically generate suggestion list
    suggestion_text = "\n".join(f"- {s}" for s in suggestions)

    # ✅ Format everything as a single string
    result_text = f"""
    Diabetes Risk: {diabetes_risk_percentage:.2f}%
    Risk Level: {risk_level}
    Suggestions:
    {suggestion_text}
    """

    return {"prediction": result_text.strip()}  # ✅ Wrapping it inside "prediction"


# ✅ Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8006)
