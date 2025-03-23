#cancer risk
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import pandas as pd
import joblib
import numpy as np
import io
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
MODEL_PATH = "rf_model.joblib"
SCALER_PATH = "scaler.joblib"
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Cancer Model Loaded Successfully!")

# ✅ Get feature order from model
feature_order = ["age", "gender_Male", "gender_Unknown", "smoking_status", "diet_quality", 
                 "bmi_range", "gene_marker1", "gene_marker2"]

# ✅ Feature extraction function
def extract_pdf_data(pdf_bytes):
    extracted_data = {}
    
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

    if not text:
        return None

    # Convert text to lowercase for case-insensitive matching
    text = text.lower()

    # ✅ Extract feature values
    if "age" in text:
        age = int("".join(filter(str.isdigit, text.split("age")[1])) or 0)
        extracted_data["age"] = age
    else:
        extracted_data["age"] = 0

    extracted_data["gender_Male"] = 1 if "male" in text else 0
    extracted_data["gender_Unknown"] = 1 if "unknown" in text else 0
    extracted_data["smoking_status"] = 1 if "smoking" in text or "smoker" in text else 0
    extracted_data["diet_quality"] = 1 if "good" in text or "healthy" in text else 0
    
    if "bmi" in text:
        bmi_value = int("".join(filter(str.isdigit, text.split("bmi")[1])) or 0)
        extracted_data["bmi_range"] = 2 if "high" in text or bmi_value > 25 else 1
    else:
        extracted_data["bmi_range"] = 0

    extracted_data["gene_marker1"] = 1 if "gene marker 1" in text or "gene_marker1" in text and "present" in text else 0
    extracted_data["gene_marker2"] = 1 if "gene marker 2" in text or "gene_marker2" in text and "present" in text else 0

    # ✅ Ensure all features exist
    feature_values = [extracted_data.get(f, 0) for f in feature_order]

    # ✅ Scale the 'age' feature
    df = pd.DataFrame([feature_values], columns=feature_order)
    df["age"] = scaler.transform(df[["age"]])

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
    cancer_risk_percentage = probabilities[1] * 100  # Probability of cancer

    # ✅ Provide suggestions based on risk and features
    suggestions = []
    cancer_types = []

    if data["smoking_status"][0] == 1:
        cancer_types.append("Lung Cancer")
        suggestions.append("Quit smoking immediately.")
    if data["bmi_range"][0] == 2:
        cancer_types.append("Colorectal/Breast Cancer")
        suggestions.append("Maintain a healthy weight through diet and exercise.")
    if data["diet_quality"][0] == 0:
        cancer_types.append("Colorectal/Stomach Cancer")
        suggestions.append("Improve diet with more fruits, vegetables, and whole grains.")
    if data["gene_marker1"][0] == 1:
        cancer_types.append("Lung Cancer (Genetic Risk)")
        suggestions.append("Consider genetic counseling and screenings.")
    if data["gene_marker2"][0] == 1 and data["gender_Male"][0] == 0:
        cancer_types.append("Breast Cancer (Genetic Risk)")
        suggestions.append("Regular mammograms and genetic testing are recommended.")

    # ✅ Additional risk-based suggestions
    if cancer_risk_percentage >= 70:
        risk_level = "High"
        suggestions.append("Schedule a cancer screening with your doctor immediately.")
    elif 30 <= cancer_risk_percentage < 70:
        risk_level = "Moderate"
        suggestions.append("Consult a healthcare provider for regular monitoring.")
    else:
        risk_level = "Low"
        suggestions.append("Maintain your current healthy habits.")

    # ✅ Dynamically generate suggestion list
    suggestion_text = "\n".join(f"- {s}" for s in suggestions) if suggestions else "- No specific suggestions needed."
    cancer_types_text = ", ".join(cancer_types) if cancer_types else "None detected"

    # ✅ Format everything as a single string
    result_text = f"""
    Cancer Risk: {cancer_risk_percentage:.2f}%
    Risk Level: {risk_level}
    Potential Cancer Types: {cancer_types_text}
    Suggestions:
    {suggestion_text}
    """

    return {"prediction": result_text.strip()}

# ✅ Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8008)