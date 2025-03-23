#brest cancer report
from fastapi import FastAPI, File, UploadFile
import pdfplumber
import re
import pandas as pd
import joblib
import numpy as nppyhton 
import io
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# ✅ Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load the trained model and scaler
MODEL_PATH = "breast_cancer_model.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Model and Scaler Loaded Successfully!")

# ✅ Feature mappings
FEATURE_NAMES = {
    "Mean Radius": "radius_mean",
    "Mean Texture": "texture_mean",
    "Mean Perimeter": "perimeter_mean",
    "Mean Area": "area_mean",
    "Mean Smoothness": "smoothness_mean",
    "Mean Compactness": "compactness_mean",
    "Mean Concavity": "concavity_mean",
    "Mean Concave Points": "concave points_mean",
    "Mean Symmetry": "symmetry_mean",
    "Mean Fractal Dimension": "fractal_dimension_mean",
    "Radius SE": "radius_se",
    "Texture SE": "texture_se",
    "Perimeter SE": "perimeter_se",
    "Area SE": "area_se",
    "Smoothness SE": "smoothness_se",
    "Compactness SE": "compactness_se",
    "Concavity SE": "concavity_se",
    "Concave Points SE": "concave points_se",
    "Symmetry SE": "symmetry_se",
    "Fractal Dimension SE": "fractal_dimension_se",
    "Worst Radius": "radius_worst",
    "Worst Texture": "texture_worst",
    "Worst Perimeter": "perimeter_worst",
    "Worst Area": "area_worst",
    "Worst Smoothness": "smoothness_worst",
    "Worst Compactness": "compactness_worst",
    "Worst Concavity": "concavity_worst",
    "Worst Concave Points": "concave points_worst",
    "Worst Symmetry": "symmetry_worst",
    "Worst Fractal Dimension": "fractal_dimension_worst",
}

# ✅ Function to extract data from a PDF
def extract_pdf_data(pdf_bytes):
    extracted_data = {}
    
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

    if not text:
        return None

    # Extract feature values using regex
    for line in text.split("\n"):
        for label, col_name in FEATURE_NAMES.items():
            match = re.search(fr"{label}:\s*([-+]?\d*\.\d+|\d+)", line)
            if match:
                extracted_data[col_name] = float(match.group(1))

    # Fill missing values with zero (or replace with median if needed)
    missing_features = set(FEATURE_NAMES.values()) - set(extracted_data.keys())
    for feature in missing_features:
        extracted_data[feature] = 0.0  # Placeholder

    # Convert to DataFrame
    df = pd.DataFrame([extracted_data])
    df = df[FEATURE_NAMES.values()]  # Ensure correct column order

    return df

# ✅ Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    
    # Extract features
    data = extract_pdf_data(pdf_bytes)
    if data is None:
        return {"error": "Invalid PDF format. Could not extract data."}
    
    # Normalize features
    data_scaled = scaler.transform(data)
    
    # Make prediction
    prediction = model.predict(data_scaled)[0]
    result = "Malignant" if prediction == 1 else "Benign"

    return {"prediction": result}

# ✅ Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8005)
