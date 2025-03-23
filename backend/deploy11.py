from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import re
import pandas as pd
import joblib
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

# ✅ Load the trained model
MODEL_PATH = "blood_report_model.pkl"
model = joblib.load(MODEL_PATH)
print("✅ Blood Report Model Loaded Successfully!")

# ✅ Define feature order
feature_order = ["WBC", "RBC", "HGB", "PLT"]

# ✅ Define component name variations
component_mapping = {
    "WBC": ["WBC", "white blood cell", "white blood count", "WBC COUNT"],
    "RBC": ["RBC", "red blood cell", "red blood count", "RBC COUNT"],
    "HGB": ["HGB", "hemoglobin", "hb", "Hemoglobin (Hb)", "HEMOGLOBIN", "HEMOGLOBIN \\(Hb\\)"],
    "PLT": ["PLT", "platelet count", "platelets", "PLATELET COUNT"]
}

# ✅ Feature extraction function
def extract_pdf_data(pdf_bytes):
    extracted_data = {}
    
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

    if not text:
        return None

    # Extract values for each component
    for component in feature_order:
        possible_names = component_mapping.get(component, [component])
        value = None
        for name in possible_names:
            escaped_name = re.escape(name)
            pattern = rf'{escaped_name}\s*(?:count)?\s*[:\s\n]+([0-9.]+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                break
        extracted_data[component] = value if value is not None else 0.0  # Default to 0 if missing

    # ✅ Convert to DataFrame
    feature_values = [extracted_data[f] for f in feature_order]
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

    # Make predictions
    predictions = model.predict(data)[0]  # [WBC_low, RBC_low, HGB_low, PLT_low, WBC_high, RBC_high, HGB_high, PLT_high]

    # Extracted values for display
    extracted_values = data.iloc[0].to_dict()

    # ✅ Define recommendations
    recommendations = {
        'WBC': {
            'low': 'Eat vitamin C-rich foods (e.g., oranges, bell peppers) to boost immunity.',
            'high': 'Consult a doctor; high WBC may indicate infection or other conditions.'
        },
        'RBC': {
            'low': 'Increase iron intake with foods like spinach, red meat, or lentils.',
            'high': 'Stay hydrated; high RBC can be due to dehydration.'
        },
        'HGB': {
            'low': 'Consume iron and vitamin B12-rich foods (e.g., eggs, fish, fortified cereals).',
            'high': 'Ensure proper hydration; high hemoglobin may require medical evaluation.'
        },
        'PLT': {
            'low': 'Add vitamin K and folate-rich foods (e.g., leafy greens, nuts) to support platelets.',
            'high': 'Consult a doctor; high platelets can increase clotting risk.'
        }
    }

    # ✅ Generate analysis and suggestions
    analysis = []
    suggestions = []
    for i, component in enumerate(feature_order):
        low_pred = predictions[i]
        high_pred = predictions[i + len(feature_order)]
        value = extracted_values[component]
        if low_pred == 1:
            status = "Low"
            suggestion = recommendations[component]['low']
        elif high_pred == 1:
            status = "High"
            suggestion = recommendations[component]['high']
        else:
            status = "Normal"
            suggestion = "No specific action needed; maintain a healthy lifestyle."
        analysis.append(f"{component}: {value} ({status})")
        if status != "Normal":
            suggestions.append(suggestion)

    # ✅ Format everything as a single string
    analysis_text = "\n".join(analysis)
    suggestions_text = "\n".join(f"- {s}" for s in suggestions) if suggestions else "- None needed."
    
    result_text = f"""
    Blood Component Analysis:
    {analysis_text}
    Suggestions:
    {suggestions_text}
    """

    return {"prediction": result_text.strip()}

# ✅ Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8011)