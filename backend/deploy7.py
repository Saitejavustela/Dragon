from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import spacy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Load and Preprocess the Dataset
file_path = "processed.cleveland.data"
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
df = pd.read_csv(file_path, names=column_names)

df.replace("?", pd.NA, inplace=True)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df.fillna(df.median(), inplace=True)

def map_risk(num):
    return "Low" if num == 0 else "Medium" if num in [1, 2] else "High"

df["risk_category"] = df["num"].apply(map_risk)
df.drop("num", axis=1, inplace=True)

# Generate Synthetic Reports
def create_report(row):
    sex = "male" if row["sex"] == 1 else "female"
    cp = {1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"}
    chest_pain = cp.get(row["cp"], "unknown")
    exang = "exercise-induced angina" if row["exang"] == 1 else "no exercise-induced angina"
    return (f"{int(row['age'])}-year-old {sex} with blood pressure of {int(row['trestbps'])} mm Hg, "
            f"cholesterol of {int(row['chol'])} mg/dl, maximum heart rate of {int(row['thalach'])}, "
            f"chest pain type: {chest_pain}, and {exang}.")

df["report"] = df.apply(create_report, axis=1)

# Feature Extraction
def extract_features(report):
    doc = nlp(report)
    features = {"age": None, "sex": None, "trestbps": None, "chol": None, "thalach": None, "cp": None, "exang": None}

    for ent in doc.ents:
        if ent.label_ == "QUANTITY":
            if "year" in ent.text:
                features["age"] = float(ent.text.split("-")[0])
            elif "mm Hg" in ent.text:
                features["trestbps"] = float(ent.text.split()[0])
            elif "mg/dl" in ent.text:
                features["chol"] = float(ent.text.split()[0])

    features["sex"] = 1 if "male" in report else 0 if "female" in report else None
    features["cp"] = 1 if "typical angina" in report else 2 if "atypical angina" in report else 3 if "non-anginal pain" in report else 4
    features["exang"] = 1 if "exercise-induced angina" in report else 0

    for token in doc:
        if token.text == "maximum" and token.nbor(2).text == "rate":
            features["thalach"] = float(token.nbor(4).text)

    for key in features:
        if features[key] is None:
            features[key] = df[key].median()

    return features

# Train the Model
extracted_features = df["report"].apply(extract_features)
features_df = pd.DataFrame(extracted_features.tolist())

X = features_df
y = df["risk_category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# API Data Model
class ReportInput(BaseModel):
    report: str

@app.post("/predict/")
def predict_risk(data: ReportInput):
    features = extract_features(data.report)
    features_df = pd.DataFrame([features], columns=X.columns)
    predicted_risk = model.predict(features_df)[0]

    recommendations = []
    if predicted_risk == "High":
        recommendations.append("Consult a cardiologist immediately.")
        if features["trestbps"] > 140:
            recommendations.append("Your blood pressure is high. Consider medication and reduce salt intake.")
        if features["chol"] > 240:
            recommendations.append("Your cholesterol is high. Adopt a low-fat diet and exercise regularly.")
    elif predicted_risk == "Medium":
        recommendations.append("Monitor your health closely and schedule a follow-up with your doctor.")
        if features["trestbps"] > 140:
            recommendations.append("Your blood pressure is high. Try stress management techniques.")
        if features["chol"] > 240:
            recommendations.append("Your cholesterol is high. Increase physical activity.")
    else:
        recommendations.append("Maintain a healthy lifestyle to keep your risk low.")

    prediction = {
        "predicted_risk": predicted_risk,
        "features": features,
        "recommendations": recommendations
    }

    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
