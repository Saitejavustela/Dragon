from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
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
MODEL_PATH ="skin_cancer_model.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Skin Cancer Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise RuntimeError(f"Model loading failed: {str(e)}")

# ✅ Function to preprocess the image
def preprocess_image(image_bytes, target_size=(224, 224)):
    """Preprocess image bytes for model input"""
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Error: Image cannot be decoded.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img_normalized = img / 255.0  # Normalize for model input
    img_uint8 = img.astype(np.uint8)  # Convert to uint8 for lesion detection
    img_array = np.expand_dims(img_normalized, axis=0)
    return img_array, img_uint8

# ✅ Function to determine recovery percentage based on stage
def get_recovery_percentage(stage):
    recovery_rates = {
        "Mild (Early Stage)": 92,
        "Moderate (Intermediate Stage)": 60,
        "Severe (Advanced Stage)": 30
    }
    return recovery_rates.get(stage, "Unknown")

# ✅ Function to find lesion size and classify severity
def find_lesion_size(img):
    """Find lesion size and severity without heatmap"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, "No lesion detected"
    
    max_contour = max(contours, key=cv2.contourArea)
    _, _, w, h = cv2.boundingRect(max_contour)
    lesion_length = max(w, h)
    
    if lesion_length < 50:
        stage = "Mild (Early Stage)"
    elif 50 <= lesion_length < 150:
        stage = "Moderate (Intermediate Stage)"
    else:
        stage = "Severe (Advanced Stage)"
    
    return lesion_length, stage, None

# ✅ Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Validate file
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return {"error": "Only JPG, JPEG, or PNG files are supported"}

    image_bytes = await file.read()
    
    # Preprocess image
    try:
        input_image, original_img = preprocess_image(image_bytes)
    except ValueError as e:
        return {"error": str(e)}

    # Make prediction
    prediction = model.predict(input_image)
    if prediction.shape[-1] == 1:
        predicted_class = "Cancerous" if prediction[0][0] > 0.7 else "Non-Cancerous"
    else:
        predicted_class = ["Non-Cancerous", "Cancerous"][np.argmax(prediction[0])]

    # Find lesion size and severity
    lesion_length, stage, error = find_lesion_size(original_img)
    
    # Generate result text
    if lesion_length is not None:
        recovery_percentage = get_recovery_percentage(stage)
        result_text = f"""
        Prediction: {predicted_class}
        Lesion Length: {lesion_length} pixels
        Cancer Stage: {stage}
        Estimated Recovery Rate: {recovery_percentage}%
        Suggestions:
        - Consult a dermatologist for a detailed evaluation.
        - Consider a biopsy if cancerous to confirm diagnosis.
        - Follow up regularly based on stage severity.
        """
    else:
        result_text = f"""
        Prediction: {predicted_class}
        Lesion Analysis: {error}
        Suggestions:
        - Consult a dermatologist if symptoms persist despite no lesion detection.
        """

    return {"prediction": result_text.strip()}

# ✅ Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8012)