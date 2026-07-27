#t2 lung
from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import io
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import cv2

app = FastAPI()

# ✅ Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load the trained model
MODEL_PATH = "pulmonary_nodule_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Pulmonary Nodule Classification Model Loaded Successfully!")

# ✅ Class labels
CLASS_LABELS = ["Benign", "Malignant", "Normal"]

# ✅ Preprocessing function
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (224, 224))  # Resize
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# ✅ Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)

    # 🔍 Model prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    return {
        "prediction": CLASS_LABELS[predicted_class],
        "confidence": f"{confidence:.2f}%"
    }

# ✅ Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
