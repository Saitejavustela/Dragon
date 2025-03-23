#nodule size t23
from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# ✅ Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000"],  # Change "*" to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load the trained model
MODEL_PATH = "E:/task1/nodule_size_predictor.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Nodule Size Prediction Model Loaded Successfully!")

# ✅ Function to preprocess image
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
        img = img.resize((256, 256))  # Resize
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 256, 256, 1)
        return img_array
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

# ✅ Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        image_bytes = await file.read()
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        if processed_image is None:
            return {"error": "Image preprocessing failed"}

        # Predict
        prediction = model.predict(processed_image)
        predicted_value = float(prediction[0][0])  # Convert to Python float

        return {"prediction": predicted_value}  # ✅ Only return prediction
    except Exception as e:
        return {"error": f"❌ Error: {str(e)}"}

# ✅ Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)
