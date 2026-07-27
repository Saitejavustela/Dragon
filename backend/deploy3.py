from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
import io
from PIL import Image
import cv2
import base64
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

# ✅ Load the trained model
MODEL_PATH = "nodule_size_predictor.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Nodule Size Prediction Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ✅ Function to preprocess image
def preprocess_image(image_bytes, target_size=(256, 256)):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
        img = img.resize(target_size)  # Resize
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel (256, 256, 1)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch (1, 256, 256, 1)
        return img_array, np.array(img)  # Return processed array and original for Grad-CAM
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None, None

# ✅ Grad-CAM implementation
def grad_cam(model, img_array, layer_name):
    """Generate Grad-CAM heatmap"""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # Regression output (nodule size)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)  # ReLU
    heatmap = heatmap / tf.reduce_max(heatmap) if tf.reduce_max(heatmap) != 0 else heatmap
    
    return heatmap.numpy()

# ✅ Overlay heatmap and mark nodule area
def overlay_heatmap(image, heatmap):
    """Overlay heatmap on image and mark nodule with a circle"""
    # Convert grayscale image to RGB for overlay
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize and colorize heatmap
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Blend images
    overlayed_img = cv2.addWeighted(image_rgb, 0.5, heatmap_colored, 0.5, 0)

    # Mark nodule area
    _, thresh = cv2.threshold(heatmap, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(overlayed_img, center, radius, (0, 0, 255), 2)  # Red circle

    _, buffer = cv2.imencode(".png", overlayed_img)
    return base64.b64encode(buffer).decode("utf-8")

# ✅ Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return {"error": "Only JPG, JPEG, or PNG files are supported"}

    if model is None:
        return {"error": f"Model not loaded. Path attempted: {MODEL_PATH}"}

    try:
        image_bytes = await file.read()
        processed_image, _ = preprocess_image(image_bytes)
        if processed_image is None:
            return {"error": "Image preprocessing failed"}

        prediction = model.predict(processed_image)
        predicted_value = float(prediction[0][0])  # Convert to Python float

        return {"prediction": predicted_value}
    except Exception as e:
        return {"error": f"❌ Error: {str(e)}"}

# ✅ Grad-CAM endpoint
@app.post("/gradcam/")
async def get_gradcam(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return {"error": "Only JPG, JPEG, or PNG files are supported"}

    if model is None:
        return {"error": f"Model not loaded. Path attempted: {MODEL_PATH}"}

    try:
        image_bytes = await file.read()
        processed_image, original_img = preprocess_image(image_bytes)
        if processed_image is None:
            return {"error": "Image preprocessing failed"}

        # Find last conv layer
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
        if not last_conv_layer_name:
            return {"error": "No convolutional layer found in the model"}

        heatmap = grad_cam(model, processed_image, last_conv_layer_name)
        heatmap_base64 = overlay_heatmap(original_img, heatmap)
        
        return {"heatmap_image": f"data:image/png;base64,{heatmap_base64}"}
    except Exception as e:
        return {"error": f"❌ Error: {str(e)}"}

# ✅ Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)