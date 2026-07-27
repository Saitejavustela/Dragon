from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import io
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import cv2
import base64

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
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    # Perform a dummy forward pass to initialize the model
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)  # Match preprocessing shape
    model.predict(dummy_input)  # Initialize layers
    print("✅ Pulmonary Nodule Classification Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ✅ Class labels
CLASS_LABELS = ["Benign", "Malignant", "Normal"]

# ✅ Preprocessing function
def preprocess_image(image_bytes, target_size=(224, 224)):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_img = np.array(img)  # Keep original for Grad-CAM
        img = cv2.resize(original_img, target_size)  # Resize
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 224, 224, 3)
        return img, original_img  # Return processed and original image
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None, None

# ✅ Grad-CAM implementation
def grad_cam(model, img_array, layer_name):
    """Generate Grad-CAM heatmap"""
    try:
        grad_model = tf.keras.models.Model(
            model.inputs,
            [model.get_layer(layer_name).output, model.output]
        )
    except ValueError as e:
        raise ValueError(f"Layer '{layer_name}' not found in model: {str(e)}")
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError(f"No gradients computed for layer '{layer_name}'")
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)  # ReLU
    heatmap = heatmap / tf.reduce_max(heatmap) if tf.reduce_max(heatmap) != 0 else heatmap
    
    return heatmap.numpy()

# ✅ Overlay heatmap and mark nodule area
def overlay_heatmap(image, heatmap):
    """Overlay heatmap on image and mark nodule with a circle"""
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    overlayed_img = cv2.addWeighted(image, 0.5, heatmap_colored, 0.5, 0)

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

    image_bytes = await file.read()
    processed_image, _ = preprocess_image(image_bytes)
    
    if processed_image is None:
        return {"error": "Image preprocessing failed"}

    # 🔍 Model prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    return {
        "prediction": CLASS_LABELS[predicted_class],
        "confidence": f"{confidence:.2f}%"
    }

# ✅ Grad-CAM endpoint
@app.post("/gradcam/")
async def get_gradcam(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return {"error": "Only JPG, JPEG, or PNG files are supported"}

    if model is None:
        return {"error": f"Model not loaded. Path attempted: {MODEL_PATH}"}

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

    try:
        heatmap = grad_cam(model, processed_image, last_conv_layer_name)
        heatmap_base64 = overlay_heatmap(original_img, heatmap)
        return {"heatmap_image": f"data:image/png;base64,{heatmap_base64}"}
    except Exception as e:
        return {"error": f"Grad-CAM failed: {str(e)}"}

# ✅ Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)