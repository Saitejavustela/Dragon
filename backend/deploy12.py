from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import io
import base64
import uvicorn

app = FastAPI()

# ✅ Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:9000"],  # Matches central API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load the trained model
MODEL_PATH = "skin_cancer_model.keras"
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

# ✅ Grad-CAM implementation for TensorFlow
def grad_cam(model, img_array, layer_name):
    """Generate Grad-CAM heatmap for the given image and layer"""
    # Create a model that outputs the activations of the specified layer and the final output
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the conv outputs with the pooled gradients
    conv_outputs = conv_outputs[0]  # Remove batch dimension
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)  # ReLU
    heatmap = heatmap / tf.reduce_max(heatmap) if tf.reduce_max(heatmap) != 0 else heatmap  # Normalize
    
    return heatmap.numpy()

# ✅ Overlay heatmap on image and mark tumor with a circle
def overlay_heatmap(image, heatmap):
    """Overlay heatmap on the image and draw a circle around the tumor"""
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Blend the heatmap with the original image
    overlayed_img = cv2.addWeighted(image, 0.5, heatmap_colored, 0.5, 0)

    # Threshold the heatmap to find the tumor region
    _, thresh = cv2.threshold(heatmap, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (assumed to be the tumor)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Draw a red circle around the tumor
        cv2.circle(overlayed_img, center, radius, (0, 0, 255), 2)  # Red circle, thickness 2

    # Encode the image
    _, buffer = cv2.imencode(".png", overlayed_img)
    return base64.b64encode(buffer).decode("utf-8")

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

# ✅ Grad-CAM endpoint
@app.post("/gradcam/")
async def get_gradcam(file: UploadFile = File(...)):
    # Validate file
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return {"error": "Only JPG, JPEG, or PNG files are supported"}

    image_bytes = await file.read()
    
    # Preprocess image
    try:
        input_image, original_img = preprocess_image(image_bytes)
    except ValueError as e:
        return {"error": str(e)}

    # Find the last convolutional layer
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
    if not last_conv_layer_name:
        return {"error": "No convolutional layer found in the model"}

    # Generate Grad-CAM heatmap
    heatmap = grad_cam(model, input_image, last_conv_layer_name)
    
    # Overlay heatmap and mark tumor
    heatmap_base64 = overlay_heatmap(original_img, heatmap)
    
    return {"heatmap_image": f"data:image/png;base64,{heatmap_base64}"}

# ✅ Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8012)  # Matches your skin model port