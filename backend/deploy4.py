from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
import numpy as np
import cv2
import base64
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Class labels
CLASS_TYPES = ['glioma', 'meningioma', 'notumor', 'pituitary']
N_CLASSES = len(CLASS_TYPES)

# ✅ Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {device}")

# ✅ Load trained model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
try:
    model.load_state_dict(torch.load("brain_tumor_classifier.pth", map_location=device))
    model = model.to(device)
    model.eval()
    print("✅ Brain Tumor Classifier Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ✅ Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ✅ Grad-CAM implementation
def grad_cam(model, img_tensor):
    model.eval()
    features = []
    gradients = []

    # Hook to capture features and gradients from the last conv layer (layer4)
    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hooks to layer4 (last conv block in ResNet18)
    hook_handle_forward = model.layer4.register_forward_hook(forward_hook)
    hook_handle_backward = model.layer4.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(img_tensor)
    predicted_idx = torch.argmax(output, 1)

    # Backward pass
    model.zero_grad()
    output[0, predicted_idx].backward()

    # Compute Grad-CAM
    pooled_grads = torch.mean(gradients[0], dim=[0, 2, 3])
    activations = features[0][0]  # Shape: [channels, H, W]

    for i in range(activations.shape[0]):
        activations[i] *= pooled_grads[i]

    heatmap = torch.mean(activations, dim=0).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1  # Normalize

    # Clean up hooks
    hook_handle_forward.remove()
    hook_handle_backward.remove()

    return heatmap

# ✅ Overlay heatmap and mark tumor area
def overlay_heatmap(image, heatmap):
    """Overlay heatmap on image and mark tumor with a circle"""
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    overlayed_img = cv2.addWeighted(image, 0.5, heatmap_colored, 0.5, 0)

    # Mark tumor area
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

# ✅ Prediction Endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return {"error": "Only JPG, JPEG, or PNG files are supported"}

    if model is None:
        return {"error": f"Model not loaded. Path attempted: brain_tumor_classifier.pth"}

    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Get Prediction
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            tumor_type = CLASS_TYPES[predicted.item()]

        return {"prediction": tumor_type}
    
    except Exception as e:
        return {"error": f"❌ Error: {str(e)}"}

# ✅ Grad-CAM Endpoint
@app.post("/gradcam/")
async def get_gradcam(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return {"error": "Only JPG, JPEG, or PNG files are supported"}

    if model is None:
        return {"error": f"Model not loaded. Path attempted: brain_tumor_classifier.pth"}

    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)  # Keep original for overlay

        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Generate Grad-CAM heatmap
        heatmap = grad_cam(model, image_tensor)
        heatmap_base64 = overlay_heatmap(image_np, heatmap)

        return {"heatmap_image": f"data:image/png;base64,{heatmap_base64}"}
    
    except Exception as e:
        return {"error": f"❌ Error: {str(e)}"}

# ✅ Run the FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004)