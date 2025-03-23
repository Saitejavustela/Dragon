from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import cv2
import base64
import uvicorn

app = FastAPI()

# ✅ Enable CORS for central API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:9000"],  # Central API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ✅ Define the model (must match your training architecture)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ✅ Load trained model
MODEL_PATH = "pdac_cnn_v3.pth"
model = CNNModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("✅ Model loaded successfully!")

# ✅ Define transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ✅ Preprocess image from bytes
def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
        image_tensor = transform(image).unsqueeze(0).to(device)
        return image_tensor, np.array(image)  # Return tensor for model and numpy array for Grad-CAM
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

# ✅ Grad-CAM implementation
def grad_cam(model, img_tensor):
    model.eval()
    features = []
    gradients = []

    # Hook to capture features from conv3 (forward)
    def forward_hook(module, input, output):
        features.append(output)

    # Hook to capture gradients from conv3 (backward)
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hooks to conv3
    hook_handle_forward = model.conv3.register_forward_hook(forward_hook)
    hook_handle_backward = model.conv3.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    
    # Backward pass for the predicted class
    model.zero_grad()
    output[0, predicted].backward()

    # Get pooled gradients and activations
    pooled_grads = torch.mean(gradients[0], dim=[0, 2, 3])  # Mean over spatial dimensions
    activations = features[0][0]  # Shape: [128, H, W]

    # Weight the channels by corresponding gradients
    for i in range(128):
        activations[i] *= pooled_grads[i]

    # Average across channels to get heatmap
    heatmap = torch.mean(activations, dim=0).detach().cpu().numpy()  # Detach before numpy()
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1  # Normalize

    # Clean up hooks
    hook_handle_forward.remove()
    hook_handle_backward.remove()

    return heatmap

# ✅ Overlay heatmap on original image and mark tumor with a circle
def overlay_heatmap(image, heatmap):
    # Convert grayscale image to 3-channel BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Resize and colorize heatmap
    heatmap_resized = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Blend the images
    overlayed_img = cv2.addWeighted(image_bgr, 0.5, heatmap_colored, 0.5, 0)

    # Threshold the heatmap to find the tumor region
    _, thresh = cv2.threshold(heatmap_uint8, 200, 255, cv2.THRESH_BINARY)
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
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return {"error": "Only JPG, JPEG, or PNG files are supported"}

    image_bytes = await file.read()
    try:
        image_tensor, _ = preprocess_image(image_bytes)
    except ValueError as e:
        return {"error": str(e)}

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        confidence = torch.softmax(output, dim=1)[0][predicted].item() * 100
        class_labels = ["Normal", "PDAC"]
        prediction = class_labels[predicted.item()]

    result_text = f"""
    Prediction: {prediction}
    Confidence: {confidence:.2f}%
    Suggestions:
    - Consult a specialist if PDAC is detected.
    """
    return {"prediction": result_text.strip()}

# ✅ Grad-CAM endpoint
@app.post("/gradcam/")
async def get_gradcam(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return {"error": "Only JPG, JPEG, or PNG files are supported"}

    image_bytes = await file.read()
    try:
        image_tensor, image_np = preprocess_image(image_bytes)
    except ValueError as e:
        return {"error": str(e)}

    heatmap = grad_cam(model, image_tensor)
    heatmap_base64 = overlay_heatmap(image_np, heatmap)
    return {"heatmap_image": f"data:image/png;base64,{heatmap_base64}"}

# ✅ Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8013)  # Matches "pancreas" model port