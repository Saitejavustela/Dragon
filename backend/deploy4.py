#brain tumor
from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000"],  # Change "*" to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Class labels
CLASS_TYPES = ['glioma', 'meningioma', 'notumor', 'pituitary']
N_CLASSES = len(CLASS_TYPES)

# ✅ Device configuration (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {device}")

# ✅ Load trained model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, N_CLASSES)  
model.load_state_dict(torch.load("E:/task1/brain_tumor_classifier.pth", map_location=device))
model = model.to(device)
model.eval()
print("✅ Brain Tumor Classifier Model Loaded Successfully!")

# ✅ Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ✅ Prediction Endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess image
        image = transform(image).unsqueeze(0).to(device)

        # Get Prediction
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            tumor_type = CLASS_TYPES[predicted.item()]

        return {"prediction": tumor_type}
    
    except Exception as e:
        return {"error": f"❌ Error: {str(e)}"}

# ✅ Run the FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004)
