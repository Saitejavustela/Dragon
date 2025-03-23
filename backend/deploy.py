from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.transforms as transforms
import io
from PIL import Image
import uvicorn
import torchvision.models as models
import torch.nn as nn
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🟢 Define available models and their paths
MODELS = {
    "lung": "pulmonary_nodule_model.keras",
    "lung_size": "nodule_size_predictor.h5",
}

# 🟢 Define class mappings (for classification models)
CLASS_MAPPING = {
    "tumor": {0: "Normal", 1: "Cyst", 2: "Stone", 3: "Tumor"},
}

# 🟢 Define Model Architectures
class TumorClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(TumorClassifier, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class NoduleSizePredictor(nn.Module):
    def __init__(self):
        super(NoduleSizePredictor, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)  # Single output for size prediction

    def forward(self, x):
        return self.model(x)

# 🟢 Load models dynamically
loaded_models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for name, path in MODELS.items():
    if name == "tumor":
        model = TumorClassifier(num_classes=4)
    elif name == "nodule_size":
        model = NoduleSizePredictor()
    elif name == "lung":
        model1 = Sequential([
            Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            MaxPooling2D(2, 2),

            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),

            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),

            Flatten(),
            Dense(512, activation="relu"),
            Dropout(0.5),
            Dense(train_generator.num_classes, activation="softmax")  # Match class count
        ])
        model = tf.keras.models.load_model(model_path)
    else:
        continue  # Skip unknown models

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    loaded_models[name] = model

# 🟢 Preprocessing Functions
#def preprocess_tumor_image(image_bytes):
 #   IMAGE_SIZE = (224, 224)
    
 #   transform = transforms.Compose([
 #       transforms.ToPILImage(),
  #      transforms.Resize(IMAGE_SIZE),
  #      transforms.ToTensor(),
    #    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  #  ])
    
   # image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
   # return transform(image).unsqueeze(0).to(device)
def lung_preprocess_image(image_path):
    if not os.path.exists(image_path):
        print("⚠️ Image file not found! Check the path and try again.")
        sys.exit()

    img = image.load_img(image_path, target_size=(224, 224))  # Resize
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

def preprocess_nodule_size_image(image_bytes):
    IMAGE_SIZE = (224, 224)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale if needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Single-channel normalization
    ])
    
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
    return transform(image).unsqueeze(0).to(device)

# 🟢 Function to dynamically select preprocessing method
def preprocess_image(image_bytes, model_name, prediction_type):
    if model_name == "lung" and prediction_type == "classification":
        return preprocess_tumor_image(image_bytes)
    elif model_name == "lung_size" and prediction_type == "regression":
        return preprocess_nodule_size_image(image_bytes)
    else:
        raise ValueError(f"Unsupported model ({model_name}) or prediction type ({prediction_type}).")

# 🟢 Prediction Endpoint
@app.post("/predict/{model_name}/{prediction_type}/")
async def predict(model_name: str, prediction_type: str, file: UploadFile = File(...)):
    print(f"🔍 Received request for model: {model_name}, prediction type: {prediction_type}, file: {file.filename}")

    if model_name not in loaded_models:
        return {"error": "Invalid model name"}

    image_bytes = await file.read()

    try:
        # Apply the correct preprocessing
        input_tensor = preprocess_image(image_bytes, model_name, prediction_type)
    except ValueError as e:
        return {"error": str(e)}

    with torch.no_grad():
        output = loaded_models[model_name](input_tensor)

        if prediction_type == "classification":
            predicted_class = torch.argmax(output, dim=1).item()
            return {"model": model_name, "prediction": CLASS_MAPPING[model_name].get(predicted_class, "Unknown")}

        elif prediction_type == "regression":
            predicted_size = output.item()  # Single floating-point value
            return {"model": model_name, "nodule_size": f"{predicted_size:.2f} mm"}

    return {"error": "Unexpected error during prediction"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
