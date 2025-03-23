from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import requests
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Matches your React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_SERVERS = {
    "brain": {"predict": "http://127.0.0.1:8004/predict/", "gradcam": "http://127.0.0.1:8004/gradcam/"},
    "lung": {"predict": "http://127.0.0.1:8002/predict/", "gradcam": "http://127.0.0.1:8002/gradcam/"},
    "lung_size": {"predict": "http://127.0.0.1:8003/predict/", "gradcam": "http://127.0.0.1:8003/gradcam/"},
    "brest": {"predict": "http://127.0.0.1:8005/predict/", "gradcam": "http://127.0.0.1:8005/gradcam/"},
    "diabetes": {"predict": "http://127.0.0.1:8006/predict/", "gradcam": None},
    "heart": {"predict": "http://127.0.0.1:8007/predict/", "gradcam": None},
    "cancer": {"predict": "http://127.0.0.1:8008/predict/", "gradcam": None},
    "psad": {"predict": "http://127.0.0.1:8009/predict/", "gradcam": None},
    "text": {"predict": "http://127.0.0.1:8010/predict/", "gradcam": None},
    "blood": {"predict": "http://127.0.0.1:8011/predict/", "gradcam": None},
    "skin": {"predict": "http://127.0.0.1:8012/predict/", "gradcam": "http://127.0.0.1:8012/gradcam/"},
    "pancreas": {"predict": "http://127.0.0.1:8013/predict/", "gradcam": "http://127.0.0.1:8013/gradcam/"},  # Added pancreas
}

@app.post("/predict/")
async def predict(model_name: str = Form(...), file: UploadFile = File(...)):
    if model_name not in MODEL_SERVERS:
        return {"error": f"Model '{model_name}' not found. Available models: {list(MODEL_SERVERS.keys())}"}

    model_url = MODEL_SERVERS[model_name]["predict"]
    files = {"file": (file.filename, await file.read(), file.content_type)}
    
    try:
        response = requests.post(model_url, files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to get prediction from {model_name} server: {str(e)}"}

@app.post("/gradcam/")
async def get_gradcam(model_name: str = Form(...), file: UploadFile = File(...)):
    if model_name not in MODEL_SERVERS:
        return {"error": f"Model '{model_name}' not found. Available models: {list(MODEL_SERVERS.keys())}"}
    
    gradcam_url = MODEL_SERVERS[model_name]["gradcam"]
    if gradcam_url is None:
        return {"error": f"Grad-CAM is not supported for the '{model_name}' model"}

    files = {"file": (file.filename, await file.read(), file.content_type)}
    
    try:
        response = requests.post(gradcam_url, files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to get Grad-CAM from {model_name} server: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)