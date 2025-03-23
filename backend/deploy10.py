from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
from gtts import gTTS
import os
import io
from transformers import pipeline
import uvicorn

app = FastAPI()

# ✅ Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load the summarization model
summarizer = pipeline("summarization")
print("✅ Summarization Model Loaded Successfully!")

# ✅ Ensure static folder exists
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# ✅ Feature extraction function
def extract_pdf_data(pdf_bytes):
    """Extract text from PDF bytes"""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    if not text:
        return None
    return text

# ✅ Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    
    # Extract text
    text = extract_pdf_data(pdf_bytes)
    if text is None:
        return {"error": "Invalid PDF format. Could not extract text."}

    # Summarize text
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]

    # Convert to speech
    audio_filename = "summary_audio.mp3"
    audio_path = os.path.join(STATIC_DIR, audio_filename)
    tts = gTTS(text=summary, lang="en")
    tts.save(audio_path)

    # ✅ Format everything as a single string
    audio_url = f"http://127.0.0.1:8010/static/{audio_filename}"
    result_text = f"""
    Summary: {summary}
    Audio URL: {audio_url}
    Instructions: Access the audio file at the URL above or download it from the server.
    """

    return {"prediction": result_text.strip()}

# ✅ Serve static files (audio)
@app.get("/static/{filename}")
async def serve_static(filename: str):
    file_path = os.path.join(STATIC_DIR, filename)
    if os.path.exists(file_path):
        return HTMLResponse(content=open(file_path, "rb").read(), media_type="audio/mp3")
    return {"error": "Audio file not found"}

# ✅ Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)