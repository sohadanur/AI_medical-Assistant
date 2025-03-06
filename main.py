from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import whisper
import os
import uuid
import sqlite3
import base64
from datetime import datetime
from pathlib import Path
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create required directories if they don't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

TEMPLATES_DIR = Path("templates")
TEMPLATES_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Whisper Transcription App")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Initialize whisper model
model = None

# Database setup
def init_db():
    try:
        db_path = "transcriptions.db"
        logger.info(f"Initializing database at {os.path.abspath(db_path)}")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transcriptions (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            audio_path TEXT NOT NULL,
            transcription TEXT NOT NULL,
            upload_date TIMESTAMP NOT NULL
        )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def save_transcription(id, filename, audio_path, transcription):
    try:
        logger.info(f"Saving transcription with ID: {id}")
        conn = sqlite3.connect("transcriptions.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO transcriptions (id, filename, audio_path, transcription, upload_date) VALUES (?, ?, ?, ?, ?)",
            (id, filename, audio_path, transcription, datetime.now())
        )
        conn.commit()
        conn.close()
        logger.info(f"Transcription saved successfully with ID: {id}")
    except Exception as e:
        logger.error(f"Error saving transcription: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def get_all_transcriptions():
    try:
        logger.info("Retrieving all transcriptions")
        conn = sqlite3.connect("transcriptions.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM transcriptions ORDER BY upload_date DESC")
        rows = cursor.fetchall()
        transcriptions = [dict(row) for row in rows]
        conn.close()
        logger.info(f"Retrieved {len(transcriptions)} transcriptions")
        return transcriptions
    except Exception as e:
        logger.error(f"Error retrieving transcriptions: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def get_transcription(id):
    try:
        logger.info(f"Retrieving transcription with ID: {id}")
        conn = sqlite3.connect("transcriptions.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM transcriptions WHERE id = ?", (id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            logger.info(f"Transcription found with ID: {id}")
            return dict(row)
        else:
            logger.warning(f"No transcription found with ID: {id}")
            return None
    except Exception as e:
        logger.error(f"Error retrieving transcription: {str(e)}")
        logger.error(traceback.format_exc())
        return None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        logger.info("Loading Whisper model...")
        model = whisper.load_model("medium")
        logger.info("Whisper model loaded successfully!")
        init_db()
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        transcriptions = get_all_transcriptions()
        logger.info(f"Rendering index page with {len(transcriptions)} transcriptions")
        return templates.TemplateResponse("index.html", {"request": request, "transcriptions": transcriptions})
    except Exception as e:
        logger.error(f"Error rendering index page: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"message": f"Internal server error: {str(e)}"})

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file for transcription: {file.filename}")
        
        # Generate a unique ID for this transcription
        transcription_id = str(uuid.uuid4())
        
        # Save the uploaded file
        file_extension = file.filename.split(".")[-1]
        audio_path = f"uploads/{transcription_id}.{file_extension}"
        
        logger.info(f"Saving file to: {audio_path}")
        with open(audio_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Transcribe the audio
        logger.info("Starting transcription...")
        result = model.transcribe(audio_path)
        transcription_text = result["text"]
        logger.info(f"Transcription complete: {transcription_text[:100]}...")
        
        # Save to database
        save_transcription(
            id=transcription_id,
            filename=file.filename,
            audio_path=audio_path,
            transcription=transcription_text
        )
        
        return {
            "id": transcription_id,
            "filename": file.filename,
            "transcription": transcription_text
        }
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"message": f"Transcription error: {str(e)}"})

@app.get("/transcription/{transcription_id}")
async def get_transcription_details(request: Request, transcription_id: str):
    try:
        logger.info(f"Retrieving details for transcription: {transcription_id}")
        transcription = get_transcription(transcription_id)
        if not transcription:
            logger.warning(f"Transcription not found: {transcription_id}")
            return JSONResponse(status_code=404, content={"message": "Transcription not found"})
        
        # Read audio file and convert to base64 for embedding in HTML
        logger.info(f"Loading audio file: {transcription['audio_path']}")
        with open(transcription["audio_path"], "rb") as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode("utf-8")
        
        transcription["audio_data"] = audio_data
        return templates.TemplateResponse(
            "transcription_details.html", 
            {"request": request, "transcription": transcription}
        )
    except Exception as e:
        logger.error(f"Error retrieving transcription details: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})