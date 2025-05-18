#!/usr/bin/env python3
"""
Self-contained web interface for audio processing pipeline
All processing functions are integrated directly in this file
"""
import os
import uuid
import yaml
import tempfile
import re
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, session

# Import necessary libraries for processing
import librosa
import soundfile as sf
import whisper
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_key_for_flask_app")
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Add template filter for newlines to <br>
@app.template_filter('nl2br')
def nl2br(value):
    if value:
        return value.replace('\n', '<br>\n')
    return value

# Default configuration
DEFAULT_CONFIG = {
    "audio_cleaning": {
        "sr": 16000,
        "silence_db": 25,
        "target_dbfs": -3.0
    },
    "text_cleaning": {
        "lowercase": True,
        "remove_non_alpha": False,
        "remove_fillers": True
    },
    "ner": {
        "model": "en_core_web_trf"
    },
    "summarization": {
        "model_name": "google/flan-t5-small"
    }
}

# Load config if exists, otherwise use defaults
ROOT = Path(__file__).resolve().parent
CFG_PATH = ROOT / "config" / "configs.yaml"
try:
    with open(CFG_PATH, "r") as f:
        CONFIG = yaml.safe_load(f)
except (FileNotFoundError, KeyError):
    CONFIG = DEFAULT_CONFIG
    # Ensure config directory exists
    os.makedirs(ROOT / "config", exist_ok=True)
    # Write default config
    with open(CFG_PATH, "w") as f:
        yaml.dump(DEFAULT_CONFIG, f)

# ---- AUDIO PROCESSING FUNCTIONS ----

def trim_silence(samples: np.ndarray, thresh_db: int = 25, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Remove silence from audio"""
    non_silent = librosa.effects.split(samples, top_db=thresh_db, frame_length=frame_length, hop_length=hop_length)
    if len(non_silent) == 0:
        return samples
    start, end = non_silent[0][0], non_silent[-1][1]
    return samples[start:end]

def peak_normalise(samples: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    """Normalize audio volume"""
    peak = np.max(np.abs(samples))
    if peak == 0:
        return samples
    target_linear = 10 ** (target_dbfs / 20)
    return samples * (target_linear / peak)

def process_audio(audio_path: str, output_path: str, sr: int, silence_db: int, target_dbfs: float) -> float:
    """Process a single audio file and return duration"""
    samples, _ = librosa.load(audio_path, sr=sr, mono=True)
    samples = trim_silence(samples, thresh_db=silence_db)
    samples = peak_normalise(samples, target_dbfs=target_dbfs)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write processed audio
    sf.write(output_path, samples, sr, subtype="PCM_16")
    
    # Return duration in seconds
    return len(samples) / sr

# ---- TEXT PROCESSING FUNCTIONS ----

def clean_text(text: str, lowercase: bool = True, remove_non_alpha: bool = False, remove_fillers: bool = True) -> str:
    """Clean and normalize text"""
    if lowercase:
        text = text.lower()
    
    if remove_fillers:
        fillers = ["uh", "um", "you know", "like", "i mean", "so", "well"]
        for word in fillers:
            text = re.sub(r"\b" + word + r"\b", "", text)
    
    if remove_non_alpha:
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---- SPEECH RECOGNITION FUNCTIONS ----

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio to text using Whisper"""
    # Load Whisper model (only load once if possible)
    model_size = "small"  # Can be tiny, base, small, medium, large
    asr = whisper.load_model(model_size, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Transcribe
    result = asr.transcribe(audio_path, fp16=torch.cuda.is_available())
    
    return result["text"].strip()

# ---- NAMED ENTITY RECOGNITION FUNCTIONS ----

# Cache for models to avoid reloading
nlp_model = None

def get_spacy_model():
    """Load spaCy model with caching"""
    global nlp_model
    if nlp_model is None:
        model_name = CONFIG.get("ner", {}).get("model", "en_core_web_trf")
        try:
            nlp_model = spacy.load(model_name)
        except OSError:
            from spacy.cli import download
            download(model_name)
            nlp_model = spacy.load(model_name)
    return nlp_model

def extract_entities(text: str) -> tuple:
    """Extract named entities from text"""
    nlp = get_spacy_model()
    doc = nlp(text)
    
    entities = []
    labels = []
    
    for ent in doc.ents:
        entities.append(ent.text)
        labels.append(ent.label_)
    
    return entities, labels

# ---- TEXT SUMMARIZATION FUNCTIONS ----

# Cache for models to avoid reloading
t5_tokenizer = None
t5_model = None
device = None

def get_t5_model():
    """Load T5 model with caching"""
    global t5_tokenizer, t5_model, device
    if t5_tokenizer is None or t5_model is None:
        model_name = CONFIG.get("summarization", {}).get("model_name", "google/flan-t5-small")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t5_tokenizer = AutoTokenizer.from_pretrained(model_name)
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return t5_tokenizer, t5_model, device

def summarize(text: str, prompt: str, max_new_tokens: int) -> str:
    """Generate summary using T5 model"""
    tokenizer, model, device = get_t5_model()
    
    inp = prompt + text
    inputs = tokenizer(
        inp,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length
    ).to(device)
    
    outs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        early_stopping=True
    )
    
    return tokenizer.decode(outs[0], skip_special_tokens=True).strip()

# ---- FLASK ROUTES ----

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    # Check if file was uploaded
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    # Check file extension
    allowed_extensions = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}
    if not file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
        flash('Invalid file type. Supported types: wav, mp3, flac, ogg, m4a')
        return redirect(url_for('index'))
    
    try:
        # Generate unique ID for this session
        session_id = str(uuid.uuid4())
        
        # Create a temp directory for this upload
        temp_dir = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded file
        input_path = os.path.join(temp_dir, file.filename)
        file.save(input_path)
        
        # Process audio file
        sr = CONFIG.get("audio_cleaning", {}).get("sr", 16000)
        silence_db = CONFIG.get("audio_cleaning", {}).get("silence_db", 25)
        target_dbfs = CONFIG.get("audio_cleaning", {}).get("target_dbfs", -3.0)
        
        # Clean audio
        cleaned_path = os.path.join(temp_dir, f"cleaned_{file.filename}")
        duration = process_audio(input_path, cleaned_path, sr, silence_db, target_dbfs)
        
        # Transcribe audio
        transcript = transcribe_audio(cleaned_path)
        
        # Clean text
        cleaned_text = clean_text(
            transcript, 
            lowercase=CONFIG.get("text_cleaning", {}).get("lowercase", True),
            remove_non_alpha=CONFIG.get("text_cleaning", {}).get("remove_non_alpha", False),
            remove_fillers=CONFIG.get("text_cleaning", {}).get("remove_fillers", True)
        )
        
        # Extract entities
        entities, labels = extract_entities(cleaned_text)
        
        # Generate summaries
        tiny_summary = summarize(cleaned_text, "Summarize in one sentence: ", 32)
        short_summary = summarize(cleaned_text, "Summarize in one paragraph: ", 128)
        long_summary = summarize(cleaned_text, "Summarize in detail (multiple paragraphs): ", 256)
        
        # Store results in session
        session['results'] = {
            'filename': file.filename,
            'duration': f"{duration:.2f}s",
            'transcript': transcript,
            'cleaned_text': cleaned_text,
            'entities': list(zip(entities, labels)),
            'tiny_summary': tiny_summary,
            'short_summary': short_summary,
            'long_summary': long_summary
        }
        
        # Clean up temp files to save space
        try:
            os.remove(input_path)
            os.remove(cleaned_path)
        except:
            pass  # Ignore cleanup errors
        
        return redirect(url_for('results'))
    
    except Exception as e:
        flash(f'Error processing file: {str(e)}')
        return redirect(url_for('index'))

@app.route('/results')
def results():
    """Display processing results"""
    if 'results' not in session:
        flash('No results to display. Please upload an audio file first.')
        return redirect(url_for('index'))
    
    return render_template('results.html', results=session['results'])

if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)
