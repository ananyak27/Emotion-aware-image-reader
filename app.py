
import os
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import pytesseract
from PIL import Image
import requests

from dotenv import load_dotenv

import tempfile
import time

             
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from nltk.corpus import words

import threading
import webbrowser

import nltk
import sys
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import pickle
import torch

print("=" * 50)
print("Current Working Directory:", os.getcwd())
print("emotion_model exists:", os.path.exists("emotion_model"))

if os.path.exists("emotion_model"):
    print("emotion_model contents:", os.listdir("emotion_model"))
else:
    print("emotion_model folder NOT FOUND")

print("=" * 50)


model = DistilBertForSequenceClassification.from_pretrained("emotion_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("emotion_model")
le = pickle.load(open("label_encoder.pkl", "rb"))

model.eval()



try:
    nltk.data.find("corpora/words")
except LookupError:
    nltk.download("words", quiet=True)



def open_browser():
    time.sleep(3)  
    webbrowser.open("http://127.0.0.1:5000")


english_words = set(words.words())

import transformers
transformers.logging.set_verbosity_error()


# Load TrOCR model
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")


def text_score(text):
    if not text:
        return 0
    tokens = text.lower().split()
    valid = sum(1 for w in tokens if w in english_words)
    return valid / len(tokens) if tokens else 0


# ============ NEW OCR SYSTEM (Printed + Handwriting) ============

def ocr_tesseract(img):
    txt = pytesseract.image_to_string(img)
    return " ".join(txt.split()).strip()

def ocr_trocr(img):
    try:
        pixel_values = trocr_processor(images=img, return_tensors="pt").pixel_values
        output_ids = trocr_model.generate(pixel_values, max_length=150)
        txt = trocr_processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return txt.strip()
    except:
        return ""

# ===== Preprocess OCR text =====
def preprocess_text(text):
    if not text:
        return ""
    # Replace "|" with "I"
    text = text.replace("|", "I")
    # Remove extra spaces
    text = " ".join(text.split())
    return text



def best_ocr(img):
    # Step 1: Try Tesseract first (fast)
    tesseract_text = ocr_tesseract(img)
    score_print = text_score(tesseract_text)

    # If printed text looks good → don't run slow TrOCR
    if score_print > 0.4:
        return "printed", tesseract_text

    # Step 2: Only run TrOCR if printed text was bad
    trocr_text = ocr_trocr(img)
    score_hand = text_score(trocr_text)

    if score_hand > score_print:
        return "handwritten", trocr_text
    else:
        return "printed", tesseract_text


load_dotenv()

tesseract_path = os.getenv("TESSERACT_PATH")

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Fetch API keys from environment variables
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY not found. Create a .env file.")

ELEVEN_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"


# ================= FUNCTIONS =================
def detect_emotion(text):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64
    )

    with torch.no_grad():
        out = model(**enc)

    pred = torch.argmax(out.logits).item()
    return le.inverse_transform([pred])[0]



def emotion_to_voice(emotion):
    emotion_voice_map = {
        # "happiness": "21m00Tcm4TlvDq8ikWAM",  # handled by Murf AI 🎉
        "sadness": "EXAVITQu4vr4xnSDxMaL",    # Sarah
        "anger": "CwhRBWXzGAHq8TQ4Fs17",      # Roger
        "fear": "SAz9YHcvj6GT2YYXdXww",       # River
        "love": "Xb7hH8MSUJpSbSDYk0k2",       # Alice
        "surprise": "XrExE9yKIg1WjnnlVkGX",   # Matilda
        "disgust": "pqHfZKP75CvOlQylNhV4",    # Bill
        "hope": "cgSgspJ2msm6clMCkdW9",       # Jessica
        "calm": "iP95p4xoKVk53GoZ742B",       # Chris
        "loneliness": "nPczCjzI2devNBz1zQrb"  # Brian
    }

    return emotion_voice_map.get(emotion.lower())



# ================= MAIN =================
from flask import Flask, request, jsonify, send_from_directory
app = Flask(__name__)

def normalize_emotion(emotion):
    if not emotion:
        return None

    e = emotion.strip().lower()

    # direct simple words
    replacements = {
        "happy": "happiness",
        "joy": "happiness",
        "joyful": "happiness",

        "sad": "sadness",
        "sorrow": "sadness",

        "angry": "anger",
        "mad": "anger",
        "furious": "anger",

        "scared": "fear",
        "afraid": "fear",

        "loving": "love",
        "affection": "love",

        "surprised": "surprise",

        "gross": "disgust",
        "disgusting": "disgust",

        "calming": "calm",
        "relaxed": "calm",

        "lonely": "loneliness",
        "missing you": "loneliness",
        "longing": "loneliness",
        "nostalgia": "loneliness",

        "hopeful": "hope",
    }

    if e in replacements:
        return replacements[e]

    # --- NEW INTELLIGENT FALLBACKS ---
    if "happy" in e or "joy" in e:
        return "happiness"
    if "sad" in e or "down" in e:
        return "sadness"
    if "ang" in e:
        return "anger"
    if "fear" in e or "scare" in e:
        return "fear"
    if "love" in e or "affection" in e:
        return "love"
    if "surpris" in e:
        return "surprise"
    if "disgust" in e or "gross" in e:
        return "disgust"
    if "calm" in e or "relax" in e:
        return "calm"
    if "lonely" in e or "alone" in e or "miss" in e:
        return "loneliness"
    if "hope" in e:
        return "hope"

    # UNKNOWN → return original (so we can debug)
    return e

def generate_tts_bytes(text, emotion):
    """
    Unified TTS generator:
    - Uses Murf AI for 'happiness'
    - Uses ElevenLabs for all other emotions
    """

    # -------------------------------
    # 1️⃣ MURF AI for happiness
    # -------------------------------
    if emotion.lower() == "happiness":
        MURF_API_KEY = os.getenv("MURF_API_KEY")

        if not MURF_API_KEY:
            raise ValueError("MURF_API_KEY not found. Create a .env file.")
        MURF_TTS_URL = "https://api.murf.ai/v1/speech/generate"

        headers = {
            "api-key": MURF_API_KEY,
            "Content-Type": "application/json"
        }

        payload = {
            "voiceId": "en-US-natalie",   # cheerful voice
            "text": text,
            "format": "mp3",
            "sampleRate": 44100
        }

        murf_response = requests.post(MURF_TTS_URL, headers=headers, json=payload)

        if murf_response.status_code != 200:
            raise Exception("Murf Error: " + murf_response.text)

        audio_url = murf_response.json().get("audioFile")
        if not audio_url:
            raise Exception("Murf error: audio file URL not found")

        # Download MP3 from Murf URL
        audio_data = requests.get(audio_url)
        return audio_data.content


    # --------------------------------
    # 2️⃣ ELEVENLABS for all other emotions
    # --------------------------------
    voice_id = emotion_to_voice(emotion)
    if not voice_id:
        raise Exception("Invalid emotion voice")

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }

    payload = {
        "model_id": "eleven_multilingual_v2",
        "text": text,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.9,
            "style": 0.8,
            "use_speaker_boost": True
        }
    }

    url = f"{ELEVEN_TTS_URL}/{voice_id}"
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception("ElevenLabs Error: " + response.text)

    return response.content


@app.route("/run", methods=["POST"])
def run_from_frontend():
    file = request.files.get("file")
    image_name = request.form.get("image_name", "").strip()

    # If file uploaded → use it
    if file:
        try:
            img = Image.open(file.stream).convert("RGB")
            text_type, clean_text = best_ocr(img)
            clean_text = preprocess_text(clean_text)

        except:
            return jsonify({"error": "Invalid uploaded image"})

    # If no file → use typed name
    elif image_name:
        possible_ext = ["png", "jpg", "jpeg"]
        image_path = None

        for ext in possible_ext:
            check = f"images/{image_name}.{ext}"
            if os.path.exists(check):
                image_path = check
                break

        if image_path is None:
            return jsonify({"error": "Image not found"})

        img = Image.open(image_path).convert("RGB")
        text_type, clean_text = best_ocr(img)
        clean_text = preprocess_text(clean_text)


    else:
        return jsonify({"error": "No image provided (neither upload nor name)"})

    if not clean_text:
        return jsonify({"error": "No text found in image"})

    # Emotion detect
    raw_emotion = detect_emotion(clean_text)
    emotion = normalize_emotion(raw_emotion)
    print("RAW EMOTION MODEL OUTPUT:", raw_emotion)
    print("NORMALIZED EMOTION:", emotion)

    # Generate TTS
    timestamp = int(time.time())
    output_filename = f"sound_{timestamp}.mp3"
    output_path = os.path.join("sounds", output_filename)

    audio_bytes = generate_tts_bytes(clean_text, emotion)

    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    return jsonify({
        "text": clean_text,
        "emotion": emotion,
        "audio_file": f"/audio/{output_filename}"
    })


@app.route("/audio/<filename>")
def serve_audio(filename):
    # Absolute path to your 'sounds' folder
    sound_folder = os.path.join(os.path.dirname(__file__), "sounds")
    full_path = os.path.join(sound_folder, filename)

    if not os.path.exists(full_path):
        return jsonify({"error": f"{filename} not found in sounds folder"}), 404

    print(f"🎧 Serving audio file: {full_path}")  # debug check
    return send_from_directory(sound_folder, filename, as_attachment=False)



@app.route("/")
def serve_frontend():
    return send_from_directory(".", "frontend.html")




if __name__ == "__main__":
    print("🌐 Starting Flask server...")

    # Start a background thread to open the browser
    threading.Thread(target=open_browser).start()

    app.run(debug=True, use_reloader=False)
