"""Flask server for the VoxGuard deepfake voice detector."""
import os
import tempfile
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("VOXGUARD_MODEL_PATH", r"D:\Project Folder\Mini Project\Files\best_model.keras"))
SAMPLE_RATE = 16000
IMG_SIZE = (128, 128)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Set VOXGUARD_MODEL_PATH to best_model.keras.")

# Load once at startup, instead of once for every uploaded recording.
model = tf.keras.models.load_model(MODEL_PATH)


def predict_audio(audio_path: Path) -> dict:
    """Use the same preprocessing and inference steps from predict_audio.py."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image_file:
        spectrogram_path = Path(image_file.name)
    try:
        plt.figure(figsize=(3, 3), dpi=128)
        librosa.display.specshow(log_mel, sr=sr, cmap="viridis")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(spectrogram_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        image = tf.keras.utils.load_img(spectrogram_path, target_size=IMG_SIZE, color_mode="rgb")
        image_array = tf.keras.utils.img_to_array(image)
        probability = float(model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0][0])
    finally:
        spectrogram_path.unlink(missing_ok=True)

    # Model labels from predict_audio.py: human = 0, nonhuman = 1.
    is_deepfake = probability >= 0.5
    confidence = (probability if is_deepfake else 1 - probability) * 100
    return {
        "label": "Deepfake Voice" if is_deepfake else "Genuine Voice",
        "confidence": round(confidence, 2),
        "risk": "High" if is_deepfake else "Low",
        "cnnScore": round(probability, 6),
        "deepfakeLikelihood": round(probability * 100, 2),
        "duration": round(len(y) / sr, 2),
    }


@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)


@app.post("/analyze")
def analyze():
    audio = request.files.get("audio")
    if audio is None or not audio.filename:
        return jsonify(error="Choose an audio file before starting analysis."), 400

    suffix = Path(secure_filename(audio.filename)).suffix.lower()
    if suffix not in {".mp3", ".wav", ".m4a", ".flac", ".ogg"}:
        return jsonify(error="Supported formats: MP3, WAV, M4A, FLAC, or OGG."), 400

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as upload_file:
        uploaded_path = Path(upload_file.name)
    try:
        audio.save(uploaded_path)
        return jsonify(predict_audio(uploaded_path))
    except Exception as error:
        app.logger.exception("Audio analysis failed")
        return jsonify(error=f"Could not analyze this audio file: {error}"), 500
    finally:
        uploaded_path.unlink(missing_ok=True)


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify(error="The audio file is larger than the 100 MB limit."), 413


if __name__ == "__main__":
    app.run(debug=True)
