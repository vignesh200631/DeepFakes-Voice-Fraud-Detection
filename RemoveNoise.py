import os
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy import signal

# ============================================================
# PATHS
# ============================================================
INPUT_FILE = r"D:\Project Folder\Mini Project\Datasets\Audios\vishal 2.wav"

OUTPUT_FILE = r"D:\Project Folder\Mini Project\Datasets\Audios\cleaned_audio.wav"

TARGET_SR = 16000

# ============================================================
# START
# ============================================================

print("=" * 60)
print("        AUDIO BACKGROUND NOISE REMOVAL")
print("=" * 60)

# Check file
if not os.path.exists(INPUT_FILE):
    print("\nERROR: Input file not found:")
    print(INPUT_FILE)
    exit()

# ============================================================
# LOAD AUDIO
# ============================================================

print("\nLoading audio...")

audio, sr = sf.read(INPUT_FILE)

print("Audio loaded successfully")
print("Original Sample Rate :", sr)
print("Duration             :", round(len(audio) / sr, 2), "seconds")

# ============================================================
# CONVERT STEREO TO MONO
# ============================================================

if audio.ndim > 1:
    print("Converting stereo to mono...")
    audio = np.mean(audio, axis=1)

# Convert to float32
audio = audio.astype(np.float32)

# ============================================================
# RESAMPLE TO 16000 Hz
# ============================================================

if sr != TARGET_SR:

    print(f"Resampling {sr} Hz -> {TARGET_SR} Hz...")

    number_of_samples = round(
        len(audio) * TARGET_SR / sr
    )

    audio = signal.resample(
        audio,
        number_of_samples
    )

    sr = TARGET_SR

# ============================================================
# REMOVE DC OFFSET
# ============================================================

print("Removing DC offset...")

audio = audio - np.mean(audio)

# ============================================================
# NORMALIZE BEFORE DENOISING
# ============================================================

max_value = np.max(np.abs(audio))

if max_value > 0:
    audio = audio / max_value

# ============================================================
# NOISE REDUCTION
# ============================================================

print("Removing background noise...")

clean_audio = nr.reduce_noise(
    y=audio,
    sr=sr,
    stationary=False,
    prop_decrease=0.80
)

# ============================================================
# NORMALIZE OUTPUT
# ============================================================

print("Normalizing cleaned audio...")

max_value = np.max(np.abs(clean_audio))

if max_value > 0:
    clean_audio = clean_audio / max_value

# Slightly reduce amplitude to prevent clipping
clean_audio = clean_audio * 0.95

# ============================================================
# SAVE
# ============================================================

output_directory = os.path.dirname(OUTPUT_FILE)

if output_directory:
    os.makedirs(output_directory, exist_ok=True)

sf.write(
    OUTPUT_FILE,
    clean_audio,
    sr,
    subtype="PCM_16"
)

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("        NOISE REMOVAL COMPLETED")
print("=" * 60)

print("\nOutput:")
print(OUTPUT_FILE)

print("\nSample Rate :", sr)
print("Format      : WAV")
print("Channels    : Mono")