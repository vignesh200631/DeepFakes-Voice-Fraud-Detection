import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = r"D:\Project Folder\Mini Project\Files\best_model.keras"

# Audio file to test
AUDIO_PATH = r"D:\Project Folder\Mini Project\Datasets\Audios\VishalRithikSingsong.mp3"

# Temporary/generated spectrogram
SPECTROGRAM_PATH = r"D:\Project Folder\Mini Project\Files\temp_spectrogram.png"

SAMPLE_RATE = 16000
IMG_SIZE = (128, 128)


# =========================================================
# LOAD CNN MODEL
# =========================================================

print("\n======================================")
print("DEEPFAKE VOICE DETECTION")
print("======================================")

print("\nLoading CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("CNN model loaded successfully!")


# =========================================================
# CHECK AUDIO FILE
# =========================================================

if not os.path.exists(AUDIO_PATH):

    print("\nERROR: Audio file not found!")
    print(AUDIO_PATH)

    exit()


# =========================================================
# LOAD AUDIO
# =========================================================

print("\nLoading audio...")

y, sr = librosa.load(
    AUDIO_PATH,
    sr=SAMPLE_RATE
)

print("Audio loaded successfully!")

print(f"Sample Rate : {sr} Hz")
print(f"Samples     : {len(y)}")
print(f"Duration    : {len(y) / sr:.2f} seconds")


# =========================================================
# CREATE MEL SPECTROGRAM
# =========================================================

print("\nGenerating Mel Spectrogram...")

mel = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_mels=128
)


# =========================================================
# CONVERT TO LOG-MEL
# =========================================================

log_mel = librosa.power_to_db(
    mel,
    ref=np.max
)


# =========================================================
# SAVE SPECTROGRAM IMAGE
# =========================================================

plt.figure(
    figsize=(3, 3),
    dpi=128
)

librosa.display.specshow(
    log_mel,
    sr=sr,
    cmap="viridis"
)

plt.axis("off")

plt.tight_layout(
    pad=0
)

plt.savefig(
    SPECTROGRAM_PATH,
    bbox_inches="tight",
    pad_inches=0
)

plt.close()

print("Spectrogram generated successfully!")

print("Spectrogram saved at:")
print(SPECTROGRAM_PATH)


# =========================================================
# LOAD SPECTROGRAM FOR CNN
# =========================================================

print("\nPreparing image for CNN...")

img = tf.keras.utils.load_img(
    SPECTROGRAM_PATH,
    target_size=IMG_SIZE,
    color_mode="rgb"
)

img_array = tf.keras.utils.img_to_array(img)


# Add batch dimension
#
# (128,128,3)
#
# becomes
#
# (1,128,128,3)

img_array = np.expand_dims(
    img_array,
    axis=0
)


# IMPORTANT:
#
# DO NOT use:
#
# img_array = img_array / 255.0
#
# because Rescaling(1./255)
# already exists inside the trained CNN model.


# =========================================================
# CNN PREDICTION
# =========================================================

print("\nAnalyzing voice...")

prediction = model.predict(
    img_array,
    verbose=0
)


# Binary sigmoid output
probability = float(
    prediction[0][0]
)


# =========================================================
# CLASSIFICATION
# =========================================================

# Training labels:
#
# human     = 0
# nonhuman  = 1

if probability >= 0.5:

    predicted_class = "NONHUMAN"

    confidence = probability * 100

    risk = "HIGH"

else:

    predicted_class = "HUMAN"

    confidence = (1 - probability) * 100

    risk = "LOW"


# =========================================================
# DISPLAY RESULT
# =========================================================

print("\n======================================")
print("       DETECTION RESULT")
print("======================================")

print(f"\nPrediction : {predicted_class}")

print(
    f"Confidence : {confidence:.2f}%"
)

print(
    f"Risk Level : {risk}"
)

print(
    f"CNN Score  : {probability:.6f}"
)


# =========================================================
# EXPLANATION
# =========================================================

print("\n--------------------------------------")

if predicted_class == "HUMAN":

    print("Voice classified as genuine human.")

else:

    print("Possible AI / synthetic voice detected.")

print("--------------------------------------")

print("\nAnalysis completed!")