import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ==========================
# INPUT AUDIO FILE
# ==========================

audio_path = r"D:\Project Folder\Mini Project\Audios\AIMohanVoice.mp3"

# ==========================
# OUTPUT IMAGE
# ==========================

output_image = r"D:\Project Folder\Mini Project\AI_Mohan_spectrogram.png"

# ==========================
# LOAD AUDIO
# ==========================

y, sr = librosa.load(audio_path, sr=16000)

# ==========================
# CREATE LOG-MEL SPECTROGRAM
# ==========================

mel = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_mels=128
)

log_mel = librosa.power_to_db(
    mel,
    ref=np.max
)

# ==========================
# SAVE IMAGE
# ==========================

plt.figure(figsize=(3, 3), dpi=128)

librosa.display.specshow(
    log_mel,
    sr=sr,
    cmap="viridis"
)

plt.axis("off")
plt.tight_layout(pad=0)

plt.savefig(
    output_image,
    bbox_inches="tight",
    pad_inches=0
)

plt.close()

print("✅ Spectrogram saved successfully!")
print(f"Image saved at:\n{output_image}")