import tensorflow as tf
import numpy as np

# ==============================
# Load Trained Model
# ==============================

model = tf.keras.models.load_model("best_model.keras")

# ==============================
# Class Names
# ==============================

class_names = ["human", "nonhuman"]

# ==============================
# Load Test Image
# ==============================

img_path = "AI_Mohan_spectrogram.png"

img = tf.keras.utils.load_img(
    img_path,
    target_size=(224, 224)
)

img_array = tf.keras.utils.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Normalize image
img_array = img_array / 255.0

# ==============================
# Prediction
# ==============================

prediction = model.predict(img_array)

print("Raw prediction:", prediction)

predicted_index = np.argmax(prediction[0])
predicted_class = class_names[predicted_index]

confidence = prediction[0][predicted_index] * 100

print("Prediction:", predicted_class)
print(f"Confidence: {confidence:.2f}%")