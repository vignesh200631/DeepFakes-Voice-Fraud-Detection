import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    Rescaling
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# =========================================================
# DATASET PATHS
# =========================================================

TRAIN_DIR = r"D:\Project Folder\Mini Project\Datasets\Dataset\train"
VAL_DIR = r"D:\Project Folder\Mini Project\Datasets\Dataset\validation"
TEST_DIR = r"D:\Project Folder\Mini Project\Datasets\Dataset\test"

# =========================================================
# PARAMETERS
# =========================================================

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001

# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading Training Dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=42
)

print("\nLoading Validation Dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("\nLoading Test Dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

# =========================================================
# CLASS INFORMATION
# =========================================================

class_names = train_ds.class_names

print("\n======================================")
print("Classes detected:", class_names)
print("======================================")

# IMPORTANT:
# TensorFlow sorts folder names alphabetically.
#
# human     -> 0
# nonhuman  -> 1

print("Class 0 =", class_names[0])
print("Class 1 =", class_names[1])

# =========================================================
# DATASET PERFORMANCE
# =========================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# =========================================================
# BUILD CNN MODEL
# =========================================================

model = Sequential([

    # Input
    Input(shape=(128, 128, 3)),

    # Normalize RGB values from 0-255 to 0-1
    Rescaling(1.0 / 255),

    # -------------------------------
    # Convolution Block 1
    # -------------------------------

    Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    # -------------------------------
    # Convolution Block 2
    # -------------------------------

    Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    # -------------------------------
    # Convolution Block 3
    # -------------------------------

    Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    # -------------------------------
    # Convolution Block 4
    # -------------------------------

    Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    # -------------------------------
    # Classification
    # -------------------------------

    Flatten(),

    Dense(
        256,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.3),

    # Binary classification
    Dense(
        1,
        activation="sigmoid"
    )
])

# =========================================================
# COMPILE MODEL
# =========================================================

model.compile(

    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

# =========================================================
# MODEL INFORMATION
# =========================================================

print("\n======================================")
print("CNN MODEL")
print("======================================")

model.summary()

# =========================================================
# CALLBACKS
# =========================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1
)

checkpoint = ModelCheckpoint(

    filepath="best_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1
)

# =========================================================
# TRAIN CNN
# =========================================================

print("\n======================================")
print("STARTING CNN TRAINING")
print("======================================\n")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=[
        early_stop,
        checkpoint
    ]
)

# =========================================================
# TEST MODEL
# =========================================================

print("\n======================================")
print("TESTING MODEL")
print("======================================")

results = model.evaluate(
    test_ds,
    verbose=1
)

print("\n======================================")
print("FINAL TEST RESULTS")
print("======================================")

for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

# =========================================================
# SAVE FINAL MODEL
# =========================================================

model.save(
    "deepfake_voice_detector.keras"
)

print("\n======================================")
print("Model saved successfully!")
print("======================================")

print("Best model  : best_model.keras")
print("Final model : deepfake_voice_detector.keras")

# =========================================================
# TRAINING ACCURACY GRAPH
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "CNN Training and Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "cnn_accuracy.png",
    dpi=300
)

plt.show()

# =========================================================
# TRAINING LOSS GRAPH
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "CNN Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "cnn_loss.png",
    dpi=300
)

plt.show()

print("\nTraining completed successfully!")