import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ==========================================
# Dataset Paths
# ==========================================

TRAIN_DIR = r"D:\Project Folder\Mini Project\Dataset\train"
VAL_DIR = r"D:\Project Folder\Mini Project\Dataset\validation"
TEST_DIR = r"D:\Project Folder\Mini Project\Dataset\test"

# ==========================================
# Parameters
# ==========================================

IMG_SIZE = (128,128)

BATCH_SIZE = 16

EPOCHS = 30

LEARNING_RATE = 0.0001

# ==========================================
# Load Dataset
# ==========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

print("\nClass Names:", train_ds.class_names)

# ==========================================
# Normalize Images
# ==========================================

normalization = Rescaling(1./255)

train_ds = train_ds.map(lambda x,y:(normalization(x),y))
val_ds = val_ds.map(lambda x,y:(normalization(x),y))
test_ds = test_ds.map(lambda x,y:(normalization(x),y))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# ==========================================
# CNN Model
# ==========================================

model = Sequential([

    Conv2D(32,(3,3),
           activation='relu',
           input_shape=(128,128,3)),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(64,(3,3),
           activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(128,(3,3),
           activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(256,(3,3),
           activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    Flatten(),

    Dense(256,
          activation='relu'),

    Dropout(0.5),

    Dense(128,
          activation='relu'),

    Dropout(0.3),

    Dense(1,
          activation='sigmoid')

])

# ==========================================
# Compile Model
# ==========================================

model.compile(

    optimizer=Adam(learning_rate=LEARNING_RATE),

    loss='binary_crossentropy',

    metrics=['accuracy']

)

model.summary()

# ==========================================
# Callbacks
# ==========================================

early_stop = EarlyStopping(

    monitor='val_loss',

    patience=5,

    restore_best_weights=True

)

checkpoint = ModelCheckpoint(

    "best_model.keras",

    monitor='val_accuracy',

    save_best_only=True

)

# ==========================================
# Train Model
# ==========================================

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=[early_stop,checkpoint]

)

# ==========================================
# Evaluate
# ==========================================

loss,accuracy = model.evaluate(test_ds)

print("\n==========================")
print("Test Accuracy :",accuracy*100)
print("==========================")

# ==========================================
# Save Model
# ==========================================

model.save("deepfake_voice_detector.keras")

print("\nModel Saved Successfully")

# ==========================================
# Accuracy Graph
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'],
         label='Training Accuracy')

plt.plot(history.history['val_accuracy'],
         label='Validation Accuracy')

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("CNN Accuracy")

plt.legend()

plt.grid()

plt.show()

# ==========================================
# Loss Graph
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(history.history['loss'],
         label='Training Loss')

plt.plot(history.history['val_loss'],
         label='Validation Loss')

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("CNN Loss")

plt.legend()

plt.grid()

plt.show()