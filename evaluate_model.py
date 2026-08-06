import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

# =========================================================
# PATHS
# =========================================================

MODEL_PATH = r"D:\Project Folder\Mini Project\Files\best_model.keras"

TEST_DIR = r"D:\Project Folder\Mini Project\Datasets\Dataset\test"

# =========================================================
# PARAMETERS
# =========================================================

IMG_SIZE = (128, 128)
BATCH_SIZE = 16

# =========================================================
# LOAD MODEL
# =========================================================

print("\n======================================")
print("LOADING CNN MODEL")
print("======================================")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# =========================================================
# LOAD TEST DATASET
# =========================================================

print("\n======================================")
print("LOADING TEST DATASET")
print("======================================")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = test_ds.class_names

print("\nClasses:", class_names)

print("Class 0 =", class_names[0])
print("Class 1 =", class_names[1])

# IMPORTANT:
#
# human     = 0
# nonhuman  = 1

# =========================================================
# GET TRUE LABELS
# =========================================================

y_true = []

for images, labels in test_ds:
    y_true.extend(labels.numpy().flatten())

y_true = np.array(y_true).astype(int)

# =========================================================
# MODEL PREDICTIONS
# =========================================================

print("\n======================================")
print("GENERATING PREDICTIONS")
print("======================================")

# Do NOT divide by 255 here.
# Rescaling(1./255) is already inside your trained model.

y_probability = model.predict(
    test_ds,
    verbose=1
).flatten()

# Convert probability to class
# < 0.5 = human
# >= 0.5 = nonhuman

y_pred = (y_probability >= 0.5).astype(int)

# =========================================================
# BASIC METRICS
# =========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

auc = roc_auc_score(
    y_true,
    y_probability
)

print("\n======================================")
print("CNN PERFORMANCE")
print("======================================")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {auc:.4f}")

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n======================================")
print("CONFUSION MATRIX")
print("======================================")

print(cm)

TN, FP, FN, TP = cm.ravel()

print("\nTrue Human predicted Human      :", TN)
print("Human predicted Nonhuman        :", FP)
print("Nonhuman predicted Human        :", FN)
print("True Nonhuman predicted Nonhuman:", TP)

# =========================================================
# CONFUSION MATRIX GRAPH
# =========================================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("CNN Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    [0, 1],
    class_names
)

plt.yticks(
    [0, 1],
    class_names
)

# Display numbers inside matrix
for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()

# =========================================================
# ROC CURVE
# =========================================================

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"CNN (AUC = {auc:.4f})"
)

# Random classifier reference line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("CNN ROC Curve")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "roc_curve.png",
    dpi=300
)

plt.show()

print("\n======================================")
print("EVALUATION COMPLETED")
print("======================================")

print("\nGenerated files:")

print("confusion_matrix.png")
print("roc_curve.png")