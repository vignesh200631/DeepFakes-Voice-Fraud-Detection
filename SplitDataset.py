import os
import random
import shutil

# ==========================================
# PATHS
# ==========================================

SOURCE_DIR = (
    r"D:\Project Folder\Mini Project\Datasets"
    r"\ImageDatas\Spectrogram Dataset"
)

OUTPUT_DIR = r"D:\Project Folder\Mini Project\Dataset"

# Source folder name : Training class name
classes = {
    "Human Voice": "human",
    "AI Voice": "nonhuman"
}

# ==========================================
# SPLIT RATIOS
# ==========================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# ==========================================
# OPTIONAL: CLEAR OLD DATASET
# ==========================================

if os.path.exists(OUTPUT_DIR):
    print("Removing old dataset...")
    shutil.rmtree(OUTPUT_DIR)

# ==========================================
# SPLIT DATASET
# ==========================================

for source_class, target_class in classes.items():

    source_folder = os.path.join(
        SOURCE_DIR,
        source_class
    )

    # Check folder exists
    if not os.path.exists(source_folder):
        print(f"Folder not found: {source_folder}")
        continue

    # Get image files
    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ]

    # Randomize
    random.shuffle(images)

    total = len(images)

    # Calculate split
    train_end = int(total * TRAIN_RATIO)

    val_end = train_end + int(
        total * VAL_RATIO
    )

    train_images = images[:train_end]

    validation_images = images[
        train_end:val_end
    ]

    test_images = images[val_end:]

    # ======================================
    # Create folders
    # ======================================

    for split in [
        "train",
        "validation",
        "test"
    ]:

        folder = os.path.join(
            OUTPUT_DIR,
            split,
            target_class
        )

        os.makedirs(
            folder,
            exist_ok=True
        )

    # ======================================
    # Copy Training Images
    # ======================================

    for img in train_images:

        shutil.copy2(
            os.path.join(
                source_folder,
                img
            ),

            os.path.join(
                OUTPUT_DIR,
                "train",
                target_class,
                img
            )
        )

    # ======================================
    # Copy Validation Images
    # ======================================

    for img in validation_images:

        shutil.copy2(
            os.path.join(
                source_folder,
                img
            ),

            os.path.join(
                OUTPUT_DIR,
                "validation",
                target_class,
                img
            )
        )

    # ======================================
    # Copy Test Images
    # ======================================

    for img in test_images:

        shutil.copy2(
            os.path.join(
                source_folder,
                img
            ),

            os.path.join(
                OUTPUT_DIR,
                "test",
                target_class,
                img
            )
        )

    # ======================================
    # Display Information
    # ======================================

    print("\n================================")
    print(f"Source Class : {source_class}")
    print(f"Model Class  : {target_class}")
    print("================================")

    print(f"Total       : {total}")
    print(f"Train       : {len(train_images)}")
    print(f"Validation  : {len(validation_images)}")
    print(f"Test        : {len(test_images)}")


print("\n================================")
print("Dataset split completed!")
print("================================")