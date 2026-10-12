# ============================================================
# FRESHORA - AI MODEL TRAINING
# ============================================================

import os

import kivy
import openai
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

PROJECT_NAME = "Freshora"

DATASET_PATH = "data/banana"
MODEL_PATH = "model/freshora_model.keras"

IMG_SIZE = (224, 224)

# 8 is deliberately modest for a 4 GB RAM laptop.
BATCH_SIZE = 8

EPOCHS = 10

print("\n===================================")
print(PROJECT_NAME)
print("AI MODEL TRAINING")
print("===================================\n")


# ============================================================
# 2. CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )


# ============================================================
# 3. LOAD TRAINING DATA
# ============================================================

print("Loading training dataset...\n")

train_data = tf.keras.utils.image_dataset_from_directory(

    DATASET_PATH,

    validation_split=0.2,

    subset="training",

    seed=123,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE

)


# ============================================================
# 4. LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation dataset...\n")

validation_data = tf.keras.utils.image_dataset_from_directory(

    DATASET_PATH,

    validation_split=0.2,

    subset="validation",

    seed=123,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE

)


# ============================================================
# 5. GET CLASS NAMES
# ============================================================

class_names = train_data.class_names

print("\n===================================")
print("CLASSES")
print("===================================")

for number, class_name in enumerate(class_names):
    print(number, "->", class_name)


# ============================================================
# 6. DATASET INFORMATION
# ============================================================

print("\n===================================")
print("DATASET INFORMATION")
print("===================================")

print(
    "Training batches:",
    tf.data.experimental.cardinality(train_data).numpy()
)

print(
    "Validation batches:",
    tf.data.experimental.cardinality(
        validation_data
    ).numpy()
)


# ============================================================
# 7. DATA PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(
    buffer_size=AUTOTUNE
)

validation_data = validation_data.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# 8. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.08
    ),

    tf.keras.layers.RandomZoom(
        0.10
    )

])


# ============================================================
# 9. BUILD CNN MODEL
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(224, 224, 3)
    ),

    data_augmentation,

    tf.keras.layers.Rescaling(
        1.0 / 255
    ),

    # --------------------------------------------------------
    # Convolution block 1
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # Convolution block 2
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # Convolution block 3
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # Classification section
    # --------------------------------------------------------

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.5
    ),

    tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    )

])


# ============================================================
# 10. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# ============================================================
# 11. MODEL SUMMARY
# ============================================================

print("\n===================================")
print("MODEL ARCHITECTURE")
print("===================================\n")

model.summary()


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\n===================================")
print("STARTING TRAINING")
print("===================================\n")

history = model.fit(

    train_data,

    validation_data=validation_data,

    epochs=EPOCHS

)


# ============================================================
# 13. EVALUATE MODEL
# ============================================================

print("\n===================================")
print("EVALUATION")
print("===================================\n")

loss, accuracy = model.evaluate(
    validation_data,
    verbose=1
)

print(
    f"\nValidation accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Validation loss: "
    f"{loss:.4f}"
)


# ============================================================
# 14. SAVE MODEL
# ============================================================

os.makedirs(
    "model",
    exist_ok=True
)

model.save(
    MODEL_PATH
)


# ============================================================
# 15. SAVE CLASS NAMES
# ============================================================

class_names_path = (
    "model/class_names.txt"
)

with open(
    class_names_path,
    "w",
    encoding="utf-8"
) as file:

    for class_name in class_names:

        file.write(
            class_name + "\n"
        )


# ============================================================
# 16. TRAINING GRAPH
# ============================================================

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Training accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Freshora Training Accuracy"
)

plt.legend()

plt.savefig(
    "model/training_accuracy.png"
)

plt.close()


# ============================================================
# 17. FINAL STATUS
# ============================================================

print("\n===================================")
print("TRAINING COMPLETE")
print("===================================")

print(
    "Model saved:",
    MODEL_PATH
)

print(
    "Classes saved:",
    class_names_path
)

print(
    "Training graph saved:",
    "model/training_accuracy.png"
)

print("\nFreshora AI model is ready.\n")
# ============================================================
# FOOD FRESHNESS AI - TRAINING
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import os

# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

PROJECT_NAME = "Food Freshness AI"

DATASET_PATH = "data/banana"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

MODEL_PATH = "food_freshness_model.keras"

print("\n===================================")
print(PROJECT_NAME)
print("===================================")

print("TensorFlow version:", tf.__version__)
print("Dataset path:", DATASET_PATH)
print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)


# ============================================================
# 2. CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {DATASET_PATH}"
    )


# ============================================================
# 3. LOAD TRAINING DATA
# ============================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# ============================================================
# 4. LOAD VALIDATION DATA
# ============================================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# ============================================================
# 5. GET CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\nClasses found:")
for i, name in enumerate(class_names):
    print(i, "->", name)


# ============================================================
# 6. IMPROVE DATA PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# 7. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])


# ============================================================
# 8. CREATE AI MODEL
# ============================================================

base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

# Freeze the pre-trained model
base_model.trainable = False


# ============================================================
# 9. BUILD FINAL MODEL
# ============================================================

model = models.Sequential([
    data_augmentation,

    layers.Rescaling(1./255),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.2),

    layers.Dense(len(class_names), activation="softmax")
])


# ============================================================
# 10. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 11. SHOW MODEL INFORMATION
# ============================================================

print("\nModel structure:")
model.summary()


# ============================================================
# 12. TRAIN THE AI
# ============================================================

print("\nStarting training...")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# ============================================================
# 13. SAVE MODEL
# ============================================================

model.save(MODEL_PATH)

print("\n===================================")
print("TRAINING COMPLETED")
print("===================================")

print("Model saved as:", MODEL_PATH)

print("\nYour AI has learned these classes:")
print(class_names)

import json
from pathlib import Path
import tensorflow as tf

# SETTINGS
DATASET_PATH = Path("data")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "food_freshness.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 123

def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_PATH.resolve()}"
        )

    # Find food folders containing condition subfolders.
    food_folders = sorted(
        p for p in DATASET_PATH.iterdir() if p.is_dir()
    )

    class_dirs = {}
    for food in food_folders:
        for condition in sorted(food.iterdir()):
            if not condition.is_dir():
                continue

            images = [
                p for p in condition.rglob("*")
                if p.is_file()
                and p.suffix.lower() in
                {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
            ]

            if images:
                class_name = f"{food.name}_{condition.name}"
                class_dirs[class_name] = condition

    if len(class_dirs) < 2:
        raise ValueError(
            "Add images to at least two condition folders "
            "inside data/food/condition/."
        )

    print("Classes found:")
    for name, folder in class_dirs.items():
        print(f"  {name}: {folder}")

    # Combine the images into a temporary class-based dataset.
    # This avoids treating each food folder as a separate condition.
    import shutil

    prepared = MODEL_DIR / "prepared_data"
    if prepared.exists():
        shutil.rmtree(prepared)
    prepared.mkdir(parents=True)

    for class_name, source in class_dirs.items():
        destination = prepared / class_name
        destination.mkdir()
        for i, img in enumerate(sorted(source.rglob("*"))):
            if img.is_file() and img.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".gif"
            }:
                shutil.copy2(img, destination / f"{i}_{img.name}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(prepared),
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(prepared),
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    class_names = train_ds.class_names
    print("Model labels:", class_names)

    # Check that every class has enough images for splitting.
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # A simple CNN for a beginner project.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(
            16, 3, activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(
            32, 3, activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(
            64, 3, activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(
            len(class_names), activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\nStarting training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    model.save(str(MODEL_PATH))
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    print("\nTraining completed!")
    print("Model saved:", MODEL_PATH)
    print("Labels saved:", LABELS_PATH)

if __name__ == "__main__":
    main()

