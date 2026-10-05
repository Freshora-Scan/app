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
