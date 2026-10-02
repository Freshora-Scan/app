import kivy
import openai
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT INFORMATION
# ============================================================

PROJECT_NAME = "Freshora"

print("\n===================================")
print(PROJECT_NAME)
print("===================================")


# ============================================================
# 2. BANANA DATASET SETTINGS
# ============================================================

DATASET_PATH = "data/banana"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("\nDataset path:", DATASET_PATH)
print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)


# ============================================================
# 3. LOAD TRAINING DATASET
# ============================================================

train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# ============================================================
# 4. LOAD VALIDATION DATASET
# ============================================================

validation_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# ============================================================
# 5. SHOW CLASSES
# ============================================================

class_names = train_data.class_names

print("\nFood: Banana")
print("Freshness classes:")

for class_name in class_names:
    print("-", class_name)


# ============================================================
# 6. IMPROVE DATA LOADING
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(
    buffer_size=AUTOTUNE
)

validation_data = validation_data.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# 7. CONFIRM DATASET
# ============================================================

print("\n===================================")
print("Banana dataset loaded successfully!")
print("Classes:", class_names)
print("===================================")
