# ============================================================
# FRESHORA - AI IMAGE PREDICTION
# ============================================================

import os
import sys

import kivy
import openai
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt


# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT_NAME = "Freshora"

MODEL_PATH = "model/freshora_model.keras"

CLASS_NAMES_PATH = "model/class_names.txt"

IMG_SIZE = (224, 224)


# ============================================================
# 2. LOAD CLASS NAMES
# ============================================================

if not os.path.exists(
    CLASS_NAMES_PATH
):

    print(
        "ERROR: class_names.txt was not found."
    )

    print(
        "Train the model first."
    )

    sys.exit(1)


with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    CLASS_NAMES = [
        line.strip()
        for line in file
        if line.strip()
    ]


# ============================================================
# 3. CHECK MODEL
# ============================================================

if not os.path.exists(
    MODEL_PATH
):

    print(
        "ERROR: Freshora model was not found."
    )

    print(
        "Train it first with:"
    )

    print(
        "py -3.12 model/train.py"
    )

    sys.exit(1)


# ============================================================
# 4. LOAD MODEL
# ============================================================

print(
    "\nLoading Freshora AI model..."
)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print(
    "Model loaded successfully."
)


# ============================================================
# 5. GET IMAGE
# ============================================================

if len(sys.argv) < 2:

    print("\nUsage:")

    print(
        "py -3.12 model/predict.py image.jpg"
    )

    sys.exit(1)


IMAGE_PATH = sys.argv[1]


# ============================================================
# 6. CHECK IMAGE
# ============================================================

if not os.path.exists(
    IMAGE_PATH
):

    print(
        "\nERROR: Image does not exist:"
    )

    print(
        IMAGE_PATH
    )

    sys.exit(1)


# ============================================================
# 7. LOAD IMAGE
# ============================================================

print(
    "\nLoading image..."
)

image = Image.open(
    IMAGE_PATH
).convert("RGB")


# ============================================================
# 8. PREPROCESS IMAGE
# ============================================================

image = image.resize(
    IMG_SIZE
)

image_array = np.array(
    image,
    dtype=np.float32
)

image_array = np.expand_dims(
    image_array,
    axis=0
)


# ============================================================
# 9. RUN AI
# ============================================================

print(
    "Analyzing image..."
)

predictions = model.predict(
    image_array,
    verbose=0
)[0]


# ============================================================
# 10. FIND PREDICTION
# ============================================================

prediction_index = int(
    np.argmax(predictions)
)

prediction = CLASS_NAMES[
    prediction_index
]

confidence = float(
    predictions[prediction_index]
)


# ============================================================
# 11. DISPLAY RESULT
# ============================================================

print("\n===================================")
print("FRESHORA RESULT")
print("===================================")

print(
    "Food:",
    "Banana"
)

print(
    "Prediction:",
    prediction
)

print(
    "Confidence:",
    f"{confidence * 100:.2f}%"
)


# ============================================================
# 12. DISPLAY ALL PROBABILITIES
# ============================================================

print("\nClass probabilities:")

for index, class_name in enumerate(
    CLASS_NAMES
):

    percentage = (
        predictions[index] * 100
    )

    print(
        f"{class_name:10s}: "
        f"{percentage:.2f}%"
    )


# ============================================================
# 13. SAFETY MESSAGE
# ============================================================

print("\n===================================")

print(
    "IMPORTANT:"
)

print(
    "Freshora's prediction is NOT a"
)

print(
    "guarantee that food is safe to eat."
)

print(
    "Use appropriate food-safety guidance"
)

print(
    "and do not rely on the AI alone."
)

print("===================================\n")

# ============================================================
# FOOD FRESHNESS AI - PREDICTION
# ============================================================

import tensorflow as tf
import numpy as np
import sys
import os


# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_PATH = "food_freshness_model.keras"

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "fresh",
    "ripe",
    "spoiled"
]


# ============================================================
# 2. CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model file not found.")
    print("Please run train.py first.")
    sys.exit()


# ============================================================
# 3. GET IMAGE PATH
# ============================================================

if len(sys.argv) < 2:
    print("Please provide an image path.")
    print("\nExample:")
    print("python predict.py test.jpg")
    sys.exit()


image_path = sys.argv[1]


# ============================================================
# 4. CHECK IMAGE
# ============================================================

if not os.path.exists(image_path):
    print("ERROR: Image not found.")
    print("Check the image path and try again.")
    sys.exit()


# ============================================================
# 5. LOAD TRAINED MODEL
# ============================================================

print("\nLoading AI model...")

model = tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# 6. LOAD IMAGE
# ============================================================

image = tf.keras.utils.load_img(
    image_path,
    target_size=IMG_SIZE
)


# ============================================================
# 7. CONVERT IMAGE TO ARRAY
# ============================================================

image_array = tf.keras.utils.img_to_array(image)


# ============================================================
# 8. ADD BATCH DIMENSION
# ============================================================

image_array = tf.expand_dims(
    image_array,
    0
)


# ============================================================
# 9. MAKE PREDICTION
# ============================================================

predictions = model.predict(image_array)

predicted_index = np.argmax(predictions[0])

confidence = float(
    predictions[0][predicted_index]
) * 100


# ============================================================
# 10. GET RESULT
# ============================================================

predicted_class = CLASS_NAMES[predicted_index]


# ============================================================
# 11. DISPLAY RESULT
# ============================================================

print("\n===================================")
print("     FOOD FRESHNESS AI")
print("===================================")

print("\nPrediction:", predicted_class)
print("Confidence:", round(confidence, 2), "%")


# ============================================================
# 12. DISPLAY MESSAGE
# ============================================================

if predicted_class == "fresh":

    print("\nThe AI predicts that the food appears fresh.")

elif predicted_class == "ripe":

    print("\nThe AI predicts that the food appears ripe.")

elif predicted_class == "spoiled":

    print("\nThe AI predicts that the food appears spoiled.")


# ============================================================
# 13. SAFETY WARNING
# ============================================================

print("\n-----------------------------------")
print("WARNING:")
print("AI predictions can be wrong.")
print("Do not rely only on this prediction")
print("to decide whether food is safe to eat.")
print("-----------------------------------")
