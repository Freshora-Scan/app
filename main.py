# ============================================================
# FRESHORA - FOOD FRESHNESS SCANNER
# ============================================================

import os

# Your original imports — KEEPING ALL OF THEM
import kivy
import openai
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

# Kivy imports
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image as KivyImage
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

PROJECT_NAME = "Freshora"

MODEL_PATH = "model/freshora_model.keras"

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "unripe",
    "ripe",
    "overripe",
    "rotten"
]


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

model = None


def load_freshora_model():

    global model

    if not os.path.exists(MODEL_PATH):

        print(
            "No trained model found."
        )

        print(
            "Train the model using:"
        )

        print(
            "py -3.12 model/train.py"
        )

        return False

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH
        )

        print(
            "Freshora model loaded successfully."
        )

        return True

    except Exception as error:

        print(
            "Could not load model:"
        )

        print(error)

        return False


# ============================================================
# 3. IMAGE PREDICTION
# ============================================================

def predict_freshness(image_path):

    if model is None:

        return None, 0.0

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

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

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        prediction_index = int(
            np.argmax(predictions)
        )

        prediction = CLASS_NAMES[
            prediction_index
        ]

        confidence = float(
            predictions[prediction_index]
        )

        return prediction, confidence

    except Exception as error:

        print(
            "Prediction error:"
        )

        print(error)

        return None, 0.0


# ============================================================
# 4. FRESHORA USER INTERFACE
# ============================================================

class FreshoraLayout(BoxLayout):

    def __init__(self, **kwargs):

        super().__init__(
            orientation="vertical",
            padding=20,
            spacing=15,
            **kwargs
        )

        self.selected_image = None

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        self.title = Label(
            text="[b]FRESHORA[/b]",
            markup=True,
            font_size=32,
            size_hint_y=None,
            height=60
        )

        self.add_widget(
            self.title
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        self.status = Label(
            text="Food Freshness AI",
            font_size=18,
            size_hint_y=None,
            height=40
        )

        self.add_widget(
            self.status
        )

        # ----------------------------------------------------
        # IMAGE PREVIEW
        # ----------------------------------------------------

        self.preview = KivyImage(
            allow_stretch=True,
            keep_ratio=True
        )

        self.add_widget(
            self.preview
        )

        # ----------------------------------------------------
        # SELECT BUTTON
        # ----------------------------------------------------

        self.select_button = Button(
            text="Select Food Image",
            size_hint_y=None,
            height=55
        )

        self.select_button.bind(
            on_press=self.open_file_picker
        )

        self.add_widget(
            self.select_button
        )

        # ----------------------------------------------------
        # ANALYZE BUTTON
        # ----------------------------------------------------

        self.scan_button = Button(
            text="Analyze Freshness",
            size_hint_y=None,
            height=55
        )

        self.scan_button.bind(
            on_press=self.analyze_image
        )

        self.add_widget(
            self.scan_button
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        self.result = Label(
            text="No image selected.",
            font_size=20
        )

        self.add_widget(
            self.result
        )

        # ----------------------------------------------------
        # DISCLAIMER
        # ----------------------------------------------------

        self.disclaimer = Label(
            text=(
                "AI predictions are not a guarantee "
                "of food safety.\n"
                "Use appropriate food-safety guidance."
            ),
            font_size=12,
            size_hint_y=None,
            height=50
        )

        self.add_widget(
            self.disclaimer
        )


    # ========================================================
    # FILE PICKER
    # ========================================================

    def open_file_picker(self, instance):

        chooser = FileChooserListView(
            filters=[
                "*.jpg",
                "*.jpeg",
                "*.png",
                "*.webp"
            ]
        )

        popup_layout = BoxLayout(
            orientation="vertical"
        )

        popup_layout.add_widget(
            chooser
        )

        choose_button = Button(
            text="Use Selected Image",
            size_hint_y=None,
            height=50
        )

        popup_layout.add_widget(
            choose_button
        )

        popup = Popup(
            title="Choose Food Image",
            content=popup_layout,
            size_hint=(0.9, 0.9)
        )


        def select_image(instance):

            if not chooser.selection:

                return

            self.selected_image = (
                chooser.selection[0]
            )

            self.preview.source = (
                self.selected_image
            )

            self.preview.reload()

            self.status.text = (
                "Image selected"
            )

            self.result.text = (
                "Ready to analyze."
            )

            popup.dismiss()


        choose_button.bind(
            on_press=select_image
        )

        popup.open()


    # ========================================================
    # ANALYZE IMAGE
    # ========================================================

    def analyze_image(self, instance):

        if self.selected_image is None:

            self.result.text = (
                "Please select an image first."
            )

            return


        if model is None:

            self.result.text = (
                "AI model not available yet.\n"
                "Train the model first."
            )

            return


        self.status.text = (
            "Analyzing..."
        )


        prediction, confidence = (
            predict_freshness(
                self.selected_image
            )
        )


        if prediction is None:

            self.status.text = (
                "Analysis failed."
            )

            self.result.text = (
                "Could not analyze this image."
            )

            return


        self.status.text = (
            "Analysis complete"
        )


        self.result.text = (

            f"Prediction: {prediction}\n"

            f"Confidence: "
            f"{confidence * 100:.2f}%"

        )


# ============================================================
# 5. FRESHORA APP
# ============================================================

class FreshoraApp(App):

    def build(self):

        self.title = PROJECT_NAME

        load_freshora_model()

        return FreshoraLayout()


# ============================================================
# 6. START APP
# ============================================================

if __name__ == "__main__":

    FreshoraApp().run()
```
