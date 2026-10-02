import os

import kivy
import openai
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image as KivyImage
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar


# ============================================================
# FRESHORA
# ============================================================

PROJECT_NAME = "Freshora"

DATASET_PATH = "data/banana"
MODEL_PATH = "model/freshora_model.keras"

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "unripe",
    "ripe",
    "overripe",
    "rotten"
]


# ============================================================
# AI MODEL
# ============================================================

model = None


def load_model():

    global model

    if not os.path.exists(MODEL_PATH):
        return False

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH
        )

        return True

    except Exception as error:

        print("Model loading failed:")
        print(error)

        return False


def predict_image(image_path):

    if model is None:

        return None, 0.0

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

    index = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[index] * 100
    )

    return CLASS_NAMES[index], confidence


# ============================================================
# MAIN KIVY UI
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

        self.status = Label(
            text="Food Freshness AI",
            font_size=18
        )

        self.add_widget(
            self.status
        )

        self.preview = KivyImage(
            allow_stretch=True,
            keep_ratio=True
        )

        self.add_widget(
            self.preview
        )

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

        self.result = Label(
            text="No image selected.",
            font_size=20
        )

        self.add_widget(
            self.result
        )

        self.disclaimer = Label(
            text=(
                "AI predictions are not a guarantee of food safety.\n"
                "Check food using appropriate food-safety guidance."
            ),
            font_size=12
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

        select_button = Button(
            text="Use Image",
            size_hint_y=None,
            height=50
        )

        popup_layout.add_widget(
            select_button
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

        select_button.bind(
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

        self.status.text = (
            "Analyzing..."
        )

        prediction, confidence = (
            predict_image(
                self.selected_image
            )
        )

        if prediction is None:

            self.status.text = (
                "Model not available"
            )

            self.result.text = (
                "Train the Freshora model first."
            )

            return

        self.status.text = (
            "Analysis complete"
        )

        self.result.text = (
            f"Prediction: {prediction}\n"
            f"Confidence: {confidence:.2f}%"
        )


# ============================================================
# KIVY APP
# ============================================================

class FreshoraApp(App):

    def build(self):

        self.title = PROJECT_NAME

        model_loaded = load_model()

        print(
            "Freshora model loaded:",
            model_loaded
        )

        return FreshoraLayout()


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    FreshoraApp().run()
