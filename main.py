# ============================================================
# FRESHORA - FOOD FRESHNESS SCANNER (Kivy app)
#
#     py -3.12 main.py
#
# The AI work (loading the model, scanning, the personality
# reply) runs in a background thread, so the window never freezes.
# ============================================================

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.utils import escape_markup

from predict import FreshoraError, FreshoraPredictor  # model/predict.py


PROJECT_NAME = "Freshora"

# Verdict colours: green = safe, amber = check first, red = don't eat.
COLOURS = {
    "yes": "4caf50",
    "maybe": "ffb300",
    "no": "e53935",
    "unknown": "9e9e9e",
}

IMAGE_FILTERS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif",
                 "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.GIF"]


def wrapped_label(**kwargs):
    """A Label whose text wraps to its own width."""
    label = Label(halign="center", valign="middle", **kwargs)
    label.bind(size=lambda lbl, size: setattr(lbl, "text_size", (size[0], None)))
    return label


class FreshoraLayout(BoxLayout):

    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", padding=20, spacing=12, **kwargs)

        self.predictor = FreshoraPredictor()
        self.selected_image = None
        self.model_ready = False
        self.busy = False

        self.add_widget(Label(
            text="[b]FRESHORA[/b]", markup=True,
            font_size=32, size_hint_y=None, height=60,
        ))

        self.status = Label(
            text="Loading AI model...", font_size=18,
            size_hint_y=None, height=40,
        )
        self.add_widget(self.status)

        # "fit_mode" is the new Kivy way, "allow_stretch" the old one.
        try:
            self.preview = KivyImage(fit_mode="contain")
        except TypeError:
            self.preview = KivyImage(allow_stretch=True, keep_ratio=True)
        self.add_widget(self.preview)

        self.select_button = Button(
            text="Select Food Image", size_hint_y=None, height=55)
        self.select_button.bind(on_press=self.open_file_picker)
        self.add_widget(self.select_button)

        self.scan_button = Button(
            text="Analyze Freshness", size_hint_y=None, height=55,
            disabled=True)
        self.scan_button.bind(on_press=self.analyze_image)
        self.add_widget(self.scan_button)

        self.result = wrapped_label(
            text="No image selected.", font_size=18, markup=True)
        self.add_widget(self.result)

        self.add_widget(wrapped_label(
            text=("AI predictions are not a guarantee of food safety.\n"
                  "Use appropriate food-safety guidance."),
            font_size=12, size_hint_y=None, height=50,
        ))

        # Load the model in the background so the window opens at once.
        threading.Thread(target=self._load_model, daemon=True).start()

    # --------------------------------------------------------
    # Background: load the model
    # --------------------------------------------------------
    def _load_model(self):
        try:
            self.predictor.load()
        except FreshoraError as error:
            message = str(error)
            Clock.schedule_once(lambda dt: self._model_failed(message))
            return
        except Exception as error:  # never let a thread die silently
            message = repr(error)
            Clock.schedule_once(lambda dt: self._model_failed(message))
            return
        Clock.schedule_once(lambda dt: self._model_loaded())

    def _model_loaded(self):
        self.model_ready = True
        self.status.text = "Ready. Choose a food image."
        self.scan_button.disabled = False

    def _model_failed(self, message):
        self.status.text = "AI model not available"
        self.result.text = escape_markup(message)

    # --------------------------------------------------------
    # File picker
    # --------------------------------------------------------
    def open_file_picker(self, instance):
        start = ROOT / "data"
        chooser = FileChooserListView(
            path=str(start if start.is_dir() else Path.home()),
            filters=IMAGE_FILTERS,
        )

        content = BoxLayout(orientation="vertical")
        content.add_widget(chooser)

        choose = Button(text="Use Selected Image", size_hint_y=None, height=50)
        content.add_widget(choose)

        popup = Popup(title="Choose Food Image", content=content,
                      size_hint=(0.9, 0.9))

        def select_image(_button):
            if not chooser.selection:
                return
            self.selected_image = chooser.selection[0]
            self.preview.source = self.selected_image
            self.preview.reload()
            self.status.text = "Image selected"
            self.result.text = "Ready to analyze."
            popup.dismiss()

        choose.bind(on_press=select_image)
        popup.open()

    # --------------------------------------------------------
    # Analyze (runs in a background thread)
    # --------------------------------------------------------
    def analyze_image(self, instance):
        if self.busy:
            return
        if self.selected_image is None:
            self.result.text = "Please select an image first."
            return
        if not self.model_ready:
            self.result.text = "The AI model is still loading, one moment."
            return

        self.busy = True
        self.scan_button.disabled = True
        self.status.text = "Analyzing..."
        self.result.text = "Freshora is squinting at your fruit..."

        threading.Thread(
            target=self._scan, args=(self.selected_image,), daemon=True
        ).start()

    def _scan(self, image_path):
        try:
            prediction, reply = self.predictor.scan(image_path)
            Clock.schedule_once(lambda dt: self._show(prediction, reply))
        except FreshoraError as error:
            message = str(error)
            Clock.schedule_once(lambda dt: self._fail(message))
        except Exception as error:
            message = repr(error)
            Clock.schedule_once(lambda dt: self._fail(message))

    def _show(self, prediction, reply):
        colour = COLOURS.get(reply.safe, COLOURS["unknown"])
        self.status.text = "Analysis complete"
        self.result.text = (
            f"{escape_markup(reply.text)}\n\n"
            f"[b][color={colour}]{escape_markup(reply.verdict_line)}[/color][/b]\n\n"
            f"[size=13]{escape_markup(prediction.label)} "
            f"({prediction.confidence * 100:.0f}% sure)[/size]"
        )
        self._done()

    def _fail(self, message):
        self.status.text = "Analysis failed"
        self.result.text = escape_markup(message)
        self._done()

    def _done(self):
        self.busy = False
        self.scan_button.disabled = False


class FreshoraApp(App):
    def build(self):
        self.title = PROJECT_NAME
        Window.minimum_width, Window.minimum_height = 360, 600
        return FreshoraLayout()


if __name__ == "__main__":
    FreshoraApp().run()
            
