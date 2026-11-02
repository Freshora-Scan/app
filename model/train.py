# ============================================================
# FRESHORA - AI MODEL TRAINING (multi-food)
#
# Run it from ANYWHERE:
#
#     py -3.12 model/train.py
#
# Expected photo layout (the file names do not matter):
#
#     data/<food>/<state>/any_name.jpg
#     e.g. data/apple/ripe/IMG_0001.jpg
#          data/mango/rotten/photo12.png
#
# Every food/state folder that contains photos becomes one class
# ("apple__ripe", "mango__rotten", ...). Empty folders (only a
# .gitkeep) are skipped, so teammates can add foods at any time.
#
# Creates:
#     model/freshora_model.keras      the trained model
#     model/class_names.txt           class order (predict.py needs it)
#     model/training_accuracy.png     accuracy graph (if matplotlib exists)
# ============================================================

import os
import shutil
import sys
from pathlib import Path

# Hide TensorFlow's very noisy start-up messages.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT_NAME = "Freshora"

# Every path is built from THIS FILE's location, not from the
# terminal's current folder.
ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data"
MODEL_DIR = ROOT / "model"
PREPARED_PATH = MODEL_DIR / "_prepared_data"      # temporary, rebuilt each run
MODEL_PATH = MODEL_DIR / "freshora_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.txt"
PLOT_PATH = MODEL_DIR / "training_accuracy.png"

SEPARATOR = "__"        # class name = food + "__" + state

IMG_SIZE = (224, 224)
BATCH_SIZE = 8          # small on purpose, friendly to low-RAM laptops
EPOCHS = 15
SEED = 123
VALIDATION_SPLIT = 0.2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

MIN_IMAGES_PER_CLASS = 2        # below this a folder is skipped
MIN_FOR_VALIDATION = 10         # per class, below this we skip validation
RECOMMENDED_PER_CLASS = 50


def stop(message):
    """Print a clear error and exit (no scary traceback)."""
    print("\n" + "=" * 60)
    print("TRAINING STOPPED")
    print("=" * 60)
    print(message)
    print("=" * 60 + "\n")
    sys.exit(1)


# ============================================================
# 2. FIND AND CHECK THE DATASET (before touching TensorFlow)
# ============================================================

def find_images(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def visible_subfolders(folder):
    """Sub-folders, ignoring hidden/private ones (.git, _prepared...)."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_dir() and not p.name.startswith((".", "_"))
    )


def inspect_dataset():
    """
    Returns {class_name: [image paths]} for every usable food/state folder.
    """
    if not DATASET_PATH.is_dir():
        stop(
            f"Dataset folder not found:\n  {DATASET_PATH}\n\n"
            "It must look like this:\n"
            "  data/<food>/<state>/photo.jpg\n"
            "  for example: data/apple/ripe/photo1.jpg\n\n"
            "If you downloaded the repo as a ZIP, check that the 'data'\n"
            "folder really is inside the project folder:\n"
            f"  {ROOT}"
        )

    classes = {}
    skipped = []
    single_state_foods = []

    for food_dir in visible_subfolders(DATASET_PATH):
        usable_states = 0
        for state_dir in visible_subfolders(food_dir):
            images = find_images(state_dir)
            label = f"{food_dir.name}{SEPARATOR}{state_dir.name}"
            if len(images) >= MIN_IMAGES_PER_CLASS:
                classes[label] = images
                usable_states += 1
            else:
                skipped.append(
                    f"{food_dir.name}/{state_dir.name} ({len(images)} image(s))"
                )
        if usable_states == 1:
            single_state_foods.append(food_dir.name)

    # Make sure every image can really be opened. A single broken
    # file (for example a half-downloaded .jpg) would crash training.
    from PIL import Image

    broken = []
    for images in classes.values():
        for path in images:
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                broken.append(path)

    if broken:
        listing = "\n".join(f"  {p}" for p in broken[:15])
        stop(
            f"{len(broken)} image(s) are broken or not real images:\n"
            f"{listing}\n\nDelete or replace them, then run again."
        )

    # ---- report ----
    print("Dataset check")
    print("-------------")
    print(f"Folder: {DATASET_PATH}\n")

    by_food = {}
    for label, images in classes.items():
        food, state = label.split(SEPARATOR, 1)
        by_food.setdefault(food, []).append((state, len(images)))

    for food in sorted(by_food):
        total = sum(count for _, count in by_food[food])
        print(f"  {food}  ({total} photos)")
        for state, count in by_food[food]:
            flag = "" if count >= RECOMMENDED_PER_CLASS else "   <- add more"
            print(f"      {state:12s}: {count:4d}{flag}")

    if skipped:
        print("\nSkipped (empty or fewer than "
              f"{MIN_IMAGES_PER_CLASS} photos):")
        for item in skipped:
            print("  -", item)

    if single_state_foods:
        print(
            "\n!! WARNING: these foods have photos in only ONE state: "
            + ", ".join(single_state_foods)
            + ".\n!! The AI cannot learn ripe vs rotten for them yet."
        )

    if len(classes) < 2:
        stop(
            f"Only {len(classes)} usable class found in {DATASET_PATH}.\n"
            "You need at least 2 folders with photos, for example\n"
            "data/banana/ripe and data/banana/rotten."
        )

    return classes


# ============================================================
# 3. PREPARE A FLAT DATASET FOR KERAS
# ============================================================
# Keras wants one folder per class. We build a temporary folder
# "_prepared_data/<food>__<state>/" using hard links, so photos are
# not duplicated on disk (falls back to copying if links fail).

def prepare_dataset(classes):
    if PREPARED_PATH.exists():
        # Safety: only ever delete our own temporary folder.
        if PREPARED_PATH.name != "_prepared_data":
            stop(f"Refusing to delete unexpected folder: {PREPARED_PATH}")
        shutil.rmtree(PREPARED_PATH)

    for label, images in classes.items():
        destination = PREPARED_PATH / label
        destination.mkdir(parents=True)
        for number, source in enumerate(images):
            target = destination / f"{number:05d}_{source.name}"
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_data(classes):
    import tensorflow as tf

    common = dict(
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    smallest = min(len(images) for images in classes.values())
    use_validation = smallest >= MIN_FOR_VALIDATION

    if use_validation:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(PREPARED_PATH),
            validation_split=VALIDATION_SPLIT,
            subset="training",
            **common,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            str(PREPARED_PATH),
            validation_split=VALIDATION_SPLIT,
            subset="validation",
            **common,
        )
    else:
        print(
            "\n!! WARNING: your smallest class has only "
            f"{smallest} image(s).\n"
            "!! A validation split would contain almost nothing, so the\n"
            "!! accuracy numbers would be meaningless. Training on ALL\n"
            "!! images without validation instead. This model will NOT be\n"
            f"!! reliable. Add {RECOMMENDED_PER_CLASS}+ photos per class."
        )
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(PREPARED_PATH), **common
        )
        val_ds = None

    class_names = list(train_ds.class_names)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    if val_ds is not None:
        val_ds = val_ds.prefetch(autotune)

    return train_ds, val_ds, class_names


# ============================================================
# 5. BUILD THE MODEL (MobileNetV2 transfer learning)
# ============================================================
# The model takes RAW pixels (0-255) and rescales them itself,
# so predict.py and main.py must NOT rescale again.

def build_model(num_classes):
    import tensorflow as tf
    from tensorflow.keras import layers

    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            # Mild brightness only. Colour IS the signal for ripeness,
            # so no hue/colour shifting here.
            layers.RandomBrightness(0.10, value_range=(0, 255)),
        ],
        name="augmentation",
    )

    def make_base():
        return tf.keras.applications.MobileNetV2(
            input_shape=IMG_SIZE + (3,),
            include_top=False,
            weights="imagenet",
        )

    base = None
    reason = ""

    try:
        base = make_base()
    except Exception as first_error:
        reason = str(first_error)

    if base is None:
        # The usual cause is a half-downloaded weights file left behind
        # by an interrupted run ("truncated file"). Delete it, try again.
        keras_home = Path(os.environ.get("KERAS_HOME", Path.home() / ".keras"))
        removed = []
        for broken in (keras_home / "models").glob("mobilenet_v2_weights*.h5"):
            try:
                broken.unlink()
                removed.append(broken)
            except OSError:
                pass

        if removed:
            print(
                "\nThe cached weights file was damaged (probably an "
                "interrupted download).\nDeleted it, downloading again...\n"
            )
            try:
                base = make_base()
            except Exception as second_error:
                reason = str(second_error)

    if base is None:
        stop(
            "Could not get the pretrained MobileNetV2 weights "
            "(a one-time ~9.4 MB download).\n"
            f"Reason: {reason}\n\n"
            "Check your internet and run again. If it keeps failing,\n"
            "download the file in a browser and put it in\n"
            f"  {Path.home() / '.keras' / 'models'}\n"
            "File name: mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
        )

    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="image")
    x = augmentation(inputs)
    # MobileNetV2 expects pixels in the range -1 .. 1.
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classes")(x)

    model = tf.keras.Model(inputs, outputs, name="freshora")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ============================================================
# 6. SAVE EVERYTHING + SELF-TEST
# ============================================================

def save_everything(model, class_names, history, has_validation):
    import numpy as np
    import tensorflow as tf

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.save(str(MODEL_PATH))

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
        for name in class_names:
            file.write(name + "\n")

    # Self-test: reload the saved file and make one prediction.
    # If this fails, you find out NOW, not when the app opens.
    reloaded = tf.keras.models.load_model(str(MODEL_PATH))
    test = reloaded.predict(
        np.zeros((1,) + IMG_SIZE + (3,), dtype="float32"), verbose=0
    )
    if test.shape != (1, len(class_names)):
        stop("Self-test failed: saved model output has the wrong shape.")

    # Optional graph, never allowed to break training.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(history.history["accuracy"], label="Training accuracy")
        if has_validation:
            plt.plot(history.history["val_accuracy"], label="Validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Freshora Training Accuracy")
        plt.legend()
        plt.savefig(str(PLOT_PATH))
        plt.close()
        print("Graph saved:", PLOT_PATH)
    except Exception as error:
        print("(Skipped the accuracy graph:", error, ")")


# ============================================================
# 7. MAIN
# ============================================================

def main():
    print("\n===================================")
    print(PROJECT_NAME, "- AI MODEL TRAINING")
    print("===================================\n")

    classes = inspect_dataset()
    prepare_dataset(classes)

    import tensorflow as tf

    train_ds, val_ds, class_names = load_data(classes)

    print("\nClasses (order the model will use):")
    for number, name in enumerate(class_names):
        print(f"  {number:2d} -> {name}")

    model = build_model(len(class_names))

    monitor = "val_loss" if val_ds is not None else "loss"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=4, restore_best_weights=True
        )
    ]

    print("\nStarting training...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    if val_ds is not None:
        loss, accuracy = model.evaluate(val_ds, verbose=0)
        print(f"\nValidation accuracy: {accuracy * 100:.2f}%")
        print(f"Validation loss:     {loss:.4f}")

    save_everything(model, class_names, history, val_ds is not None)

    # The temporary folder is no longer needed.
    shutil.rmtree(PREPARED_PATH, ignore_errors=True)

    print("\n===================================")
    print("TRAINING COMPLETE")
    print("===================================")
    print("Model saved:  ", MODEL_PATH)
    print("Classes saved:", CLASS_NAMES_PATH)
    print("\nFreshora AI model is ready.\n")


if __name__ == "__main__":
    main()
# ============================================================
# FRESHORA - FOOD FRESHNESS SCANNER (Kivy app: laptop + Android)
#
#     py -3.12 main.py
#
# Screens:  Welcome -> Home -> Camera -> Analysing -> Result
#           Home -> History -> Result
#
# On a laptop the "camera" button opens a file picker instead.
# The AI work (loading the model, scanning, the personality reply)
# runs in a background thread, so the window never freezes.
#
# Files used:  model/predict.py (AI)   model/personality.py (voice)
#              freshora_store.py (history)   greetings.py   theme.py
# ============================================================

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from kivy.animation import Animation
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.metrics import dp, sp
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import FadeTransition, Screen, ScreenManager
from kivy.uix.scrollview import ScrollView
from kivy.utils import escape_markup, platform

import theme
from freshora_store import Store, make_entry, time_ago
from greetings import greeting
from predict import FreshoraError, FreshoraPredictor  # model/predict.py
from theme import rgba

ANDROID = platform == "android"
PROJECT_NAME = "Freshora"

IMAGE_FILTERS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif",
                 "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.GIF"]

NOTICE = ("Important: Freshora only sees the outside of food. It can be "
          "wrong. Smell it, check the date, and trust your own senses.")

ANALYSING_LINES = [
    "Freshora is squinting at it...",
    "Consulting the fruit council...",
    "Interrogating the pixels...",
    "Sniffing the photo, spiritually...",
    "Asking the vegetables for comment...",
]


# ============================================================
# 1. SMALL BUILDING BLOCKS
# ============================================================

def paint(widget, hex_code):
    """Give a widget a solid background colour."""
    with widget.canvas.before:
        Color(*rgba(hex_code))
        rect = Rectangle(pos=widget.pos, size=widget.size)
    widget.bind(
        pos=lambda w, value: setattr(rect, "pos", value),
        size=lambda w, value: setattr(rect, "size", value),
    )


def wrap_label(**kwargs):
    """A Label that wraps to its width and grows to fit its text."""
    halign = kwargs.pop("halign", "left")
    color = kwargs.pop("color", rgba(theme.FOREST))
    label = Label(halign=halign, valign="top", color=color,
                  size_hint_y=None, **kwargs)
    label.bind(width=lambda lbl, w: setattr(lbl, "text_size", (w, None)))
    label.bind(texture_size=lambda lbl, size: setattr(lbl, "height", size[1]))
    return label


def make_image(**kwargs):
    """Image that fits inside its box. 'fit_mode' is new Kivy, the other old."""
    try:
        return KivyImage(fit_mode="contain", **kwargs)
    except TypeError:
        return KivyImage(allow_stretch=True, keep_ratio=True, **kwargs)


class Card(BoxLayout):
    """A rounded box."""

    def __init__(self, bg=theme.PAPER, radius=16, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(*rgba(bg))
            self._rect = RoundedRectangle(
                pos=self.pos, size=self.size, radius=[dp(radius)])
        self.bind(pos=self._sync, size=self._sync)

    def _sync(self, *args):
        self._rect.pos = self.pos
        self._rect.size = self.size


class TextCard(Card):
    """A rounded box that grows to fit the text inside it."""

    def __init__(self, bg=theme.PAPER, pad=14, **label_kwargs):
        super().__init__(bg=bg, orientation="vertical",
                         padding=dp(pad), size_hint_y=None)
        self._pad = dp(pad)
        self.label = wrap_label(markup=True, **label_kwargs)
        self.add_widget(self.label)
        self.label.bind(height=self._resize)
        self.height = self.label.height + 2 * self._pad

    def _resize(self, label, height):
        self.height = height + 2 * self._pad

    def set_text(self, text):
        self.label.text = text


class PillButton(Button):
    """A rounded, coloured button."""

    def __init__(self, bg=theme.MINT, fg=theme.FOREST, **kwargs):
        kwargs.setdefault("background_normal", "")
        kwargs.setdefault("background_down", "")
        kwargs.setdefault("background_color", (0, 0, 0, 0))
        kwargs.setdefault("font_size", sp(17))
        kwargs.setdefault("bold", True)
        super().__init__(color=rgba(fg), **kwargs)
        with self.canvas.before:
            self._colour = Color(*rgba(bg))
            self._rect = RoundedRectangle(
                pos=self.pos, size=self.size, radius=[dp(26)])
        self.bind(pos=self._sync, size=self._sync, disabled=self._dim)

    def _sync(self, *args):
        self._rect.pos = self.pos
        self._rect.size = self.size

    def _dim(self, button, disabled):
        self._colour.a = 0.35 if disabled else 1.0


class ScanRow(ButtonBehavior, Card):
    """One line in a list of scans: photo, food, state, time."""

    def __init__(self, entry, on_open, **kwargs):
        super().__init__(orientation="horizontal", padding=dp(8),
                         spacing=dp(12), size_hint_y=None, height=dp(72),
                         **kwargs)
        self.add_widget(make_image(
            source=entry.get("image", ""), size_hint_x=None, width=dp(56)))

        colour = theme.verdict_hex(entry.get("safe"))
        state = (entry.get("state") or "?").replace("_", " ").upper()
        if not entry.get("reliable", True):
            state = "NOT SURE"
        middle = Label(
            markup=True, halign="left", valign="middle",
            color=rgba(theme.FOREST),
            text=(f"[b][size={int(sp(16))}]"
                  f"{escape_markup(str(entry.get('food', 'food')).title())}"
                  f"[/size][/b]\n[color={colour}][b]{escape_markup(state)}"
                  f"[/b][/color]"),
        )
        middle.bind(size=lambda lbl, s: setattr(lbl, "text_size", s))
        self.add_widget(middle)

        right = Label(
            text=time_ago(entry.get("time", time.time())),
            font_size=sp(12), color=rgba(theme.MUTED),
            size_hint_x=None, width=dp(80), halign="right", valign="middle",
        )
        right.bind(size=lambda lbl, s: setattr(lbl, "text_size", s))
        self.add_widget(right)

        self.bind(on_release=lambda *_: on_open(entry))


# ============================================================
# 2. SCREENS
# ============================================================

class WelcomeScreen(Screen):

    def __init__(self, fa, **kwargs):
        super().__init__(name="welcome", **kwargs)
        root = BoxLayout(orientation="vertical", padding=dp(28),
                         spacing=dp(16))
        paint(root, theme.DEEP)

        root.add_widget(Label(size_hint_y=0.25))
        root.add_widget(Label(
            text="Freshora", bold=True, font_size=sp(48),
            color=rgba(theme.CREAM), size_hint_y=None, height=dp(70)))
        root.add_widget(Label(
            text="Scan food. Know better.\nMake safer choices.",
            font_size=sp(18), color=rgba(theme.MINT),
            halign="center", size_hint_y=None, height=dp(60)))
        root.add_widget(Label())

        start = PillButton(text="Get started", size_hint_y=None,
                           height=dp(56))
        start.bind(on_release=lambda *_: fa.finish_welcome())
        root.add_widget(start)

        note = wrap_label(
            text=("Freshora looks at the outside of food. It can be wrong "
                  "and is not a food-safety guarantee."),
            font_size=sp(12), halign="center", color=rgba(theme.MUTED))
        root.add_widget(note)
        self.add_widget(root)


class HomeScreen(Screen):

    def __init__(self, fa, **kwargs):
        super().__init__(name="home", **kwargs)
        self.fa = fa
        root = BoxLayout(orientation="vertical", padding=dp(20),
                         spacing=dp(12))
        paint(root, theme.CREAM)

        self.title = wrap_label(text="", bold=True, font_size=sp(26))
        root.add_widget(self.title)
        self.subtitle = wrap_label(
            text="", font_size=sp(15), color=rgba(theme.MUTED))
        root.add_widget(self.subtitle)

        # The big scan card
        card = Card(bg=theme.FOREST, orientation="vertical",
                    padding=dp(16), spacing=dp(8),
                    size_hint_y=None, height=dp(150))
        card.add_widget(Label(
            text="Scan Food", bold=True, font_size=sp(24),
            color=rgba(theme.CREAM), halign="left"))
        card.add_widget(Label(
            text="Take a photo to check freshness.",
            font_size=sp(14), color=rgba(theme.MINT)))
        self.scan_button = PillButton(
            text="Open camera", size_hint_y=None, height=dp(48))
        self.scan_button.bind(on_release=lambda *_: fa.open_camera())
        card.add_widget(self.scan_button)
        root.add_widget(card)

        # Gallery + History
        row = BoxLayout(size_hint_y=None, height=dp(52), spacing=dp(12))
        self.gallery_button = PillButton(
            text="Gallery", bg=theme.PAPER, fg=theme.FOREST)
        self.gallery_button.bind(on_release=lambda *_: fa.open_gallery())
        history = PillButton(
            text="History", bg=theme.PAPER, fg=theme.FOREST)
        history.bind(on_release=lambda *_: fa.show_history())
        row.add_widget(self.gallery_button)
        row.add_widget(history)
        root.add_widget(row)

        self.status = wrap_label(
            text="Loading AI model...", font_size=sp(13),
            color=rgba(theme.MUTED))
        root.add_widget(self.status)

        root.add_widget(wrap_label(
            text="Recent scans", bold=True, font_size=sp(18)))

        scroll = ScrollView(do_scroll_x=False)
        self.recent = GridLayout(cols=1, spacing=dp(8), size_hint_y=None)
        self.recent.bind(minimum_height=self.recent.setter("height"))
        scroll.add_widget(self.recent)
        root.add_widget(scroll)

        root.add_widget(wrap_label(
            text="AI predictions are not a guarantee of food safety.",
            font_size=sp(11), halign="center", color=rgba(theme.MUTED)))
        self.add_widget(root)

        self.set_ready(False)

    def on_pre_enter(self, *args):
        self.refresh()

    def refresh(self):
        title, line = greeting()
        self.title.text = title
        self.subtitle.text = line

        self.recent.clear_widgets()
        entries = self.fa.store.history()[:4]
        if not entries:
            self.recent.add_widget(wrap_label(
                text="Nothing scanned yet. Feed me a banana.",
                font_size=sp(14), color=rgba(theme.MUTED)))
        for entry in entries:
            self.recent.add_widget(
                ScanRow(entry, on_open=self.fa.open_saved_entry))

    def set_ready(self, ready, message=None):
        self.scan_button.disabled = not ready
        self.gallery_button.disabled = not ready
        if message is not None:
            self.status.text = message
        elif ready:
            self.status.text = "Ready."


class CameraScreen(Screen):
    """Live camera (Android only, uses Camera4Kivy)."""

    def __init__(self, fa, **kwargs):
        super().__init__(name="camera", **kwargs)
        self.fa = fa
        self.preview = None
        self._connected = False

        root = BoxLayout(orientation="vertical", padding=dp(8),
                         spacing=dp(8))
        paint(root, theme.DEEP)

        bar = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        back = PillButton(text="Back", bg=theme.FOREST, fg=theme.CREAM)
        back.bind(on_release=lambda *_: fa.go("home"))
        self.flash_button = PillButton(
            text="Flash", bg=theme.FOREST, fg=theme.CREAM)
        self.flash_button.bind(on_release=lambda *_: self._flash())
        gallery = PillButton(text="Gallery", bg=theme.FOREST, fg=theme.CREAM)
        gallery.bind(on_release=lambda *_: fa.open_gallery())
        for widget in (back, self.flash_button, gallery):
            bar.add_widget(widget)
        root.add_widget(bar)

        self.slot = BoxLayout()          # the camera picture goes in here
        root.add_widget(self.slot)

        self.hint = wrap_label(
            text="Put the whole food in the frame, in good light.",
            font_size=sp(14), halign="center", color=rgba(theme.CREAM))
        root.add_widget(self.hint)

        self.shutter = PillButton(text="Take photo", size_hint_y=None,
                                  height=dp(56))
        self.shutter.bind(on_release=lambda *_: self._take_photo())
        root.add_widget(self.shutter)
        self.add_widget(root)

    # ---- screen events
    def on_enter(self, *args):
        self.fa.ask_camera_permission(self._permission_done)

    def on_leave(self, *args):
        self._disconnect()

    # ---- camera plumbing
    def _permission_done(self, granted):
        if not granted:
            self.hint.text = ("Camera permission was refused. Allow it in "
                              "Android Settings, or use the Gallery button.")
            self.shutter.disabled = True
            return
        # Camera4Kivy asks to connect a moment AFTER the screen shows.
        Clock.schedule_once(lambda dt: self._connect(), 0.3)

    def _connect(self):
        try:
            if self.preview is None:
                from camera4kivy import Preview
                self.preview = Preview()
                self.slot.add_widget(self.preview)
            self.preview.connect_camera(
                filepath_callback=self._photo_saved,
                enable_video=False,
                enable_analyze_pixels=False,
            )
            self._connected = True
            self.shutter.disabled = False
        except Exception as error:
            self.hint.text = ("The camera could not start. Use the Gallery "
                              f"button instead.\n({error})")
            self.shutter.disabled = True

    def _disconnect(self):
        if self.preview is not None and self._connected:
            try:
                self.preview.disconnect_camera()
            except Exception:
                pass
        self._connected = False

    def _flash(self):
        try:
            self.preview.flash()
        except Exception:
            pass

    def _take_photo(self):
        if self.preview is None or not self._connected:
            return
        self.shutter.disabled = True
        # If the camera never answers, let the user try again.
        Clock.schedule_once(self._unlock_shutter, 5)
        try:
            self.preview.capture_photo(
                location="private", subdir="shots",
                name=f"shot_{int(time.time() * 1000)}")
        except Exception as error:
            self.hint.text = f"Could not take the photo: {error}"
            self._unlock_shutter()

    def _unlock_shutter(self, *args):
        if self._connected:
            self.shutter.disabled = False

    def _photo_saved(self, path):
        # May be called from a camera thread -> hop to the main thread.
        Clock.schedule_once(lambda dt: self.fa.start_scan(path))


class AnalysingScreen(Screen):

    def __init__(self, fa, **kwargs):
        super().__init__(name="analysing", **kwargs)
        root = BoxLayout(orientation="vertical", padding=dp(28),
                         spacing=dp(12))
        paint(root, theme.DEEP)
        root.add_widget(Label())
        self.line = Label(
            text="", bold=True, font_size=sp(22), halign="center",
            color=rgba(theme.CREAM), size_hint_y=None, height=dp(90))
        self.line.bind(
            width=lambda lbl, w: setattr(lbl, "text_size", (w, None)))
        root.add_widget(self.line)
        root.add_widget(wrap_label(
            text="Working out which food it is, then how fresh it looks.",
            font_size=sp(14), halign="center", color=rgba(theme.MINT)))
        root.add_widget(Label())
        self.add_widget(root)

    def on_pre_enter(self, *args):
        import random
        self.line.text = random.choice(ANALYSING_LINES)

    def on_enter(self, *args):
        pulse = (Animation(opacity=0.35, duration=0.7)
                 + Animation(opacity=1.0, duration=0.7))
        pulse.repeat = True
        pulse.start(self.line)

    def on_leave(self, *args):
        Animation.cancel_all(self.line)
        self.line.opacity = 1.0


class ResultScreen(Screen):

    def __init__(self, fa, **kwargs):
        super().__init__(name="result", **kwargs)
        self.fa = fa
        root = BoxLayout(orientation="vertical")
        paint(root, theme.CREAM)

        scroll = ScrollView(do_scroll_x=False)
        self.content = GridLayout(
            cols=1, spacing=dp(12), padding=dp(16), size_hint_y=None)
        self.content.bind(minimum_height=self.content.setter("height"))
        scroll.add_widget(self.content)
        root.add_widget(scroll)

        self.picture = make_image(size_hint_y=None, height=dp(240))
        self.header = TextCard(font_size=sp(16))
        self.numbers = TextCard(font_size=sp(13), color=rgba(theme.MUTED))
        self.voice = TextCard(font_size=sp(17))
        self.verdict = TextCard(font_size=sp(16))
        self.notice = TextCard(bg="FFF3D6", font_size=sp(13))
        for widget in (self.picture, self.header, self.numbers, self.voice,
                       self.verdict, self.notice):
            self.content.add_widget(widget)

        buttons = BoxLayout(size_hint_y=None, height=dp(64),
                            padding=dp(8), spacing=dp(12))
        self.again_button = PillButton(
            text="Scan again", bg=theme.PAPER, fg=theme.FOREST)
        self.again_button.bind(on_release=lambda *_: fa.open_camera())
        self.save_button = PillButton(text="Save result")
        self.save_button.bind(on_release=lambda *_: fa.save_current())
        home = PillButton(text="Home", bg=theme.PAPER, fg=theme.FOREST)
        home.bind(on_release=lambda *_: fa.go("home"))
        for widget in (home, self.again_button, self.save_button):
            buttons.add_widget(widget)
        root.add_widget(buttons)
        self.add_widget(root)

    def show(self, entry, saved):
        colour = theme.verdict_hex(entry.get("safe"))
        food = escape_markup(str(entry.get("food", "food")).title())
        state = escape_markup(
            (entry.get("state") or "?").replace("_", " ").upper())

        self.picture.source = entry.get("image", "")
        try:
            self.picture.reload()
        except Exception:
            pass

        if entry.get("reliable", True):
            chip = f"[color={colour}][b]{state}[/b][/color]"
            food_line = f"[b][size={int(sp(26))}]{food}[/size][/b]"
        else:
            chip = f"[color={theme.verdict_hex('unknown')}][b]NOT SURE[/b][/color]"
            food_line = (f"[b][size={int(sp(22))}]Best guess: "
                         f"{food}[/size][/b]")
        self.header.set_text(f"{food_line}\n{chip}")

        self.numbers.set_text(
            f"{entry.get('food_conf', 0) * 100:.0f}% sure about the food  |  "
            f"{entry.get('state_conf', 0) * 100:.0f}% sure about the freshness"
            "\n(How sure the AI is. This is not the chance that it is safe.)")

        self.voice.set_text
        
