# ============================================================
# FRESHORA - AI IMAGE PREDICTION (laptop AND phone)
#
#     py -3.12 model/predict.py path\to\photo.jpg
#     py -3.12 model/predict.py path\to\photo.jpg --no-ai
#     py -3.12 model/predict.py --classes        (what did it learn?)
#
# Also imported by main.py (the app), so everything lives in
# the FreshoraPredictor class and nothing runs on import.
#
# ANDROID RUNTIME
#   Android uses the bundled LiteRT Interpreter 1.4.2 Java runtime through
#   PyJNIus. Desktop keeps using a Python TFLite/TensorFlow interpreter.
#   The phone only loads the exported .tflite file; training stays offline.
#
# The model's classes are "<food>__<state>" (apple__ripe ...).
# We answer two questions separately:
#   1. WHAT food is it?      -> add up all states of each food
#   2. HOW fresh is it?      -> best state among that food's states
# ============================================================

import importlib
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image, ImageOps

# Make "import personality" work no matter where we are started from.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import personality  # noqa: E402


# ============================================================
# 1. SETTINGS
# ============================================================

TFLITE_PATH = HERE / "freshora_model.tflite"
KERAS_PATH = HERE / "freshora_model.keras"
# export_tflite.py writes labels.txt (a copy of class_names.txt).
CLASS_NAME_FILES = [HERE / "class_names.txt", HERE / "labels.txt"]
IMG_SIZE = (224, 224)

# Below these the scan is reported as "not sure" instead of guessing.
MIN_FOOD_CONFIDENCE = 0.80     # Freshora only reports food IDs at 80%+
MIN_STATE_CONFIDENCE = 0.80    # and only reports freshness at 80%+
                               # among the states of that food

# Safety brake: if the model gives this much of a food's probability
# to "rotten" (or to a state we have no safety rule for) we never say
# "safe", even when another state wins. A 60% ripe / 40% rotten scan
# must come out as "not sure", not as "safe to eat".
MAX_RISKY_SHARE = 0.25

# The runtimes we try, in this order, to run the .tflite file.
TFLITE_RUNTIMES = ("tflite_runtime.interpreter", "ai_edge_litert.interpreter")


_NATIVE_CLASS = None


def resolve_android_class():
    """
    Finds the Java class FreshoraTFLite. MUST be called once from the main
    (UI) thread, e.g. in App.build(). A background thread uses Android's
    system class loader, which cannot see classes inside the APK, and fails
    with ClassNotFoundException. Safe to call again.
    """
    global _NATIVE_CLASS
    if _NATIVE_CLASS is None:
        from jnius import autoclass
        _NATIVE_CLASS = autoclass("org.freshora.ai.FreshoraTFLite")
    return _NATIVE_CLASS


class FreshoraError(Exception):
    """A problem with a plain-English message for the user."""


@dataclass
class Prediction:
    label: str                 # full class, e.g. "apple__ripe"
    food: str                  # "apple"
    state: str                 # "ripe"
    confidence: float          # model probability of that exact class
    food_confidence: float     # 0-1, is it really this food?
    state_confidence: float    # 0-1, among this food's states
    food_reliable: bool
    state_reliable: bool
    judgeable: bool            # False if only ONE state was ever learned
    scores: dict               # every class -> probability

    @property
    def reliable(self):
        return self.food_reliable and self.state_reliable and self.judgeable

    @property
    def food_name(self):
        return personality.display_name(self.food)


# ============================================================
# 2. HELPERS
# ============================================================

def read_class_names():
    for path in CLASS_NAME_FILES:
        if path.is_file():
            with open(path, "r", encoding="utf-8") as file:
                names = [line.strip() for line in file if line.strip()]
            if names:
                return names
    raise FreshoraError(
        "class_names.txt was not found.\n"
        "Train the model first:  py -3.12 model/train.py"
    )


def _find_tflite_interpreter():
    """Returns the Interpreter class of the first runtime that works."""
    for module_name in TFLITE_RUNTIMES:
        try:
            return importlib.import_module(module_name).Interpreter
        except Exception:
            continue
    try:  # the laptop: full TensorFlow contains the same interpreter
        import tensorflow as tf
        return tf.lite.Interpreter
    except Exception:
        return None


def _resize_filter():
    # Training resized with bilinear; do the same here.
    resampling = getattr(Image, "Resampling", Image)
    return resampling.BILINEAR


# ============================================================
# 3. THE PREDICTOR
# ============================================================

class FreshoraPredictor:

    def __init__(self, android=False):
        self.class_names = []
        self.android = bool(android)
        self.backend = None          # android_tflite, tflite, or keras
        self._interpreter = None
        self._android_interpreter = None
        self._input_index = None
        self._output_index = None
        self._keras_model = None
        # Loading and running must never happen on two threads at once.
        self._lock = threading.RLock()

    @property
    def loaded(self):
        return self.backend is not None

    # --------------------------------------------------------
    # Load model + class names (call once, takes a few seconds)
    # --------------------------------------------------------
    def load(self):
        with self._lock:
            if self.loaded:
                return

            names = read_class_names()

            if self.android:
                # Use the Android-native TensorFlow Lite AAR, not the old
                # python-for-android tflite-runtime recipe. The submitted
                # model uses FULLY_CONNECTED v12, supported by TFLite 2.16.1+.
                if not TFLITE_PATH.is_file():
                    raise FreshoraError(
                        "The Android model is missing: freshora_model.tflite.\n"
                        "Put it in the model/ folder and rebuild the APK."
                    )
                self._load_android_tflite(names)
            else:
                interpreter_class = _find_tflite_interpreter()
                if TFLITE_PATH.is_file() and interpreter_class is not None:
                    self._load_tflite(interpreter_class, names)
                elif KERAS_PATH.is_file():
                    self._load_keras(names)
                elif TFLITE_PATH.is_file():
                    raise FreshoraError(
                        "The phone model exists but no TFLite runtime is "
                        "installed.\nOn the laptop run:  pip install tensorflow."
                    )
                else:
                    raise FreshoraError(
                        "No trained model was found.\n"
                        "Train it:   py -3.12 model/train.py\n"
                        "Then export: py -3.12 model/export_tflite.py"
                    )
            self.class_names = names

    def _load_android_tflite(self, names):
        """Load the model through the bundled LiteRT Java runtime."""
        interpreter = None
        try:
            NativeInterpreter = resolve_android_class()
            interpreter = NativeInterpreter(str(TFLITE_PATH))
            input_bytes = int(interpreter.getInputByteCount())
            outputs = int(interpreter.getOutputSize())
        except Exception as error:
            if interpreter is not None:
                try:
                    interpreter.close()
                except Exception:
                    pass
            raise FreshoraError(
                "Freshora's Android LiteRT Interpreter 1.4.2 runtime could not "
                "load the model.\n"
                "The APK must include android_src/ and the Gradle dependency "
                "com.google.ai.edge.litert:litert:1.4.2.\n"
                f"Details: {error}"
            )

        expected_input_bytes = 1 * IMG_SIZE[0] * IMG_SIZE[1] * 3 * 4
        if input_bytes != expected_input_bytes:
            try:
                interpreter.close()
            except Exception:
                pass
            raise FreshoraError(
                "The Android model has an unexpected input tensor.\n"
                f"Expected {expected_input_bytes} bytes for a 224x224 RGB "
                f"float32 image; the model reports {input_bytes}."
            )

        if outputs != len(names):
            try:
                interpreter.close()
            except Exception:
                pass
            raise FreshoraError(
                f"The Android model has {outputs} outputs but the class list "
                f"has {len(names)} names.\n"
                "The model and class_names.txt must come from the same "
                "training run."
            )

        self._android_interpreter = interpreter
        self.backend = "android_tflite"

    def _load_tflite(self, interpreter_class, names):
        try:
            interpreter = interpreter_class(model_path=str(TFLITE_PATH))
            interpreter.allocate_tensors()
            inp = interpreter.get_input_details()[0]
            out = interpreter.get_output_details()[0]
        except Exception as error:
            raise FreshoraError(
                "The phone model (freshora_model.tflite) could not be "
                "loaded.\nExport it again:  py -3.12 model/export_tflite.py\n"
                f"Details: {error}"
            )

        outputs = int(out["shape"][-1])
        if outputs != len(names):
            raise FreshoraError(
                f"The model has {outputs} outputs but the class list has "
                f"{len(names)} names.\nThey come from different training "
                "runs. Train, then export again."
            )
        if np.dtype(inp["dtype"]) != np.float32:
            raise FreshoraError(
                "The phone model expects "
                f"{np.dtype(inp['dtype']).name} input, but Freshora sends "
                "float32.\nExport it again with export_tflite.py."
            )

        self._interpreter = interpreter
        self._input_index = inp["index"]
        self._output_index = out["index"]
        self.backend = "tflite"

    def _load_keras(self, names):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(str(KERAS_PATH))
        except Exception as error:
            raise FreshoraError(
                "The model file could not be loaded.\n"
                "Retrain it with:  py -3.12 model/train.py\n"
                f"Details: {error}"
            )
        outputs = int(model.output_shape[-1])
        if outputs != len(names):
            raise FreshoraError(
                f"The model has {outputs} outputs but class_names.txt lists "
                f"{len(names)} classes.\n"
                "They come from different training runs. "
                "Retrain with:  py -3.12 model/train.py"
            )
        self._keras_model = model
        self.backend = "keras"

    # --------------------------------------------------------
    # Turn an image into the array the model expects
    # --------------------------------------------------------
    @staticmethod
    def _prepare(source):
        """source: a file path OR an already opened PIL image."""
        try:
            if isinstance(source, Image.Image):
                img = source
            else:
                path = Path(source)
                if not path.is_file():
                    raise FreshoraError(f"Image does not exist:\n{path}")
                img = Image.open(path)
            with img:
                # Phone photos are often rotated through EXIF data.
                img = ImageOps.exif_transpose(img).convert("RGB")
                img = img.resize(IMG_SIZE, _resize_filter())
                array = np.array(img, dtype=np.float32)
        except FreshoraError:
            raise
        except Exception as error:
            raise FreshoraError(f"Could not open this image: {error}")

        # NO "/ 255" here. The model rescales by itself.
        return np.expand_dims(array, axis=0)

    # --------------------------------------------------------
    # Run the model: image batch -> one probability per class
    # --------------------------------------------------------
    def _run(self, batch):
        with self._lock:
            if self.backend == "android_tflite":
                try:
                    # Send the contiguous float32 tensor as bytes. This avoids
                    # converting 150k pixels into a large Python list/JNI array.
                    raw_input = bytearray(
                        np.ascontiguousarray(batch, dtype=np.float32).tobytes()
                    )
                    raw_output = self._android_interpreter.runBytes(raw_input)
                    probabilities = np.asarray(list(raw_output), dtype=np.float32)
                except Exception as error:
                    raise FreshoraError(
                        "Android LiteRT could not analyse this photo.\n"
                        f"Details: {error}"
                    )
                if probabilities.size != len(self.class_names):
                    raise FreshoraError(
                        "The Android model returned an unexpected number of "
                        "scores. Check that the APK contains the matching "
                        "model and class_names.txt."
                    )
                return probabilities

            if self.backend == "tflite":
                self._interpreter.set_tensor(self._input_index, batch)
                self._interpreter.invoke()
                return np.array(
                    self._interpreter.get_tensor(self._output_index)[0]
                )
            return self._keras_model.predict(batch, verbose=0)[0]

    def close(self):
        """Release native model memory when the app is closing."""
        with self._lock:
            if self._android_interpreter is not None:
                try:
                    self._android_interpreter.close()
                finally:
                    self._android_interpreter = None
            self._interpreter = None
            self._keras_model = None
            self.backend = None

    # --------------------------------------------------------
    # Turn raw probabilities into food + state
    # --------------------------------------------------------
    def interpret(self, probabilities):
        """probabilities: one number per class, same order as class_names."""
        by_food = {}
        for index, class_name in enumerate(self.class_names):
            food, state = personality.split_label(class_name)
            by_food.setdefault(food, []).append(
                (state, float(probabilities[index]), class_name)
            )

        food_totals = {
            food: sum(p for _, p, _ in items) for food, items in by_food.items()
        }
        best_food = max(food_totals, key=food_totals.get)
        food_conf = food_totals[best_food]

        items = by_food[best_food]
        state, state_p, label = max(items, key=lambda item: item[1])
        state_conf = state_p / food_conf if food_conf > 0 else 0.0

        # How much of this food's probability sits on risky states?
        risky = 0.0
        for item_state, item_p, _ in items:
            if personality.classify_state(item_state) in ("rotten", None):
                risky += item_p
        risky_share = risky / food_conf if food_conf > 0 else 0.0
        winner_is_risky = personality.classify_state(state) in ("rotten", None)
        brake = (not winner_is_risky) and risky_share >= MAX_RISKY_SHARE

        return Prediction(
            label=label,
            food=best_food,
            state=state,
            confidence=state_p,
            food_confidence=food_conf,
            state_confidence=state_conf,
            food_reliable=food_conf >= MIN_FOOD_CONFIDENCE,
            state_reliable=(state_conf >= MIN_STATE_CONFIDENCE) and not brake,
            # With a single learned state the "state confidence" is
            # always 100% and means nothing. Never trust it.
            judgeable=len(items) >= 2,
            scores={
                name: float(probabilities[i])
                for i, name in enumerate(self.class_names)
            },
        )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------
    def predict(self, source):
        """source: image file path or a PIL image."""
        self.load()
        batch = self._prepare(source)
        return self.interpret(self._run(batch))

    # --------------------------------------------------------
    # Predict + personality in one call (what the app uses)
    # --------------------------------------------------------
    def scan(self, source, use_ai=True):
        p = self.predict(source)
        reply = personality.make_reply(
            p.food,
            p.state,
            food_conf=p.food_confidence,
            state_conf=p.state_confidence,
            food_ok=p.food_reliable,
            state_ok=p.state_reliable,
            judgeable=p.judgeable,
            use_ai=use_ai,
        )
        return p, reply


# ============================================================
# 4. COMMAND LINE
# ============================================================

def show_classes():
    """List what the model learned and how each state will be treated."""
    names = read_class_names()

    foods = {}
    for class_name in names:
        food, state = personality.split_label(class_name)
        foods.setdefault(food, []).append(state)

    print(f"\nFreshora knows {len(foods)} food(s), {len(names)} classes:\n")
    problems = 0
    for food in sorted(foods):
        print(f"  {personality.display_name(food) or '(no food name)'}")
        for state in sorted(foods[food]):
            group = personality.classify_state(state)
            if group is None:
                note = ("UNKNOWN state name: scans will say 'check it "
                        "yourself'. Add it to STATE_GROUPS in personality.py")
                problems += 1
            else:
                note = f"treated as {group}"
            print(f"      {state:14s} -> {note}")
        if len(foods[food]) < 2:
            print("      !! only ONE state: cannot judge freshness yet")
            problems += 1

    if problems:
        print(f"\n{problems} thing(s) to fix above.\n")
    else:
        print("\nAll state names are understood.\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Freshora food scanner")
    parser.add_argument("image", nargs="?", help="path to a photo of the food")
    parser.add_argument(
        "--no-ai", action="store_true",
        help="use the offline personality only (no OpenAI call)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="show why the AI personality was skipped",
    )
    parser.add_argument(
        "--classes", action="store_true",
        help="list what the trained model knows (no photo needed)",
    )
    args = parser.parse_args()

    if args.classes:
        try:
            show_classes()
        except FreshoraError as error:
            print("\nERROR:", error)
            sys.exit(1)
        return

    if not args.image:
        parser.error("give a photo path, or use --classes")

    predictor = FreshoraPredictor()

    try:
        print("\nLoading Freshora AI model...")
        predictor.load()
        print(f"Model type: {predictor.backend}")
        print("Analysing image...")
        p, reply = predictor.scan(args.image, use_ai=not args.no_ai)
    except FreshoraError as error:
        print("\nERROR:", error)
        sys.exit(1)

    print("\n==========================================")
    print("FRESHORA RESULT")
    print("==========================================")
    print(f"Food:  {p.food_name or '?'} ({p.food_confidence * 100:.1f}% sure)")
    print(f"State: {p.state} ({p.state_confidence * 100:.1f}% of this food's states)")

    print("\nTop guesses:")
    top = sorted(p.scores.items(), key=lambda item: item[1], reverse=True)[:5]
    for name, score in top:
        print(f"  {name:30s}: {score * 100:5.1f}%")

    print("\n" + str(reply.full()))

    if args.debug:
        print(f"\n[voice source: {reply.source}] {reply.note}")

    print("\n------------------------------------------")
    print("Freshora can be wrong. It only sees the outside of the food.")
    print("Never rely on the AI alone to decide if food is safe to eat.")
    print("------------------------------------------\n")

if __name__ == "__main__":
    main()
    
