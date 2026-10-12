# ============================================================
# FRESHORA - AI IMAGE PREDICTION
#
#     py -3.12 model/predict.py path\to\banana.jpg
#     py -3.12 model/predict.py path\to\banana.jpg --no-ai
#
# Also imported by main.py (the app), so everything lives in
# the FreshoraPredictor class and nothing runs on import.
# ============================================================

import os
import sys
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

MODEL_PATH = HERE / "freshora_model.keras"
CLASS_NAMES_PATH = HERE / "class_names.txt"
IMG_SIZE = (224, 224)

# Below this the scan is reported as "not sure" instead of guessing.
MIN_CONFIDENCE = 0.55


class FreshoraError(Exception):
    """A problem with a plain-English message for the user."""


@dataclass
class Prediction:
    label: str
    confidence: float
    scores: dict
    reliable: bool


# ============================================================
# 2. THE PREDICTOR
# ============================================================

class FreshoraPredictor:

    def __init__(self):
        self.model = None
        self.class_names = []

    # --------------------------------------------------------
    # Load model + class names (call once, takes a few seconds)
    # --------------------------------------------------------
    def load(self):
        if self.model is not None:
            return

        if not CLASS_NAMES_PATH.is_file():
            raise FreshoraError(
                "class_names.txt was not found.\n"
                "Train the model first:  py -3.12 model/train.py"
            )

        if not MODEL_PATH.is_file():
            raise FreshoraError(
                "The trained model (freshora_model.keras) was not found.\n"
                "Train the model first:  py -3.12 model/train.py"
            )

        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            names = [line.strip() for line in file if line.strip()]

        import tensorflow as tf  # imported here so the app window opens fast

        try:
            model = tf.keras.models.load_model(str(MODEL_PATH))
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

        self.model = model
        self.class_names = names

    # --------------------------------------------------------
    # Turn an image file into the array the model expects
    # --------------------------------------------------------
    @staticmethod
    def _prepare(image_path):
        path = Path(image_path)
        if not path.is_file():
            raise FreshoraError(f"Image does not exist:\n{path}")

        try:
            with Image.open(path) as img:
                # Phone photos are often rotated through EXIF data.
                img = ImageOps.exif_transpose(img).convert("RGB")
                img = img.resize(IMG_SIZE)
                array = np.array(img, dtype=np.float32)
        except Exception as error:
            raise FreshoraError(f"Could not open this image: {error}")

        # NO "/ 255" here. The model rescales by itself.
        return np.expand_dims(array, axis=0)

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------
    def predict(self, image_path):
        self.load()
        batch = self._prepare(image_path)

        probabilities = self.model.predict(batch, verbose=0)[0]
        best = int(np.argmax(probabilities))
        confidence = float(probabilities[best])

        return Prediction(
            label=self.class_names[best],
            confidence=confidence,
            scores={
                name: float(probabilities[i])
                for i, name in enumerate(self.class_names)
            },
            reliable=confidence >= MIN_CONFIDENCE,
        )

    # --------------------------------------------------------
    # Predict + personality in one call (what the app uses)
    # --------------------------------------------------------
    def scan(self, image_path, use_ai=True):
        prediction = self.predict(image_path)
        reply = personality.make_reply(
            prediction.label,
            prediction.confidence,
            reliable=prediction.reliable,
            use_ai=use_ai,
        )
        return prediction, reply


# ============================================================
# 3. COMMAND LINE
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Freshora banana scanner")
    parser.add_argument("image", help="path to a photo of the food")
    parser.add_argument(
        "--no-ai", action="store_true",
        help="use the offline personality only (no OpenAI call)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="show why the AI personality was skipped",
    )
    args = parser.parse_args()

    predictor = FreshoraPredictor()

    try:
        print("\nLoading Freshora AI model...")
        predictor.load()
        print("Analysing image...")
        prediction, reply = predictor.scan(args.image, use_ai=not args.no_ai)
    except FreshoraError as error:
        print("\nERROR:", error)
        sys.exit(1)

    print("\n===================================")
    print("FRESHORA RESULT")
    print("===================================")
    print("Food:      Banana")
    print("Prediction:", prediction.label)
    print(f"Confidence: {prediction.confidence * 100:.2f}%")

    print("\nAll class scores:")
    for name, score in sorted(
        prediction.scores.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"  {name:10s}: {score * 100:.2f}%")

    print("\n" + reply.full())

    if args.debug:
        print(f"\n[voice source: {reply.source}] {reply.note}")

    print("\n-----------------------------------")
    print("Freshora can be wrong. It only sees the outside of the food.")
    print("Never rely on the AI alone to decide if food is safe to eat.")
    print("-----------------------------------\n")


if __name__ == "__main__":
    main()
      
