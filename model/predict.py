# ============================================================
# FRESHORA - AI IMAGE PREDICTION (multi-food)
#
#     py -3.12 model/predict.py path\to\photo.jpg
#     py -3.12 model/predict.py path\to\photo.jpg --no-ai
#     py -3.12 model/predict.py --classes        (what did it learn?)
#
# Also imported by main.py (the app), so everything lives in
# the FreshoraPredictor class and nothing runs on import.
#
# The model's classes are "<food>__<state>" (apple__ripe ...).
# We answer two questions separately:
#   1. WHAT food is it?      -> add up all states of each food
#   2. HOW fresh is it?      -> best state among that food's states
# Splitting it this way is much more reliable than trusting one
# single winner among dozens of classes.
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

# Below these the scan is reported as "not sure" instead of guessing.
MIN_FOOD_CONFIDENCE = 0.50     # how sure it is WHICH food this is
MIN_STATE_CONFIDENCE = 0.55    # how sure it is about the freshness,
                               # among the states of that food

# Safety brake: if the model gives this much of a food's probability
# to "rotten" (or to a state we have no safety rule for) we never say
# "safe", even when another state wins. A 60% ripe / 40% rotten scan
# must come out as "not sure", not as "safe to eat".
MAX_RISKY_SHARE = 0.25


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
# 2. THE PREDICTOR
# ============================================================

def read_class_names():
    if not CLASS_NAMES_PATH.is_file():
        raise FreshoraError(
            "class_names.txt was not found.\n"
            "Train the model first:  py -3.12 model/train.py"
        )
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


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

        names = read_class_names()

        if not MODEL_PATH.is_file():
            raise FreshoraError(
                "The trained model (freshora_model.keras) was not found.\n"
                "Train the model first:  py -3.12 model/train.py"
            )

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
    def predict(self, image_path):
        self.load()
        batch = self._prepare(image_path)
        probabilities = self.model.predict(batch, verbose=0)[0]
        return self.interpret(probabilities)

    # --------------------------------------------------------
    # Predict + personality in one call (what the app uses)
    # --------------------------------------------------------
    def scan(self, image_path, use_ai=True):
        p = self.predict(image_path)
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
# 3. COMMAND LINE
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
        print("Analysing image...")
        p, reply = predictor.scan(args.image, use_ai=not args.no_ai)
    except FreshoraError as error:
        print("\nERROR:", error)
        sys.exit(1)

    print("\n===================================")
    print("FRESHORA RESULT")
    print("===================================")
    print(f"Food:   {p.food_name or '?'}  ({p.food_confidence * 100:.1f}% sure)")
    print(f"State:  {p.state}  ({p.state_confidence * 100:.1f}% of "
          f"this food's states)")

    print("\nTop guesses:")
    top = sorted(p.scores.items(), key=lambda item: item[1], reverse=True)[:5]
    for name, score in top:
        print(f"  {name:30s}: {score * 100:5.1f}%")

    print("\n" + reply.full())

    if args.debug:
        print(f"\n[voice source: {reply.source}] {reply.note}")

    print("\n-----------------------------------")
    print("Freshora can be wrong. It only sees the outside of the food.")
    print("Never rely on the AI alone to decide if food is safe to eat.")
    print("-----------------------------------\n")


if __name__ == "__main__":
    main()
    
