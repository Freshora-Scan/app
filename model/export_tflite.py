# ============================================================
# FRESHORA - EXPORT THE MODEL FOR ANDROID
#
#     py -3.12 model\export_tflite.py
#
# Turns freshora_model.keras (laptop format) into
# freshora_model.tflite (phone format) and then CHECKS that the
# phone version gives the same answers as the laptop version.
# Run it once after every training.
# ============================================================

import os
import random
import shutil
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
KERAS_PATH = HERE / "freshora_model.keras"
CLASS_NAMES_PATH = HERE / "class_names.txt"
TFLITE_PATH = HERE / "freshora_model.tflite"
LABELS_PATH = HERE / "labels.txt"
IMG_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def fail(message):
    print("\nERROR:", message)
    sys.exit(1)


def load_image(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE)
        # Raw 0-255 numbers: the model rescales by itself.
        return np.expand_dims(np.array(img, dtype=np.float32), axis=0)


def find_test_images(count=12):
    data = ROOT / "data"
    files = [p for p in data.rglob("*")
             if p.suffix.lower() in IMAGE_EXTENSIONS
             and "_prepared_data" not in p.parts]
    random.Random(1).shuffle(files)
    return files[:count]


def make_inference_model(model, tf):
    """Rebuild the trained graph without its training-only augmentation."""
    try:
        rescale = model.get_layer("rescale")
        classifier = model.get_layer("classes")
    except ValueError as error:
        fail(f"The trained model is missing an expected layer: {error}")

    base_models = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model) and layer.name != "augmentation"
    ]
    if not base_models:
        fail("Could not find the MobileNetV2 base model in the trained model.")
    base = base_models[0]

    pooling_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)
    ]
    dropout_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dropout)
    ]
    if not pooling_layers or not dropout_layers:
        fail("The trained model is missing its pooling or dropout layer.")

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="image")
    x = rescale(inputs)
    x = base(x, training=False)
    x = pooling_layers[0](x)
    x = dropout_layers[0](x, training=False)
    outputs = classifier(x)
    inference_model = tf.keras.Model(
        inputs, outputs, name="freshora_inference")

    return inference_model


def main():
    if not KERAS_PATH.is_file():
        fail("freshora_model.keras not found. Train first: "
             "py -3.12 model\\train.py")
    if not CLASS_NAMES_PATH.is_file():
        fail("class_names.txt not found. Train first.")

    import tensorflow as tf

    # The Android app packages LiteRT Interpreter 1.4.2. Pin the converter
    # family so a future desktop TensorFlow upgrade cannot silently create
    # a model that requires newer operators than the APK runtime provides.
    if not tf.__version__.startswith("2.20."):
        fail(
            "Freshora's pinned exporter requires TensorFlow 2.20.x, but this "
            f"export is running TensorFlow {tf.__version__}.\n"
            "Install the pinned development dependencies from "
            "requirements.txt, then export again."
        )

    print("Loading the laptop model...")
    model = tf.keras.models.load_model(str(KERAS_PATH))
    print("Preparing the phone model without training-only augmentation...")
    inference_model = make_inference_model(model, tf)

    print("Converting to phone format (this takes a minute)...")
    export_dir = HERE / "_export_tmp"
    if export_dir.exists():
        shutil.rmtree(export_dir)

    try:
        inference_model.export(str(export_dir))
        converter = tf.lite.TFLiteConverter.from_saved_model(str(export_dir))
    except Exception as error:
        print("(first route failed, trying the second:", error, ")")
        converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)

    # Keep the exported graph inside the standard builtin operator set.
    # The Android APK uses LiteRT Interpreter 1.4.2, which supports this TFLite model format.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.allow_custom_ops = False

    # Dynamic-range weight quantization makes the model smaller while keeping
    # float32 image input and float32 class scores.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    TFLITE_PATH.write_bytes(tflite_bytes)

    if export_dir.exists():
        shutil.rmtree(export_dir, ignore_errors=True)

    shutil.copyfile(CLASS_NAMES_PATH, LABELS_PATH)
    print(f"Saved {TFLITE_PATH.name}  ({len(tflite_bytes) / 1e6:.1f} MB)")
    print(f"Saved {LABELS_PATH.name}")

    # ---------------- check: phone version == laptop version
    print("\nChecking both versions give the same answers...")
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    print("Phone model input :", inp["shape"], inp["dtype"].__name__)
    print("Phone model output:", out["shape"])

    if tuple(int(x) for x in inp["shape"]) != (1, 224, 224, 3):
        fail("Android model input must be [1, 224, 224, 3].")
    if np.dtype(inp["dtype"]) != np.float32:
        fail("Android model input must be float32 RGB pixels.")
    if np.dtype(out["dtype"]) != np.float32:
        fail("Android model output must be float32 class scores.")

    names = [l.strip() for l in open(CLASS_NAMES_PATH, encoding="utf-8")
             if l.strip()]
    if int(out["shape"][-1]) != len(names):
        fail("The phone model has a different number of classes than "
             "class_names.txt. Train again, then export again.")

    images = find_test_images()
    if not images:
        print("No images found in data/ to check with. Skipping the check.")
        return

    same = 0
    worst = 0.0
    for path in images:
        batch = load_image(path)
        keras_p = model.predict(batch, verbose=0)[0]
        interpreter.set_tensor(inp["index"], batch.astype(inp["dtype"]))
        interpreter.invoke()
        lite_p = interpreter.get_tensor(out["index"])[0]
        agree = int(np.argmax(keras_p)) == int(np.argmax(lite_p))
        same += agree
        worst = max(worst, float(np.max(np.abs(keras_p - lite_p))))
        print(f"  {'OK ' if agree else 'DIFF'} {names[int(np.argmax(keras_p))]:28s}"
              f" laptop {keras_p.max() * 100:5.1f}%  phone {lite_p.max() * 100:5.1f}%")

    print(f"\n{same}/{len(images)} same top answer. "
          f"Biggest probability gap: {worst * 100:.1f} points.")
    if same == len(images) and worst < 0.05:
        print("GOOD. The phone model matches the laptop model.")
    else:
        fail(
            "The exported phone model differs from the laptop model on the "
            "test images. Do not publish this APK until the mismatch is "
            "investigated."
        )


if __name__ == "__main__":
    main()
