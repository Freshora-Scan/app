# ============================================================
# FRESHORA - AI MODEL TRAINING
#
# Run it from ANYWHERE, the paths no longer depend on where
# your terminal is:
#
#     py -3.12 model/train.py
#     py -3.12 train.py          (from inside the model folder)
#
# Creates:
#     model/freshora_model.keras      the trained model
#     model/class_names.txt           class order (predict.py needs it)
#     model/training_accuracy.png     accuracy graph (if matplotlib exists)
# ============================================================

import os
import sys
from pathlib import Path

# Hide TensorFlow's very noisy start-up messages.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT_NAME = "Freshora"

# Every path is built from THIS FILE's location, not from the
# terminal's current folder. This was the cause of:
#   FileNotFoundError: Dataset not found: data/banana
ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "banana"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "freshora_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.txt"
PLOT_PATH = MODEL_DIR / "training_accuracy.png"

IMG_SIZE = (224, 224)
BATCH_SIZE = 8          # small on purpose, friendly to low-RAM laptops
EPOCHS = 15
SEED = 123
VALIDATION_SPLIT = 0.2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

MIN_IMAGES_PER_CLASS = 2        # below this, training is impossible
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
# 2. CHECK THE DATASET BEFORE TOUCHING TENSORFLOW
# ============================================================

def find_images(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def inspect_dataset():
    if not DATASET_PATH.is_dir():
        stop(
            f"Dataset folder not found:\n  {DATASET_PATH}\n\n"
            "It must look like this:\n"
            "  data/banana/unripe/*.jpg\n"
            "  data/banana/ripe/*.jpg\n"
            "  data/banana/overripe/*.jpg\n"
            "  data/banana/rotten/*.jpg\n\n"
            "If you downloaded the repo as a ZIP, check that the 'data'\n"
            "folder really is inside the project folder:\n"
            f"  {ROOT}"
        )

    class_folders = sorted(p for p in DATASET_PATH.iterdir() if p.is_dir())

    if len(class_folders) < 2:
        stop(
            f"Found {len(class_folders)} class folder(s) in {DATASET_PATH}.\n"
            "You need at least 2 (for example: ripe and rotten)."
        )

    counts = {}
    images_by_class = {}
    for folder in class_folders:
        images = find_images(folder)
        images_by_class[folder.name] = images
        counts[folder.name] = len(images)

    # Make sure every image can really be opened. A single broken
    # file (for example a half-downloaded .jpg) would crash training.
    from PIL import Image

    broken = []
    for images in images_by_class.values():
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

    print("Dataset check")
    print("-------------")
    print(f"Folder: {DATASET_PATH}")
    for name, count in counts.items():
        print(f"  {name:10s}: {count} images")

    empty = [n for n, c in counts.items() if c < MIN_IMAGES_PER_CLASS]
    if empty:
        stop(
            f"These classes have fewer than {MIN_IMAGES_PER_CLASS} images: "
            f"{', '.join(empty)}\n"
            "Add more photos to those folders (aim for "
            f"{RECOMMENDED_PER_CLASS}+ per class)."
        )

    return counts


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data(counts):
    import tensorflow as tf

    common = dict(
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    smallest = min(counts.values())
    use_validation = smallest >= MIN_FOR_VALIDATION

    if use_validation:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(DATASET_PATH),
            validation_split=VALIDATION_SPLIT,
            subset="training",
            **common,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            str(DATASET_PATH),
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
            str(DATASET_PATH), **common
        )
        val_ds = None

    class_names = list(train_ds.class_names)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    if val_ds is not None:
        val_ds = val_ds.prefetch(autotune)

    return train_ds, val_ds, class_names


# ============================================================
# 4. BUILD THE MODEL (MobileNetV2 transfer learning)
# ============================================================
# Why not the old hand-made CNN? With a few dozen photos a CNN
# built from scratch just memorises them. MobileNetV2 already
# knows what edges, colours and textures look like from ImageNet,
# so we only teach it the last step: ripe vs rotten.
#
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

    try:
        base = tf.keras.applications.MobileNetV2(
            input_shape=IMG_SIZE + (3,),
            include_top=False,
            weights="imagenet",
        )
    except Exception as error:
        stop(
            "Could not download the pretrained MobileNetV2 weights "
            "(a one-time ~14 MB download).\n"
            f"Reason: {error}\n\n"
            "Connect to the internet and run again. After the first\n"
            "successful run the weights are cached on your computer."
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
# 5. SAVE EVERYTHING + SELF-TEST
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
# 6. MAIN
# ============================================================

def main():
    print("\n===================================")
    print(PROJECT_NAME, "- AI MODEL TRAINING")
    print("===================================\n")

    counts = inspect_dataset()

    import tensorflow as tf

    train_ds, val_ds, class_names = load_data(counts)

    print("\nClasses (order the model will use):")
    for number, name in enumerate(class_names):
        print(f"  {number} -> {name}")

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

    print("\n===================================")
    print("TRAINING COMPLETE")
    print("===================================")
    print("Model saved:  ", MODEL_PATH)
    print("Classes saved:", CLASS_NAMES_PATH)
    print("\nFreshora AI model is ready.\n")


if __name__ == "__main__":
    main()
  
