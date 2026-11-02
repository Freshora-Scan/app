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
        