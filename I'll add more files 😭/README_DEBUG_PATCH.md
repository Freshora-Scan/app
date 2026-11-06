# Freshora Android debug patch

This is an **overlay patch** for the existing Freshora project, not a replacement for the entire repository. Apply the files at the same relative paths and keep the other application modules that were not part of the upload (for example `theme.py`, `greetings.py`, `freshora_store.py`, and `model/personality.py`).

## What was fixed

1. The screenshot error is a runtime/model operator mismatch: the supplied TFLite model needs `FULLY_CONNECTED` operator version 12, but the Android interpreter in the existing APK is older.
2. Android inference now uses Google's published LiteRT Interpreter AAR (`com.google.ai.edge.litert:litert:1.4.2`) through a small Java bridge. The bridge validates the model's tensor shapes and types, then performs a zero-input inference during startup. Freshora only enables scanning after the APK's actual native runtime successfully executes the full model. LiteRT's 1.x Interpreter API keeps the `org.tensorflow.lite` Java API, so the bridge uses those imports.
3. The Android build no longer depends on the old Python `tflite-runtime` package. Desktop prediction continues to use the Python TensorFlow/TFLite path.
4. `main.py` loads the already-trained model once in a background thread at app startup. It does **not** train on the phone. If model loading or the native startup inference fails, scanning stays blocked; the guarded camera/gallery actions remain available to display the underlying error rather than starting a scan that is certain to fail.
5. The build workflow validates the already-exported `.tflite` model with TensorFlow 2.20, including a real test inference and the label count, then builds the APK. It does not train or convert the model during an APK build. The patched `export_tflite.py` is pinned to TensorFlow 2.20 for the next time you retrain on a computer.
6. The Android target covers ARM 32-bit and ARM 64-bit devices running Android 7.0/API 24 or newer. It does not target x86 emulator devices, and no APK can be guaranteed to work on literally every Android model.

## Apply the patch

Copy these paths into the root of your existing GitHub repository, replacing the matching files:

- `main.py`
- `buildozer.spec`
- `requirements.txt`
- `model/predict.py`
- `model/train.py`
- `model/export_tflite.py`
- `model/class_names.txt`
- `model/labels.txt`
- `model/freshora_model.keras`
- `model/freshora_model.tflite`
- `android_src/org/freshora/ai/FreshoraTFLite.java`
- `.github/workflows/build-apk.yml`

Keep your existing non-uploaded Python modules and assets in place. The `.tflite` model and label files must be inside the `model/` directory because the project code and Buildozer configuration refer to those exact paths. The `.keras` file is included in this patch for future re-training/export, but the APK workflow does not require or package it.

## Build without a laptop

After pushing the replaced files to GitHub:

1. Open the repository's **Actions** tab.
2. Select **Build Android APK**.
3. Select **Run workflow**.
4. When it completes, open the run and download the `freshora-apk` artifact.

The workflow will stop before APK packaging if the model/label smoke test fails. That is intentional: it avoids producing an APK that cannot load the model.

## Checks performed here

- Python syntax compilation passed for the patched Python files.
- The GitHub Actions YAML parsed successfully.
- The predictor's Android branch passed a mocked PyJNIus bridge test for model load, input byte count, a 40-score output, and interpreter cleanup.
- The Java bridge passed a Java syntax compilation against API-shaped TensorFlow Lite stubs; this is not a substitute for compiling against the real LiteRT AAR.
- The supplied `class_names.txt` and `labels.txt` each contain 40 entries and match exactly.

**Not verified here:** a complete Gradle/Buildozer APK build, the real LiteRT AAR linkage on an Android device, and an actual camera/gallery scan. This environment does not have the Android SDK/NDK build chain or a connected phone. The GitHub Actions workflow performs the real build; the first device install is still the final compatibility test. If it fails, send the Actions error log or the phone screenshot.
