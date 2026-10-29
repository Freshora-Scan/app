# Freshora: how to run and build

## A. Run the app on a laptop (Windows or Mac)

Needs Python 3.10, 3.11 or 3.12.

    python -m venv venv
    (Windows)  venv\Scripts\activate        (Mac)  source venv/bin/activate
    pip install -r requirements.txt
    python main.py

If `model/freshora_model.tflite` exists the app uses it, otherwise it falls
back to `model/freshora_model.keras`. On a laptop the camera button opens a
file picker (the real camera only exists in the Android build).

## B. Make the phone model (once after every training)

    python model/train.py
    python model/export_tflite.py

Check that it prints `GOOD. The phone model matches the laptop model.`
Then commit `model/freshora_model.tflite` and `model/labels.txt`.

## C. Build the APK (no installs needed)

1. Push everything to GitHub (including the two model files above).
2. GitHub -> Actions -> **Build Android APK** -> Run workflow.
3. Wait 20-40 min (the first build is the slowest).
4. Open the finished run -> Artifacts -> **freshora-apk** -> download, unzip.
5. Copy the .apk to an Android phone, open it, allow "install unknown apps".

To publish for testers: create a tag such as `v0.1.0`. The APK is attached
to a GitHub Release automatically.

Every build is signed with a fresh debug key, so a new APK cannot install
over an older one. Uninstall the old Freshora first.

## D. If the build fails

Open the failed run, expand the red step, copy the last 40 lines and send
them. Do not guess-fix `buildozer.spec`.

## E. Building on a Mac instead (optional, slower to set up)

Buildozer supports macOS, but GitHub's build (C) needs no setup. If you
still want it: follow https://buildozer.readthedocs.io/en/latest/installation/
then in the project folder run

    git clone --depth 1 https://github.com/Android-for-Python/camerax_provider.git
    rm -rf camerax_provider/.git
    buildozer -v android debug
