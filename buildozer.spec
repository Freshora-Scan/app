[app]

# ---- identity
title = Freshora
package.name = freshora
package.domain = org.freshora
version = 0.1.0
orientation = portrait
fullscreen = 0
# icon.filename = %(source.dir)s/assets/icon.png
# presplash.filename = %(source.dir)s/assets/presplash.png

# ---- what goes INTO the app
# (data/ holds thousands of training photos: it must NOT be packed.)
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt,json
source.exclude_dirs = data,tests,bin,.git,.github,.buildozer,venv,__pycache__
source.exclude_patterns = model/LEGACY*,model/train.py,model/export_tflite.py,model/*.keras,model/*.png,model/_prepared_data/*,requirements.txt,BUILD_NOTES.md

# ---- libraries on the phone
# tflite-runtime runs the .tflite model (NOT full TensorFlow).
# camera4kivy + gestures4kivy = camera, androidstorage4kivy = gallery.
requirements = python3,kivy,numpy,pillow,camera4kivy,gestures4kivy,androidstorage4kivy,tflite-runtime

# ---- Android
android.api = 33
android.minapi = 24
android.archs = arm64-v8a
android.permissions = CAMERA
android.accept_sdk_license = True
# Sets up the camera library (the folder is cloned by the build, see
# .github/workflows/build-apk.yml or BUILD_NOTES.md).
p4a.hook = camerax_provider/gradle_options.py


[buildozer]
log_level = 2
warn_on_root = 1
