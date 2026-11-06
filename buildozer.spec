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
# The training dataset is intentionally not packaged.
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt,json
source.exclude_dirs = data,tests,bin,.git,.github,.buildozer,venv,__pycache__
source.exclude_patterns = model/LEGACY*,model/train.py,model/export_tflite.py,model/*.keras,model/*.png,model/_prepared_data/*,requirements.txt,BUILD_NOTES.md

# ---- Python libraries on the phone
# LiteRT runs natively on Android through the Java AAR below.
# Kivy includes PyJNIus, which connects Python to FreshoraTFLite.java.
requirements = python3,kivy,numpy,pillow,camera4kivy,gestures4kivy,androidstorage4kivy

# ---- Native Android LiteRT runtime
# The bundled LiteRT Interpreter runtime supports the model's built-in operators.
# LiteRT 1.4.2 provides the compatible Android Interpreter API (including FULLY_CONNECTED v12).
android.gradle_dependencies = com.google.ai.edge.litert:litert:1.4.2
android.add_src = android_src
android.enable_androidx = True
android.add_compile_options = "sourceCompatibility = 1.8", "targetCompatibility = 1.8"

# ---- Android compatibility
android.api = 33
android.minapi = 24
# Include both 64-bit and 32-bit ARM phones. x86 emulators are not targeted.
android.archs = arm64-v8a,armeabi-v7a
android.permissions = CAMERA
android.accept_sdk_license = True
p4a.branch = v2024.01.21
android.ndk = 25b
p4a.hook = camerax_provider/gradle_options.py

[buildozer]
log_level = 2
warn_on_root = 1
