# ============================================================
# FRESHORA - LOCAL STORAGE (scan history + settings)
#
# Everything is saved on the PHONE only, in two small files:
#     history.json   the saved scans
#     settings.json  small flags (e.g. "welcome screen already seen")
# plus a small thumbnail photo per saved scan in the "scans" folder.
# Nothing is ever sent anywhere.
# ============================================================

import json
import os
import time
from pathlib import Path

from PIL import Image, ImageOps

MAX_HISTORY = 200
THUMB_SIZE = 640


def make_entry(prediction, reply, image_path):
    """Everything the result screen needs, as plain data."""
    return {
        "id": int(time.time() * 1000),
        "time": time.time(),
        "image": str(image_path),
        "food": prediction.food_name or "food",
        "state": prediction.state,
        "reliable": bool(prediction.reliable),
        "safe": reply.safe,
        "text": reply.text,
        "verdict_line": reply.verdict_line,
        "food_conf": round(float(prediction.food_confidence), 4),
        "state_conf": round(float(prediction.state_confidence), 4),
    }


def time_ago(then, now=None):
    seconds = max(0, int((now if now is not None else time.time()) - then))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86400:
        hours = seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = seconds // 86400
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


def _read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return default
    except Exception:
        # A damaged file must never crash the app. Keep a copy, start fresh.
        try:
            os.replace(path, str(path) + ".broken")
        except OSError:
            pass
        return default


def _write_json(path, data):
    """Write to a temp file first so a crash cannot leave half a file."""
    temp = str(path) + ".tmp"
    with open(temp, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)
    os.replace(temp, path)


class Store:

    def __init__(self, folder):
        self.folder = Path(folder)
        self.scans = self.folder / "scans"
        self.scans.mkdir(parents=True, exist_ok=True)
        self.history_path = self.folder / "history.json"
        self.settings_path = self.folder / "settings.json"

    # ---------------- settings
    def get_setting(self, key, default=None):
        return _read_json(self.settings_path, {}).get(key, default)

    def set_setting(self, key, value):
        data = _read_json(self.settings_path, {})
        data[key] = value
        _write_json(self.settings_path, data)

    # ---------------- history
    def history(self):
        """Newest first."""
        items = _read_json(self.history_path, [])
        if not isinstance(items, list):
            return []
        return sorted(items, key=lambda e: e.get("time", 0), reverse=True)

    def add(self, entry):
        """Saves the scan + a small photo copy. Returns the saved entry."""
        entry = dict(entry)
        thumb = self.scans / f"{entry['id']}.jpg"
        try:
            with Image.open(entry["image"]) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.thumbnail((THUMB_SIZE, THUMB_SIZE))
                img.save(thumb, "JPEG", quality=85)
            entry["image"] = str(thumb)
        except Exception:
            pass  # keep the result even if the photo copy fails

        items = [e for e in self.history() if e.get("id") != entry["id"]]
        items.insert(0, entry)
        for old in items[MAX_HISTORY:]:
            self._remove_photo(old)
        _write_json(self.history_path, items[:MAX_HISTORY])
        return entry

    def clear(self):
        # Remove every thumbnail in our own folder, even orphaned ones
        # left behind if the history file was ever damaged.
        for photo in self.scans.glob("*.jpg"):
            try:
                photo.unlink()
            except OSError:
                pass
        _write_json(self.history_path, [])

    def _remove_photo(self, entry):
        try:
            path = Path(entry.get("image", ""))
            if path.parent == self.scans and path.is_file():
                path.unlink()
        except OSError:
            pass
