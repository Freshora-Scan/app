# ============================================================
# FRESHORA - FOOD FRESHNESS SCANNER (Kivy app: laptop + Android)
#
#     py -3.12 main.py
#
# Screens:  Welcome -> Home -> Camera -> Analysing -> Result
#           Home -> History -> Result
#
# On a laptop the "camera" button opens a file picker instead.
# The AI work (loading the model, scanning, the personality reply)
# runs in a background thread, so the window never freezes.
#
# Files used:  model/predict.py (AI)   model/personality.py (voice)
#              freshora_store.py (history)   greetings.py   theme.py
# ============================================================

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from kivy.animation import Animation
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.metrics import dp, sp
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import FadeTransition, Screen, ScreenManager
from kivy.uix.scrollview import ScrollView
from kivy.utils import escape_markup, platform

import theme
from freshora_store import Store, make_entry, time_ago
from greetings import greeting
from predict import FreshoraError, FreshoraPredictor  # model/predict.py
from theme import rgba

ANDROID = platform == "android"
PROJECT_NAME = "Freshora"

IMAGE_FILTERS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif",
                 "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.GIF"]

NOTICE = ("Important: Freshora only sees the outside of food. It can be "
          "wrong. Smell it, check the date, and trust your own senses.")

ANALYSING_LINES = [
    "Freshora is squinting at it...",
    "Consulting the fruit council...",
    "Interrogating the pixels...",
    "Sniffing the photo, spiritually...",
    "Asking the vegetables for comment...",
]


# ============================================================
# 1. SMALL BUILDING BLOCKS
# ============================================================

def paint(widget, hex_code):
    """Give a widget a solid background colour."""
    with widget.canvas.before:
        Color(*rgba(hex_code))
        rect = Rectangle(pos=widget.pos, size=widget.size)
    widget.bind(
        pos=lambda w, value: setattr(rect, "pos", value),
        size=lambda w, value: setattr(rect, "size", value),
    )


def wrap_label(**kwargs):
    """A Label that wraps to its width and grows to fit its text."""
    halign = kwargs.pop("halign", "left")
    color = kwargs.pop("color", rgba(theme.FOREST))
    label = Label(halign=halign, valign="top", color=color,
                  size_hint_y=None, **kwargs)
    label.bind(width=lambda lbl, w: setattr(lbl, "text_size", (w, None)))
    label.bind(texture_size=lambda lbl, size: setattr(lbl, "height", size[1]))
    return label


def make_image(**kwargs):
    """Image that fits inside its box. 'fit_mode' is new Kivy, the other old."""
    try:
        return KivyImage(fit_mode="contain", **kwargs)
    except TypeError:
        return KivyImage(allow_stretch=True, keep_ratio=True, **kwargs)


class Card(BoxLayout):
    """A rounded box."""

    def __init__(self, bg=theme.PAPER, radius=16, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(*rgba(bg))
            self._rect = RoundedRectangle(
                pos=self.pos, size=self.size, radius=[dp(radius)])
        self.bind(pos=self._sync, size=self._sync)

    def _sync(self, *args):
        self._rect.pos = self.pos
        self._rect.size = self.size


class TextCard(Card):
    """A rounded box that grows to fit the text inside it."""

    def __init__(self, bg=theme.PAPER, pad=14, **label_kwargs):
        super().__init__(bg=bg, orientation="vertical",
                         padding=dp(pad), size_hint_y=None)
        self._pad = dp(pad)
        self.label = wrap_label(markup=True, **label_kwargs)
        self.add_widget(self.label)
        self.label.bind(height=self._resize)
        self.height = self.label.height + 2 * self._pad

    def _resize(self, label, height):
        self.height = height + 2 * self._pad

    def set_text(self, text):
        self.label.text = text


class PillButton(Button):
    """A rounded, coloured button."""

    def __init__(self, bg=theme.MINT, fg=theme.FOREST, **kwargs):
        kwargs.setdefault("background_normal", "")
        kwargs.setdefault("background_down", "")
        kwargs.setdefault("background_color", (0, 0, 0, 0))
        kwargs.setdefault("font_size", sp(17))
        kwargs.setdefault("bold", True)
        super().__init__(color=rgba(fg), **kwargs)
        with self.canvas.before:
            self._colour = Color(*rgba(bg))
            self._rect = RoundedRectangle(
                pos=self.pos, size=self.size, radius=[dp(26)])
        self.bind(pos=self._sync, size=self._sync, disabled=self._dim)

    def _sync(self, *args):
        self._rect.pos = self.pos
        self._rect.size = self.size

    def _dim(self, button, disabled):
        self._colour.a = 0.35 if disabled else 1.0


class ScanRow(ButtonBehavior, Card):
    """One line in a list of scans: photo, food, state, time."""

    def __init__(self, entry, on_open, **kwargs):
        super().__init__(orientation="horizontal", padding=dp(8),
                         spacing=dp(12), size_hint_y=None, height=dp(72),
                         **kwargs)
        self.add_widget(make_image(
            source=entry.get("image", ""), size_hint_x=None, width=dp(56)))

        colour = theme.verdict_hex(entry.get("safe"))
        state = (entry.get("state") or "?").replace("_", " ").upper()
        if not entry.get("reliable", True):
            state = "NOT SURE"
        middle = Label(
            markup=True, halign="left", valign="middle",
            color=rgba(theme.FOREST),
            text=(f"[b][size={int(sp(16))}]"
                  f"{escape_markup(str(entry.get('food', 'food')).title())}"
                  f"[/size][/b]\n[color={colour}][b]{escape_markup(state)}"
                  f"[/b][/color]"),
        )
        middle.bind(size=lambda lbl, s: setattr(lbl, "text_size", s))
        self.add_widget(middle)

        right = Label(
            text=time_ago(entry.get("time", time.time())),
            font_size=sp(12), color=rgba(theme.MUTED),
            size_hint_x=None, width=dp(80), halign="right", valign="middle",
        )
        right.bind(size=lambda lbl, s: setattr(lbl, "text_size", s))
        self.add_widget(right)

        self.bind(on_release=lambda *_: on_open(entry))


# ============================================================
# 2. SCREENS
# ============================================================

class WelcomeScreen(Screen):

    def __init__(self, fa, **kwargs):
        super().__init__(name="welcome", **kwargs)
        root = BoxLayout(orientation="vertical", padding=dp(28),
                         spacing=dp(16))
        paint(root, theme.DEEP)

        root.add_widget(Label(size_hint_y=0.25))
        root.add_widget(Label(
            text="Freshora", bold=True, font_size=sp(48),
            color=rgba(theme.CREAM), size_hint_y=None, height=dp(70)))
        root.add_widget(Label(
            text="Scan food. Know better.\nMake safer choices.",
            font_size=sp(18), color=rgba(theme.MINT),
            halign="center", size_hint_y=None, height=dp(60)))
        root.add_widget(Label())

        start = PillButton(text="Get started", size_hint_y=None,
                           height=dp(56))
        start.bind(on_release=lambda *_: fa.finish_welcome())
        root.add_widget(start)

        note = wrap_label(
            text=("Freshora looks at the outside of food. It can be wrong "
                  "and is not a food-safety guarantee."),
            font_size=sp(12), halign="center", color=rgba(theme.MUTED))
        root.add_widget(note)
        self.add_widget(root)


class HomeScreen(Screen):

    def __init__(self, fa, **kwargs):
        super().__init__(name="home", **kwargs)
        self.fa = fa
        root = BoxLayout(orientation="vertical", padding=dp(20),
                         spacing=dp(12))
        paint(root, theme.CREAM)

        self.title = wrap_label(text="", bold=True, font_size=sp(26))
        root.add_widget(self.title)
        self.subtitle = wrap_label(
            text="", font_size=sp(15), color=rgba(theme.MUTED))
        root.add_widget(self.subtitle)

        # The big scan card
        card = Card(bg=theme.FOREST, orientation="vertical",
                    padding=dp(16), spacing=dp(8),
                    size_hint_y=None, height=dp(150))
        card.add_widget(Label(
            text="Scan Food", bold=True, font_size=sp(24),
            color=rgba(theme.CREAM), halign="left"))
        card.add_widget(Label(
            text="Take a photo to check freshness.",
            font_size=sp(14), color=rgba(theme.MINT)))
        self.scan_button = PillButton(
            text="Open camera", size_hint_y=None, height=dp(48))
        self.scan_button.bind(on_release=lambda *_: fa.open_camera())
        card.add_widget(self.scan_button)
        root.add_widget(card)

        # Gallery + History
        row = BoxLayout(size_hint_y=None, height=dp(52), spacing=dp(12))
        self.gallery_button = PillButton(
            text="Gallery", bg=theme.PAPER, fg=theme.FOREST)
        self.gallery_button.bind(on_release=lambda *_: fa.open_gallery())
        history = PillButton(
            text="History", bg=theme.PAPER, fg=theme.FOREST)
        history.bind(on_release=lambda *_: fa.show_history())
        row.add_widget(self.gallery_button)
        row.add_widget(history)
        root.add_widget(row)

        self.status = wrap_label(
            text="Loading AI model...", font_size=sp(13),
            color=rgba(theme.MUTED))
        root.add_widget(self.status)

        root.add_widget(wrap_label(
            text="Recent scans", bold=True, font_size=sp(18)))

        scroll = ScrollView(do_scroll_x=False)
        self.recent = GridLayout(cols=1, spacing=dp(8), size_hint_y=None)
        self.recent.bind(minimum_height=self.recent.setter("height"))
        scroll.add_widget(self.recent)
        root.add_widget(scroll)

        root.add_widget(wrap_label(
            text="AI predictions are not a guarantee of food safety.",
            font_size=sp(11), halign="center", color=rgba(theme.MUTED)))
        self.add_widget(root)

        self.set_ready(False)

    def on_pre_enter(self, *args):
        self.refresh()

    def refresh(self):
        title, line = greeting()
        self.title.text = title
        self.subtitle.text = line

        self.recent.clear_widgets()
        entries = self.fa.store.history()[:4]
        if not entries:
            self.recent.add_widget(wrap_label(
                text="Nothing scanned yet. Feed me a banana.",
                font_size=sp(14), color=rgba(theme.MUTED)))
        for entry in entries:
            self.recent.add_widget(
                ScanRow(entry, on_open=self.fa.open_saved_entry))

    def set_ready(self, ready, message=None):
        self.scan_button.disabled = not ready
        self.gallery_button.disabled = not ready
        if message is not None:
            self.status.text = message
        elif ready:
            self.status.text = "Ready."


class CameraScreen(Screen):
    """Live camera (Android only, uses Camera4Kivy)."""

    def __init__(self, fa, **kwargs):
        super().__init__(name="camera", **kwargs)
        self.fa = fa
        self.preview = None
        self._connected = False

        root = BoxLayout(orientation="vertical", padding=dp(8),
                         spacing=dp(8))
        paint(root, theme.DEEP)

        bar = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        back = PillButton(text="Back", bg=theme.FOREST, fg=theme.CREAM)
        back.bind(on_release=lambda *_: fa.go("home"))
        self.flash_button = PillButton(
            text="Flash", bg=theme.FOREST, fg=theme.CREAM)
        self.flash_button.bind(on_release=lambda *_: self._flash())
        gallery = PillButton(text="Gallery", bg=theme.FOREST, fg=theme.CREAM)
        gallery.bind(on_release=lambda *_: fa.open_gallery())
        for widget in (back, self.flash_button, gallery):
            bar.add_widget(widget)
        root.add_widget(bar)

        self.slot = BoxLayout()          # the camera picture goes in here
        root.add_widget(self.slot)

        self.hint = wrap_label(
            text="Put the whole food in the frame, in good light.",
            font_size=sp(14), halign="center", color=rgba(theme.CREAM))
        root.add_widget(self.hint)

        self.shutter = PillButton(text="Take photo", size_hint_y=None,
                                  height=dp(56))
        self.shutter.bind(on_release=lambda *_: self._take_photo())
        root.add_widget(self.shutter)
        self.add_widget(root)

    # ---- screen events
    def on_enter(self, *args):
        self.fa.ask_camera_permission(self._permission_done)

    def on_leave(self, *args):
        self._disconnect()

    # ---- camera plumbing
    def _permission_done(self, granted):
        if not granted:
            self.hint.text = ("Camera permission was refused. Allow it in "
                              "Android Settings, or use the Gallery button.")
            self.shutter.disabled = True
            return
        # Camera4Kivy asks to connect a moment AFTER the screen shows.
        Clock.schedule_once(lambda dt: self._connect(), 0.3)

    def _connect(self):
        try:
            if self.preview is None:
                from camera4kivy import Preview
                self.preview = Preview()
                self.slot.add_widget(self.preview)
            self.preview.connect_camera(
                filepath_callback=self._photo_saved,
                enable_video=False,
                enable_analyze_pixels=False,
            )
            self._connected = True
            self.shutter.disabled = False
        except Exception as error:
            self.hint.text = ("The camera could not start. Use the Gallery "
                              f"button instead.\n({error})")
            self.shutter.disabled = True

    def _disconnect(self):
        if self.preview is not None and self._connected:
            try:
                self.preview.disconnect_camera()
            except Exception:
                pass
        self._connected = False

    def _flash(self):
        try:
            self.preview.flash()
        except Exception:
            pass

    def _take_photo(self):
        if self.preview is None or not self._connected:
            return
        self.shutter.disabled = True
        # If the camera never answers, let the user try again.
        Clock.schedule_once(self._unlock_shutter, 5)
        try:
            self.preview.capture_photo(
                location="private", subdir="shots",
                name=f"shot_{int(time.time() * 1000)}")
        except Exception as error:
            self.hint.text = f"Could not take the photo: {error}"
            self._unlock_shutter()

    def _unlock_shutter(self, *args):
        if self._connected:
            self.shutter.disabled = False

    def _photo_saved(self, path):
        # May be called from a camera thread -> hop to the main thread.
        Clock.schedule_once(lambda dt: self.fa.start_scan(path))


class AnalysingScreen(Screen):

    def __init__(self, fa, **kwargs):
        super().__init__(name="analysing", **kwargs)
        root = BoxLayout(orientation="vertical", padding=dp(28),
                         spacing=dp(12))
        paint(root, theme.DEEP)
        root.add_widget(Label())
        self.line = Label(
            text="", bold=True, font_size=sp(22), halign="center",
            color=rgba(theme.CREAM), size_hint_y=None, height=dp(90))
        self.line.bind(
            width=lambda lbl, w: setattr(lbl, "text_size", (w, None)))
        root.add_widget(self.line)
        root.add_widget(wrap_label(
            text="Working out which food it is, then how fresh it looks.",
            font_size=sp(14), halign="center", color=rgba(theme.MINT)))
        root.add_widget(Label())
        self.add_widget(root)

    def on_pre_enter(self, *args):
        import random
        self.line.text = random.choice(ANALYSING_LINES)

    def on_enter(self, *args):
        pulse = (Animation(opacity=0.35, duration=0.7)
                 + Animation(opacity=1.0, duration=0.7))
        pulse.repeat = True
        pulse.start(self.line)

    def on_leave(self, *args):
        Animation.cancel_all(self.line)
        self.line.opacity = 1.0


class ResultScreen(Screen):

    def __init__(self, fa, **kwargs):
        super().__init__(name="result", **kwargs)
        self.fa = fa
        root = BoxLayout(orientation="vertical")
        paint(root, theme.CREAM)

        scroll = ScrollView(do_scroll_x=False)
        self.content = GridLayout(
            cols=1, spacing=dp(12), padding=dp(16), size_hint_y=None)
        self.content.bind(minimum_height=self.content.setter("height"))
        scroll.add_widget(self.content)
        root.add_widget(scroll)

        self.picture = make_image(size_hint_y=None, height=dp(240))
        self.header = TextCard(font_size=sp(16))
        self.numbers = TextCard(font_size=sp(13), color=rgba(theme.MUTED))
        self.voice = TextCard(font_size=sp(17))
        self.verdict = TextCard(font_size=sp(16))
        self.notice = TextCard(bg="FFF3D6", font_size=sp(13))
        for widget in (self.picture, self.header, self.numbers, self.voice,
                       self.verdict, self.notice):
            self.content.add_widget(widget)

        buttons = BoxLayout(size_hint_y=None, height=dp(64),
                            padding=dp(8), spacing=dp(12))
        self.again_button = PillButton(
            text="Scan again", bg=theme.PAPER, fg=theme.FOREST)
        self.again_button.bind(on_release=lambda *_: fa.open_camera())
        self.save_button = PillButton(text="Save result")
        self.save_button.bind(on_release=lambda *_: fa.save_current())
        home = PillButton(text="Home", bg=theme.PAPER, fg=theme.FOREST)
        home.bind(on_release=lambda *_: fa.go("home"))
        for widget in (home, self.again_button, self.save_button):
            buttons.add_widget(widget)
        root.add_widget(buttons)
        self.add_widget(root)

    def show(self, entry, saved):
        colour = theme.verdict_hex(entry.get("safe"))
        food = escape_markup(str(entry.get("food", "food")).title())
        state = escape_markup(
            (entry.get("state") or "?").replace("_", " ").upper())

        self.picture.source = entry.get("image", "")
        try:
            self.picture.reload()
        except Exception:
            pass

        if entry.get("reliable", True):
            chip = f"[color={colour}][b]{state}[/b][/color]"
            food_line = f"[b][size={int(sp(26))}]{food}[/size][/b]"
        else:
            chip = f"[color={theme.verdict_hex('unknown')}][b]NOT SURE[/b][/color]"
            food_line = (f"[b][size={int(sp(22))}]Best guess: "
                         f"{food}[/size][/b]")
        self.header.set_text(f"{food_line}\n{chip}")

        self.numbers.set_text(
            f"{entry.get('food_conf', 0) * 100:.0f}% sure about the food  |  "
            f"{entry.get('state_conf', 0) * 100:.0f}% sure about the freshness"
            "\n(How sure the AI is. This is not the chance that it is safe.)")

        self.voice.set_text(escape_markup(entry.get("text", "")))
        self.verdict.set_text(
            f"[b][color={colour}]"
            f"{escape_markup(entry.get('verdict_line', ''))}[/color][/b]")
        self.notice.set_text(escape_markup(NOTICE))
        self.mark_saved(saved)

    def mark_saved(self, saved):
        self.save_button.text = "Saved" if saved else "Save result"
        self.save_button.disabled = saved


class HistoryScreen(Screen):

    def __init__(self, fa, **kwargs):
        super().__init__(name="history", **kwargs)
        self.fa = fa
        root = BoxLayout(orientation="vertical", padding=dp(16),
                         spacing=dp(10))
        paint(root, theme.CREAM)

        bar = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        back = PillButton(text="Back", bg=theme.PAPER, fg=theme.FOREST,
                          size_hint_x=None, width=dp(90))
        back.bind(on_release=lambda *_: fa.go("home"))
        bar.add_widget(back)
        bar.add_widget(Label(
            text="Scan history", bold=True, font_size=sp(20),
            color=rgba(theme.FOREST)))
        clear = PillButton(text="Clear all", bg=theme.PAPER, fg="E53935",
                           size_hint_x=None, width=dp(100))
        clear.bind(on_release=lambda *_: self._confirm_clear())
        bar.add_widget(clear)
        root.add_widget(bar)

        scroll = ScrollView(do_scroll_x=False)
        self.list = GridLayout(cols=1, spacing=dp(8), size_hint_y=None)
        self.list.bind(minimum_height=self.list.setter("height"))
        scroll.add_widget(self.list)
        root.add_widget(scroll)

        root.add_widget(wrap_label(
            text="Saved only on this phone. Nothing is sent anywhere.",
            font_size=sp(11), halign="center", color=rgba(theme.MUTED)))
        self.add_widget(root)

    def on_pre_enter(self, *args):
        self.refresh()

    def refresh(self):
        self.list.clear_widgets()
        entries = self.fa.store.history()
        if not entries:
            self.list.add_widget(wrap_label(
                text="No saved scans yet.", font_size=sp(15),
                color=rgba(theme.MUTED)))
        for entry in entries:
            self.list.add_widget(
                ScanRow(entry, on_open=self.fa.open_saved_entry))

    def _confirm_clear(self):
        box = BoxLayout(orientation="vertical", padding=dp(12),
                        spacing=dp(12))
        box.add_widget(Label(text="Delete ALL saved scans from this phone?"))
        row = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(12))
        keep = Button(text="Keep them")
        delete = Button(text="Delete all")
        row.add_widget(keep)
        row.add_widget(delete)
        box.add_widget(row)
        popup = Popup(title="Clear history", content=box,
                      size_hint=(0.85, None), height=dp(210))

        def do_delete(*_):
            self.fa.store.clear()
            popup.dismiss()
            self.refresh()

        keep.bind(on_release=popup.dismiss)
        delete.bind(on_release=do_delete)
        popup.open()


# ============================================================
# 3. THE APP
# ============================================================

class FreshoraApp(App):

    def build(self):
        self.title = PROJECT_NAME
        Window.clearcolor = rgba(theme.CREAM)
        if not ANDROID:
            Window.size = (400, 780)      # phone-shaped window on a laptop
            Window.minimum_width, Window.minimum_height = 360, 600

        self.predictor = FreshoraPredictor()
        self.store = Store(self.user_data_dir)
        self.model_error = ""
        self.busy = False
        self.current_entry = None
        self.saved = False
        self.result_origin = "scan"
        self._chooser = None

        self.sm = ScreenManager(transition=FadeTransition(duration=0.12))
        self.home = HomeScreen(self)
        self.result = ResultScreen(self)
        self.analysing = AnalysingScreen(self)
        self.history = HistoryScreen(self)
        self.sm.add_widget(WelcomeScreen(self))
        self.sm.add_widget(self.home)
        self.sm.add_widget(self.analysing)
        self.sm.add_widget(self.result)
        self.sm.add_widget(self.history)
        if ANDROID:
            self.sm.add_widget(CameraScreen(self))

        self.sm.current = ("home" if self.store.get_setting("welcome_seen")
                           else "welcome")

        Window.bind(on_keyboard=self._on_key)
        threading.Thread(target=self._load_model, daemon=True).start()
        return self.sm

    def on_stop(self):
        pass

    # --------------------------------------------------------
    # Navigation
    # --------------------------------------------------------
    def go(self, name):
        self.sm.current = name

    def finish_welcome(self):
        try:
            self.store.set_setting("welcome_seen", True)
        except Exception:
            pass
        self.go("home")

    def show_history(self):
        self.go("history")

    def _on_key(self, window, key, *args):
        """Android back button (and Esc on a laptop)."""
        if key != 27:
            return False
        current = self.sm.current
        if current in ("home", "welcome"):
            return False                  # let the system close the app
        if current == "analysing":
            return True                   # busy, ignore
        if current == "result" and self.result_origin == "history":
            self.go("history")
        else:
            self.go("home")
        return True

    def show_message(self, title, text):
        box = BoxLayout(orientation="vertical", padding=dp(12),
                        spacing=dp(12))
        label = Label(text=text, halign="center", valign="middle")
        label.bind(size=lambda lbl, s: setattr(lbl, "text_size", s))
        box.add_widget(label)
        ok = Button(text="OK", size_hint_y=None, height=dp(48))
        box.add_widget(ok)
        popup = Popup(title=title, content=box, size_hint=(0.9, 0.5))
        ok.bind(on_release=popup.dismiss)
        popup.open()

    # --------------------------------------------------------
    # Model loading (background)
    # --------------------------------------------------------
    def _load_model(self):
        try:
            self.predictor.load()
        except FreshoraError as error:
            message = str(error)
        except Exception as error:   # never let a thread die silently
            message = repr(error)
        else:
            Clock.schedule_once(lambda dt: self.home.set_ready(True))
            return
        self.model_error = message
        Clock.schedule_once(lambda dt: self.home.set_ready(
            False, "AI model not available. Tap Gallery or Camera for details."))
        # Buttons stay usable so the user can read the real reason.
        Clock.schedule_once(lambda dt: self._enable_for_error())

    def _enable_for_error(self):
        self.home.scan_button.disabled = False
        self.home.gallery_button.disabled = False

    # --------------------------------------------------------
    # Getting a photo: camera, gallery, laptop file picker
    # --------------------------------------------------------
    def ask_camera_permission(self, callback):
        """Asks for the camera ONLY when the user opens the camera."""
        if not ANDROID:
            callback(True)
            return
        try:
            from android.permissions import (
                Permission, check_permission, request_permissions)
            if check_permission(Permission.CAMERA):
                callback(True)
                return

            def answered(permissions, grants):
                allowed = bool(grants) and all(grants)
                Clock.schedule_once(lambda dt: callback(allowed))

            request_permissions([Permission.CAMERA], answered)
        except Exception:
            callback(False)

    def open_camera(self):
        if self._model_unavailable():
            return
        if ANDROID:
            self.go("camera")
        else:
            self._desktop_picker()

    def open_gallery(self):
        if self._model_unavailable():
            return
        if not ANDROID:
            self._desktop_picker()
            return
        try:
            from androidstorage4kivy import Chooser
            if self._chooser is None:
                self._chooser = Chooser(self._gallery_chosen)
            self._chooser.choose_content("image/*")
        except Exception as error:
            self.show_message("Gallery not available", str(error))

    def _gallery_chosen(self, shared_files):
        # Called by Android -> hop to the main thread.
        Clock.schedule_once(lambda dt: self._copy_chosen(shared_files))

    def _copy_chosen(self, shared_files):
        if not shared_files:
            return
        try:
            from androidstorage4kivy import SharedStorage
            path = SharedStorage().copy_from_shared(shared_files[0])
        except Exception as error:
            self.show_message("Could not open that photo", str(error))
            return
        if not path:
            self.show_message("Could not open that photo",
                              "Android did not give access to it. "
                              "Try another photo.")
            return
        self.start_scan(path)

    def _desktop_picker(self):
        start = ROOT / "data"
        chooser = FileChooserListView(
            path=str(start if start.is_dir() else Path.home()),
            filters=IMAGE_FILTERS,
        )
        content = BoxLayout(orientation="vertical")
        content.add_widget(chooser)
        choose = Button(text="Use Selected Image", size_hint_y=None,
                        height=dp(50))
        content.add_widget(choose)
        popup = Popup(title="Choose Food Image", content=content,
                      size_hint=(0.95, 0.9))

        def select_image(_button):
            if not chooser.selection:
                return
            path = chooser.selection[0]
            popup.dismiss()
            self.start_scan(path)

        choose.bind(on_release=select_image)
        popup.open()

    def _model_unavailable(self):
        if self.model_error:
            self.show_message("AI model not available", self.model_error)
            return True
        return False

    # --------------------------------------------------------
    # Scanning (background thread)
    # --------------------------------------------------------
    def start_scan(self, path):
        if self.busy:
            return
        self.busy = True
        self.go("analysing")
        threading.Thread(
            target=self._scan_worker, args=(str(path),), daemon=True
        ).start()

    def _scan_worker(self, path):
        try:
            # No OpenAI call on the phone: its key can never be hidden there.
            prediction, reply = self.predictor.scan(path, use_ai=not ANDROID)
            entry = make_entry(prediction, reply, path)
        except FreshoraError as error:
            message = str(error)
            Clock.schedule_once(lambda dt: self._scan_failed(message))
            return
        except Exception as error:
            message = repr(error)
            Clock.schedule_once(lambda dt: self._scan_failed(message))
            return
        Clock.schedule_once(lambda dt: self._scan_done(entry))

    def _scan_done(self, entry):
        self.busy = False
        self.current_entry = entry
        self.saved = False
        self.result_origin = "scan"
        self.result.show(entry, saved=False)
        self.go("result")

    def _scan_failed(self, message):
        self.busy = False
        self.go("home")
        self.show_message("The scan failed", message)

    # --------------------------------------------------------
    # Saving / history
    # --------------------------------------------------------
    def save_current(self):
        if self.current_entry is None or self.saved:
            return
        try:
            self.current_entry = self.store.add(self.current_entry)
            self.saved = True
            self.result.mark_saved(True)
        except Exception as error:
            self.show_message("Could not save", str(error))

    def open_saved_entry(self, entry):
        self.current_entry = entry
        self.saved = True
        self.result_origin = "history"
        self.result.show(entry, saved=True)
        self.go("result")


if __name__ == "__main__":
    FreshoraApp().run()
