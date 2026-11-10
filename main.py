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
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from kivy.animation import Animation
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Ellipse, Line, Rectangle, RoundedRectangle
from kivy.metrics import dp, sp
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.colorpicker import ColorPicker
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import FadeTransition, Screen, ScreenManager
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.utils import escape_markup, platform

import theme
from freshora_store import Store, make_entry, time_ago
from greetings import greeting
from predict import FreshoraError, FreshoraPredictor  # model/predict.py
from theme import rgba

ANDROID = platform == "android"
PROJECT_NAME = "Freshora"
RATE_URL = "https://docs.google.com/forms/d/e/1FAIpQLSf30fuYnB4r_b0i4iOltV9sqi7JLtBEXWNi03sg4pLgAgReog/viewform?usp=publish-editor"
FEEDBACK_URL = "https://docs.google.com/forms/d/e/1FAIpQLSfyEGHy_rEjHoDD9WglpxmDN5xjSIJPy9acgOR5EDS1xTc4SQ/viewform?usp=header"

IMAGE_FILTERS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.webp",
                 "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.GIF", "*.WEBP"]

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


def entry_is_reliable(entry):
    """One consistent display gate for new and previously saved predictions."""
    try:
        return (
            bool(entry.get("reliable", True))
            and float(entry.get("food_conf", 0.0)) >= 0.80
            and float(entry.get("state_conf", 0.0)) >= 0.80
        )
    except (TypeError, ValueError):
        return False


class Card(BoxLayout):
    """A rounded box."""

    def __init__(self, bg=None, radius=16, **kwargs):
        bg = bg or theme.PAPER
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

    def __init__(self, bg=None, pad=14, **label_kwargs):
        bg = bg or theme.PAPER
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

    def __init__(self, bg=None, fg=None, **kwargs):
        bg = bg or theme.MINT
        fg = fg or theme.FOREST
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


class MetricTile(Card):
    """A small dashboard metric that follows the existing Freshora theme."""

    def __init__(self, title, value="0", **kwargs):
        super().__init__(bg=theme.PAPER, orientation="vertical",
                         padding=dp(10), spacing=dp(3), **kwargs)
        self.value_label = Label(
            text=str(value), bold=True, font_size=sp(20),
            color=rgba(theme.FOREST), halign="center", valign="middle")
        self.value_label.bind(
            size=lambda lbl, size: setattr(lbl, "text_size", size))
        self.title_label = Label(
            text=title, font_size=sp(11), color=rgba(theme.MUTED),
            halign="center", valign="middle")
        self.title_label.bind(
            size=lambda lbl, size: setattr(lbl, "text_size", size))
        self.add_widget(self.value_label)
        self.add_widget(self.title_label)


class NavItem(ButtonBehavior, BoxLayout):
    """Centered bottom-navigation icon and label."""

    def __init__(self, label, destination, icon_file, on_press,
                 active=False, **kwargs):
        super().__init__(orientation="vertical", spacing=dp(1),
                         padding=(dp(2), dp(3)), size_hint_y=None,
                         height=dp(70), **kwargs)
        background = theme.MINT if active else theme.PAPER
        foreground = theme.FOREST
        with self.canvas.before:
            self._nav_colour = Color(*rgba(background))
            self._nav_rect = RoundedRectangle(
                pos=self.pos, size=self.size, radius=[dp(12)])
        self.bind(
            pos=lambda widget, value: setattr(self._nav_rect, "pos", value),
            size=lambda widget, value: setattr(self._nav_rect, "size", value),
        )

        # The original artwork had uneven transparent margins. Dedicated,
        # tightly-cropped nav assets plus an AnchorLayout make the visible
        # artwork consistently sized and horizontally centered.
        icon_holder = AnchorLayout(
            anchor_x="center", anchor_y="center",
            size_hint=(1, None), height=dp(40))
        self.icon = make_image(
            source=str(ROOT / "assets" / icon_file),
            size_hint=(None, None), size=(dp(34), dp(34)),
        )
        icon_holder.add_widget(self.icon)
        self.add_widget(icon_holder)

        self.caption = Label(
            text=label, font_size=sp(10), bold=active,
            color=rgba(foreground), halign="center", valign="middle",
            size_hint=(1, None), height=dp(18),
        )
        self.caption.bind(
            size=lambda widget, value: setattr(widget, "text_size", value))
        self.add_widget(self.caption)
        self.bind(on_release=lambda *_: on_press(destination))


class BottomNav(BoxLayout):
    """Four-icon navigation: Home, Scan, History and Settings."""

    def __init__(self, fa, active="home", **kwargs):
        super().__init__(orientation="horizontal", spacing=dp(5),
                         size_hint_y=None, height=dp(76), **kwargs)
        self.padding = (dp(3), dp(3))
        for label, destination, icon_file in (
            ("Home", "home", "nav_home.png"),
            ("Scan", "scan", "nav_scan.png"),
            ("History", "history", "nav_history.png"),
            ("Settings", "settings", "nav_settings.png"),
        ):
            self.add_widget(NavItem(
                label, destination, icon_file, fa.navigate_tab,
                active=(destination == active), size_hint_x=1,
            ))


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
        if not entry_is_reliable(entry):
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
        root = BoxLayout(orientation="vertical", padding=dp(26),
                         spacing=dp(12))
        paint(root, theme.DEEP)

        root.add_widget(Label(size_hint_y=0.12))
        root.add_widget(make_image(
            source=str(ROOT / "assets" / "icon.png"),
            size_hint_y=0.34,
        ))
        root.add_widget(Label(
            text="Freshora", bold=True, font_size=sp(40),
            color=rgba(theme.CREAM), size_hint_y=None, height=dp(58)))
        root.add_widget(Label(
            text="Scan food. Know better.\nMake safer choices.",
            font_size=sp(17), color=rgba(theme.MINT),
            halign="center", size_hint_y=None, height=dp(58)))

        start = PillButton(text="Get started", size_hint_y=None,
                           height=dp(54))
        start.bind(on_release=lambda *_: fa.go("name"))
        root.add_widget(start)

        note = wrap_label(
            text=("Use Freshora only to scan fruits and vegetables. "
                  "Do not use it to scan people, animals, objects, or "
                  "anything other than food. AI predictions can be wrong "
                  "and are not a food-safety guarantee."),
            font_size=sp(12), halign="center", color=rgba(theme.CREAM))
        root.add_widget(note)
        self.add_widget(root)


class NameScreen(Screen):
    """First-run nickname entry."""

    def __init__(self, fa, **kwargs):
        super().__init__(name="name", **kwargs)
        self.fa = fa
        root = BoxLayout(orientation="vertical", padding=dp(24),
                         spacing=dp(16))
        paint(root, theme.CREAM)
        root.add_widget(Label(size_hint_y=0.18))
        root.add_widget(wrap_label(
            text="Enter your name/nickname", bold=True,
            font_size=sp(25), halign="center", color=rgba(theme.FOREST)))
        root.add_widget(wrap_label(
            text=("We'll use it in your Freshora greetings, such as "
                  "'Good morning, Alex'. You can change it later in Settings."),
            font_size=sp(14), halign="center", color=rgba(theme.MUTED)))
        self.name_input = TextInput(
            hint_text="Name or nickname", multiline=False,
            size_hint_y=None, height=dp(54),
            padding=(dp(12), dp(14)), font_size=sp(18),
            foreground_color=rgba(theme.FOREST),
            cursor_color=rgba(theme.FOREST),
            background_normal="", background_active="",
            background_color=rgba(theme.PAPER),
        )
        self.name_input.bind(on_text_validate=lambda *_: self.continue_setup())
        root.add_widget(self.name_input)
        button = PillButton(text="Continue", size_hint_y=None, height=dp(54))
        button.bind(on_release=lambda *_: self.continue_setup())
        root.add_widget(button)
        root.add_widget(Label())
        self.add_widget(root)

    def on_pre_enter(self, *args):
        self.name_input.text = self.fa.nickname()
        self.name_input.focus = True

    def continue_setup(self):
        name = self.name_input.text.strip()
        if not name:
            self.fa.show_message("Name needed", "Enter a name or nickname to continue.")
            return
        self.fa.save_onboarding_name(name)


class TutorialScreen(Screen):
    """Eight short, image-led steps using the screenshots in assets/."""

    def __init__(self, fa, **kwargs):
        super().__init__(name="tutorial", **kwargs)
        self.fa = fa
        self.index = 0
        self.slides = [
            ("icon.png",
             "Welcome! Freshora is an AI model that tries to guess whether "
             "a fruit or vegetable looks good or bad. Expect it to be a bit "
             "dumb sometimes: it can make mistakes. This is the Freshora logo."),
            ("tutorial1.png",
             "This is your Home page. The buttons do what they say. To scan, "
             "press Open camera or Add photo."),
            ("tutorial2.png",
             "Allow the camera or photo permission when your phone asks. "
             "Without the permission, that feature may not load."),
            ("tutorial3.png",
             "Choose a photo or take a picture of a fruit or vegetable. "
             "Scan only one type of food at a time, and keep it clearly in frame."),
            ("tutorial4.png",
             "There you go: an AI guess! It may give a sloppy response. "
             "Check the confidence and press Save result to keep the data."),
            ("tutorial5.png",
             "Want to see earlier scans? Press History. Your saved scan "
             "records stay on this device."),
            ("SCAN.png",
             "You can also open the scanner at any time by pressing the "
             "Scan icon in the bottom navigation bar."),
            ("Settings.png",
             "Settings lets you choose a theme, change your name, rate "
             "Freshora, or report a bug. And that's it, that's how you "
             '"scan fruits"! Happy scanning!'),
        ]

        root = BoxLayout(orientation="vertical", padding=dp(14),
                         spacing=dp(8))
        paint(root, theme.CREAM)
        self.counter = Label(
            text="Tutorial 1 of 8", bold=True, font_size=sp(17),
            color=rgba(theme.FOREST), size_hint_y=None, height=dp(28))
        root.add_widget(self.counter)
        self.picture = make_image(size_hint_y=0.57)
        root.add_widget(self.picture)
        self.copy = wrap_label(
            text="", font_size=sp(15), color=rgba(theme.FOREST),
            halign="center")
        root.add_widget(self.copy)

        controls = BoxLayout(size_hint_y=None, height=dp(52), spacing=dp(10))
        self.back_button = PillButton(
            text="Back", bg=theme.PAPER, fg=theme.FOREST)
        self.back_button.bind(on_release=lambda *_: self.previous_slide())
        self.next_button = PillButton(text="Next")
        self.next_button.bind(on_release=lambda *_: self.next_slide())
        controls.add_widget(self.back_button)
        controls.add_widget(self.next_button)
        root.add_widget(controls)
        self.add_widget(root)
        self._render_slide()

    def _render_slide(self):
        filename, copy = self.slides[self.index]
        self.counter.text = f"Tutorial {self.index + 1} of {len(self.slides)}"
        self.picture.source = str(ROOT / "assets" / filename)
        try:
            self.picture.reload()
        except Exception:
            pass
        self.copy.text = copy
        self.back_button.disabled = self.index == 0
        self.next_button.text = (
            "Get scanning" if self.index == len(self.slides) - 1 else "Next")

    def previous_slide(self):
        if self.index > 0:
            self.index -= 1
            self._render_slide()

    def next_slide(self):
        if self.index < len(self.slides) - 1:
            self.index += 1
            self._render_slide()
        else:
            self.fa.finish_tutorial()

    def on_pre_enter(self, *args):
        self.index = 0
        self._render_slide()


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

        metrics = GridLayout(cols=2, spacing=dp(10),
                             size_hint_y=None, height=dp(78))
        self.saved_metric = MetricTile(
            "Saved scans", "0", size_hint_y=None, height=dp(74))
        self.latest_metric = MetricTile(
            "Latest food", "—", size_hint_y=None, height=dp(74))
        metrics.add_widget(self.saved_metric)
        metrics.add_widget(self.latest_metric)
        root.add_widget(metrics)

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
            text="Add photo", bg=theme.PAPER, fg=theme.FOREST)
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
        root.add_widget(BottomNav(fa, active="home"))
        self.add_widget(root)

        self.set_ready(False)

    def on_pre_enter(self, *args):
        self.refresh()

    def refresh(self):
        title, line = greeting()
        nickname = self.fa.nickname()
        self.title.text = f"{title}, {nickname}" if nickname else title
        self.subtitle.text = line

        self.recent.clear_widgets()
        all_entries = self.fa.store.history()
        entries = all_entries[:4]
        self.saved_metric.value_label.text = str(len(all_entries))
        if all_entries:
            latest = all_entries[0]
            self.latest_metric.value_label.text = (
                str(latest.get("food", "—")).title()
                if entry_is_reliable(latest) else "Not sure")
        else:
            self.latest_metric.value_label.text = "—"
        if not entries:
            self.recent.add_widget(wrap_label(
                text="Nothing saved yet. Add a food photo to begin.",
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
        if self.fa.sm.current != self.name:
            return
        if not granted:
            self.hint.text = ("Camera permission was refused. Allow it in "
                              "Android Settings, or use the Gallery button.")
            self.shutter.disabled = True
            return
        # Camera4Kivy must connect after the app is visible and permission
        # has been granted. The short delay also lets Android lay out Preview.
        Clock.schedule_once(lambda dt: self._connect(), 0.3)

    def _connect(self):
        if self.fa.sm.current != self.name or self._connected:
            return
        try:
            if self.preview is None:
                from camera4kivy import Preview
                self.preview = Preview(aspect_ratio="4:3")
                self.slot.add_widget(self.preview)
            self.preview.connect_camera(
                camera_id="back",
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
        if path:
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
        self.batch_label = wrap_label(
            text="", font_size=sp(12), halign="center",
            color=rgba(theme.MUTED))
        self.header = TextCard(font_size=sp(16))
        self.numbers = TextCard(font_size=sp(13), color=rgba(theme.MUTED))
        self.voice = TextCard(font_size=sp(17))
        self.verdict = TextCard(font_size=sp(16))
        self.notice = TextCard(bg="FFF3D6", font_size=sp(13))
        for widget in (self.batch_label, self.picture, self.header,
                       self.numbers, self.voice,
                       self.verdict, self.notice):
            self.content.add_widget(widget)

        buttons = BoxLayout(size_hint_y=None, height=dp(64),
                            padding=dp(8), spacing=dp(12))
        self.again_button = PillButton(
            text="Scan again", bg=theme.PAPER, fg=theme.FOREST)
        self.again_button.bind(on_release=lambda *_: fa.result_primary_action())
        self.save_button = PillButton(text="Save result")
        self.save_button.bind(on_release=lambda *_: fa.save_current())
        home = PillButton(text="Home", bg=theme.PAPER, fg=theme.FOREST)
        home.bind(on_release=lambda *_: fa.go("home"))
        for widget in (home, self.again_button, self.save_button):
            buttons.add_widget(widget)
        root.add_widget(buttons)
        self.add_widget(root)

    def show(self, entry, saved, batch_progress=None):
        colour = theme.verdict_hex(entry.get("safe"))
        food = escape_markup(str(entry.get("food", "food")).title())
        state = escape_markup(
            (entry.get("state") or "?").replace("_", " ").upper())

        self.picture.source = entry.get("image", "")
        try:
            self.picture.reload()
        except Exception:
            pass

        self.batch_label.text = batch_progress or ""
        self.again_button.text = (
            "Next photo" if batch_progress and
            self.fa.has_next_photo() else "Scan again")

        if entry_is_reliable(entry):
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


class ScanFailedScreen(Screen):
    """Blocks uncertain model output and clearly shows both confidence scores."""

    def __init__(self, fa, **kwargs):
        super().__init__(name="scan_failed", **kwargs)
        self.fa = fa
        root = BoxLayout(orientation="vertical", padding=dp(22),
                         spacing=dp(14))
        paint(root, theme.CREAM)
        root.add_widget(Label(size_hint_y=0.12))
        root.add_widget(Label(
            text="Scan failed", font_size=sp(30), bold=True,
            color=rgba("B42318"), size_hint_y=None, height=dp(48)))
        root.add_widget(wrap_label(
            text=("Freshora is not sure enough about this image, so it "
                  "has blocked the prediction instead of showing a guess."),
            font_size=sp(16), halign="center", color=rgba(theme.FOREST)))
        self.food_confidence = TextCard(font_size=sp(16), halign="center")
        self.freshness_confidence = TextCard(font_size=sp(16), halign="center")
        root.add_widget(self.food_confidence)
        root.add_widget(self.freshness_confidence)
        root.add_widget(wrap_label(
            text=("Both confidence scores must be at least 80%, and the model "
                  "must be able to assess freshness reliably. Try again with "
                  "one fruit or vegetable clearly visible, good lighting, and "
                  "no other food in the frame. Confidence is not a food-safety guarantee."),
            font_size=sp(13), halign="center", color=rgba(theme.MUTED)))
        root.add_widget(Label())
        controls = BoxLayout(size_hint_y=None, height=dp(54), spacing=dp(10))
        retry = PillButton(text="Try again")
        retry.bind(on_release=lambda *_: fa.retry_last_scan())
        home = PillButton(text="Home", bg=theme.PAPER, fg=theme.FOREST)
        home.bind(on_release=lambda *_: fa.go("home"))
        controls.add_widget(retry)
        controls.add_widget(home)
        root.add_widget(controls)
        self.add_widget(root)

    def show(self, entry):
        food = float(entry.get("food_conf", 0.0)) * 100
        freshness = float(entry.get("state_conf", 0.0)) * 100
        self.food_confidence.set_text(
            f"[b]Food identification confidence: {food:.1f}%[/b]")
        self.freshness_confidence.set_text(
            f"[b]Freshness confidence: {freshness:.1f}%[/b]")


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
        root.add_widget(BottomNav(fa, active="history"))
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


class SettingsScreen(Screen):
    """Local preferences, theme selection, rating and feedback links."""

    def __init__(self, fa, **kwargs):
        super().__init__(name="settings", **kwargs)
        self.fa = fa
        root = BoxLayout(orientation="vertical", padding=dp(16),
                         spacing=dp(8))
        paint(root, theme.CREAM)

        root.add_widget(wrap_label(
            text="Settings", bold=True, font_size=sp(25),
            color=rgba(theme.FOREST)))

        scroll = ScrollView(do_scroll_x=False)
        body = BoxLayout(orientation="vertical", spacing=dp(12),
                         size_hint_y=None, padding=(0, dp(4)))
        body.bind(minimum_height=body.setter("height"))

        body.add_widget(wrap_label(
            text="Your name", bold=True, font_size=sp(16),
            color=rgba(theme.FOREST)))
        name_card = Card(bg=theme.PAPER, orientation="horizontal",
                         size_hint_y=None, height=dp(76), padding=dp(10),
                         spacing=dp(8))
        self.name_label = wrap_label(
            text="", font_size=sp(15), color=rgba(theme.FOREST))
        name_card.add_widget(self.name_label)
        pencil = make_image(
            source=str(ROOT / "assets" / "pencil.png"),
            size_hint=(None, None), width=dp(20), height=dp(20))
        name_card.add_widget(pencil)
        change_name = PillButton(
            text="Change the name", size_hint_x=None, width=dp(154),
            font_size=sp(12))
        change_name.bind(on_release=lambda *_: fa.change_name_dialog())
        name_card.add_widget(change_name)
        body.add_widget(name_card)

        body.add_widget(wrap_label(
            text="Theme", bold=True, font_size=sp(16),
            color=rgba(theme.FOREST)))
        self.theme_button = PillButton(
            text=f"Choose theme  ·  {fa.theme_name}",
            bg=theme.PAPER, fg=theme.FOREST,
            size_hint_y=None, height=dp(52), font_size=sp(15))
        self.theme_button.bind(on_release=lambda *_: fa.go("themes"))
        body.add_widget(self.theme_button)

        custom_row = BoxLayout(orientation="horizontal", spacing=dp(8),
                               size_hint_y=None, height=dp(48))
        custom_row.add_widget(make_image(
            source=str(ROOT / "assets" / "pencil.png"),
            size_hint=(None, None), width=dp(18), height=dp(18)))
        self.custom_button = PillButton(
            text="Customize colours", bg=theme.PAPER, fg=theme.FOREST)
        self.custom_button.bind(on_release=lambda *_: fa.open_custom_theme())
        custom_row.add_widget(self.custom_button)
        body.add_widget(custom_row)

        rate = PillButton(text="Rate us", size_hint_y=None, height=dp(50))
        rate.bind(on_release=lambda *_: fa.open_form(RATE_URL))
        feedback = PillButton(
            text="Feedback / Bug report", bg=theme.PAPER,
            fg=theme.FOREST, size_hint_y=None, height=dp(50))
        feedback.bind(on_release=lambda *_: fa.open_form(FEEDBACK_URL))
        body.add_widget(rate)
        body.add_widget(feedback)

        body.add_widget(wrap_label(
            text="Made with love by a group of students",
            font_size=sp(12), halign="center", color=rgba(theme.MUTED)))
        scroll.add_widget(body)
        root.add_widget(scroll)
        root.add_widget(BottomNav(fa, active="settings"))
        self.add_widget(root)
        self.refresh()

    def on_pre_enter(self, *args):
        self.refresh()

    def refresh(self):
        self.name_label.text = f"Nickname: {self.fa.nickname() or 'Not set'}"
        self.theme_button.text = f"Choose theme  ·  {self.fa.theme_name}"
        self.custom_button.disabled = self.fa.theme_name != "Custom"



class ThemeCircle(Widget):
    """A true circular colour swatch with a ring for the active theme."""

    def __init__(self, colour, selected=False, **kwargs):
        super().__init__(**kwargs)
        self._selected = selected
        with self.canvas:
            Color(*rgba(colour))
            self._disc = Ellipse(pos=self.pos, size=self.size)
        if selected:
            with self.canvas.after:
                Color(*rgba(theme.MINT))
                self._ring = Line(circle=(0, 0, 1), width=dp(2.4))
        else:
            self._ring = None
        self.bind(pos=self._sync, size=self._sync)
        self._sync()

    def _sync(self, *args):
        self._disc.pos = self.pos
        self._disc.size = self.size
        if self._ring is not None:
            radius = min(self.width, self.height) / 2 + dp(2)
            self._ring.circle = (self.center_x, self.center_y, radius)


class ThemeOption(ButtonBehavior, Card):
    """One tappable theme tile: centered colour circle and name below."""

    def __init__(self, fa, name, colour, selected=False, **kwargs):
        super().__init__(
            bg=theme.PAPER, radius=16, orientation="vertical",
            padding=(dp(4), dp(7)), spacing=dp(3),
            size_hint=(1, None), height=dp(122), **kwargs)
        circle_holder = AnchorLayout(
            anchor_x="center", anchor_y="center",
            size_hint=(1, None), height=dp(76))
        circle_holder.add_widget(ThemeCircle(
            colour, selected=selected,
            size_hint=(None, None), size=(dp(62), dp(62))))
        self.add_widget(circle_holder)
        name_label = Label(
            text=name, font_size=sp(12), bold=selected,
            color=rgba(theme.FOREST), halign="center", valign="middle",
            size_hint=(1, None), height=dp(29),
            shorten=True, shorten_from="right")
        name_label.bind(
            size=lambda widget, value: setattr(widget, "text_size", value))
        self.add_widget(name_label)
        self.bind(on_release=lambda *_: fa.choose_theme(name))


class ThemeScreen(Screen):
    """Scrollable three-column theme chooser with colour-only circles."""

    def __init__(self, fa, **kwargs):
        super().__init__(name="themes", **kwargs)
        self.fa = fa
        root = BoxLayout(orientation="vertical", padding=dp(16),
                         spacing=dp(10))
        paint(root, theme.CREAM)

        header = BoxLayout(orientation="horizontal", spacing=dp(10),
                           size_hint_y=None, height=dp(48))
        back = PillButton(
            text="Back", bg=theme.PAPER, fg=theme.FOREST,
            size_hint_x=None, width=dp(88), font_size=sp(14))
        back.bind(on_release=lambda *_: fa.go("settings"))
        header.add_widget(back)
        header.add_widget(Label(
            text="Choose a theme", bold=True, font_size=sp(22),
            color=rgba(theme.FOREST), halign="left", valign="middle"))
        root.add_widget(header)
        root.add_widget(wrap_label(
            text="Tap a colour to apply its theme.",
            font_size=sp(13), color=rgba(theme.MUTED)))

        scroll = ScrollView(do_scroll_x=False, bar_width=dp(4),
                            scroll_type=["bars", "content"])
        grid = GridLayout(
            cols=3, spacing=(dp(10), dp(12)), padding=(dp(2), dp(6)),
            size_hint=(1, None))
        grid.bind(minimum_height=grid.setter("height"))
        for name in theme.THEME_NAMES:
            if name == "Custom":
                colour = (fa.custom_theme_color or "2F80ED").lstrip("#")
            else:
                colour = theme.PALETTES[name]["FOREST"]
            grid.add_widget(ThemeOption(
                fa, name, colour, selected=(fa.theme_name == name)))
        scroll.add_widget(grid)
        root.add_widget(scroll)
        self.add_widget(root)


# ============================================================
# 3. THE APP
# ============================================================

class FreshoraApp(App):

    def build(self):
        self.title = PROJECT_NAME
        if not ANDROID:
            Window.size = (400, 780)
            Window.minimum_width, Window.minimum_height = 360, 600

        self.predictor = FreshoraPredictor(android=ANDROID)
        self.store = Store(self.user_data_dir)
        self._migrate_old_nickname()
        self.theme_name = self.store.get_setting("theme_name", "Default")
        if self.theme_name not in theme.THEME_NAMES:
            self.theme_name = "Default"
        self.custom_theme_color = str(self.store.get_setting(
            "theme_custom_color", "2F80ED") or "2F80ED").lstrip("#").upper()
        try:
            rgba(self.custom_theme_color)
        except Exception:
            self.custom_theme_color = "2F80ED"
        try:
            theme.apply_theme(self.theme_name, self.custom_theme_color)
        except Exception:
            self.theme_name = "Default"
            theme.apply_theme("Default")
        Window.clearcolor = rgba(theme.CREAM)

        self.model_error = ""
        self.model_loading = True
        self.busy = False
        self.current_entry = None
        self.failure_entry = None
        self.last_scan_path = None
        self.saved = False
        self.result_origin = "scan"
        self._chooser = None
        self._last_picker_path = str(Path.home())
        self.scan_queue = []
        self.scan_index = 0

        self.sm = ScreenManager(transition=FadeTransition(duration=0.12))
        self._construct_screens()
        if not self.nickname():
            first_screen = "welcome"
        elif self.store.get_setting("onboarding_complete", False):
            first_screen = "home"
        else:
            first_screen = "tutorial"
        self.sm.current = first_screen

        Window.bind(on_keyboard=self._on_key)
        if ANDROID:
            # Resolve the bundled Java LiteRT class on the UI thread before
            # the model loader starts in a worker thread.
            try:
                from predict import resolve_android_class
                resolve_android_class()
            except Exception:
                pass
        threading.Thread(target=self._load_model, daemon=True).start()
        return self.sm

    def _construct_screens(self):
        """Build all screens from the current palette."""
        for screen in list(self.sm.screens):
            self.sm.remove_widget(screen)

        self.welcome_screen = WelcomeScreen(self)
        self.name_screen = NameScreen(self)
        self.tutorial_screen = TutorialScreen(self)
        self.home = HomeScreen(self)
        self.analysing = AnalysingScreen(self)
        self.result = ResultScreen(self)
        self.history = HistoryScreen(self)
        self.settings = SettingsScreen(self)
        self.themes = ThemeScreen(self)
        self.scan_failed = ScanFailedScreen(self)

        for screen in (
            self.welcome_screen, self.name_screen, self.tutorial_screen,
            self.home, self.analysing, self.result, self.history,
            self.settings, self.themes, self.scan_failed,
        ):
            self.sm.add_widget(screen)

        if ANDROID:
            self.camera_screen = CameraScreen(self)
            self.sm.add_widget(self.camera_screen)
        else:
            self.camera_screen = None

    def on_stop(self):
        if ANDROID and self.camera_screen is not None:
            self.camera_screen._disconnect()
        try:
            self.predictor.close()
        except Exception:
            pass

    def on_pause(self):
        if ANDROID and self.sm.current == "camera" and self.camera_screen:
            self.camera_screen._disconnect()
        return True

    def on_resume(self):
        if ANDROID and self.sm.current == "camera" and self.camera_screen:
            try:
                from android.permissions import Permission, check_permission
                if check_permission(Permission.CAMERA):
                    self.camera_screen._connected = False
                    Clock.schedule_once(
                        lambda dt: self.camera_screen._connect(), 0.3)
            except Exception:
                pass

    # --------------------------------------------------------
    # Navigation and onboarding
    # --------------------------------------------------------
    def go(self, name):
        if name in ("home", "history", "settings"):
            self.scan_queue = []
            self.scan_index = 0
        self.sm.current = name

    def navigate_tab(self, destination):
        if destination == "scan":
            self.open_camera()
        elif destination == "history":
            self.show_history()
        elif destination == "settings":
            self.go("settings")
        else:
            self.go("home")

    def _migrate_old_nickname(self):
        """Carry forward a nickname saved in the old local Profile screen."""
        current_name = self.store.get_setting("nickname", "")
        if str(current_name or "").strip():
            return
        old_profile = self.store.get_setting("profile", {})
        if isinstance(old_profile, dict):
            old_name = str(old_profile.get("name", "")).strip()
            if old_name:
                self.store.set_setting("nickname", old_name)
                # Existing users have already used the app; do not force the
                # new tutorial on an upgrade from the old local Profile flow.
                self.store.set_setting("onboarding_complete", True)

    def nickname(self):
        value = self.store.get_setting("nickname", "")
        return str(value).strip() if value else ""

    def save_onboarding_name(self, name):
        clean_name = str(name).strip()[:30]
        if not clean_name:
            self.show_message("Name needed", "Enter a name or nickname to continue.")
            return
        try:
            self.store.set_setting("nickname", clean_name)
        except Exception as error:
            self.show_message("Could not save name", str(error))
            return
        self.tutorial_screen.index = 0
        self.tutorial_screen._render_slide()
        self.go("tutorial")

    def finish_tutorial(self):
        self.store.set_setting("onboarding_complete", True)
        self.store.set_setting("welcome_seen", True)
        self.home.refresh()
        self.go("home")

    def set_nickname(self, name):
        clean_name = str(name).strip()[:30]
        if not clean_name:
            self.show_message("Name needed", "Please enter a name or nickname.")
            return False
        try:
            self.store.set_setting("nickname", clean_name)
        except Exception as error:
            self.show_message("Could not save name", str(error))
            return False
        self.home.refresh()
        self.settings.refresh()
        return True

    def change_name_dialog(self):
        box = BoxLayout(orientation="vertical", padding=dp(12),
                        spacing=dp(10))
        box.add_widget(Label(
            text="Enter your name/nickname", size_hint_y=None, height=dp(34)))
        name_input = TextInput(
            text=self.nickname(), multiline=False,
            size_hint_y=None, height=dp(50),
            foreground_color=rgba(theme.FOREST),
            cursor_color=rgba(theme.FOREST),
            background_normal="", background_active="",
            background_color=rgba(theme.PAPER),
        )
        box.add_widget(name_input)
        controls = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(10))
        cancel = PillButton(text="Cancel", bg=theme.PAPER, fg=theme.FOREST)
        save = PillButton(text="Save")
        controls.add_widget(cancel)
        controls.add_widget(save)
        box.add_widget(controls)
        popup = Popup(title="Change the name", content=box,
                      size_hint=(0.9, None), height=dp(205))
        cancel.bind(on_release=popup.dismiss)

        def save_name(*_):
            if self.set_nickname(name_input.text):
                popup.dismiss()

        save.bind(on_release=save_name)
        popup.open()

    def choose_theme(self, name):
        if name == "Custom":
            existing = self.store.get_setting("theme_custom_color", "")
            if existing:
                self.set_theme("Custom", existing)
            else:
                self.open_custom_theme()
            return
        self.set_theme(name)

    def set_theme(self, name, custom_primary=None):
        if name not in theme.THEME_NAMES:
            name = "Default"
        if name == "Custom":
            custom_primary = (custom_primary or self.custom_theme_color or
                             "2F80ED").lstrip("#").upper()
            try:
                rgba(custom_primary)
            except Exception:
                custom_primary = "2F80ED"
            self.custom_theme_color = custom_primary
            self.store.set_setting("theme_custom_color", custom_primary)
        self.theme_name = name
        self.store.set_setting("theme_name", name)
        theme.apply_theme(name, self.custom_theme_color)
        Window.clearcolor = rgba(theme.CREAM)
        destination = self.sm.current if self.sm.current in (
            "home", "history", "settings", "themes", "result", "scan_failed") else "settings"
        if self.sm.current == "camera" and self.camera_screen:
            self.camera_screen._disconnect()
        self._construct_screens()
        self.sm.current = destination
        self.home.refresh()
        if destination == "history":
            self.history.refresh()
        elif destination == "result" and self.current_entry:
            self.result.show(self.current_entry, saved=self.saved)
        elif destination == "scan_failed" and self.failure_entry:
            self.scan_failed.show(self.failure_entry)

    def open_custom_theme(self):
        picker = ColorPicker()
        try:
            picker.color = (*rgba(self.custom_theme_color)[:3], 1)
        except Exception:
            pass
        content = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(8))
        content.add_widget(Label(
            text="Choose your custom theme colour", size_hint_y=None,
            height=dp(32), color=rgba(theme.FOREST)))
        content.add_widget(picker)
        controls = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        cancel = PillButton(text="Cancel", bg=theme.PAPER, fg=theme.FOREST)
        apply = PillButton(text="Apply colour")
        controls.add_widget(cancel)
        controls.add_widget(apply)
        content.add_widget(controls)
        popup = Popup(title="Custom theme", content=content,
                      size_hint=(0.95, 0.88))
        def cancel_picker(*_):
            popup.dismiss()
            if getattr(self, "settings", None):
                self.settings.refresh()

        cancel.bind(on_release=cancel_picker)

        def apply_colour(*_):
            red, green, blue = picker.color[:3]
            hex_colour = "".join(
                f"{round(channel * 255):02X}" for channel in (red, green, blue))
            popup.dismiss()
            self.set_theme("Custom", hex_colour)

        apply.bind(on_release=apply_colour)
        popup.open()

    def open_form(self, url):
        try:
            if ANDROID:
                from jnius import autoclass
                Intent = autoclass("android.content.Intent")
                Uri = autoclass("android.net.Uri")
                PythonActivity = autoclass("org.kivy.android.PythonActivity")
                intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
                PythonActivity.mActivity.startActivity(intent)
            else:
                webbrowser.open(url)
        except Exception as error:
            self.show_message("Could not open the form", str(error))

    def _maybe_prompt_rating(self):
        if self.store.get_setting("rating_prompted", False):
            return
        count = int(self.store.get_setting("completed_scans", 0) or 0) + 1
        self.store.set_setting("completed_scans", count)
        if count < 3:
            return

        box = BoxLayout(orientation="vertical", padding=dp(12), spacing=dp(12))
        box.add_widget(Label(
            text="Does our app work good? Rate us!",
            halign="center", valign="middle"))
        row = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(10))
        rate = PillButton(text="Rate us")
        no_thanks = PillButton(
            text="No thanks", bg=theme.PAPER, fg=theme.FOREST)
        row.add_widget(rate)
        row.add_widget(no_thanks)
        box.add_widget(row)
        popup = Popup(title="A quick question", content=box,
                      size_hint=(0.9, None), height=dp(190), auto_dismiss=False)

        def finish_rating(*_):
            self.store.set_setting("rating_prompted", True)
            popup.dismiss()
            self.open_form(RATE_URL)

        def skip_rating(*_):
            self.store.set_setting("rating_prompted", True)
            popup.dismiss()

        rate.bind(on_release=finish_rating)
        no_thanks.bind(on_release=skip_rating)
        popup.open()

    def retry_last_scan(self):
        path = self.last_scan_path
        if path and Path(path).is_file():
            self.start_scan(path)
        else:
            self.open_camera()

    def show_history(self):
        self.go("history")

    def has_next_photo(self):
        return self.scan_index + 1 < len(self.scan_queue)

    def result_primary_action(self):
        if self.has_next_photo():
            self.scan_index += 1
            self._scan_current_photo()
            return
        self.scan_queue = []
        self.scan_index = 0
        self.open_camera()

    def _on_key(self, window, key, *args):
        """Android back button (and Esc on a laptop)."""
        if key != 27:
            return False
        current = self.sm.current
        if current in ("home", "welcome"):
            return False
        if current in ("name", "tutorial"):
            self.go("welcome" if current == "name" else "name")
            return True
        if current == "analysing":
            return True
        if current == "themes":
            self.go("settings")
            return True
        if current == "result" and self.result_origin == "history":
            self.go("history")
        else:
            self.go("home")
        return True

    def show_message(self, title, text):
        box = BoxLayout(orientation="vertical", padding=dp(12),
                        spacing=dp(12))
        label = Label(text=text, halign="center", valign="middle")
        label.bind(size=lambda lbl, value: setattr(lbl, "text_size", value))
        box.add_widget(label)
        ok = PillButton(text="OK", size_hint_y=None, height=dp(48))
        box.add_widget(ok)
        popup = Popup(title=title, content=box, size_hint=(0.9, 0.5))
        ok.bind(on_release=popup.dismiss)
        popup.open()

    # --------------------------------------------------------
    # Model loading (background)
    # --------------------------------------------------------
    def _load_model(self):
        """Load the already-trained model once; never train on the phone."""
        try:
            self.predictor.load()
        except FreshoraError as error:
            self.model_error = str(error)
            self.model_loading = False
            Clock.schedule_once(lambda dt: self._show_model_load_error())
            return
        except Exception as error:
            self.model_error = repr(error)
            self.model_loading = False
            Clock.schedule_once(lambda dt: self._show_model_load_error())
            return

        self.model_error = ""
        self.model_loading = False
        Clock.schedule_once(lambda dt: self.home.set_ready(
            True, "AI model loaded. Ready to scan."))

    def _show_model_load_error(self):
        self.home.set_ready(
            False, "AI model unavailable. Tap Open camera or Add photo "
                   "to see the loading error.")
        self.home.scan_button.disabled = False
        self.home.gallery_button.disabled = False

    # --------------------------------------------------------
    # Getting a photo: camera, gallery, laptop file picker
    # --------------------------------------------------------
    def ask_camera_permission(self, callback):
        """Asks for camera permission only when the camera is opened."""
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
            self._chooser.choose_content("image/*", multiple=True)
        except Exception as error:
            self.show_message("Gallery not available", str(error))

    def _gallery_chosen(self, shared_files):
        Clock.schedule_once(lambda dt: self._copy_chosen(shared_files))

    def _copy_chosen(self, shared_files):
        if not shared_files:
            return
        if isinstance(shared_files, (str, bytes)):
            shared_files = [shared_files]
        try:
            from androidstorage4kivy import SharedStorage
            storage = SharedStorage()
            paths = []
            for shared_file in shared_files:
                path = storage.copy_from_shared(shared_file)
                if path and Path(str(path)).is_file():
                    paths.append(str(path))
        except Exception as error:
            self.show_message("Could not open that photo", str(error))
            return
        if not paths:
            self.show_message("Could not open that photo",
                              "Android did not give access to it. Try another photo.")
            return
        self.start_scan(paths)

    def _desktop_picker(self):
        start = Path(self._last_picker_path)
        if not start.is_dir():
            start = Path.home()
        chooser = FileChooserListView(
            path=str(start), filters=IMAGE_FILTERS, multiselect=True)
        content = BoxLayout(orientation="vertical", spacing=dp(6),
                            padding=dp(6))
        shortcuts = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(6))
        for label, folder in (
            ("Home", Path.home()),
            ("Downloads", Path.home() / "Downloads"),
            ("Pictures", Path.home() / "Pictures"),
            ("Desktop", Path.home() / "Desktop"),
        ):
            if folder.is_dir():
                shortcut = Button(text=label)
                shortcut.bind(
                    on_release=lambda _button, target=folder:
                    setattr(chooser, "path", str(target)))
                shortcuts.add_widget(shortcut)
        content.add_widget(shortcuts)
        content.add_widget(chooser)
        choose = PillButton(
            text="Use Selected Photo(s)", size_hint_y=None, height=dp(50))
        content.add_widget(choose)
        popup = Popup(title="Choose Food Image", content=content,
                      size_hint=(0.95, 0.9))

        def select_image(_button):
            if not chooser.selection:
                return
            paths = list(chooser.selection)
            self._last_picker_path = chooser.path
            popup.dismiss()
            self.start_scan(paths)

        choose.bind(on_release=select_image)
        popup.open()

    def _model_unavailable(self):
        if self.predictor.loaded:
            return False
        if self.model_loading:
            self.show_message(
                "AI model loading",
                "Freshora is loading its pre-trained model. Try again in a moment.")
        else:
            self.show_message(
                "AI model not available",
                self.model_error or "The packaged AI model could not be loaded.")
        return True

    # --------------------------------------------------------
    # Scanning (background thread)
    # --------------------------------------------------------
    def start_scan(self, path):
        if self.busy:
            return
        if self._model_unavailable():
            return
        if isinstance(path, (list, tuple)):
            self.scan_queue = [str(item) for item in path if item]
        else:
            self.scan_queue = [str(path)] if path else []
        self.scan_index = 0
        if not self.scan_queue:
            self.show_message("No photo selected", "Choose a photo first.")
            return
        self._scan_current_photo()

    def _scan_current_photo(self):
        if self.busy or not self.scan_queue:
            return
        path = self.scan_queue[self.scan_index]
        self.last_scan_path = str(path)
        self.busy = True
        self.go("analysing")
        threading.Thread(
            target=self._scan_worker, args=(str(path),), daemon=True
        ).start()

    def _scan_worker(self, path):
        try:
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
        # Never display a prediction unless both independent confidence
        # measures reach 80% and the predictor's safety checks pass.
        if not entry_is_reliable(entry):
            self.failure_entry = entry
            self.scan_queue = []
            self.scan_index = 0
            self.scan_failed.show(entry)
            self.go("scan_failed")
            return

        self.current_entry = entry
        self.saved = False
        self.result_origin = "scan"
        progress = None
        if len(self.scan_queue) > 1:
            progress = (f"Photo {self.scan_index + 1} of "
                        f"{len(self.scan_queue)}")
        self.result.show(entry, saved=False, batch_progress=progress)
        self.go("result")
        Clock.schedule_once(lambda dt: self._maybe_prompt_rating(), 0.45)

    def _scan_failed(self, message):
        self.busy = False
        self.scan_queue = []
        self.scan_index = 0
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
        self.scan_queue = []
        self.scan_index = 0
        self.saved = True
        self.result_origin = "history"
        if not entry_is_reliable(entry):
            # Older versions could save sub-80% guesses. Do not display those
            # as confident results after this update either.
            self.failure_entry = entry
            self.last_scan_path = entry.get("image")
            self.scan_failed.show(entry)
            self.go("scan_failed")
            return
        self.current_entry = entry
        self.result.show(entry, saved=True)
        self.go("result")


if __name__ == "__main__":
    FreshoraApp().run()
