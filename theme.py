# ============================================================
# FRESHORA - APP COLOUR THEMES
# Theme names and palettes used throughout the Kivy app.
# ============================================================


def rgba(hex_code, alpha=1.0):
    """Convert a six-digit hex colour to Kivy's RGBA tuple."""
    h = str(hex_code).lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected a six-digit hex colour, got {hex_code!r}")
    return (
        int(h[0:2], 16) / 255.0,
        int(h[2:4], 16) / 255.0,
        int(h[4:6], 16) / 255.0,
        alpha,
    )


def _darken(hex_code, amount=0.55):
    rgb = rgba(hex_code)[:3]
    return "".join(f"{max(0, min(255, round(channel * amount * 255))):02X}"
                   for channel in rgb)


def _lighten(hex_code, amount=0.82):
    rgb = rgba(hex_code)[:3]
    return "".join(
        f"{max(0, min(255, round((channel + (1 - channel) * amount) * 255))):02X}"
        for channel in rgb
    )


THEME_NAMES = (
    "Default",
    "Dark",
    "Aquatic",
    "Forest",
    "Modern",
    "I like red",
    "Pinkpink",
    "Custom",
)

PALETTES = {
    "Default": {
        "FOREST": "143A2D",
        "DEEP": "08241B",
        "CREAM": "F5F1E6",
        "MINT": "A8ED75",
        "LEAF": "5B9B47",
        "PAPER": "FFFFF8",
        "MUTED": "6B7B6F",
    },
    "Dark": {
        "FOREST": "DCE9E3",
        "DEEP": "09120F",
        "CREAM": "141E19",
        "MINT": "8BC9A8",
        "LEAF": "4B8065",
        "PAPER": "202D27",
        "MUTED": "A2B5AA",
    },
    "Aquatic": {
        "FOREST": "123D5A",
        "DEEP": "061D30",
        "CREAM": "EAF7FB",
        "MINT": "70D8E8",
        "LEAF": "2D90B7",
        "PAPER": "F8FDFF",
        "MUTED": "56717D",
    },
    "Forest": {
        "FOREST": "174B32",
        "DEEP": "082719",
        "CREAM": "EDF5E9",
        "MINT": "A8E6A0",
        "LEAF": "4C9658",
        "PAPER": "FBFFF9",
        "MUTED": "607B65",
    },
    "Modern": {
        "FOREST": "3C424A",
        "DEEP": "1E2329",
        "CREAM": "F0F1F2",
        "MINT": "C5CCD3",
        "LEAF": "707A85",
        "PAPER": "FFFFFF",
        "MUTED": "6D747B",
    },
    "I like red": {
        "FOREST": "7F1D1D",
        "DEEP": "3B0B0B",
        "CREAM": "FFF1F1",
        "MINT": "FCA5A5",
        "LEAF": "DC2626",
        "PAPER": "FFFBFB",
        "MUTED": "8A5B5B",
    },
    "Pinkpink": {
        "FOREST": "831843",
        "DEEP": "3A1028",
        "CREAM": "FFF0F6",
        "MINT": "F9A8D4",
        "LEAF": "DB5A91",
        "PAPER": "FFFBFD",
        "MUTED": "90677C",
    },
}

# The default palette is applied at import time. FreshoraApp applies the
# user's saved palette before constructing any screens.
FOREST = PALETTES["Default"]["FOREST"]
DEEP = PALETTES["Default"]["DEEP"]
CREAM = PALETTES["Default"]["CREAM"]
MINT = PALETTES["Default"]["MINT"]
LEAF = PALETTES["Default"]["LEAF"]
PAPER = PALETTES["Default"]["PAPER"]
MUTED = PALETTES["Default"]["MUTED"]


def apply_theme(name="Default", custom_primary=None):
    """Apply a named palette, optionally using a user-picked custom colour."""
    global FOREST, DEEP, CREAM, MINT, LEAF, PAPER, MUTED

    if name == "Custom":
        primary = (custom_primary or "2F80ED").lstrip("#").upper()
        # Validate before updating globals so corrupt settings cannot crash UI.
        rgba(primary)
        values = {
            "FOREST": primary,
            "DEEP": _darken(primary, 0.48),
            "CREAM": "F1F5F9",
            "MINT": _lighten(primary, 0.78),
            "LEAF": _darken(primary, 0.82),
            "PAPER": "FFFFFF",
            "MUTED": "64748B",
        }
    else:
        values = PALETTES.get(name, PALETTES["Default"])

    FOREST = values["FOREST"]
    DEEP = values["DEEP"]
    CREAM = values["CREAM"]
    MINT = values["MINT"]
    LEAF = values["LEAF"]
    PAPER = values["PAPER"]
    MUTED = values["MUTED"]
    return name


# Verdict colours remain semantically consistent in every theme.
VERDICT_HEX = {
    "yes": "4caf50",
    "maybe": "ffb300",
    "no": "e53935",
    "unknown": "9e9e9e",
}


def verdict_hex(safe):
    return VERDICT_HEX.get(safe, VERDICT_HEX["unknown"])
