# ============================================================
# FRESHORA - COLOURS (from the UI concept deck)
# Change a colour here and it changes everywhere in the app.
# ============================================================


def rgba(hex_code, alpha=1.0):
    """'143A2D' -> (0.078, 0.227, 0.176, 1.0), the format Kivy wants."""
    h = hex_code.lstrip("#")
    return (
        int(h[0:2], 16) / 255.0,
        int(h[2:4], 16) / 255.0,
        int(h[4:6], 16) / 255.0,
        alpha,
    )


FOREST = "143A2D"     # main brand colour
DEEP = "08241B"       # dark backgrounds
CREAM = "F5F1E6"      # light backgrounds
MINT = "A8ED75"       # main buttons
LEAF = "5B9B47"       # secondary green
PAPER = "FFFFF8"      # cards
MUTED = "6B7B6F"      # small grey-green text

# Verdict colours: green = fine, amber = check first, red = don't eat.
VERDICT_HEX = {
    "yes": "4caf50",
    "maybe": "ffb300",
    "no": "e53935",
    "unknown": "9e9e9e",
}


def verdict_hex(safe):
    return VERDICT_HEX.get(safe, VERDICT_HEX["unknown"])
