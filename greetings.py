# ============================================================
# FRESHORA - WEIRD HOME-SCREEN GREETINGS (works offline)
#
# greeting() gives a plain "Good morning," style title plus a weird
# second line that is never the same twice in a row.
# ADD YOUR OWN LINES to the lists below. More lines = more weird.
# ============================================================

import random
from datetime import datetime

TITLES = {
    "morning": ["Good morning,", "Morning, sunshine,", "Rise and ripen,"],
    "afternoon": ["Good afternoon,", "Hello, midday human,", "Afternoon,"],
    "evening": ["Good evening,", "Evening, kitchen explorer,", "Welcome back,"],
    "night": ["Late-night snacker,", "Still awake?", "Midnight fridge visit?"],
}

WEIRD = [
    "I got bored writing messages :P",
    "the fruit bowl has been talking about you.",
    "a banana in the next room just sneezed.",
    "your fridge light is judging you, gently.",
    "the vegetables held a meeting. you were mentioned.",
    "somewhere, an onion is thinking about its choices.",
    "the oracle has had coffee. be afraid.",
    "I counted your apples. they counted back.",
    "bring me something suspicious to look at.",
    "my crystal ball smells faintly of lemons today.",
    "let us interrogate some produce.",
    "the tomatoes say hello. they are not to be trusted.",
    "today's forecast: ninety percent chance of snacks.",
    "I have prepared a tiny speech for your mangoes.",
    "show me the food. I will pretend to be calm.",
]

_last = {"weird": None, "title": None}


def _slot(hour):
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def _different(options, key):
    choices = [o for o in options if o != _last[key]] or options
    chosen = random.choice(choices)
    _last[key] = chosen
    return chosen


def greeting(now=None):
    """Returns (title, weird_line)."""
    now = now or datetime.now()
    title = _different(TITLES[_slot(now.hour)], "title")
    weird = _different(WEIRD, "weird")
    return title, weird
