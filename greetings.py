# ============================================================
# FRESHORA - OFFLINE HOME-SCREEN GREETINGS
# The app adds the nickname after the time-specific greeting, e.g.
# "Good morning, Alex".
# ============================================================

import random
from datetime import datetime

TITLES = {
    "morning": "Good morning",
    "afternoon": "Good afternoon",
    "evening": "Good evening",
    "night": "Good evening",
}

WEIRD = [
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

_last_weird = None


def _slot(hour):
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def greeting(now=None):
    """Return a stable time-based title and a varied second line."""
    global _last_weird
    now = now or datetime.now()
    choices = [line for line in WEIRD if line != _last_weird] or WEIRD
    weird = random.choice(choices)
    _last_weird = weird
    return TITLES[_slot(now.hour)], weird
