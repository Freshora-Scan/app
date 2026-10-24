# ============================================================
# FRESHORA - PERSONALITY (multi-food)
#
# Two jobs, kept strictly apart:
#
#   1. THE VERDICT  (VERDICT_TEXT below)
#      Fixed text written by YOU. The AI can never change it,
#      so a rotten apple can never be called "safe".
#
#   2. THE VOICE
#      A fresh, weird reply every time. Written by an OpenAI model
#      when OPENAI_API_KEY is set, otherwise by the built-in
#      offline generator (no internet, no cost, never crashes).
#
# The final message = voice + verdict line.
#
# Class names look like  "<food>__<state>"  (apple__ripe,
# bitter gourd__rotten). Folder names decide everything, so the
# state folder names your team uses should be in STATE_GROUPS below.
# A state that is NOT listed is never called safe: the verdict
# says "check it yourself".
# ============================================================

import os
import random
import re
from collections import deque
from dataclasses import dataclass

SEPARATOR = "__"      # must match train.py


# ============================================================
# 1. STATE NAMES -> HOW FRESH THEY MEAN
# ============================================================
# Add your own folder names here (lower case, use _ for spaces).
# Be careful: only list a state as "ripe"/"unripe" if it really is
# safe. Words that can hide a danger (green potato, sprouted
# potato...) are deliberately NOT listed.

STATE_GROUPS = {
    "unripe": {
        "unripe", "underripe", "under_ripe", "not_ripe", "immature",
    },
    "ripe": {
        "ripe", "fresh", "good", "healthy", "ready",
    },
    "overripe": {
        "overripe", "over_ripe", "old", "wilted", "soft", "stale",
        "shriveled", "shrivelled", "wrinkled", "bruised", "aged", "limp",
    },
    "rotten": {
        "rotten", "rotting", "rot", "spoiled", "spoilt", "bad",
        "mouldy", "moldy", "mould", "mold", "decayed", "fungus",
        "fungal", "expired",
    },
}


def normalise(text):
    return text.strip().lower().replace("-", "_").replace(" ", "_")


def split_label(label):
    """'apple__ripe' -> ('apple', 'ripe').  'ripe' -> ('', 'ripe')."""
    if SEPARATOR in label:
        food, state = label.split(SEPARATOR, 1)
    else:
        food, state = "", label
    return food.strip().lower(), state.strip().lower()


def display_name(text):
    """'bitter_gourd' / 'bitter gourd' -> 'bitter gourd'."""
    return text.replace("_", " ").strip()


def classify_state(state):
    """Returns 'unripe' | 'ripe' | 'overripe' | 'rotten' | None."""
    key = normalise(state)
    for group, names in STATE_GROUPS.items():
        if key in names:
            return group
    return None


# ============================================================
# 2. VERDICTS (facts, never written by the AI)
# ============================================================
# safe: "yes" | "maybe" | "no" | "unknown"   (the app uses it for colours)

VERDICT_TEXT = {
    "unripe": (
        "yes",
        "Usually safe to eat, but not at its best yet. "
        "Give it a few days to ripen.",
    ),
    "ripe": (
        "yes",
        "Safe to eat. This is peak freshness.",
    ),
    "overripe": (
        "maybe",
        "Usually fine if there is no mould, leaking or sour smell. "
        "Use it soon, for cooking, smoothies or baking.",
    ),
    "rotten": (
        "no",
        "Do NOT eat it. Bin it or compost it.",
    ),
}


def get_verdict(food, state, food_ok=True, state_ok=True, judgeable=True):
    """
    The fixed verdict line. Doubt always beats a confident-sounding
    answer: if anything is uncertain, the verdict says so.
    """
    name = display_name(food)
    subject = f" {name}" if name else ""

    if not food_ok:
        return {
            "safe": "unknown",
            "line": "Verdict: NOT SURE what this is. Retake the photo in "
                    "good light, with the whole food in frame on a plain "
                    "background, and check it yourself.",
        }

    if not judgeable:
        return {
            "safe": "unknown",
            "line": f"Verdict: this looks like{subject or ' food'}, but "
                    "Freshora has not learned how to judge its freshness "
                    "yet. Check it yourself (look, smell, texture).",
        }

    if not state_ok:
        return {
            "safe": "unknown",
            "line": f"Verdict: NOT SURE how fresh this{subject or ' food'} "
                    "is. Retake the photo in good light with the whole "
                    "thing in frame, and trust your eyes and nose.",
        }

    group = classify_state(state)
    if group is None:
        return {
            "safe": "unknown",
            "line": f"Verdict: {display_name(state).upper()}{subject}. "
                    "Freshora has no safety rule for this condition. "
                    "Check it yourself (look, smell, texture).",
        }

    safe, text = VERDICT_TEXT[group]
    return {
        "safe": safe,
        "line": f"Verdict: {display_name(state).upper()}{subject}. {text}",
    }


# ============================================================
# 3. THE PERSONA (used by the AI model)
# ============================================================

PERSONA = """You are FRESHORA, the resident oracle of a food-scanner app.
You are a dramatic, slightly unhinged kitchen prophet. You treat fruit
and vegetables as if they were tiny celebrities, rivals, or exes. You
are funny, specific, and strange, never corporate, never generic.

Rules:
- 2 to 3 short sentences. Maximum 45 words.
- Be original every time. Never reuse a phrase from the AVOID list.
- Never say "As an AI", never apologise, no emojis, no hashtags.
- The VERDICT you are given is a fact. Never contradict it. Never
  suggest eating something whose verdict is ROTTEN, NOT SURE or
  has no safety rule.
- Do not write the verdict line yourself. It is added separately.
- Weird is good. Mean to the user is not."""

MOODS = [
    "smug", "heartbroken", "suspiciously calm", "caffeinated",
    "dramatically exhausted", "conspiratorial", "overly proud",
    "mysteriously poetic", "mildly offended", "giddy",
    "like a bored wizard", "like a nature documentary narrator",
]

FORMS = [
    "a tiny prophecy",
    "a fake weather report for the kitchen counter",
    "a one-line courtroom ruling",
    "a sports commentator losing their mind",
    "a message from the food itself",
    "a dramatic whisper",
    "a fortune-cookie that went off the rails",
    "a breakup text from the fruit bowl",
    "a very confident wrong-sounding fact that is still harmless",
]

WEIRD_THINGS = [
    "a moth in a tuxedo", "a haunted spoon", "the moon", "a retired pirate",
    "a damp sock parliament", "a tax auditor", "a jazz trumpet",
    "a lighthouse", "a confused goose", "the ghost of a smoothie",
    "a pocket-sized thunderstorm", "a very polite volcano",
    "a disco ball", "grandma's secret", "a tiny accountant",
]


# ============================================================
# 4. OFFLINE VOICE (no internet needed)
# ============================================================
# reply = opener + middle + closer (+ sometimes an aside).
# {food} / {Food} are filled in; lines are written so they also
# read fine for plural foods like "grapes".

OFFLINE = {
    "unripe": {
        "openers": [
            "Ah. {Food}, still in the teenage phase.",
            "Stubborn. Firm. Wearing a school uniform.",
            "This one is not ready, and it knows it.",
            "I sense someone skipped the ripening meeting.",
            "Behold: commitment issues, in edible form.",
            "The skin whispers, 'not yet, not yet.'",
        ],
        "middles": [
            "It's edible, but it will taste like chalk with ambitions.",
            "Bite it now and it bites back, spiritually.",
            "It's safe enough, just deeply unimpressed with itself.",
            "Give it time. Even legends start out starchy.",
            "The texture is a determined eraser.",
            "Nothing wrong here, only a shortage of sweetness and drama.",
        ],
        "closers": [
            "Check back in a few days.",
            "Leave it on the counter and leave it alone.",
            "Patience, little scanner.",
            "Do not poke it. It hates being poked.",
            "Ripeness cannot be rushed, only respected.",
            "Come back when it has finished growing up.",
        ],
    },
    "ripe": {
        "openers": [
            "Oh. OH. This is the one.",
            "Golden hour has arrived, in edible form.",
            "I have seen empires fall, but never anything this on time.",
            "Something just walked in like it owns the kitchen.",
            "Peak performance. No notes.",
            "Applause. The moment has peaked.",
        ],
        "middles": [
            "Fresh, fragrant, structurally confident.",
            "It's safe, and a little smug about it.",
            "Peak {food}. The kind poets write tragic letters about.",
            "It is exactly as ready as it can legally be.",
            "Zero drama, maximum flavour.",
            "Go on. The universe has signed the permission slip.",
        ],
        "closers": [
            "Eat it before it gets ideas.",
            "Do not waste this moment on a scan.",
            "Wash, bite, nod slowly.",
            "This is your sign.",
            "Eat. Now.",
            "Enjoy the good timeline.",
        ],
    },
    "overripe": {
        "openers": [
            "This one has seen things. Mostly Tuesdays.",
            "Soft, spotted, and philosophical.",
            "A dramatic third act.",
            "Freckles everywhere. A true veteran.",
            "The kitchen's retired jazz musician has arrived.",
            "Ooh, this one is basically a puddle with a plan.",
        ],
        "middles": [
            "It's probably fine if there's no mould or sour smell.",
            "Too tired for a dignified snack, perfect for cooking.",
            "Sweetness at maximum, structure at a minimum.",
            "Smoothie destiny is calling.",
            "Sniff first. Your nose is the real judge here.",
            "Still has dignity, just not much shape.",
        ],
        "closers": [
            "Mash it. Cook it. Respect it.",
            "Freeze it before it files a complaint.",
            "Check for mould, then proceed.",
            "Its final form is dinner.",
            "Do not wait another day.",
            "Use it today or lose it.",
        ],
    },
    "rotten": {
        "openers": [
            "No. Absolutely not.",
            "This one has left the building and taken the lease.",
            "I'm sorry for your loss. It is gone.",
            "That is not food anymore, that is a cautionary tale.",
            "The prophecy is dark, and it smells worse.",
            "This is a funeral.",
        ],
        "middles": [
            "Whatever lives in there is paying no rent.",
            "It has crossed a line that cannot be uncrossed.",
            "Even the fruit flies are checking their options.",
            "It has become a science project without consent.",
            "The kitchen wants a divorce.",
            "Hope is gone. Mould is not.",
        ],
        "closers": [
            "Bin it. Wash your hands. Move on.",
            "Compost it with honour.",
            "Do not eat it. I mean it.",
            "Say goodbye, then say it to the trash.",
            "Let it go. Literally. Into the bin.",
            "Not edible. Not negotiable.",
        ],
    },
    # Doubt about the photo, the food, or the freshness.
    "unsure": {
        "openers": [
            "The fog is thick today.",
            "I squinted. The food squinted back.",
            "My crystal ball is buffering.",
            "This photo is shy.",
            "The oracle needs better lighting.",
            "Hmm. The signal is wobbling.",
        ],
        "middles": [
            "I cannot judge what I cannot see.",
            "It could be anything, even a very convincing spoon.",
            "My confidence has left for a snack.",
            "Shadows are lying to me.",
            "I'd rather admit doubt than lie to you.",
            "The pixels are being mysterious.",
        ],
        "closers": [
            "Try again with the whole thing in view.",
            "Find better light and a plain background.",
            "Take another photo, no dramatic angles.",
            "Use your nose in the meantime.",
            "Another shot, please.",
            "Closer, brighter, calmer.",
        ],
    },
    # A state Freshora has no safety rule for.
    "other": {
        "openers": [
            "I know this one, but not its mood.",
            "A condition I have no rulebook for.",
            "Intriguing. Unlisted. Slightly suspicious.",
            "My scrolls do not cover this state.",
            "The oracle consulted the manual. The manual shrugged.",
            "Unfamiliar territory, ahoy.",
        ],
        "middles": [
            "I can name the food, but not the verdict.",
            "I will not gamble with your stomach.",
            "Facts are missing, and I refuse to invent them.",
            "Safety questions deserve real answers.",
            "Better a cautious oracle than a sorry one.",
            "The data is polite but unhelpful.",
        ],
        "closers": [
            "Check it yourself: look, smell, touch.",
            "When in doubt, throw it out.",
            "Use your eyes and nose.",
            "Ask a human with a nose.",
            "Trust your senses over mine.",
            "Let your nose make the call.",
        ],
    },
}

ASIDES = [
    "Also, {thing} says hello.",
    "I blame {thing}.",
    "Somewhere, {thing} nods in approval.",
    "This message was reviewed by {thing}.",
    "(Do not tell {thing} I said that.)",
    "Reminder: {thing} is watching.",
]


# ============================================================
# 5. REPLY OBJECT
# ============================================================

@dataclass
class Reply:
    text: str            # the weird voice part
    verdict_line: str    # the fixed, safe, factual part
    safe: str            # "yes" | "maybe" | "no" | "unknown"
    source: str          # "ai" or "offline"
    note: str = ""       # why AI was skipped (for debugging)

    def full(self):
        return f"{self.text}\n\n{self.verdict_line}"


# Remember recent replies so neither the AI nor the offline
# generator repeats itself.
_recent = deque(maxlen=6)
_last_parts = {}


# ============================================================
# 6. OFFLINE GENERATOR
# ============================================================

def _pick_different(options, key):
    """random.choice that never returns what it returned last time."""
    choices = [o for o in options if o != _last_parts.get(key)] or options
    chosen = random.choice(choices)
    _last_parts[key] = chosen
    return chosen


def _offline_reply(group, food):
    bank = OFFLINE[group]
    parts = [
        _pick_different(bank["openers"], (group, "o")),
        _pick_different(bank["middles"], (group, "m")),
        _pick_different(bank["closers"], (group, "c")),
    ]
    text = " ".join(parts)
    if random.random() < 0.5:
        text += " " + random.choice(ASIDES)

    name = display_name(food) or "food"
    return text.format(
        food=name,
        Food=name[:1].upper() + name[1:],
        thing=random.choice(WEIRD_THINGS),
    )


# ============================================================
# 7. AI GENERATOR (OpenAI)
# ============================================================

DEFAULT_MODEL = "gpt-4o-mini"

# Words that would contradict a "do not eat" / "not sure" verdict.
_BAD_FOR_UNSAFE = re.compile(
    r"\b(go ahead|devour|bon app|enjoy it|eat it up|dig in|safe to eat)\b",
    re.IGNORECASE,
)


def _ai_reply(food, state, food_conf, state_conf, verdict):
    """Returns (text, note). text is None when the AI could not be used."""
    if not os.environ.get("OPENAI_API_KEY"):
        return None, "OPENAI_API_KEY is not set"

    try:
        from openai import OpenAI
    except ImportError:
        return None, "openai package is not installed"

    mood = random.choice(MOODS)
    form = random.choice(FORMS)
    thing = random.choice(WEIRD_THINGS)
    avoid = "\n".join(f"- {r}" for r in _recent) or "- (nothing yet)"
    name = display_name(food) or "a food item"

    user_message = (
        f"Scan result: {name}, condition: {display_name(state) or 'unknown'} "
        f"(food confidence {food_conf * 100:.0f}%, "
        f"condition confidence {state_conf * 100:.0f}%).\n"
        f"{verdict['line']}\n\n"
        f"Your mood right now: {mood}.\n"
        f"Write it as: {form}.\n"
        f"Weird ingredient you may include (optional): {thing}.\n\n"
        f"AVOID these earlier replies completely:\n{avoid}"
    )

    try:
        client = OpenAI(timeout=15.0, max_retries=1)
        response = client.chat.completions.create(
            model=os.environ.get("FRESHORA_MODEL", DEFAULT_MODEL),
            messages=[
                {"role": "system", "content": PERSONA},
                {"role": "user", "content": user_message},
            ],
            temperature=1.15,
            top_p=0.95,
            presence_penalty=0.7,
            frequency_penalty=0.4,
            max_tokens=120,
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"

    text = text.strip('"').strip()

    if not text:
        return None, "AI returned an empty reply"
    if len(text) > 400:
        return None, "AI reply was too long"
    if verdict["safe"] in ("no", "unknown") and _BAD_FOR_UNSAFE.search(text):
        return None, "AI reply contradicted the verdict, discarded"

    return text, ""


# ============================================================
# 8. MAIN ENTRY POINT
# ============================================================

def make_reply(food, state, food_conf=1.0, state_conf=1.0,
               food_ok=True, state_ok=True, judgeable=True, use_ai=True):
    """
    food, state   parsed from the class name ("apple", "ripe")
    *_conf        0.0 - 1.0 confidence for the food and for the state
    food_ok       False when the model is not sure what the food is
    state_ok      False when the model is not sure how fresh it is
    judgeable     False when the model only knows ONE state for this
                  food (so it cannot tell fresh from rotten)
    use_ai        False to force the offline voice
    """
    verdict = get_verdict(food, state, food_ok, state_ok, judgeable)

    if not food_ok or not state_ok or not judgeable:
        group = "unsure"
    else:
        group = classify_state(state) or "other"

    text, note = None, "AI disabled"
    if use_ai:
        text, note = _ai_reply(food, state, food_conf, state_conf, verdict)

    source = "ai"
    if text is None:
        source = "offline"
        text = _offline_reply(group, food)

    _recent.append(text)

    return Reply(
        text=text,
        verdict_line=verdict["line"],
        safe=verdict["safe"],
        source=source,
        note=note,
    )
                   
