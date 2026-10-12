# ============================================================
# FRESHORA - PERSONALITY
#
# Two jobs, kept strictly apart:
#
#   1. THE VERDICT  (VERDICTS below)
#      Fixed text written by YOU. The AI can never change it,
#      so a rotten banana can never be called "safe".
#
#   2. THE VOICE
#      A fresh, weird reply every time. Written by an OpenAI model
#      when OPENAI_API_KEY is set, otherwise by the built-in
#      offline generator (no internet, no cost, never crashes).
#
# The final message = voice + verdict line.
# ============================================================

import os
import random
import re
from collections import deque
from dataclasses import dataclass


# ============================================================
# 1. VERDICTS (facts, never written by the AI)
# ============================================================
# safe: "yes" | "maybe" | "no" | "unknown"   (the app uses it for colours)

VERDICTS = {
    "unripe": {
        "safe": "yes",
        "line": "Verdict: UNRIPE. Safe to eat, but firm and starchy. "
                "Give it a few days at room temperature.",
    },
    "ripe": {
        "safe": "yes",
        "line": "Verdict: RIPE. Safe to eat. This is peak banana.",
    },
    "overripe": {
        "safe": "maybe",
        "line": "Verdict: OVERRIPE. Usually fine if there is no mould, "
                "leaking or sour smell. Best for smoothies or banana bread.",
    },
    "rotten": {
        "safe": "no",
        "line": "Verdict: ROTTEN. Do NOT eat it. Bin it or compost it.",
    },
}

UNSURE = {
    "safe": "unknown",
    "line": "Verdict: NOT SURE. Retake the photo in good light with the "
            "whole banana in frame, and trust your eyes and nose.",
}

UNKNOWN_LABEL = {
    "safe": "unknown",
    "line": "Verdict: UNKNOWN. Do not rely on this scan. Check the food "
            "yourself (look, smell, texture).",
}


def get_verdict(label, reliable=True):
    """Fixed verdict for a class. Low confidence always wins."""
    if not reliable:
        return UNSURE
    return VERDICTS.get(label, UNKNOWN_LABEL)


# ============================================================
# 2. THE PERSONA (used by the AI model)
# ============================================================

PERSONA = """You are FRESHORA, the resident oracle of a food-scanner app.
You are a dramatic, slightly unhinged fruit-bowl prophet. You treat
bananas as if they were tiny celebrities, rivals, or exes. You are
funny, specific, and strange, never corporate, never generic.

Rules:
- 2 to 3 short sentences. Maximum 45 words.
- Be original every time. Never reuse a phrase from the AVOID list.
- Never say "As an AI", never apologise, no emojis, no hashtags.
- The VERDICT you are given is a fact. Never contradict it. Never
  suggest eating something whose verdict is ROTTEN or NOT SURE.
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
    "a fake weather report for the fruit bowl",
    "a one-line courtroom ruling",
    "a sports commentator losing their mind",
    "a message from the banana itself",
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
# 3. OFFLINE VOICE (no internet needed)
# ============================================================
# reply = opener + middle + closer (+ sometimes an aside).
# 6 x 6 x 6 per class = 216 combinations, times the asides.

OFFLINE = {
    "unripe": {
        "openers": [
            "Ah. A banana still in its teenage phase.",
            "Green. Stubborn. Wearing its school uniform.",
            "This one is not ready, and it knows it.",
            "I sense a banana that skipped the meeting.",
            "Behold: a fruit with commitment issues.",
            "The peel whispers, 'not yet, not yet.'",
        ],
        "middles": [
            "It is eatable, but it will taste like chalk with ambitions.",
            "Bite it now and it bites back, spiritually.",
            "It's safe, just deeply unimpressed with itself.",
            "Give it time. Even legends start out starchy.",
            "It has the texture of a determined eraser.",
            "Nothing wrong here, only a lack of sweetness and drama.",
        ],
        "closers": [
            "Check back in a few days.",
            "Leave it on the counter and leave it alone.",
            "Patience, little scanner.",
            "Do not poke it. It hates being poked.",
            "Ripeness cannot be rushed, only respected.",
            "Come back when it's wearing yellow.",
        ],
    },
    "ripe": {
        "openers": [
            "Oh. OH. This is the one.",
            "Golden hour has arrived in peel form.",
            "I have seen empires fall, but never a banana this on time.",
            "This banana walked in like it owns the bowl.",
            "Sunshine, but edible.",
            "Applause. The fruit has peaked.",
        ],
        "middles": [
            "Sweet, soft, structurally confident.",
            "It's safe, fragrant, and a little smug about it.",
            "Peak banana. The kind poets write tragic letters about.",
            "It is exactly as ready as a banana can legally be.",
            "Zero drama, maximum sweetness.",
            "Go on. The universe has signed the permission slip.",
        ],
        "closers": [
            "Eat it before it gets ideas.",
            "Do not waste this moment on a scan.",
            "Peel, chew, nod slowly.",
            "This is your sign. It's yellow.",
            "Banana. Mouth. Now.",
            "Enjoy the good timeline.",
        ],
    },
    "overripe": {
        "openers": [
            "This banana has seen things. Mostly Tuesdays.",
            "Spotted, soft, and philosophical.",
            "A banana in its dramatic third act.",
            "Freckles everywhere. A true veteran.",
            "The fruit bowl's retired jazz musician has arrived.",
            "Ooh, this one is basically a sweet puddle with a plan.",
        ],
        "middles": [
            "It's probably fine if there's no mould or sour smell.",
            "Too mushy for a dignified snack, perfect for banana bread.",
            "Sweetness at maximum, structure at a minimum.",
            "Smoothie destiny is calling.",
            "Sniff first. Your nose is the real judge here.",
            "Still has dignity, just not much shape.",
        ],
        "closers": [
            "Mash it. Bake it. Respect it.",
            "Freeze it before it files a complaint.",
            "Check for mould, then proceed.",
            "Its final form is bread.",
            "Do not wait another day.",
            "Use it today or lose it.",
        ],
    },
    "rotten": {
        "openers": [
            "No. Absolutely not.",
            "This banana has left the building and taken the lease.",
            "I'm sorry for your loss. The banana is gone.",
            "That is not fruit anymore, that is a cautionary tale.",
            "The prophecy is dark, and it smells worse.",
            "This is a banana funeral.",
        ],
        "middles": [
            "Whatever lives in there is paying no rent.",
            "It has crossed a line that cannot be uncrossed.",
            "Even the fruit flies are checking their options.",
            "It has become a science project without consent.",
            "The bowl wants a divorce.",
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
    "unsure": {
        "openers": [
            "The fog is thick today.",
            "I squinted. The banana squinted back.",
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
            "Try again with the whole banana in view.",
            "Find better light and a plain background.",
            "Take another photo, no dramatic angles.",
            "Use your nose in the meantime.",
            "Another shot, please.",
            "Closer, brighter, calmer.",
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
# 4. REPLY OBJECT
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
# 5. OFFLINE GENERATOR
# ============================================================

def _pick_different(options, key):
    """random.choice that never returns what it returned last time."""
    choices = [o for o in options if o != _last_parts.get(key)] or options
    chosen = random.choice(choices)
    _last_parts[key] = chosen
    return chosen


def _offline_reply(group):
    bank = OFFLINE[group]
    parts = [
        _pick_different(bank["openers"], (group, "o")),
        _pick_different(bank["middles"], (group, "m")),
        _pick_different(bank["closers"], (group, "c")),
    ]
    text = " ".join(parts)
    if random.random() < 0.5:
        text += " " + random.choice(ASIDES).format(thing=random.choice(WEIRD_THINGS))
    return text


# ============================================================
# 6. AI GENERATOR (OpenAI)
# ============================================================

DEFAULT_MODEL = "gpt-4o-mini"

# Words that would contradict a "do not eat" verdict.
_BAD_FOR_UNSAFE = re.compile(
    r"\b(go ahead|devour|bon app|enjoy it|eat it up|dig in|safe to eat)\b",
    re.IGNORECASE,
)


def _ai_reply(label, confidence, verdict, group):
    """Returns (text, error). text is None when the AI could not be used."""
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

    user_message = (
        f"Scan result: the banana is {label.upper()} "
        f"(confidence {confidence * 100:.0f}%).\n"
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
# 7. MAIN ENTRY POINT
# ============================================================

def make_reply(label, confidence, reliable=True, use_ai=True):
    """
    label       class name from the model ("ripe", "rotten", ...)
    confidence  0.0 - 1.0
    reliable    False when the model was not confident enough
    use_ai      False to force the offline voice
    """
    verdict = get_verdict(label, reliable)

    if not reliable:
        group = "unsure"
    elif label in OFFLINE:
        group = label
    else:
        group = "unsure"

    text, note = None, "AI disabled"
    if use_ai:
        text, note = _ai_reply(label, confidence, verdict, group)

    source = "ai"
    if text is None:
        source = "offline"
        text = _offline_reply(group)

    _recent.append(text)

    return Reply(
        text=text,
        verdict_line=verdict["line"],
        safe=verdict["safe"],
        source=source,
        note=note,
  )
  
