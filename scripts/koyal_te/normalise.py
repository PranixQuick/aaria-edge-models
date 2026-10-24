"""
Text normalisation for voice measurement.

WHY THIS FILE EXISTS
--------------------
NVIDIA's Indic and Code-Mixed session is explicit that measuring accuracy
without normalising BOTH sides produces a number that is mostly formatting:

    "The evaluation script applies no normalisation of its own - it is a bare
     whitespace split, case and punctuation sensitive. Scoring a
     punctuation-aware model against lowercase references reports catastrophic
     error that is entirely formatting."

So before any word or character error rate is computed, reference and
hypothesis pass through exactly the same function. Any asymmetry here is
measured as model error, which is why there is one normaliser and not two.

WHAT THE FIRST REAL MEASUREMENT CAUGHT IN THIS FILE
---------------------------------------------------
The first live run scored this pair at 50% word error:

    reference   "Paid one thousand eight hundred and fifty rupees for petrol"
    hypothesis  "Paid Rs.1850 for petrol"

The recogniser was completely correct. The error was entirely mine: number
words were mapped one at a time, so "one thousand eight hundred and fifty"
became "1 1000 8 100 and 50" instead of "1850", and a currency symbol glued to
its digits was left untouched.

That is precisely the failure NVIDIA warned about, built into the tool whose
whole purpose is to avoid it. Two fixes, both below:

  1. Number words are COMPOSED into a value, not substituted individually.
  2. Currency markers are removed from both sides, so "1850 rupees" and
     "Rs.1850" compare equal on the amount.

Removing the currency marker is a deliberate trade: it means we cannot detect
a system that drops the word "rupees". For a single-currency product that is
worth it, because the alternative is scoring correct transcriptions as errors
and then chasing a model problem that does not exist.

WHAT IT DOES, AND DELIBERATELY DOES NOT DO
------------------------------------------
Does:   case folding, punctuation stripping, whitespace collapsing, Indic
        danda handling, number composition, currency handling.
Does NOT: transliterate between scripts. NVIDIA describe script confusion as a
        distinct failure ("heard correctly, written in the wrong script") that
        must be measured, not normalised away. If we transliterated here we
        would hide the exact defect we are trying to detect.
"""

import re
import unicodedata

# ---------------------------------------------------------------- scripts
#
# Which alphabet a piece of text is written in. This is NOT transliteration -
# see the docstring above for why we refuse to do that. It exists so the
# harness can tell two different things apart that currently score identically:
#
#   1. The engine heard Telugu and wrote it in Devanagari.
#      A real failure. Must score as one.
#
#   2. The engine heard Telugu and wrote it in Telugu, while our answer sheet
#      happened to be typed in English letters.
#      Not a failure at all. Currently scores 100% word error.
#
# On the committed run of 14 September, every code-mixed utterance was case 2
# and the slice reported 95% word error on four perfect transcriptions.
_SCRIPT_BLOCKS = (
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("odia", 0x0B00, 0x0B7F),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
)


def dominant_script(text):
    """The script most of this text's letters belong to, or None.

    Counts letters only: digits, punctuation and spaces are shared between
    scripts and would drag every verdict toward whichever side had more of
    them. Returns None for text with no letters at all - an empty
    transcription, or a bare amount - because "no script" is not a script and
    must not be compared as though it were.
    """
    if not text:
        return None
    counts = {}
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        name = "latin" if cp < 0x0250 else None
        if name is None:
            for block, lo, hi in _SCRIPT_BLOCKS:
                if lo <= cp <= hi:
                    name = block
                    break
        if name is None:
            name = "other"
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]

# Small number words -> value. Deliberately explicit rather than a general
# grammar: NVIDIA note Indic normalisation grammars are weeks of work per
# language and that "formatting moves to the application" is a legitimate
# choice at this stage.
_SMALL = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    # Romanised Hindi/Telugu as our users actually say them
    "ek": 1, "do": 2, "teen": 3, "chaar": 4, "char": 4, "paanch": 5,
    "panch": 5, "che": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10,
    "okati": 1, "rendu": 2, "moodu": 3, "naalugu": 4, "aidu": 5,
    # Telugu script. The reference sheets write amounts as digits; a correct
    # transcription spells them, and without these the scorer calls a correct
    # transcription an error. Measured on the 15 Sept FLEURS run: at least five
    # of fifty Telugu utterances were penalised for being right.
    #
    # NOTE ON WHAT IS DELIBERATELY ABSENT: "ఒక" is not here. It is the
    # adjectival "a/an", not the numeral - "ఒక కొత్త చొరవ" is "a new
    # initiative", not "1 new initiative". The numeral is "ఒకటి", which is
    # here. Adding "ఒక" would trade one class of false error for another.
    "సున్నా": 0, "ఒకటి": 1, "రెండు": 2, "మూడు": 3, "నాలుగు": 4, "నాల్గు": 4,
    "ఐదు": 5, "అయిదు": 5, "ఆరు": 6, "ఏడు": 7, "ఎనిమిది": 8, "తొమ్మిది": 9,
    "పది": 10, "పదకొండు": 11, "పన్నెండు": 12, "పదమూడు": 13, "పద్నాలుగు": 14,
    "పదిహేను": 15, "పదహారు": 16, "పదిహేడు": 17, "పద్దెనిమిది": 18,
    "పంతొమ్మిది": 19, "ఇరవై": 20, "ముప్పై": 30, "ముప్పయి": 30,
    "నలభై": 40, "నలబై": 40, "యాభై": 50, "ఏభై": 50, "అరవై": 60,
    "డెబ్బై": 70, "డెబ్భై": 70, "ఎనభై": 80, "ఎనబై": 80,
    "తొంభై": 90, "తొంబై": 90,
    # Devanagari. Same defect, same fix: "रमेश से पाँच सौ रुपये" against a
    # reference of "500" scored 43% word error on a perfect transcription.
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5,
    "छह": 6, "छः": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
    "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15,
    "पन्द्रह": 15, "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19,
    "बीस": 20, "तीस": 30, "चालीस": 40, "पचास": 50, "साठ": 60,
    "सत्तर": 70, "अस्सी": 80, "नब्बे": 90,
}

# Scale words. 100 multiplies what came before; the rest close a group.
_SCALES = {
    "hundred": 100, "thousand": 1000, "lakh": 100000, "lakhs": 100000,
    "crore": 10000000, "crores": 10000000, "million": 1000000,
    "sau": 100, "vandha": 100, "hazaar": 1000, "hazar": 1000, "vela": 1000,
    # Telugu script scales. "ఎనిమిది లక్షల" -> 800000.
    "వంద": 100, "వందల": 100, "వందలు": 100,
    "వేయి": 1000, "వెయ్యి": 1000, "వేల": 1000, "వేలు": 1000,
    "లక్ష": 100000, "లక్షల": 100000, "లక్షలు": 100000,
    "కోటి": 10000000, "కోట్ల": 10000000, "కోట్లు": 10000000,
    # Devanagari scales. "पाँच सौ" -> 500.
    "सौ": 100, "हज़ार": 1000, "हजार": 1000,
    "लाख": 100000, "करोड़": 10000000, "करोड": 10000000,
}

# Removed from both sides. See the module docstring for why the currency
# marker goes rather than being canonicalised.
_CURRENCY_WORDS = {
    "rupees", "rupee", "rupaye", "rupay", "rupaiya", "rs", "inr", "₹",
    # Same trade as the Latin forms above, in the scripts the product uses.
    # Without these, "पाँच सौ रुपये" against "₹500" still differs by one word
    # even once the number composes, because one side kept its currency word
    # and the other had it stripped.
    "रुपये", "रुपए", "रुपया", "रूपये", "रु",
    "రూపాయలు", "రూపాయిలు", "రూపాయి", "రూ",
}

_JOINERS = {"and"}

# Dotted abbreviations must collapse BEFORE punctuation is stripped, or
# "a.m." becomes two tokens ("a", "m") while "am" stays one - which scores as
# a word error on text that is identical in meaning.
ABBREV = [
    (re.compile(r"\ba\.\s*m\.?", re.I), "am"),
    (re.compile(r"\bp\.\s*m\.?", re.I), "pm"),
    (re.compile(r"\bo\'\s*clock\b", re.I), "oclock"),
    (re.compile(r"\bno\.\s*", re.I), "number "),
    # "Rs.1850" / "Rs 1850" / "₹1,850" -> the amount, currency word separated
    # so the token filter below can drop it.
    (re.compile(r"(?:₹|\brs\.?\b|\binr\b)\s*(?=\d)", re.I), "rupees "),
]

# Punctuation to strip. Includes the Devanagari/Telugu danda and double danda.
# The rupee sign is handled above and then dropped as a currency word.
_PUNCT = re.compile(r"[.,!?;:\"'`()\[\]{}<>/\\|@#$%^&*_+=~।॥॰-]")
_WS = re.compile(r"\s+")
_DIGITS_WITH_SEPARATORS = re.compile(r"(?<=\d),(?=\d)")


def _compose_numbers(tokens):
    """Fold runs of number words into a single value.

    "one thousand eight hundred and fifty" -> "1850"
    "chaar hazaar"                          -> "4000"

    A joiner ("and") is only swallowed when a number word follows it, so
    "salt and pepper" is untouched.
    """
    out = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok not in _SMALL and tok not in _SCALES:
            out.append(tok)
            i += 1
            continue

        total = 0
        current = 0
        seen_any = False
        while i < n:
            t = tokens[i]
            if t in _SMALL:
                current += _SMALL[t]
                seen_any = True
                i += 1
            elif t in _SCALES:
                scale = _SCALES[t]
                if scale == 100:
                    current = (current or 1) * 100
                else:
                    total += (current or 1) * scale
                    current = 0
                seen_any = True
                i += 1
            elif t in _JOINERS and i + 1 < n and (
                tokens[i + 1] in _SMALL or tokens[i + 1] in _SCALES
            ):
                i += 1          # swallow "and" inside a number
            else:
                break
        if seen_any:
            out.append(str(total + current))
    return out


def normalise(text: str) -> str:
    """
    Canonical form used on BOTH reference and hypothesis before scoring.

    Idempotent: normalise(normalise(x)) == normalise(x).
    """
    if not text:
        return ""

    # NFKC folds compatibility forms and composes Indic vowel signs
    # consistently, so the same word is one sequence.
    t = unicodedata.normalize("NFKC", text)
    t = t.casefold()
    for pat, rep in ABBREV:
        t = pat.sub(rep, t)
    t = _DIGITS_WITH_SEPARATORS.sub("", t)   # 1,850 -> 1850
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()

    tokens = [tok for tok in t.split(" ") if tok]
    tokens = _compose_numbers(tokens)
    tokens = [tok for tok in tokens if tok not in _CURRENCY_WORDS]
    return " ".join(tokens)


def _levenshtein(a, b) -> int:
    """Edit distance over any sequence. Used for both WER and CER."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate on normalised text. 0.0 is perfect; >1.0 is possible."""
    r = normalise(reference).split()
    h = normalise(hypothesis).split()
    if not r:
        return 0.0 if not h else 1.0
    return _levenshtein(r, h) / len(r)


def cer(reference: str, hypothesis: str) -> float:
    """
    Character error rate on normalised text.

    NVIDIA are emphatic that this is the more honest number for Indic:
    Dravidian morphology packs into single words what English spreads over
    five, the same utterance is validly written joined or split, and
    "character error responds before word error does". For Telugu we lead
    with this figure.
    """
    r = normalise(reference).replace(" ", "")
    h = normalise(hypothesis).replace(" ", "")
    if not r:
        return 0.0 if not h else 1.0
    return _levenshtein(r, h) / len(r)


def compute_edits_and_lengths(reference: str, hypothesis: str) -> dict:
    """
    Compute word edits, reference word count, character edits, reference character count
    on normalised text, along with per-clip error rates.
    """
    r_words = normalise(reference).split()
    h_words = normalise(hypothesis).split()
    w_edits = _levenshtein(r_words, h_words)
    w_len = len(r_words)

    r_chars = normalise(reference).replace(" ", "")
    h_chars = normalise(hypothesis).replace(" ", "")
    c_edits = _levenshtein(r_chars, h_chars)
    c_len = len(r_chars)

    w_err = (w_edits / w_len) if w_len > 0 else (0.0 if not h_words else 1.0)
    c_err = (c_edits / c_len) if c_len > 0 else (0.0 if not h_chars else 1.0)

    return {
        "word_edits": w_edits,
        "ref_words": w_len,
        "clip_wer": w_err,
        "char_edits": c_edits,
        "ref_chars": c_len,
        "clip_cer": c_err,
    }


if __name__ == "__main__":

    # Cases that must score ZERO: identical content, different formatting.
    same = [
        ("Remind me to pay the electricity bill tomorrow at 10 a.m.",
         "remind me to pay the electricity bill tomorrow at 10 am"),
        # The pair the first live measurement scored at 50% error.
        ("Paid one thousand eight hundred and fifty rupees for petrol at Shell via UPI",
         "Paid Rs.1850 for petrol at Shell via UPI."),
        ("Paid one thousand eight hundred and fifty rupees",
         "Paid ₹1,850"),
        ("Suresh ko chaar hazaar rupaye diye", "Suresh ko 4000 rupees diye"),
        ("Ramesh se paanch sau rupaye aaye", "Ramesh se 500 rupees aaye"),
        ("Ravi ki forty-five thousand rupees invoice cheyyi",
         "Ravi ki 45000 rupees invoice cheyyi"),
        ("రేపు ఉదయం పది గంటలకు కరెంట్ బిల్లు కట్టాలి।",
         "రేపు ఉదయం పది గంటలకు కరెంట్ బిల్లు కట్టాలి"),
    ]
    # Cases that must NOT be collapsed - a real difference must stay visible.
    differ = [
        ("Paid one thousand eight hundred and fifty rupees", "Paid 1500 rupees"),
        ("salt and pepper", "salt pepper"),
        ("call Ravi", "call Kiran"),
    ]

    bad = 0
    for ref, hyp in same:
        w = wer(ref, hyp)
        if w != 0:
            bad += 1
            print(f"FAIL WER={w:.3f}  {ref[:52]}")
            print(f"       ref -> {normalise(ref)}")
            print(f"       hyp -> {normalise(hyp)}")
        else:
            print(f"ok   WER={w:.3f}  {ref[:52]}")
    for ref, hyp in differ:
        w = wer(ref, hyp)
        if w == 0:
            bad += 1
            print(f"FAIL (collapsed a real difference)  {ref[:52]}")
        else:
            print(f"ok   WER={w:.3f}  {ref[:52]}")

    assert normalise(normalise(same[0][0])) == normalise(same[0][0]), "not idempotent"
    assert normalise(normalise(same[1][0])) == normalise(same[1][0]), "not idempotent"
    print("\nidempotency: OK")
    print("failures:", bad)
    raise SystemExit(1 if bad else 0)
