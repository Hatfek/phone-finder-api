import re

BRAND_ALIASES: dict[str, tuple[str, ...]] = {
    "samsung": ("samsung", "galaxy"),
    "apple": ("apple", "iphone"),
    "google": ("google", "pixel"),
    "motorola": ("motorola", "moto"),
    "oneplus": ("oneplus", "one plus"),
    "xiaomi": ("xiaomi", "redmi", "poco", "mi "),
    "nothing": ("nothing phone",),
    "sony": ("sony", "xperia"),
    "asus": ("asus", "rog phone", "zenfone"),
    "honor": ("honor",),
    "huawei": ("huawei",),
    "oppo": ("oppo",),
    "vivo": ("vivo",),
    "realme": ("realme",),
    "nokia": ("nokia",),
    "tcl": ("tcl",),
    "tecno": ("tecno",),
    "infinix": ("infinix",),
}

BRAND_OS = {"apple": "iOS"}

_ALIAS_RE = {
    brand: re.compile(
        r"\b(" + "|".join(re.escape(a.strip()) for a in aliases) + r")\b", re.IGNORECASE
    )
    for brand, aliases in BRAND_ALIASES.items()
}

_CLAUSE_SPLIT = re.compile(r"[.;,!?\n]|\bbut\b|\bthough\b|\bhowever\b", re.IGNORECASE)

_NEGATION_CUES = (
    "not ", "no ", "never", "avoid", "don't", "dont", "do not", "doesn't", "hate",
    "dislike", "except", "other than", "anything but", "rather not", "no more",
    "sick of", "tired of", "away from", "switch from", "switching from", "moving from",
    "moved from", "coming from", "upgrade from", "upgrading from", "used to have",
    "used to use", "leaving", "give up on", "anti", "won't", "wont ", "nothing from",
)

_PREFERENCE_CUES = (
    "prefer", "only", "stick with", "stick to", "stay with", "loyal", "love",
    "i like", "we like", "fan of", "always buy", "always been", "want a", "want an",
    "wants a", "looking for a", "looking for an", "must be", "has to be", "needs to be",
    "should be", "keen on", "set on", "go with", "buy another", "another one",
    "replace it with", "again",
)


def _clauses(text: str) -> list[str]:
    return [c for c in _CLAUSE_SPLIT.split(text or "") if c and c.strip()]


def _has_cue(fragment: str, cues: tuple[str, ...]) -> bool:
    low = fragment.lower()
    return any(cue in low for cue in cues)


def detect_brands(text: str, *, direct: bool = False) -> tuple[list[str], list[str], list[str]]:
    """Return (prefer, exclude, mentioned) brands read from free text.

    `prefer` and `exclude` are hard constraints and need an explicit cue, unless
    `direct` marks the text as a reply to a question, where naming a brand is the answer.
    `mentioned` is every other brand seen, which only nudges ranking.
    """
    prefer: list[str] = []
    exclude: list[str] = []
    mentioned: list[str] = []

    for clause in _clauses(text):
        for brand, pattern in _ALIAS_RE.items():
            match = pattern.search(clause)
            if not match:
                continue
            before = clause[: match.start()]
            if _has_cue(before, _NEGATION_CUES):
                bucket = exclude
            elif direct or _has_cue(clause, _PREFERENCE_CUES):
                bucket = prefer
            else:
                bucket = mentioned
            if brand not in bucket:
                bucket.append(brand)

    prefer = [b for b in prefer if b not in exclude]
    mentioned = [b for b in mentioned if b not in prefer and b not in exclude]
    return prefer, exclude, mentioned


def brand_of(name: str) -> str | None:
    for brand, pattern in _ALIAS_RE.items():
        if pattern.search(name):
            return brand
    return None


def matches_brands(name: str, prefer: list[str], exclude: list[str]) -> bool:
    for brand in exclude:
        if _ALIAS_RE[brand].search(name):
            return False
    if not prefer:
        return True
    return any(_ALIAS_RE[brand].search(name) for brand in prefer)


def implied_os(prefer: list[str]) -> str | None:
    if not prefer:
        return None
    systems = {BRAND_OS.get(brand, "Android") for brand in prefer}
    return systems.pop() if len(systems) == 1 else None


def query_words(prefer: list[str]) -> str:
    return " ".join(prefer[:2])
