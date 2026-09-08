from app.schemas import Question

FALLBACK_QUESTIONS: dict[str, Question] = {
    "os": Question(
        question="Which system do you get on with?",
        options=["Android", "iOS", "No preference"],
        hint="Or tell us what you use now and why",
        slot="os",
    ),
    "max_price_usd": Question(
        question="What are you comfortable spending?",
        options=["Under $300", "$300-600", "$600-900", "No limit"],
        hint="Or give us a number",
        slot="max_price_usd",
    ),
    "priority": Question(
        question="What does this phone need to be good at?",
        options=["Camera", "Battery", "Gaming", "A bit of everything"],
        hint="Or describe what you use it for most",
        slot="priority",
    ),
    "size": Question(
        question="How big do you want it in the hand?",
        options=["Compact", "Large screen", "Doesn't matter"],
        hint="Or tell us how you carry it",
        slot="size",
    ),
}

SLOT_LABELS: dict[str, str] = {
    "os": "System",
    "max_price_usd": "Budget",
    "priority": "Priority",
    "size": "Size",
    "brands": "Brand",
    "exclude_brands": "Excluded brand",
}

BRAND_QUESTIONS: dict[str, Question] = {
    "brands": Question(
        question="Which maker should we stick to?",
        options=["Samsung", "Google", "Apple", "Any maker"],
        hint="Or name the one you want",
        slot="brands",
    ),
    "exclude_brands": Question(
        question="Which maker should we keep out?",
        options=["Not Apple", "Not Samsung", "Nothing is off the table"],
        hint="Or name the one you would not buy",
        slot="exclude_brands",
    ),
}


def revision_questions() -> dict[str, Question]:
    """The canned question behind every revisable filter, keyed by slot."""
    return {**FALLBACK_QUESTIONS, **BRAND_QUESTIONS}


PRICE_BRACKETS = [(0, 300, "Under $300"), (300, 600, "$300-600"), (600, 900, "$600-900")]


def price_options(prices: list[float]) -> list[str]:
    if not prices:
        return FALLBACK_QUESTIONS["max_price_usd"].options
    reachable = [label for lo, hi, label in PRICE_BRACKETS if any(lo <= p < hi for p in prices)]
    reachable.append("No limit")
    return reachable if len(reachable) >= 2 else FALLBACK_QUESTIONS["max_price_usd"].options
