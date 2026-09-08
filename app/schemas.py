from typing import Literal, TypedDict

from pydantic import BaseModel, Field, field_validator

OS = Literal["Android", "iOS", "any"]
FitFilter = Literal["brand", "budget", "priority", "size", "os", "market"]
Priority = Literal["camera", "battery", "gaming", "allround"]
Size = Literal["compact", "large", "any"]
SLOTS = ("os", "max_price_usd", "priority", "size")
BRAND_SLOTS = ("brands", "exclude_brands")
REVISABLE_SLOTS = (*SLOTS, *BRAND_SLOTS)
DEFAULTS: dict[str, object] = {
    "os": "any",
    "max_price_usd": None,
    "priority": None,
    "size": "any",
}


class SlotFilters(BaseModel):
    os: OS = "any"
    max_price_usd: int | None = None
    priority: Priority | None = None
    size: Size = "any"


class Filters(SlotFilters):
    brands: list[str] = Field(default_factory=list)
    exclude_brands: list[str] = Field(default_factory=list)
    soft_brands: list[str] = Field(default_factory=list)
    resolved: list[str] = Field(default_factory=list)

    def is_set(self, slot: str) -> bool:
        return slot in self.resolved or getattr(self, slot) != DEFAULTS[slot]

    def is_revisable(self, slot: str) -> bool:
        """True when the shopper has something to change: a live constraint or a settled slot."""
        if slot in BRAND_SLOTS:
            return bool(getattr(self, slot))
        return slot in SLOTS and self.is_set(slot)

    def clear_slot(self, slot: str) -> "Filters":
        """Drop one constraint and unsettle it, leaving every other slot exactly as it was."""
        data = self.model_dump()
        data[slot] = [] if slot in BRAND_SLOTS else DEFAULTS[slot]
        data["resolved"] = [s for s in data["resolved"] if s != slot]
        return Filters.model_validate(data)

    def unfilled_slots(self) -> list[str]:
        return [s for s in SLOTS if not self.is_set(s)]

    def fill_from(self, other: SlotFilters, skip: tuple[str, ...] = ()) -> "Filters":
        """Take values from `other` for slots the shopper has not settled yet."""
        data = self.model_dump()
        for slot in SLOTS:
            if slot in skip or self.is_set(slot):
                continue
            value = getattr(other, slot)
            if value != DEFAULTS[slot]:
                data[slot] = value
        return Filters.model_validate(data)

    def with_slot(self, slot: str, value: object, *, resolve: bool = True) -> "Filters":
        data = self.model_dump()
        data[slot] = value
        if resolve and slot not in data["resolved"]:
            data["resolved"] = [*data["resolved"], slot]
        return Filters.model_validate(data)


class Question(BaseModel):
    question: str
    options: list[str] = Field(min_length=2, max_length=5)
    hint: str = ""
    slot: str

    @field_validator("options")
    @classmethod
    def strip_blank(cls, v: list[str]) -> list[str]:
        cleaned = [o.strip() for o in v if o and o.strip()]
        if len(cleaned) < 2:
            raise ValueError("need at least 2 options")
        return cleaned


class FitFactor(BaseModel):
    """One filter's contribution to a phone's rank, in the shopper's own terms."""

    filter: FitFilter
    label: str
    points: float


class PhoneResult(BaseModel):
    name: str
    price_usd: float = Field(gt=0, lt=10000)
    retailer: str
    url: str
    image: str | None = None
    fetched_at: str
    why: str = ""
    fit_summary: str = ""
    fit_factors: list[FitFactor] = Field(default_factory=list)


class ExtractedPhone(BaseModel):
    name: str
    price_usd: float | None = None


class ExtractionBatch(BaseModel):
    phones: list[ExtractedPhone] = Field(default_factory=list)


class PlannerOutput(BaseModel):
    filters: SlotFilters
    next_question: Question | None = None
    reasoning: str = ""


class SearchQueries(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=4)


class WhyLine(BaseModel):
    name: str
    why: str


class WhyBatch(BaseModel):
    lines: list[WhyLine] = Field(default_factory=list)


class GraphState(TypedDict, total=False):
    profile: str
    answers: list[str]
    filters: dict
    results: list[dict]
    question: dict | None
    question_count: int
    queries: list[str]
    urls: list[str]
    done: bool
    notice: str | None
    widened: bool
    relaxed: bool
    degraded: list[str]
    merged_answers: int
