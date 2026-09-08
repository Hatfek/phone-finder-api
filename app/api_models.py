import re
import unicodedata

from pydantic import BaseModel, Field, field_validator

from app.schemas import REVISABLE_SLOTS, Filters, PhoneResult, Question

THREAD_ID_PATTERN = r"^[0-9a-f]{12}$"

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")


def sanitize_text(value: str) -> str:
    """Normalises free text before it can reach a search query or a prompt.

    Strips control characters and zero-width joiners, folds runs of whitespace, and
    normalises to NFKC so visually identical strings compare equal.
    """
    text = unicodedata.normalize("NFKC", value)
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    return _WHITESPACE_RE.sub(" ", text).strip()


def _require_text(value: str) -> str:
    text = sanitize_text(value)
    if not text:
        raise ValueError("must not be blank")
    return text


class StartRequest(BaseModel):
    profile: str = Field(min_length=1, max_length=4000)

    @field_validator("profile")
    @classmethod
    def _check_profile(cls, value: str) -> str:
        return _require_text(value)


class AnswerRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=500)

    @field_validator("answer")
    @classmethod
    def _check_answer(cls, value: str) -> str:
        return _require_text(value)


class ResetRequest(BaseModel):
    profile: str | None = Field(default=None, max_length=4000)

    @field_validator("profile")
    @classmethod
    def _check_profile(cls, value: str | None) -> str | None:
        return None if value is None else _require_text(value)


class ReviseRequest(BaseModel):
    """Change one earlier answer. A blank `answer` clears the slot and re-asks it."""

    slot: str
    answer: str | None = Field(default=None, max_length=500)

    @field_validator("slot")
    @classmethod
    def _check_slot(cls, value: str) -> str:
        if value not in REVISABLE_SLOTS:
            raise ValueError(f"slot must be one of {', '.join(REVISABLE_SLOTS)}")
        return value

    @field_validator("answer")
    @classmethod
    def _check_answer(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text or None


class SlotOption(BaseModel):
    """One revisable filter, as the client should draw it."""

    slot: str
    label: str
    question: str
    options: list[str]
    hint: str = ""


class SlotCatalogue(BaseModel):
    slots: list[SlotOption] = []


class TurnResponse(BaseModel):
    thread_id: str
    step: str
    done: bool
    price_reference: str
    notice: str | None = None
    ask_question: Question | None = None
    show_phones: list[PhoneResult] = []
    filters: Filters = Filters()
    degraded: list[str] = []
    warnings: list[str] = []
