import asyncio
import logging
from dataclasses import dataclass
from typing import Generic, TypeVar

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.llm import get_llm

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

RETRIES = 2
REPAIR_HINT = (
    "Your previous reply was not valid for the required schema ({error}). "
    "Reply again with a single JSON object only. No prose, no markdown fence."
)


@dataclass(frozen=True)
class StructuredResult(Generic[T]):
    """A parsed value, or the reason there is none: bad answer vs. no answer at all."""

    value: T | None = None
    unreachable: bool = False

    def __bool__(self) -> bool:
        return self.value is not None


async def structured_call(
    schema: type[T],
    system: str,
    user: str,
    *,
    run_name: str,
    temperature: float = 0.0,
) -> StructuredResult[T]:
    llm = get_llm(temperature).with_structured_output(schema, method="json_schema").with_config(
        {"run_name": run_name}
    )
    messages = [SystemMessage(system), HumanMessage(user)]
    timeout = get_settings().llm_timeout

    for attempt in range(RETRIES + 1):
        try:
            result = await asyncio.wait_for(llm.ainvoke(messages), timeout=timeout)
            if isinstance(result, schema):
                return StructuredResult(result)
            return StructuredResult(schema.model_validate(result))
        except (ValidationError, ValueError, TypeError) as exc:
            log.warning("structured_call %s attempt %s rejected: %s", run_name, attempt, exc)
            if attempt == RETRIES:
                return StructuredResult()
            messages.append(HumanMessage(REPAIR_HINT.format(error=str(exc)[:200])))
        except asyncio.TimeoutError:
            log.error("structured_call %s timed out after %ss", run_name, timeout)
            return StructuredResult(unreachable=True)
        except (httpx.HTTPError, OSError) as exc:
            log.error("structured_call %s could not reach the model: %s", run_name, exc)
            return StructuredResult(unreachable=True)
        except Exception:
            log.exception("structured_call %s failed unexpectedly", run_name)
            return StructuredResult(unreachable=True)
    return StructuredResult()
