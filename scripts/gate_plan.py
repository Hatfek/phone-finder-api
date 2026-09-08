import asyncio
import os
import sys
import time

sys.path.insert(0, ".")

from app.config import Settings, get_settings

if "--quality" in sys.argv:
    os.environ["OLLAMA_MODEL"] = Settings().ollama_model_quality

from app.nodes import intake, plan

PROFILE = (
    "I'm 34, a field engineer. I drive between sites all day, phone is on maps and calls "
    "for hours and I'm rarely near a charger. I take a lot of photos of equipment. "
    "I keep a phone about four years. I'd rather not go over 600 dollars."
)

RUNS = 10


async def main() -> int:
    print(f"model: {get_settings().ollama_model}")

    ok_intake = 0
    ok_plan = 0
    latencies = []

    for i in range(RUNS):
        t0 = time.perf_counter()
        seed = await intake({"profile": PROFILE})
        filters = seed["filters"]
        intake_valid = isinstance(filters, dict) and "os" in filters
        ok_intake += intake_valid

        out = await plan({**seed, "profile": PROFILE, "results": []})
        question = out.get("question")
        plan_valid = bool(question) and 2 <= len(question["options"]) <= 5
        ok_plan += plan_valid
        latencies.append(time.perf_counter() - t0)

        print(
            f"{i + 1:>2}. intake={'ok ' if intake_valid else 'FAIL'} "
            f"plan={'ok ' if plan_valid else 'FAIL'} "
            f"{latencies[-1]:.1f}s  filters={filters}  "
            f"q={question['question'] if question else None}"
        )

    print(f"\nintake {ok_intake}/{RUNS}   plan {ok_plan}/{RUNS}")
    print(f"latency avg {sum(latencies) / len(latencies):.1f}s  max {max(latencies):.1f}s")
    passed = ok_intake == RUNS and ok_plan == RUNS
    print("GATE PASS" if passed else "GATE FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
