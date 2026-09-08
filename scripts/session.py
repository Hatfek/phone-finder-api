import asyncio
import sys
import time
import uuid

sys.path.insert(0, ".")

from langgraph.types import Command

from app.graph import get_graph
from app.serialize import to_response

PROFILE = (
    "I'm 34, a field engineer. I drive between sites all day, phone is on maps and calls "
    "for hours and I'm rarely near a charger. I take a lot of photos of equipment. "
    "I keep a phone about four years."
)


def render(response) -> None:
    print(f"\n[{response.step}] done={response.done}  ({response.price_reference})")
    if response.notice:
        print(f"  notice: {response.notice}")
    for phone in response.show_phones[:5]:
        print(f"  ${phone.price_usd:<8} {phone.name[:40]:<42} {phone.retailer:<10} {phone.why}")
    if response.ask_question:
        print(f"  Q: {response.ask_question.question}")
        print(f"     {response.ask_question.options}")


async def main() -> None:
    graph = await get_graph()
    thread_id = uuid.uuid4().hex[:12]
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
    print(f"thread {thread_id}")

    t0 = time.perf_counter()
    await graph.ainvoke({"profile": PROFILE}, config=config)
    snapshot = await graph.aget_state(config)
    state = snapshot.values
    response = to_response(thread_id, state, bool(snapshot.next))
    render(response)
    print(f"  turn {time.perf_counter() - t0:.1f}s")

    while not response.done:
        answer = response.ask_question.options[0]
        print(f"\n>>> {answer}   (slot: {response.ask_question.slot})")
        t0 = time.perf_counter()
        await graph.ainvoke(Command(resume=answer), config=config)
        snapshot = await graph.aget_state(config)
        state = snapshot.values
        response = to_response(thread_id, state, bool(snapshot.next))
        render(response)
        print(f"  turn {time.perf_counter() - t0:.1f}s")

    print(f"\nfinal: done={response.done} questions_asked={state.get('question_count')}")


if __name__ == "__main__":
    asyncio.run(main())
