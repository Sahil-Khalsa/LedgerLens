"""
LangGraph agent graph — wiring and conditional edges.

Phase 1: planner → verifier (fast path) OR retriever stub → synthesizer → critic.
Phase 2: visual_retriever and extractor nodes fully wired.
"""

from typing import Literal

from langgraph.graph import END, StateGraph

from agents.nodes import (
    critic,
    extractor,
    numerical_verifier,
    planner,
    synthesizer,
    visual_retriever,
)
from agents.state import State


def _route_after_planner(state: State) -> Literal["verifier", "retriever"]:
    return "verifier" if state.get("route") == "fast" else "retriever"


def _route_after_critic(state: State) -> Literal["retriever", "__end__"]:
    critique = state.get("critique")
    if critique and not critique["grounded"] and (state.get("retries", 0) < 2):
        return "retriever"
    return END


def build_graph() -> StateGraph:  # type: ignore[type-arg]
    g: StateGraph = StateGraph(State)  # type: ignore[type-arg]

    g.add_node("planner", planner)
    g.add_node("retriever", visual_retriever)
    g.add_node("extractor", extractor)
    g.add_node("verifier", numerical_verifier)
    g.add_node("synth", synthesizer)
    g.add_node("critic", critic)

    g.set_entry_point("planner")

    g.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"verifier": "verifier", "retriever": "retriever"},
    )
    g.add_edge("retriever", "extractor")
    g.add_edge("extractor", "verifier")
    g.add_edge("verifier", "synth")
    g.add_edge("synth", "critic")
    g.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"retriever": "retriever", END: END},
    )

    return g


def _get_app() -> object:
    """Return a compiled graph. Adds PostgresSaver checkpointer when DB is available."""
    g = build_graph()
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from config import settings
        checkpointer = PostgresSaver.from_conn_string(settings.database_url)
        return g.compile(checkpointer=checkpointer)
    except Exception:
        return g.compile()


app = _get_app()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run a query through the agent graph")
    parser.add_argument("--q", required=True, help="Question to ask")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--pipeline", default="baseline", choices=["baseline", "visual"])
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    initial_state: State = {
        "question": args.q,
        "ticker": args.ticker,
        "filing_accn": None,
        "thread_id": None,
        "plan": [],
        "route": "document",
        "pages": [],
        "facts": [],
        "draft": None,
        "answer": None,
        "answer_status": None,
        "critique": None,
        "retries": 0,
        "cost_usd": 0.0,
        "latency_ms": 0,
    }

    result = app.invoke(initial_state)
    print(f"\nRoute:  {result.get('route')}")
    print(f"Status: {result.get('answer_status')}")
    print(f"\nAnswer:\n{result.get('answer') or 'No answer generated'}")
    print(f"\nFacts:")
    for f in result.get("facts", []):
        print(f"  [{f['verified']}] {f['text']} — {f['page_ref']}")


if __name__ == "__main__":
    _cli()
