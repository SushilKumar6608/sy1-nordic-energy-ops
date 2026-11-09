"""End-to-end test of the approval gate inside a real LangGraph graph.

LangGraph pauses a node listed in ``interrupt_before`` BEFORE the node runs.
The operator supplies the approval token by writing it into the graph state
(``update_state``) and resuming with ``invoke(None)``. ``Command(resume=...)``
feeds ``interrupt()`` calls, not static breakpoints, so it does not reach the
gate and is not used here.
"""

import pytest

pytest.importorskip("langgraph")

from typing import Any, Optional  # noqa: E402

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402

from src.policy import audit  # noqa: E402
from src.tools.approval_gate import ApprovalGateNode, request_approval  # noqa: E402

TOKEN = "tok-e2e-123"


class GraphState(TypedDict, total=False):
    pending_approval: Optional[dict[str, Any]]
    approval_history: list[dict[str, Any]]
    executed: bool


def _supervisor(state: GraphState) -> GraphState:
    pending = request_approval(
        tool_name="dispatch_maintenance_crew",
        inputs={"crew_id": "C-04", "location": "node_7"},
        policy_reason="irreversible tool",
        trace_id="trace-e2e",
    )
    return {"pending_approval": pending.model_dump()}


def _executor(state: GraphState) -> GraphState:
    # Stand-in for the real executor: it only acts if the gate approved.
    pending = state.get("pending_approval") or {}
    return {"executed": bool(pending.get("approved"))}


def _build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("supervisor", _supervisor)
    builder.add_node("approval_gate", ApprovalGateNode())
    builder.add_node("tool_executor_node", _executor)
    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "approval_gate")
    builder.add_edge("approval_gate", "tool_executor_node")
    builder.add_edge("tool_executor_node", END)
    return builder.compile(
        checkpointer=MemorySaver(), interrupt_before=["approval_gate"]
    )


@pytest.fixture
def audit_file(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(audit, "AUDIT_LOG_PATH", path)
    return path


def _audit_text(path) -> str:
    return path.read_text() if path.exists() else ""


def _start(thread_id: str):
    graph = _build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke({}, config)
    return graph, config


def _give_token(graph, config, token: str) -> None:
    pending = dict(graph.get_state(config).values["pending_approval"])
    graph.update_state(
        config, {"pending_approval": {**pending, "approval_token": token}}
    )


def test_graph_pauses_before_the_gate(audit_file):
    graph, config = _start("pauses")
    assert graph.get_state(config).next == ("approval_gate",)
    assert "executed" not in graph.get_state(config).values


def test_resume_without_token_does_not_execute(audit_file):
    graph, config = _start("no-token")
    result = graph.invoke(None, config)
    assert result["executed"] is False
    text = _audit_text(audit_file)
    assert "PENDING_HUMAN_APPROVAL" in text
    assert "APPROVED" not in text


def test_resume_with_token_executes_and_redacts_token(audit_file):
    graph, config = _start("with-token")
    _give_token(graph, config, TOKEN)
    result = graph.invoke(None, config)
    assert result["executed"] is True
    assert result["pending_approval"]["approved"] is True
    text = _audit_text(audit_file)
    assert "APPROVED" in text
    assert TOKEN not in text  # approval_token is redacted in the audit log
    ok, problems = audit.verify_chain_integrity()
    assert ok, problems


def test_whitespace_token_is_denied(audit_file):
    graph, config = _start("blank-token")
    _give_token(graph, config, "   ")
    result = graph.invoke(None, config)
    assert result["executed"] is False
    assert "DENIED_EMPTY_TOKEN" in _audit_text(audit_file)
