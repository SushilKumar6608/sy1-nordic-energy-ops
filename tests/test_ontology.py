"""
Tests for the fault ontology query layer.
Skipped if rdflib is not installed.
"""

import pytest

try:
    import rdflib  # noqa: F401

    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RDFLIB_AVAILABLE, reason="rdflib not installed")


@pytest.fixture(autouse=True)
def reset_graph():
    from src.knowledge.ontology_query import reset_graph

    reset_graph()
    yield
    reset_graph()


def test_query_high_severity_faults_returns_results():
    from src.knowledge.ontology_query import query_fault_candidates

    results = query_fault_candidates("high")
    assert len(results) > 0


def test_fault_candidate_has_required_keys():
    from src.knowledge.ontology_query import query_fault_candidates

    results = query_fault_candidates("high")
    for r in results:
        assert "label" in r
        assert "recommended_tool" in r
        assert "severity" in r
        assert "requires_immediate_action" in r


def test_query_critical_faults_all_require_immediate_action():
    from src.knowledge.ontology_query import query_fault_candidates

    results = query_fault_candidates("critical")
    for r in results:
        assert r["requires_immediate_action"] is True


def test_recommended_tool_for_known_fault():
    from src.knowledge.ontology_query import query_recommended_tool

    tool = query_recommended_tool("Pipe Leak Fault")
    assert tool == "dispatch_maintenance_crew"


def test_recommended_tool_unknown_fault_returns_none():
    from src.knowledge.ontology_query import query_recommended_tool

    tool = query_recommended_tool("Nonexistent Fault XYZ")
    assert tool is None


def test_immediate_action_faults_not_empty():
    from src.knowledge.ontology_query import query_immediate_action_faults

    results = query_immediate_action_faults()
    assert len(results) > 0
    for r in results:
        assert r["requires_immediate_action"] is True


def test_list_all_fault_types_not_empty():
    from src.knowledge.ontology_query import list_all_fault_types

    faults = list_all_fault_types()
    assert len(faults) >= 5


def test_low_severity_fault_does_not_require_immediate_action():
    from src.knowledge.ontology_query import query_fault_candidates

    results = query_fault_candidates("low")
    for r in results:
        assert r["requires_immediate_action"] is False
