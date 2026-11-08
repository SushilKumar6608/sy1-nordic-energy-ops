"""
SPARQL query wrapper for the SY1 fault ontology.

Loads fault_ontology.ttl and exposes structured query methods
used by the diagnosis agent. No LLM calls - pure graph reasoning.
"""

from pathlib import Path
from typing import Any

try:
    from rdflib import Graph

    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False

SY1 = "http://sy1-gov.local/ontology#"
ONTOLOGY_PATH = Path(__file__).parent.parent.parent / "knowledge" / "fault_ontology.ttl"

_graph: "Graph | None" = None


def _get_graph() -> "Graph":
    """Load and cache the ontology graph."""
    global _graph
    if _graph is None:
        if not RDFLIB_AVAILABLE:
            raise ImportError(
                "rdflib is required for ontology queries. "
                "Install it with: pip install rdflib"
            )
        g = Graph()
        g.parse(str(ONTOLOGY_PATH), format="turtle")
        _graph = g
    return _graph


def query_fault_candidates(severity: str) -> list[dict[str, Any]]:
    """
    Return all fault instances matching the given severity level.
    severity: one of 'low', 'medium', 'high', 'critical'
    """
    g = _get_graph()
    sparql = f"""
        PREFIX sy1: <{SY1}>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?fault ?label ?tool ?immediate
        WHERE {{
            ?fault sy1:hasSeverity "{severity}" .
            ?fault rdfs:label ?label .
            ?fault sy1:recommendedTool ?tool .
            ?fault sy1:requiresImmediateAction ?immediate .
        }}
        ORDER BY ?label
    """
    results = []
    for row in g.query(sparql):
        results.append(
            {
                "fault": str(row.fault),
                "label": str(row.label),
                "recommended_tool": str(row.tool),
                "requires_immediate_action": str(row.immediate).lower() == "true",
                "severity": severity,
            }
        )
    return results


def query_recommended_tool(fault_label: str) -> str | None:
    """
    Return the recommended tool for a fault by its rdfs:label.
    Returns None if the fault is not found.
    """
    g = _get_graph()
    sparql = f"""
        PREFIX sy1: <{SY1}>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?tool
        WHERE {{
            ?fault rdfs:label "{fault_label}" .
            ?fault sy1:recommendedTool ?tool .
        }}
        LIMIT 1
    """
    for row in g.query(sparql):
        return str(row.tool)
    return None


def query_immediate_action_faults() -> list[dict[str, Any]]:
    """Return all faults that require immediate action."""
    g = _get_graph()
    sparql = f"""
        PREFIX sy1: <{SY1}>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        SELECT ?fault ?label ?tool ?severity
        WHERE {{
            ?fault sy1:requiresImmediateAction "true"^^xsd:boolean .
            ?fault rdfs:label ?label .
            ?fault sy1:recommendedTool ?tool .
            ?fault sy1:hasSeverity ?severity .
        }}
        ORDER BY ?severity ?label
    """
    results = []
    for row in g.query(sparql):
        results.append(
            {
                "fault": str(row.fault),
                "label": str(row.label),
                "recommended_tool": str(row.tool),
                "severity": str(row.severity),
                "requires_immediate_action": True,
            }
        )
    return results


def list_all_fault_types() -> list[str]:
    """Return rdfs:labels of all fault instances in the ontology."""
    g = _get_graph()
    sparql = f"""
        PREFIX sy1: <{SY1}>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?label
        WHERE {{
            ?fault a ?type .
            ?type rdfs:subClassOf* sy1:Fault .
            ?fault rdfs:label ?label .
        }}
        ORDER BY ?label
    """
    return [str(row.label) for row in g.query(sparql)]


def reset_graph() -> None:
    """Reset the cached graph (test helper)."""
    global _graph
    _graph = None
