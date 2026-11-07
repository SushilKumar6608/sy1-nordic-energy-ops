"""
SPARQL query wrapper for the SY1-Gov fault ontology.

The diagnosis agent calls this BEFORE the LLM to narrow fault candidates
structurally. The ontology query is deterministic and injection-resistant -
no LLM output influences which fault types are considered as candidates.

Usage:
    from src.knowledge.ontology_query import OntologyQuery
    oq = OntologyQuery()
    candidates = oq.get_fault_candidates(
        symptom_labels=["Acute demand spike (>= 40%)", "Pressure drop in pipe network"],
        temperature_c=-8.3,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rdflib import Graph, Namespace, URIRef
from rdflib.plugins.sparql import prepareQuery

ONTOLOGY_PATH = Path(__file__).parent / "fault_ontology.ttl"
NS = Namespace("http://sy1-gov.local/ontology#")


class OntologyQuery:
    def __init__(self, ontology_path: Path = ONTOLOGY_PATH):
        self._g = Graph()
        self._g.parse(str(ontology_path), format="turtle")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_fault_candidates(
        self,
        symptom_labels: list[str],
        temperature_c: Optional[float] = None,
        node_type_label: Optional[str] = None,
    ) -> list[dict]:
        """
        Return fault types that match the observed symptoms and conditions.
        Results are ordered by number of matching symptoms (most matches first).

        Each result dict:
          {
            "fault_uri":       str,
            "fault_label":     str,
            "matched_symptoms": list[str],
            "onset_pattern":   str,
            "deviation_range": str,   e.g. "40-80%"
            "sop_label":       str,
          }
        """
        # 1. Resolve symptom URIs from labels
        symptom_uris = self._resolve_symptom_uris(symptom_labels)

        # 2. Query: faults that have at least one matching symptom
        raw = self._query_faults_by_symptoms(symptom_uris)

        # 3. Filter by temperature if provided
        if temperature_c is not None:
            raw = self._filter_by_temperature(raw, temperature_c)

        # 4. Filter by node type if provided
        if node_type_label:
            raw = self._filter_by_node_type(raw, node_type_label)

        # 5. Sort by matched symptom count
        raw.sort(key=lambda x: len(x["matched_symptoms"]), reverse=True)

        return raw

    def get_sop_for_fault(self, fault_label: str) -> list[str]:
        """Return SOP labels triggered by a named fault type."""
        q = """
        PREFIX : <http://sy1-gov.local/ontology#>
        SELECT ?sopLabel WHERE {
            ?fault rdfs:label ?faultLabel .
            ?fault :triggersSOPFor ?sop .
            ?sop rdfs:label ?sopLabel .
            FILTER(str(?faultLabel) = ?target)
        }
        """
        results = self._g.query(q, initBindings={"target": fault_label})
        return [str(row.sopLabel) for row in results]

    def get_all_fault_types(self) -> list[str]:
        """Return labels of all fault types in the ontology."""
        q = """
        PREFIX : <http://sy1-gov.local/ontology#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?label WHERE {
            ?f a :FaultType ; rdfs:label ?label .
        }
        """
        return [str(row.label) for row in self._g.query(q)]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_symptom_uris(self, labels: list[str]) -> list[str]:
        """Map symptom label strings to their ontology URIs."""
        q = """
        PREFIX : <http://sy1-gov.local/ontology#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?s ?label WHERE { ?s a :Symptom ; rdfs:label ?label . }
        """
        label_to_uri = {str(row.label): str(row.s) for row in self._g.query(q)}
        return [label_to_uri[l] for l in labels if l in label_to_uri]

    def _query_faults_by_symptoms(self, symptom_uris: list[str]) -> list[dict]:
        if not symptom_uris:
            return []

        # Build VALUES block
        values_block = " ".join(f"<{u}>" for u in symptom_uris)
        q = f"""
        PREFIX : <http://sy1-gov.local/ontology#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?fault ?faultLabel ?symptomLabel ?onsetPattern ?devMin ?devMax ?sopLabel WHERE {{
            VALUES ?symptom {{ {values_block} }}
            ?fault a :FaultType ;
                   rdfs:label ?faultLabel ;
                   :hasSymptom ?symptom .
            ?symptom rdfs:label ?symptomLabel .
            OPTIONAL {{ ?fault :onsetPattern ?onsetPattern . }}
            OPTIONAL {{ ?fault :deviationMinPct ?devMin . }}
            OPTIONAL {{ ?fault :deviationMaxPct ?devMax . }}
            OPTIONAL {{ ?fault :triggersSOPFor ?sop . ?sop rdfs:label ?sopLabel . }}
        }}
        """
        rows = list(self._g.query(q))

        # Group by fault URI
        faults: dict[str, dict] = {}
        for row in rows:
            uri = str(row.fault)
            if uri not in faults:
                faults[uri] = {
                    "fault_uri": uri,
                    "fault_label": str(row.faultLabel),
                    "matched_symptoms": [],
                    "onset_pattern": (
                        str(row.onsetPattern) if row.onsetPattern else "unknown"
                    ),
                    "deviation_range": (
                        f"{float(row.devMin):.0f}-{float(row.devMax):.0f}%"
                        if row.devMin and row.devMax
                        else "unknown"
                    ),
                    "sop_label": str(row.sopLabel) if row.sopLabel else "unknown",
                }
            sym = str(row.symptomLabel)
            if sym not in faults[uri]["matched_symptoms"]:
                faults[uri]["matched_symptoms"].append(sym)

        return list(faults.values())

    def _filter_by_temperature(
        self, faults: list[dict], temperature_c: float
    ) -> list[dict]:
        """
        Keep faults whose temperature condition is satisfied.
        AnyTemperature faults always pass.
        """
        q = """
        PREFIX : <http://sy1-gov.local/ontology#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?fault ?condLabel ?threshold WHERE {
            ?fault :isLikelyWhen ?cond .
            ?cond rdfs:label ?condLabel .
            OPTIONAL { ?cond :tempThresholdC ?threshold . }
        }
        """
        fault_conditions: dict[str, list[tuple[str, float | None]]] = {}
        for row in self._g.query(q):
            uri = str(row.fault)
            cond_label = str(row.condLabel)
            threshold = float(row.threshold) if row.threshold else None
            fault_conditions.setdefault(uri, []).append((cond_label, threshold))

        kept = []
        for fault in faults:
            uri = fault["fault_uri"]
            conditions = fault_conditions.get(uri, [])
            passes = False
            for cond_label, threshold in conditions:
                if "Temperature-independent" in cond_label:
                    passes = True
                    break
                if "Below" in cond_label and threshold is not None:
                    if temperature_c < threshold:
                        passes = True
                        break
                if "Above" in cond_label and threshold is not None:
                    if temperature_c > threshold:
                        passes = True
                        break
            if passes or not conditions:
                kept.append(fault)
        return kept

    def _filter_by_node_type(
        self, faults: list[dict], node_type_label: str
    ) -> list[dict]:
        """Keep faults that affect the given node type."""
        q = """
        PREFIX : <http://sy1-gov.local/ontology#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?fault ?nodeLabel WHERE {
            ?fault :affectsNodeType ?nt .
            ?nt rdfs:label ?nodeLabel .
        }
        """
        fault_nodes: dict[str, list[str]] = {}
        for row in self._g.query(q):
            fault_nodes.setdefault(str(row.fault), []).append(str(row.nodeLabel))

        return [
            f
            for f in faults
            if node_type_label in fault_nodes.get(f["fault_uri"], [node_type_label])
        ]
