import sys
import os
import json
import uuid
from typing import List, Dict

from promptAnonymizer.analyzer.pii.agent.main import Agent
from promptAnonymizer.analyzer.pii.presidio.main import presidio_analyze
from promptAnonymizer.analyzer.purpose.agent.main import PurposeAgent
from promptAnonymizer.anonymizer.fpe.operator import FPEOperator
from presidio_anonymizer.entities import OperatorConfig
from presidio_anonymizer.operators import Operator, OperatorType

from presidio_anonymizer import AnonymizerEngine, RecognizerResult

class PseudonymizationOperator(Operator):
    def operate(self, text: str = None, params: Dict = None) -> str:
        entity_type = params.get("entity_type")
        mapping = params.get("mapping")
        
        if mapping is not None and text in mapping:
            return mapping[text]
            
        token = uuid.uuid4().hex[:4]
        pseudonym = f"<{entity_type}_{token}>"
        
        if mapping is not None:
            mapping[text] = pseudonym
            
        return pseudonym

    def validate(self, params: Dict = None) -> None:
        pass

    def operator_name(self) -> str:
        return "Pseudonymization"

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize

# Initialize anonymizer engine once
engine = AnonymizerEngine()
engine.add_anonymizer(FPEOperator)
engine.add_anonymizer(PseudonymizationOperator)


def _resolve_conflicts(presidio_entities: List[RecognizerResult], agent_entities: List[RecognizerResult]) -> List[RecognizerResult]:
    """
    Resolve conflicts between Presidio and Agent entities.
    Strategy:
    1. If both Presidio and Agent detect the same entity type for an overlapping span, prioritize that type.
    2. Otherwise, prioritize the entity with the highest score.
    """
    # Tag entities with source
    all_items = []
    for e in presidio_entities:
        all_items.append({"entity": e, "source": "presidio"})
    for e in agent_entities:
        all_items.append({"entity": e, "source": "agent"})
    
    # Sort by start position
    all_items.sort(key=lambda x: x["entity"].start)
    
    resolved = []
    
    # Cluster overlapping entities
    i = 0
    while i < len(all_items):
        current_cluster = [all_items[i]]
        cluster_end = all_items[i]["entity"].end
        
        j = i + 1
        while j < len(all_items):
            next_item = all_items[j]
            # Check for overlap (start < end of cluster)
            if next_item["entity"].start < cluster_end:
                current_cluster.append(next_item)
                cluster_end = max(cluster_end, next_item["entity"].end)
                j += 1
            else:
                break
        
        # Resolve the current cluster
        best_entity = _pick_best_from_cluster(current_cluster)
        if best_entity:
            resolved.append(best_entity)
        
        i = j
        
    return resolved


def _pick_best_from_cluster(cluster: List[Dict]) -> RecognizerResult:
    if not cluster:
        return None
    if len(cluster) == 1:
        return cluster[0]["entity"]
        
    # Check for agreement on entity type
    agent_types = set(item["entity"].entity_type for item in cluster if item["source"] == "agent")
    presidio_types = set(item["entity"].entity_type for item in cluster if item["source"] == "presidio")
    
    common_types = agent_types.intersection(presidio_types)
    
    candidates = []
    if common_types:
        # Filter for common types if agreement exists
        candidates = [item for item in cluster if item["entity"].entity_type in common_types]
    else:
        # No agreement, consider all
        candidates = cluster
        
    # Pick highest score from candidates
    # Tie-breaker: Prefer Agent (source="agent") if scores are equal
    best = max(candidates, key=lambda x: (getattr(x["entity"], "score", 0), 1 if x["source"] == "agent" else 0))
    return best["entity"]


def _entity_to_dict(e: RecognizerResult) -> Dict:
    return {
        "entity_type": e.entity_type,
        "start": e.start,
        "end": e.end,
        "score": getattr(e, "score", None)
    }


def anonymize_text_with_purpose(text: str, use_llm_agent: bool = True, model: str = "llama3.2", purpose_info: Dict | None = None, anonymization_method: str = "pseudonymization", ollama_port: int = 11434, use_purpose_awareness: bool = True) -> Dict:
    """Detect PII, classify purpose, and anonymize only non-relevant PII.

    Returns a dictionary containing purpose info, entities kept, entities anonymized and final text.
    """

    # Presidio detection
    presidio_results = presidio_analyze(text)

    # Optional LLM-based detection
    agent_entities: List[RecognizerResult] = []
    if use_llm_agent:
        agent = Agent(model=model, ollama_port=ollama_port)
        llm_json_entities = agent.analyze(text) or []
        # Convert to RecognizerResult objects
        for item in llm_json_entities:
            try:
                agent_entities.append(RecognizerResult.from_json(item))
            except Exception:
                pass

    # Combine & deduplicate
    combined_entities = _resolve_conflicts(list(presidio_results), agent_entities)

    relevant_types = set()

    if use_purpose_awareness:
        # Detect purpose with agent if not provided
        if purpose_info is None:
            purpose_info = PurposeAgent(model, ollama_port).analyze(text)
        
        # Use the agent's determined relevant PII types
        relevant_types = set(purpose_info.get("relevant_pii_types", []))
    else:
        if purpose_info is None:
            purpose_info = {}

    # Partition entities into relevant (kept) vs anonymized (removed from anonymization list)
    kept_entities = []
    to_anonymize = []
    for ent in combined_entities:
        if ent.entity_type in relevant_types:
            kept_entities.append(ent)
        else:
            to_anonymize.append(ent)

    # Configure operators
    operators = {}
    deanonymization_mapping = {}
    
    types_to_anonymize = set(e.entity_type for e in to_anonymize)

    if anonymization_method == "fpe":
        for t in types_to_anonymize:
            operators[t] = OperatorConfig("FPE", {"entity_type": t})
    elif anonymization_method == "masking":
        for t in types_to_anonymize:
            operators[t] = OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 100, "from_end": False})
    else: # pseudonymization (default)
        for t in types_to_anonymize:
            operators[t] = OperatorConfig("Pseudonymization", {"entity_type": t, "mapping": deanonymization_mapping})

    # Perform anonymization only on non-relevant entities
    anonymized_result = engine.anonymize(
        text=text, 
        analyzer_results=to_anonymize,
        operators=operators
    )

    # Build mapping Anonymized -> Original
    # For Pseudonymization, mapping is already populated by the operator as Original -> Anonymized (for consistency check).
    # We need to invert it to be Anonymized -> Original for the output.
    if (anonymization_method == "pseudonymization" or not anonymization_method) and deanonymization_mapping:
        # Invert the mapping
        deanonymization_mapping_inverted = {v: k for k, v in deanonymization_mapping.items()}
        # Clear and update the original dictionary to keep the reference if needed, or just reassign
        deanonymization_mapping.clear()
        deanonymization_mapping.update(deanonymization_mapping_inverted)
    
    if not deanonymization_mapping:
        # to_anonymize contains original spans. anonymized_result.items contains new spans and text.
        # We assume they correspond 1-to-1 if sorted by start index.
        to_anonymize_sorted = sorted(to_anonymize, key=lambda x: x.start)
        items_sorted = sorted(anonymized_result.items, key=lambda x: x.start)
        
        if len(to_anonymize_sorted) == len(items_sorted):
            for orig_ent, new_ent in zip(to_anonymize_sorted, items_sorted):
                orig_text_val = text[orig_ent.start:orig_ent.end]
                new_text_val = new_ent.text
                deanonymization_mapping[new_text_val] = orig_text_val

    return {
        "original_text": text,
        "anonymized_text": anonymized_result.text,
        "purpose": purpose_info,
        "kept_entities": [_entity_to_dict(e) for e in kept_entities],
        "anonymized_entities": [_entity_to_dict(e) for e in to_anonymize],
        "deanonymization_mapping": deanonymization_mapping
    }


def demo():
    sample = (
        "A student's assessment was found on device bearing IMEI: 06-184755-866851-3. "
        "The document falls under the various topics discussed in our Optimization curriculum. "
        "Can you please collect it?"
    )
    result = anonymize_text_with_purpose(sample)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    demo()