from typing import TypedDict, Any, Dict
import time

from langgraph.graph import StateGraph, END

from promptAnonymizer.analyzer.purpose.agent.main import PurposeAgent
from promptAnonymizer.anonymizer.presidio.main import anonymize_text_with_purpose
from promptAnonymizer.analyzer.context.agent.main import ContextAgent
from promptAnonymizer.reidentifier.agent.main import ReidentificationAgent


class GraphState(TypedDict, total=False):
    text: str
    purpose: Dict[str, Any]
    pii: Dict[str, Any]
    context: Dict[str, Any]
    final_text: str
    model: str
    ollama_port: int
    use_llm_agent_for_pii: bool
    anonymization_method: str
    use_reidentification: bool
    reidentification_attempts: int
    reidentification_breach: bool
    reidentification_findings: Any
    pii_node_execution_time: float
    use_purpose_awareness: bool


def purpose_node(state: GraphState) -> GraphState:
    if not state.get("use_purpose_awareness", True):
        return {"purpose": {}}
        
    agent = PurposeAgent(state.get("model", "llama3.2"), state.get("ollama_port", 11434))
    purpose = agent.analyze(state["text"])
    return {"purpose": purpose}


def pii_node(state: GraphState) -> GraphState:
    start_time = time.time()
    result = anonymize_text_with_purpose(
        text=state["text"],
        use_llm_agent=state.get("use_llm_agent_for_pii", True),
        model=state.get("model", "llama3.2"),
        purpose_info=state["purpose"],
        anonymization_method=state.get("anonymization_method", "pseudonymization"),
        ollama_port=state.get("ollama_port", 11434),
        use_purpose_awareness=state.get("use_purpose_awareness", True)
    )
    end_time = time.time()
    exec_time = end_time - start_time
    return {"pii": result, "final_text": result["anonymized_text"], "pii_node_execution_time": exec_time}


def context_node(state: GraphState) -> GraphState:
    context_agent = ContextAgent(state.get("model", "llama3.2"), state.get("ollama_port", 11434))
    # Analyze the text that has potentially already been processed by PII node
    current_text = state.get("final_text", state["text"])
    analysis = context_agent.analyze(current_text)

    # Relevance check is inside pipeline.main; reimplement minimal here
    primary = state["purpose"].get("primary_purpose", "GENERAL_CHAT")
    from promptAnonymizer.pipeline.config import PURPOSE_RELEVANT_CONTEXT_TYPES  # reuse mapping

    relevant = analysis.get("context_type", "GENERAL") in PURPOSE_RELEVANT_CONTEXT_TYPES.get(primary, set())

    # No anonymization applied in this node anymore, just analysis
    anonymized_text = None
    applied = False
    final_text = current_text

    return {
        "context": {
            "analysis": analysis,
            "relevant_to_purpose": relevant,
            "anonymization_applied": applied,
            "anonymized_text": anonymized_text,
        },
        "final_text": final_text,
    }


def reidentification_node(state: GraphState) -> GraphState:
    if not state.get("use_reidentification", False):
        return {"reidentification_breach": False}
    
    attempts = state.get("reidentification_attempts", 0)
    if attempts >= 3:
        # Stop trying after 3 attempts to avoid infinite loops
        return {"reidentification_breach": False}

    agent = ReidentificationAgent(state.get("model", "llama3.2"), state.get("ollama_port", 11434))
    final_text = state.get("final_text", "")
    findings = agent.analyze(final_text)
    
    # Verification logic
    mapping = state.get("pii", {}).get("deanonymization_mapping", {})
    original_values = set(mapping.values())
    
    breach = False
    
    # 1. Check Re-identifications
    re_ids = findings.get("re_identifications", [])
    for item in re_ids:
        guessed = item.get("guessed_original", "")
        if guessed:
            # Check if guessed value matches any original PII
            for orig in original_values:
                if orig and (orig.lower() in guessed.lower() or guessed.lower() in orig.lower()):
                    breach = True
                    break
        if breach: break
        
    # 2. Check Missed PII
    if not breach:
        missed_pii = findings.get("missed_pii", [])
        kept_entities = state.get("pii", {}).get("kept_entities", [])
        kept_texts = {e.get("text", "").lower() for e in kept_entities} # Assuming kept_entities has 'text' field? 
        # Wait, kept_entities from pii_node are RecognizerResult dicts: {entity_type, start, end, score}
        # They don't have 'text' field directly in _entity_to_dict.
        # We need to extract text from original text using start/end.
        
        original_text = state.get("text", "")
        kept_texts = set()
        for e in kept_entities:
            start, end = e.get("start"), e.get("end")
            if start is not None and end is not None:
                kept_texts.add(original_text[start:end].lower())

        # Filter out missed PII that was actually kept intentionally
        real_missed = []
        for item in missed_pii:
            text_val = item.get("text", "").lower()
            if text_val not in kept_texts:
                real_missed.append(item)
        
        if real_missed:
            breach = True
            # Update findings to only show real missed
            findings["missed_pii"] = real_missed
        
    # 3. Check Weak Pseudonyms
    if not breach and findings.get("weak_pseudonyms"):
        breach = True
            
    if breach:
        print(f"Re-identification breach detected! Attempt {attempts + 1}")
        return {
            "reidentification_attempts": attempts + 1,
            "reidentification_breach": True,
            "reidentification_findings": findings
        }
    
    return {
        "reidentification_attempts": attempts + 1,
        "reidentification_breach": False,
        "reidentification_findings": findings
    }


def build_app() -> Any:
    graph = StateGraph(GraphState)
    graph.add_node("purpose", purpose_node)
    graph.add_node("pii", pii_node)
    graph.add_node("context", context_node)
    graph.add_node("reidentification", reidentification_node)

    graph.set_entry_point("purpose")
    graph.add_edge("purpose", "pii")
    graph.add_edge("pii", "context")
    graph.add_edge("context", "reidentification")
    
    def check_breach(state: GraphState):
        if state.get("reidentification_breach", False):
            return "pii"
        return END

    graph.add_conditional_edges(
        "reidentification",
        check_breach,
        {
            "pii": "pii",
            END: END
        }
    )

    return graph.compile()

def run_pipeline(text: str, model: str = "llama3.2", use_llm_agent_for_pii: bool = True, anonymization_method: str = "pseudonymization", use_reidentification: bool = False, ollama_port: int = 11434, use_purpose_awareness: bool = True) -> Dict[str, Any]:
    app = build_app()
    state: GraphState = {
        "text": text,
        "model": model,
        "ollama_port": ollama_port,
        "use_llm_agent_for_pii": use_llm_agent_for_pii,
        "anonymization_method": anonymization_method,
        "use_reidentification": use_reidentification,
        "reidentification_attempts": 0,
        "use_purpose_awareness": use_purpose_awareness
    }
    result = app.invoke(state)
    # Stitch a uniform output similar to pipeline.main.process_text
    return {
        "original_text": text,
        "purpose": result.get("purpose"),
        "pii": result.get("pii"),
        "context": result.get("context"),
        "final_text": result.get("final_text", text),
        "pii_node_execution_time": result.get("pii_node_execution_time", 0),
        "reidentification": {
            "breach": result.get("reidentification_breach", False),
            "attempts": result.get("reidentification_attempts", 0),
            "findings": result.get("reidentification_findings", [])
        }
    }
