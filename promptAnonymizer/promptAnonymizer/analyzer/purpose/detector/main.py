from typing import Dict, List, Set

# Common purposes (used as examples or fallbacks)
COMMON_PURPOSES: List[str] = [
    "SUMMARIZATION",
    "PARAPHRASING",
    "OPEN_ENDED_GENERATION",
    "QUESTION_ANSWERING",
    "TRANSLATION",
    "GENERAL_CHAT",
]

# Default map of purposes to PII entity types (fallback if agent doesn't specify)
DEFAULT_PURPOSE_RELEVANT_PII: Dict[str, Set[str]] = {
    "SUMMARIZATION": set(),
    "PARAPHRASING": set(),
    "OPEN_ENDED_GENERATION": set(),
    "QUESTION_ANSWERING": set(),
    "TRANSLATION": set(),
    "GENERAL_CHAT": set(),
}

# Backwards compatibility aliases
ALL_PURPOSES = COMMON_PURPOSES
PURPOSE_RELEVANT_PII = DEFAULT_PURPOSE_RELEVANT_PII

def is_pii_relevant(entity_type: str, primary_purpose: str) -> bool:

    return entity_type in PURPOSE_RELEVANT_PII.get(primary_purpose, set())


__all__ = ["is_pii_relevant", "PURPOSE_RELEVANT_PII", "ALL_PURPOSES"]
