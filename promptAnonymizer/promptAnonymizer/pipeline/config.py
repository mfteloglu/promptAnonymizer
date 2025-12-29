from typing import Dict, Set

PURPOSE_RELEVANT_CONTEXT_TYPES: Dict[str, Set[str]] = {
    "SUMMARIZATION": set(),
    "PARAPHRASING": set(),
    "OPEN_ENDED_GENERATION": set(),
    "QUESTION_ANSWERING": set(),
    "TRANSLATION": set(),
    "GENERAL_CHAT": set(),
    # Legacy mappings kept for compatibility if needed
    "MEDICAL_ADVICE": {"MEDICAL"},
    "FINANCIAL_ADVICE": {"FINANCIAL"},
    "LEGAL_ADVICE": {"LEGAL"},
    "TECH_SUPPORT": {"TECHNICAL"},
    "BUSINESS_STRATEGY": {"BUSINESS"},
    "ACADEMIC_QUERY": {"ACADEMIC"},
    "PERSONAL_ASSISTANCE": {"PERSONAL", "EMPLOYMENT"},
    "CODING": {"TECHNICAL"},
}
