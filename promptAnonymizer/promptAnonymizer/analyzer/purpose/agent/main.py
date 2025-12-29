import sys
import os
import requests
import json
import re
from typing import Dict, List

from promptAnonymizer.analyzer.purpose.detector.main import (
    ALL_PURPOSES,
    PURPOSE_RELEVANT_PII,
)
from promptAnonymizer.analyzer.pii.presidio.constants import PRESIDIO_PII_TYPES


class PurposeAgent:
    def __init__(self, model: str = "llama3.2", ollama_port: int = 11434):
        self.model = model
        self.ollama_port = ollama_port

    def analyze(self, text: str) -> Dict:
        """Use local LLM to classify the primary purpose of the text and determine relevant PII.
        Returns a dict compatible with `detect_purpose` output and includes `relevant_pii_types`.
        """
        url = f"http://localhost:{self.ollama_port}/api/chat"
        pii_types_str = ", ".join(PRESIDIO_PII_TYPES)

        data = {
            "model": "llama3.2",
            "think": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an intelligent privacy assistant specialized in NLP task classification and data minimization. Your goal is to analyze the user's text to determine its primary NLP purpose and identify which Personally Identifiable Information (PII) types are strictly necessary to preserve for that specific purpose."
                },
                {
                    "role": "user",
                    "content": f"""
Analyze the following text to determine its primary NLP task and the PII types that must be preserved to maintain utility.

1. **Classify the Primary Purpose** into one of these categories:
   - SUMMARIZATION (e.g., summarizing emails, articles)
   - PARAPHRASING (e.g., rewriting for tone, clarity)
   - OPEN_ENDED_GENERATION (e.g., writing stories, blogs, creative content)
   - QUESTION_ANSWERING (e.g., answering factual questions)
   - TRANSLATION (e.g., translating between languages)

2. **Determine Relevant PII Types**:
   - Analyze the text content and the identified purpose.
   - Decide which PII types (from the allowed list below) are critical for the task asked from you.
   - Only focus on the task requested from you, nothing else. Ignore the other irrelevant text, just focus on your task.

Allowed PII Types: [{pii_types_str}]

- For example if the text is "Write a story about a person named John who lives in Paris", then PERSON and LOCATION are relevant PII types.
- For example if the text is "My name is Fatih and what was the weather like in New York yesterday?", then LOCATION and DATE_TIME are relevant PII types.
- For example if the text is "Please check if this credit card number is valid", then CREDIT_CARD is a relevant PII type.

Respond ONLY with a JSON object in this exact format:
{{
  "primary_purpose": "CATEGORY_NAME",
  "relevant_pii_types": ["TYPE1", "TYPE2"],
  "confidence": 0-100,
  "explanation": "Brief reasoning for the purpose and PII selection."
}}

Text to analyze:
{text}
"""
                },
            ],
            "stream": False,
        }

        try:
            response = requests.post(url, json=data)
            if not response.ok:
                return self._default_purpose()

            content = response.json()
            llm_text = content["message"]["content"]
            parsed = self._parse_response(llm_text)

            # Ensure defaults
            parsed.setdefault("primary_purpose", "GENERAL_CHAT")
            parsed.setdefault("relevant_pii_types", [])
            
            return parsed
        except Exception:
            return self._default_purpose()

    def _parse_response(self, response_text: str) -> Dict:
        try:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not json_match:
                return {}
            obj = json.loads(json_match.group())

            # Basic normalization
            obj.setdefault("primary_purpose", "GENERAL_CHAT")
            obj.setdefault("relevant_pii_types", [])
            obj.setdefault("confidence", 0)
            obj.setdefault("explanation", "")
            return obj
        except Exception:
            return {}

    def _default_purpose(self) -> Dict:
        primary = "GENERAL_CHAT"
        return {
            "primary_purpose": primary,
            "relevant_pii_types": [],
            "confidence": 0,
            "explanation": "Fallback default",
        }

__all__ = ["PurposeAgent"]
