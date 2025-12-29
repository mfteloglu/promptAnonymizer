import requests
import json
from typing import List, Dict, Any

class ReidentificationAgent:
    def __init__(self, model: str = "llama3.2", ollama_port: int = 11434):
        self.model = model
        self.ollama_port = ollama_port

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze the anonymized text to attempt re-identification or detection of anonymized entities.
        
        :param text: The anonymized text.
        :return: Dictionary with keys 're_identifications', 'missed_pii', 'weak_pseudonyms'.
        """
        
        url = f"http://localhost:{self.ollama_port}/api/chat"
        
        system_prompt = (
            "You are a data privacy researcher and re-identification expert. "
            "You are given a text that has been anonymized. "
            "Your goal is to perform three specific tasks:\n"
            "1. **Re-identify Entities**: Try to guess the original value of any masked or pseudonymized entity based on the context. "
            "For example, if you see '<PERSON_1>' and the context mentions 'email: john.doe@example.com', you might guess 'John Doe'.\n"
            "2. **Identify Missed PII**: Detect any PII that was NOT anonymized and remains in the text (e.g., names, emails, locations, dates, etc. that are still visible). "
            "IMPORTANT: Do NOT report text that has been successfully masked (e.g. with '***') or pseudonymized (e.g. '<PERSON_1>') as missed PII. Only report PII that is still in its original, raw form.\n"
            "3. **Detect Weak Pseudonyms**: Identify pseudonyms that are too revealing (e.g., 'John Doe' used as a generic name, or 'PERSON_123' which looks like a placeholder but might be considered weak if not fully opaque).\n\n"
            "Output the result as a JSON object with three keys: 're_identifications', 'missed_pii', and 'weak_pseudonyms'.\n"
            "- 're_identifications': List of objects { 'anonymized_text': '...', 'guessed_original': '...', 'confidence': 0.0-1.0 }\n"
            "- 'missed_pii': List of objects { 'text': '...', 'type': '...' }\n"
            "- 'weak_pseudonyms': List of objects { 'text': '...', 'reason': '...' }\n\n"
            "Do not output any explanation, only the JSON object."
        )

        data = {
            "model": self.model,
            "think": False,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Analyze this text: {text}"
                }
            ],
            "stream": False,
            "format": "json"
        }

        try:
            response = requests.post(url, json=data)
            if response.ok:
                content = response.json()
                message_content = content["message"]["content"]
                try:
                    parsed = json.loads(message_content)
                    if isinstance(parsed, dict):
                        # Ensure all keys exist
                        parsed.setdefault("re_identifications", [])
                        parsed.setdefault("missed_pii", [])
                        parsed.setdefault("weak_pseudonyms", [])
                        return parsed
                    else:
                        print(f"Unexpected JSON structure: {type(parsed)}")
                        return {"re_identifications": [], "missed_pii": [], "weak_pseudonyms": []}
                except json.JSONDecodeError:
                    print("Failed to parse JSON response from LLM")
                    return {"re_identifications": [], "missed_pii": [], "weak_pseudonyms": []}
            else:
                print(f"Error calling LLM: {response.text}")
                return {"re_identifications": [], "missed_pii": [], "weak_pseudonyms": []}
        except Exception as e:
            print(f"Exception during re-identification: {e}")
            return {"re_identifications": [], "missed_pii": [], "weak_pseudonyms": []}

if __name__ == "__main__":
    agent = ReidentificationAgent()
    sample_text = "Hello, my name is <PERSON> and I live in <LOCATION>."
    print(json.dumps(agent.analyze(sample_text), indent=2))
