import sys
import os
import requests

from promptAnonymizer.analyzer.pii.agent.regex_helper import detect_pii_indices

class Agent:
    def __init__(self, model: str = "llama3.2", ollama_port: int = 11434):
        self.model = model
        self.ollama_port = ollama_port

    def analyze(self, text: str):
        """
        Analyze the text for PII using the specified model.
        
        :param text: The text to analyze.
        :param model: The model to use for analysis.
        :return: List of detected PII or indices based on the model.
        """
        
        url = f"http://localhost:{self.ollama_port}/api/chat"
        
        data = {
            "model": self.model,
            "think": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an intelligent privacy assistant specialized in detecting Personally Identifiable Information (PII). Your goal is to analyze text and identify sensitive entities with high accuracy."
                },
                {
                    "role": "user",
                    "content": "Please analyze the following text and identify all PII entities. "
                               "For each detected entity, provide the exact text, its PII type, its start and end indices in the original text, and a confidence score (0.0-1.0). "
                               "Do not include any explanations, conversational text, or markdown. Output ONLY a valid JSON array.\n\n"
                               "Allowed PII Types:\n"
                               "- CREDIT_CARD\n- CRYPTO\n- DATE_TIME\n- EMAIL_ADDRESS\n- IBAN_CODE\n- IP_ADDRESS\n- NRP\n- LOCATION\n- PERSON\n- PHONE_NUMBER\n- MEDICAL_LICENSE\n- URL\n\n"
                               "JSON Format:\n"
                               "[{\"text\": \"PII_TEXT\", \"type\": \"PII_TYPE\", \"start\": START_INDEX, \"end\": END_INDEX, \"score\": CONFIDENCE_SCORE}]\n\n"
                               "Example:\n"
                               "Input: My name is John and my email is john@gmail.com.\n"
                               "Output: [{\"text\": \"John\", \"type\": \"PERSON\", \"start\": 11, \"end\": 15, \"score\": 0.98}, {\"text\": \"john@gmail.com\", \"type\": \"EMAIL_ADDRESS\", \"start\": 39, \"end\": 53, \"score\": 0.99}]\n\n"
                               "Important: Ensure the output is a valid JSON array of objects. Do not wrap objects in strings.\n"
                               f"Text to analyze:\n{text}"
                }
            ],
            "stream": False,
        }

        response = requests.post(url, json=data)

        if response.ok:
            content = response.json()
            print(content["message"]["content"])
            print("-" * 50)
            results = detect_pii_indices(text, content["message"]["content"])
            return results
        else:
            print("Error:", response.text)
            return None

