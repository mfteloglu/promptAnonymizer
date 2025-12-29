import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import requests
import json
import re

class ContextAgent:
    def __init__(self, model: str = "llama3.2", ollama_port: int = 11434):
        self.model = "llama3.2"
        self.ollama_port = ollama_port
        
        # Define context categories we want to detect
        self.context_types = [
            "RACIAL_OR_ETHNIC_ORIGIN",
            "POLITICAL_OPINIONS",
            "RELIGIOUS_OR_PHILOSOPHICAL_BELIEFS",
            "TRADE_UNION_MEMBERSHIP",
            "GENETIC_DATA",
            "BIOMETRIC_DATA",
            "HEALTH_DATA",
            "SEX_LIFE_OR_SEXUAL_ORIENTATION",
            "GENERAL"       # Other sensitive but not categorized
        ]

    def analyze(self, text: str):
        """
        Analyze the text for sensitive context using the specified model.
        
        :param text: The text to analyze.
        :return: Dictionary with context analysis results.
        """
        
        url = f"http://localhost:{self.ollama_port}/api/chat"
        
        context_types_str = ", ".join(self.context_types)
        
        data = {
            "model": self.model,
            "think": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a context analysis assistant. You analyze text to determine if it contains sensitive information and classify the type of sensitivity."
                },
                {
                    "role": "user",
                    "content": f"""Analyze the following text for sensitive content and context. 
                    
Determine:
1. If the text contains sensitive information (true/false)
2. The primary context type from these categories, taken from GDPR: {context_types_str}
3. A confidence score (0-100) for your assessment
4. A brief explanation of why the text is considered sensitive or not
5. If you are not sure, default to "GENERAL" context type

Respond ONLY with a JSON object in this exact format:
{{
    "is_sensitive": true/false,
    "context_type": "CONTEXT_TYPE",
    "confidence": 85
    "explanation": "Brief explanation here"
}}

Examples:
- "I need to schedule my chemotherapy appointment" → {{"is_sensitive": true, "context_type": "HEALTH_DATA", "confidence": 95, "explanation": "Contains health data regarding treatment"}}
- "I am a member of the Green Party" → {{"is_sensitive": true, "context_type": "POLITICAL_OPINIONS", "confidence": 95, "explanation": "Reveals political affiliation"}}
- "The weather is nice today" → {{"is_sensitive": false, "context_type": "GENERAL", "confidence": 99, "explanation": "General conversation about weather"}}
- "I have fever and a cough" → {{"is_sensitive": true, "context_type": "HEALTH_DATA", "confidence": 90, "explanation": "Mentions health symptoms"}}

Now analyze this text: {text}"""
                }
            ],
            "stream": False,
        }

        try:
            response = requests.post(url, json=data)
            
            if response.ok:
                content = response.json()
                llm_response = content["message"]["content"]
                print("LLM Response:", llm_response)
                print("-" * 50)
                
                # Parse the JSON response
                result = self._parse_response(llm_response)
                return result
            else:
                print("Error:", response.text)
                return self._default_response("API Error")
                
        except Exception as e:
            print(f"Exception occurred: {e}")
            return self._default_response("Exception occurred")

    def _parse_response(self, response_text: str):
        """
        Parse the LLM response and extract structured data.
        """
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # Validate the response structure
                required_keys = ["is_sensitive", "context_type", "confidence", "explanation"]
                if all(key in result for key in required_keys):
                    # Ensure confidence is within valid range
                    result["confidence"] = max(0, min(100, result["confidence"]))
                    
                    # Ensure context_type is valid
                    if result["context_type"] not in self.context_types:
                        result["context_type"] = "GENERAL"
                    
                    return result
                else:
                    print("Invalid response structure")
                    return self._default_response("Invalid response structure")
            else:
                print("No JSON found in response")
                return self._default_response("No JSON found")
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return self._default_response("JSON parsing error")
        except Exception as e:
            print(f"Parsing error: {e}")
            return self._default_response("Parsing error")

    def _default_response(self, error_reason: str):
        """
        Return a default response when parsing fails.
        """
        return {
            "is_sensitive": False,
            "context_type": "GENERAL",
            "confidence": 0,
            "explanation": f"Analysis failed: {error_reason}"
        }

    def analyze_batch(self, texts: list):
        """
        Analyze multiple texts for context.
        
        :param texts: List of texts to analyze.
        :return: List of analysis results.
        """
        results = []
        for i, text in enumerate(texts):
            print(f"Analyzing text {i+1}/{len(texts)}")
            result = self.analyze(text)
            results.append(result)
        return results
