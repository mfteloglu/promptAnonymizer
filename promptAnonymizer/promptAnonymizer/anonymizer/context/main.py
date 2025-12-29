from promptAnonymizer.analyzer.context.agent.main import ContextAgent

class ContextAnonymizerAgent:
    def __init__(self, model: str = "llama3.2", ollama_port: int = 11434):
        self.analyzer = ContextAgent(model, ollama_port)

    def analyze_and_anonymize(self, text: str):
        """
        Analyze the text for sensitive context and anonymize if necessary.
        
        :param text: The text to analyze and anonymize.
        :return: Dictionary with context analysis and anonymized text.
        """
        analysis = self.analyzer.analyze(text)
        
        # For now, we don't implement actual contextual anonymization (rewriting),
        # so we just return the original text.
        # In a future implementation, this would use an LLM to rewrite the text
        # to remove the sensitive context while preserving utility.
        
        return {
            "context_analysis": analysis,
            "anonymized_text": text 
        }
