from presidio_analyzer import AnalyzerEngine


# Set up the engine, loads the NLP module (spaCy model by default) and other PII recognizers
analyzer = AnalyzerEngine()

# Set up the anonymizer engine


def presidio_analyze(text: str):
    # Call analyzer to get results
    results = analyzer.analyze(text=text,
                            language='en')
    return results


