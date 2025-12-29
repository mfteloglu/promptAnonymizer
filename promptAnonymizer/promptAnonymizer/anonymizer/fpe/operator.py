from presidio_anonymizer.operators import Operator, OperatorType
from typing import Dict, Any
from promptAnonymizer.anonymizer.fpe.engine import FPEEngine

class FPEOperator(Operator):
    """
    Custom Presidio Operator that uses FPEEngine.
    """
    
    def __init__(self):
        self.engine = FPEEngine()

    def operate(self, text: str = None, params: Dict = None) -> str:
        """
        Anonymize the text using FPEEngine.
        :param text: The text to anonymize (the entity value).
        :param params: Dictionary containing 'entity_type'.
        :return: The anonymized text.
        """
        entity_type = params.get("entity_type")
        return self.engine.anonymize(entity_type, text)

    def validate(self, params: Dict = None) -> None:
        pass

    def operator_name(self) -> str:
        return "FPE"

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize
