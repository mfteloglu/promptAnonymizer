from faker import Faker
from typing import Dict, Any, List
import random
import pycountry

class FPEEngine:
    """
    Engine to apply Format-Preserving Encryption (simulated via Faker) 
    or Pseudonymization to PII entities.
    """
    
    def __init__(self, seed: int = 42):
        self.faker = Faker()
        self.faker.seed_instance(seed)
        random.seed(seed)
        
        # Cache for consistent mapping if needed (simple pseudonymization)
        self.mapping: Dict[str, str] = {}

        self.countries = [c.name for c in pycountry.countries]
        self.languages = [l.name for l in pycountry.languages if hasattr(l, 'name')]
        # NRP is not really evaluated at the end so just FPE it with some examples
        self.other_nrp = [ 
            "Christian", "Muslim", "Jewish", "Buddhist", "Hindu", "Sikh", "Atheist", "Catholic",
            "Democrat", "Republican", "Liberal", "Conservative", "Socialist", "Libertarian"
        ]

    def anonymize(self, entity_type: str, text: str) -> str:
        """
        Anonymize the given text based on its entity type using FPE/Faker.
        """
        # Check cache for consistency
        key = f"{entity_type}:{text}"
        if key in self.mapping:
            return self.mapping[key]

        replacement = self._generate_replacement(entity_type)
        self.mapping[key] = replacement
        return replacement

    def _generate_replacement(self, entity_type: str) -> str:
        """
        Generate a format-preserving replacement using Faker.
        """
        if entity_type == "CREDIT_CARD":
            return self.faker.credit_card_number()
        elif entity_type == "CRYPTO":
            return self.faker.sha256() # Approximation
        elif entity_type == "DATE_TIME":
            return str(self.faker.date_this_decade())
        elif entity_type == "EMAIL_ADDRESS":
            return self.faker.email()
        elif entity_type == "IBAN_CODE":
            return self.faker.iban()
        elif entity_type == "IP_ADDRESS":
            return self.faker.ipv4()
        elif entity_type == "NRP":
            # Mix of countries, languages (nationalities), and other groups
            source = random.choice([self.countries, self.languages, self.other_nrp])
            return random.choice(source)
        elif entity_type == "LOCATION":
            return self.faker.city()
        elif entity_type == "PERSON":
            return self.faker.name()
        elif entity_type == "PHONE_NUMBER":
            return self.faker.phone_number()
        elif entity_type == "MEDICAL_LICENSE":
            return self.faker.bothify(text="??-#######")
        elif entity_type == "URL":
            return self.faker.url()
        else:
            # Fallback
            return f"<{entity_type}>"

__all__ = ["FPEEngine"]
