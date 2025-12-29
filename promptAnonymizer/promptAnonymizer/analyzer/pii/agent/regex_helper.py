
import re
import json

def detect_pii_indices(text, pii_json):
    """
    Detect PII in the text based on the provided JSON output.
    
    :param text: The text to analyze for PII.
    :param pii_json: JSON string containing PII information.
    :return: List of dictionaries with PII type and indices.
    """

    # Robustly extract JSON objects
    pii_list = []
    
    # 1. Try full JSON load
    try:
        parsed = json.loads(pii_json)
        if isinstance(parsed, list):
            pii_list = parsed
    except Exception:
        pass

    # 2. If failed, try finding all {...} patterns (fallback for broken arrays)
    if not pii_list:
        # Regex to find JSON objects. This is a simple approximation.
        matches = re.finditer(r"\{[^{}]*\}", pii_json) 
        try:
            match = re.search(r"\[\s*\{.*?\}\s*\]", pii_json, re.DOTALL)
            if match:
                pii_list = json.loads(match.group(0))
        except:
            pass
            
    if not pii_list:
        # Fallback: Find individual objects
        # We look for { followed by "text" ... }
        # Also handle malformed objects using [] instead of {} like ["text": "val"]
        
        # Pattern 1: Standard objects { ... }
        matches = re.finditer(r"(\{.*?\})", pii_json, re.DOTALL)
        for match in matches:
            try:
                obj = json.loads(match.group(1))
                if isinstance(obj, dict):
                    pii_list.append(obj)
            except:
                pass
                
        # Pattern 2: Malformed objects using [] like ["text": "val", ...] or mixed ["text": ... }
        # We look for [ followed by "text" and ending with ] or }
        matches_brackets = re.finditer(r"(\[\s*\"text\".*?[\]\}])", pii_json, re.DOTALL)
        for match in matches_brackets:
            content = match.group(1)
            # Replace outer brackets with { }
            # We strip the first and last char and wrap in {}
            fixed_content = "{" + content[1:-1] + "}"
            try:
                obj = json.loads(fixed_content)
                if isinstance(obj, dict):
                    pii_list.append(obj)
            except:
                pass

    if not pii_list:
        print("PII JSON parsing failed; no array or objects found")
        return []

    results = []

    for pii in pii_list:
        if isinstance(pii, str):
            # Handle case where LLM returns list of strings or list of JSON strings
            try:
                # Try to parse the string as a JSON object
                parsed_pii = json.loads(pii)
                if isinstance(parsed_pii, dict):
                    text_val = parsed_pii.get("text", "")
                    type_val = parsed_pii.get("type", "UNKNOWN")
                    score_val = parsed_pii.get("score", 0.8)
                else:
                    # It was a string but not a JSON object
                    text_val = pii
                    type_val = "UNKNOWN"
                    score_val = 0.5
            except json.JSONDecodeError:
                # Not a JSON string, treat as raw text
                text_val = pii
                type_val = "UNKNOWN"
                score_val = 0.5
        else:
            text_val = pii.get("text", "")
            type_val = pii.get("type", "UNKNOWN")
            score_val = pii.get("score", 0.8)
            # We intentionally ignore 'start' and 'end' from the LLM response 
            # and recalculate them below using regex for reliability.
            
        if not text_val:
            continue

        for match in re.finditer(re.escape(text_val), text):
            results.append({
                "entity_type": type_val,
                "start": match.start(),
                "end": match.end(),
                "score": score_val
            })

    
    print(json.dumps(results, indent=2))
    return results
