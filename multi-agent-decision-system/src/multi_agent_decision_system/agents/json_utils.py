import json
import re

def extract_json_object(text: str) -> str:
    """
    Extract the first valid JSON object from model output.
    Raises ValueError if none found.
    """
    # Fast path
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    # Robust fallback: find first {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in model output")

    return match.group(0)