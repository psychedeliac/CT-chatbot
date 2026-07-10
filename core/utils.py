import re

def clean_response_prefix(text: str) -> str:
    """
    Remove common RAG prefix phrases (e.g. 'Based on the information provided...') 
    that LLMs generate automatically before their actual response.
    """
    if not text:
        return text
        
    # Pattern to match "Based on <something>," or "According to <something>," at the beginning of the text,
    # followed by optional spaces and a capitalized letter.
    pattern = r"^\s*(based\s+on|according\s+to)\s+[^,.:\n]+,\s*"
    
    cleaned = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # Capitalize the first letter of the cleaned string if it starts with a lowercase letter
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
        
    return cleaned
