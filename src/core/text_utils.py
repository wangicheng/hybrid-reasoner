import re
import html

def clean_html(text: str) -> str:
    """
    Removes HTML tags from the input text and decodes HTML entities.
    
    Args:
        text: The input string containing HTML.
        
    Returns:
        The cleaned text string.
    """
    if not text:
        return ""
        
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', text)
    
    # Decode HTML entities (e.g., &nbsp;, &amp;)
    clean_text = html.unescape(clean_text)
    
    # Remove extra whitespace often left behind
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text
