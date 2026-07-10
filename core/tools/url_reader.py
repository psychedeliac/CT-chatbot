from langchain.tools import tool
import requests
from bs4 import BeautifulSoup

@tool
def url_reader(url: str) -> str:
    """Reads the content of a specific webpage URL and returns its text. Use this when you have a specific URL to analyze."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        # Limit to 10000 characters to avoid context window overflow
        return text[:10000]
    except Exception as e:
        return f"Error reading URL: {str(e)}"
