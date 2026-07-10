from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

@tool
def web_search(query: str) -> str:
    """Searches the web using DuckDuckGo to answer questions about current events, facts, or any general topic."""
    search = DuckDuckGoSearchRun()
    return search.run(query)
