from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

@lru_cache(maxsize=100) # This stores the last 100 Wikipedia searches in RAM
def get_trusted_context(query: str) -> str:
    # Your Wikipedia logic here...
    return wiki.run(query)

def get_trusted_context(query: str) -> str:
    """Fetches real-time data from Wikipedia."""
    api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
    try:
        return wiki.run(query)
    except Exception as e:
        return f"Error fetching context: {str(e)}"