"""
Tavily Search Module

This module provides functionality to perform web searches using the Tavily API.
It retrieves search results including titles, URLs, and content snippets for
display or further processing.
"""

from tavily import TavilyClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Tavily client with API key from environment variables
client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


def tavily_search(query):
    """
    Perform a web search using the Tavily API and return formatted results.
    
    This function sends a search query to the Tavily API, retrieves the top
    search results, and formats them with titles, URLs, and truncated content
    snippets for readability.
    
    Args:
        query (str): The search query string to submit to Tavily.
    
    Returns:
        str: A formatted string containing search results. Each result includes:
             - Result number (1-based index)
             - Title in bold formatting
             - Full URL of the result
             - Content snippet (truncated to 300 characters for readability)
             Results are separated by double newlines for clear visual separation.
             Returns an empty string if no results are found.
    
    Example:
        >>> results = tavily_search("artificial intelligence news")
        >>> print(results)
        1. **AI Breakthrough in Healthcare**
           https://example.com/ai-healthcare
           Scientists have developed a new AI model that can detect diseases...
        
        2. **New AI Regulations Proposed**
           https://example.com/ai-regulations
           Government officials announced new regulatory frameworks...
    """
    
    # Perform search using Tavily client
    response = client.search(
        query=query,           # The search query string
        max_results=5          # Maximum number of results to return
    )
    
    # Initialize list to store formatted search results
    results = []
    
    # Iterate through search results with enumeration starting from 1
    for i, r in enumerate(response["results"], 1):
        
        # Extract title with fallback to "Unknown" if not present
        title = r.get("title", "Unknown")
        
        # Extract URL (no fallback needed as it's optional)
        url = r.get("url", "")
        
        # Extract and clean content snippet
        snippet = r.get("content", "").strip()
        
        # Truncate snippet to first 300 characters to prevent wall-of-text
        # Use rsplit to cut at the last complete word within the limit
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."
        
        # Format result with number, bold title, URL, and truncated snippet
        results.append(f"{i}. **{title}**\n   {url}\n   {snippet}")
    
    # Return all results as a single string with double newline separation
    return "\n\n".join(results)