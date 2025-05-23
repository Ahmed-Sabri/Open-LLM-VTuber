from duckduckgo_search import DDGS
from loguru import logger

def search_web(query: str, num_results: int = 3) -> str:
    """
    Performs a web search using DuckDuckGo and returns formatted results.

    Args:
        query: The search query.
        num_results: The maximum number of results to return.

    Returns:
        A string containing the formatted search results, or an error message.
    """
    results_output = []
    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(
                keywords=query,
                max_results=num_results
            )
            if search_results:
                for i, result in enumerate(search_results):
                    results_output.append(
                        f"Result {i+1}:\n"
                        f"Title: {result.get('title', 'N/A')}\n"
                        f"Snippet: {result.get('body', 'N/A')}\n"
                        f"URL: {result.get('href', 'N/A')}\n"
                    )
            else:
                return "No results found for your query."

        if not results_output: # Should be covered by the above but as a safeguard
            return "No results found for your query."
            
        return "\n".join(results_output)

    except Exception as e:
        logger.error(f"Web search failed for query '{query}': {e}")
        return f"An error occurred during the web search: {str(e)}"

if __name__ == '__main__':
    # Test the function
    test_query = "What is the weather like in London?"
    print(f"Searching for: '{test_query}'")
    search_results_output = search_web(test_query, num_results=3)
    print("\nSearch Results:")
    print(search_results_output)

    print("-" * 20)

    test_query_no_results = "asdflkjhasiudfhoaiusdfh" # unlikely to find results
    print(f"Searching for: '{test_query_no_results}'")
    search_results_output_no_results = search_web(test_query_no_results, num_results=3)
    print("\nSearch Results (no results expected):")
    print(search_results_output_no_results)

    print("-" * 20)
    # To test network error, one might disable network temporarily
    # For now, we assume it works or raises an exception caught by the handler.
    # Example of how it might be called by the agent:
    # formatted_results = search_web("latest AI news")
    # prompt_with_results = f"Based on the following search results:\n{formatted_results}\n\nPlease answer the user's question."
    # print(prompt_with_results)
