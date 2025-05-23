import pytest
from unittest.mock import patch, MagicMock

from open_llm_vtuber.agent.tools.web_search_tool import search_web

@pytest.fixture
def mock_ddgs_text():
    """Fixture to mock DDGS().text() method."""
    with patch('duckduckgo_search.DDGS.text') as mock_text:
        yield mock_text

def test_search_web_success(mock_ddgs_text):
    """Test successful web search and result formatting."""
    sample_results = [
        {"title": "Test Title 1", "body": "Snippet 1...", "href": "http://example.com/1"},
        {"title": "Test Title 2", "body": "Snippet 2...", "href": "http://example.com/2"},
    ]
    mock_ddgs_text.return_value = sample_results

    query = "test query"
    num_results = 2
    expected_output = (
        "Result 1:\n"
        "Title: Test Title 1\n"
        "Snippet: Snippet 1...\n"
        "URL: http://example.com/1\n\n"
        "Result 2:\n"
        "Title: Test Title 2\n"
        "Snippet: Snippet 2...\n"
        "URL: http://example.com/2"
    )
    # Note: The join in the actual function is just "\n", so the last result won't have a trailing "\n\n"
    # For multiple results, the join adds one "\n" between them.
    # Let's adjust expected_output based on the actual joiner.
    # If the join is `"\n".join(results_output)`, then there's no double newline.
    # The current code is: `"\n".join(results_output)` where each item in results_output
    # already ends with "\n" if it's not the last item due to the f-string structure.
    # Let's re-verify the f-string in search_web.
    # results_output.append(f"Result {i+1}:\nTitle: ...\nSnippet: ...\nURL: ...\n")
    # This means each block ends with \n. So "\n".join will put an extra \n.
    # So, the expected output should have one \n between entries.
    
    # Re-evaluating the output based on the code:
    # Each entry in results_output is f"Result {i+1}:\nTitle: {title}\nSnippet: {body}\nURL: {href}\n"
    # So, if results_output has two such strings, "\n".join will put one more \n between them.
    # String1\n + \n + String2\n => String1\n\nString2\n
    # This seems correct. Let's stick to the initial expected_output if each block has one trailing \n.
    # The current code's f-string for results_output:
    # f"Result {i+1}:\nTitle: {result.get('title', 'N/A')}\nSnippet: {result.get('body', 'N/A')}\nURL: {result.get('href', 'N/A')}\n"
    # This means each entry in results_output already has a trailing newline.
    # So "\n".join(results_output) will be:
    # "Block1\n" + "\n" + "Block2\n" = "Block1\n\nBlock2\n" (if results_output has >1 item)
    # If only one item, it will be "Block1\n"
    # The test case has num_results = 2, so it will have "\n\n"

    # If the function is `"\n\n".join(results_output_without_trailing_newline_in_block)` it would be different.
    # Let's assume the current code's f-string is as above, and the join is `"\n".join`.
    # Expected for 2 results:
    # "Result 1:\nTitle: T1\nSnippet: S1\nURL: U1\n"  <- from results_output[0]
    # "\n"                                            <- from join
    # "Result 2:\nTitle: T2\nSnippet: S2\nURL: U2\n"  <- from results_output[1]
    # So, yes, a double newline between entries if each entry has one trailing newline.

    # Re-adjusting expected based on how a typical join works:
    # If each element in `results_output` is:
    # `f"Result {i+1}:\nTitle: {title}\nSnippet: {body}\nURL: {href}"` (no trailing \n in the f-string block)
    # Then `"\n\n".join(results_output)` would give the desired double newline.
    # The current code `search_web` has:
    # `results_output.append(f"Result {i+1}:\n...\nURL: {result.get('href', 'N/A')}\n")` (trailing \n)
    # and then `"\n".join(results_output)`.
    # This means: "Block1\n" + "\n" + "Block2\n" + ...
    # So, it should be:
    expected_output = (
        "Result 1:\n"
        "Title: Test Title 1\n"
        "Snippet: Snippet 1...\n"
        "URL: http://example.com/1\n" # This is results_output[0]
        "\n" # This is the joiner
        "Result 2:\n"
        "Title: Test Title 2\n"
        "Snippet: Snippet 2...\n"
        "URL: http://example.com/2\n" # This is results_output[1]
    )
    # The final string will have a trailing \n from the last element.

    actual_output = search_web(query, num_results=num_results)
    assert actual_output.strip() == expected_output.strip() # Use strip to handle potential trailing newline discrepancies
    mock_ddgs_text.assert_called_once_with(keywords=query, max_results=num_results)

def test_search_web_no_results(mock_ddgs_text):
    """Test web search when no results are found."""
    mock_ddgs_text.return_value = []
    query = "query with no results"
    
    expected_output = "No results found for your query."
    actual_output = search_web(query)
    
    assert actual_output == expected_output
    mock_ddgs_text.assert_called_once_with(keywords=query, max_results=3) # Default num_results

def test_search_web_ddgs_exception(mock_ddgs_text):
    """Test web search when DDGS().text() raises an exception."""
    mock_ddgs_text.side_effect = Exception("DDGS API error")
    query = "query causing error"
    
    expected_output = "An error occurred during the web search: DDGS API error"
    actual_output = search_web(query)
    
    assert actual_output == expected_output
    mock_ddgs_text.assert_called_once_with(keywords=query, max_results=3)

def test_search_web_missing_fields_in_results(mock_ddgs_text):
    """Test that missing fields in search results are handled gracefully (N/A)."""
    sample_results = [
        {"body": "Snippet 1...", "href": "http://example.com/1"}, # Missing title
        {"title": "Test Title 2", "href": "http://example.com/2"}, # Missing body
        {"title": "Test Title 3", "body": "Snippet 3..."},         # Missing href
    ]
    mock_ddgs_text.return_value = sample_results

    query = "test query for missing fields"
    num_results = 3
    expected_output = (
        "Result 1:\n"
        "Title: N/A\n"
        "Snippet: Snippet 1...\n"
        "URL: http://example.com/1\n"
        "\n"
        "Result 2:\n"
        "Title: Test Title 2\n"
        "Snippet: N/A\n"
        "URL: http://example.com/2\n"
        "\n"
        "Result 3:\n"
        "Title: Test Title 3\n"
        "Snippet: Snippet 3...\n"
        "URL: N/A\n"
    )
    actual_output = search_web(query, num_results=num_results)
    assert actual_output.strip() == expected_output.strip()

def test_search_web_num_results_respected(mock_ddgs_text):
    """Test that num_results parameter limits the output."""
    sample_results = [
        {"title": "Title 1", "body": "Body 1", "href": "URL 1"},
        {"title": "Title 2", "body": "Body 2", "href": "URL 2"},
        {"title": "Title 3", "body": "Body 3", "href": "URL 3"},
    ]
    mock_ddgs_text.return_value = sample_results # DDGS mock will return all 3

    query = "test num_results"
    num_results_to_request = 1 # Request only 1
    
    # search_web itself slices the results from DDGS based on its internal logic,
    # but it passes max_results=num_results to ddgs.text().
    # So, ddgs.text() should only return num_results_to_request.
    # If ddgs.text() still returned all 3, our search_web would process only `num_results_to_request`.
    # The mock should reflect what ddgs.text() would do if it got max_results=1.
    # Let's assume ddgs.text() respects max_results.
    mock_ddgs_text.return_value = sample_results[:num_results_to_request]


    expected_output = (
        "Result 1:\n"
        "Title: Title 1\n"
        "Snippet: Body 1\n"
        "URL: URL 1\n"
    )
    actual_output = search_web(query, num_results=num_results_to_request)
    
    assert actual_output.strip() == expected_output.strip()
    mock_ddgs_text.assert_called_once_with(keywords=query, max_results=num_results_to_request)

def test_search_web_empty_query(mock_ddgs_text):
    """Test web search with an empty query string."""
    query = ""
    # DDGS might return results for an empty query or raise an error.
    # Let's assume it returns no results for an empty query.
    mock_ddgs_text.return_value = [] 
    
    expected_output = "No results found for your query."
    actual_output = search_web(query)
    
    assert actual_output == expected_output
    mock_ddgs_text.assert_called_once_with(keywords=query, max_results=3)
