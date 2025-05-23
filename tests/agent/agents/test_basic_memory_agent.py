import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

from open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from open_llm_vtuber.agent.stateless_llm.stateless_llm_interface import StatelessLLMInterface
from open_llm_vtuber.agent.input_types import BatchInput, TextData, TextSource
from open_llm_vtuber.agent.output_types import SentenceOutput, DisplayText, Actions

# Minimal mock for Live2D model if needed by action extractor
@pytest.fixture
def mock_live2d_model():
    model = MagicMock()
    model.emo_str = "neutral,happy,sad" # Example emo_str
    model.set_expression_from_action = MagicMock()
    return model

@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=StatelessLLMInterface)
    # The chat_completion method needs to be an async generator
    async def mock_chat_completion(messages, system_prompt):
        # Default behavior: echo back the last user message content or a generic response
        if hasattr(mock_chat_completion, 'custom_response_iterator'):
            async for token in mock_chat_completion.custom_response_iterator:
                yield token
        else:
            last_user_message = "Default LLM response."
            if messages and messages[-1]["role"] == "user":
                content = messages[-1]["content"]
                if isinstance(content, list): # For multimodal input
                    for item in content:
                        if item.get("type") == "text":
                            last_user_message = item["text"]
                            break
                else:
                    last_user_message = content
            
            # Simulate token streaming
            for char_token in last_user_message:
                yield char_token
    
    llm.chat_completion = mock_chat_completion
    return llm

@pytest.fixture
def basic_memory_agent_no_search(mock_llm, mock_live2d_model):
    return BasicMemoryAgent(
        llm=mock_llm,
        system="You are a helpful assistant.",
        live2d_model=mock_live2d_model,
        enable_web_search=False # Web search disabled
    )

@pytest.fixture
def basic_memory_agent_with_search(mock_llm, mock_live2d_model):
    return BasicMemoryAgent(
        llm=mock_llm,
        system="You are a helpful assistant. Use [SEARCH: query] for web search.",
        live2d_model=mock_live2d_model,
        enable_web_search=True # Web search enabled
    )

async def collect_response_from_agent(agent, text_input):
    """Helper to collect all SentenceOutput from agent.chat into a single string."""
    full_response_text = ""
    input_batch = BatchInput(texts=[TextData(source=TextSource.INPUT, content=text_input)])
    async for sentence_output in agent.chat(input_batch):
        if isinstance(sentence_output, SentenceOutput) and sentence_output.display_text:
            full_response_text += sentence_output.display_text.text
    return full_response_text

@pytest.mark.asyncio
async def test_agent_no_search_normal_response(basic_memory_agent_no_search, mock_llm):
    """Test that the agent responds normally when web search is disabled and no search command is issued."""
    user_input = "Hello, world!"
    
    # Configure mock_llm to stream this specific response
    async def specific_response_iterator():
        for char_token in "Echo: " + user_input:
            yield char_token
    mock_llm.chat_completion.custom_response_iterator = specific_response_iterator()

    response = await collect_response_from_agent(basic_memory_agent_no_search, user_input)
    assert response == "Echo: " + user_input
    # Check that the user message and assistant response are in memory
    assert len(basic_memory_agent_no_search._memory) == 3 # System, User, Assistant
    assert basic_memory_agent_no_search._memory[1]["content"] == user_input
    assert basic_memory_agent_no_search._memory[2]["content"] == "Echo: " + user_input

@pytest.mark.asyncio
async def test_agent_no_search_receives_search_command(basic_memory_agent_no_search, mock_llm):
    """Test that agent ignores [SEARCH:] command if web_search is disabled."""
    user_input = "Tell me about LLMs"
    llm_response_with_search = "I should search for this. [SEARCH: Large Language Models]"
    
    async def specific_response_iterator():
        for char_token in llm_response_with_search:
            yield char_token
    mock_llm.chat_completion.custom_response_iterator = specific_response_iterator()

    response = await collect_response_from_agent(basic_memory_agent_no_search, user_input)
    assert response == llm_response_with_search # Agent should output the LLM response as is

@pytest.mark.asyncio
@patch('open_llm_vtuber.agent.agents.basic_memory_agent.search_web')
async def test_agent_with_search_executes_search(mock_search_web, basic_memory_agent_with_search, mock_llm):
    user_input = "What's the latest news on AI?"
    llm_first_response_asks_for_search = "I need to find that out. [SEARCH: latest AI news]"
    search_results = "Result 1: AI is advancing rapidly.\nURL: example.com/ai_news"
    llm_final_response_after_search = "AI is advancing rapidly according to recent news."

    # Mock search_web
    mock_search_web.return_value = search_results

    # Configure LLM responses:
    # 1. First call: LLM asks to search
    # 2. Second call (after search results): LLM gives final answer
    llm_call_count = 0
    async def multi_step_llm_response(messages, system_prompt):
        nonlocal llm_call_count
        llm_call_count += 1
        if llm_call_count == 1:
            for char_token in llm_first_response_asks_for_search:
                await asyncio.sleep(0) # Yield control to allow event loop to run
                yield char_token
        elif llm_call_count == 2:
            # Check if the search results are in the messages for the second call
            last_message_content = messages[-1]["content"]
            assert search_results in last_message_content 
            for char_token in llm_final_response_after_search:
                await asyncio.sleep(0)
                yield char_token
        else:
            for char_token in "Unexpected LLM call": # Should not happen
                await asyncio.sleep(0)
                yield char_token
                
    mock_llm.chat_completion = multi_step_llm_response # Assign the new mock behavior directly

    response = await collect_response_from_agent(basic_memory_agent_with_search, user_input)
    
    assert response == llm_final_response_after_search
    mock_search_web.assert_called_once_with("latest AI news", num_results=3)
    
    # Verify memory: System, User, LLM attempt (with search), User (search results), Final LLM response
    # The BasicMemoryAgent's current implementation adds the LLM's attempt (with [SEARCH:])
    # and then adds the search results as a user message.
    assert len(basic_memory_agent_with_search._memory) == 5 
    assert basic_memory_agent_with_search._memory[0]["role"] == "system"
    assert basic_memory_agent_with_search._memory[1]["role"] == "user"
    assert basic_memory_agent_with_search._memory[1]["content"] == user_input
    assert basic_memory_agent_with_search._memory[2]["role"] == "assistant" # LLM's first attempt
    assert basic_memory_agent_with_search._memory[2]["content"] == llm_first_response_asks_for_search
    assert basic_memory_agent_with_search._memory[3]["role"] == "user" # Search results injected as user
    assert search_results in basic_memory_agent_with_search._memory[3]["content"]
    assert basic_memory_agent_with_search._memory[4]["role"] == "assistant" # Final LLM response
    assert basic_memory_agent_with_search._memory[4]["content"] == llm_final_response_after_search


@pytest.mark.asyncio
@patch('open_llm_vtuber.agent.agents.basic_memory_agent.search_web')
async def test_agent_search_loop_limit(mock_search_web, basic_memory_agent_with_search, mock_llm):
    user_input = "Keep searching for stuff."
    llm_response_always_searches = "Okay, I'll search again. [SEARCH: something else]"
    
    # Mock search_web to return some generic result
    mock_search_web.return_value = "Search result: found something."

    # Configure LLM to always respond with a search command
    async def llm_always_searches_iterator(messages, system_prompt):
        for char_token in llm_response_always_searches:
            yield char_token
    mock_llm.chat_completion = llm_always_searches_iterator

    # The agent should hit MAX_SEARCH_ITERATIONS (currently 2)
    # It will perform search twice, then the third time it gets the search command,
    # it should just output that command as the final response.
    response = await collect_response_from_agent(basic_memory_agent_with_search, user_input)
    
    assert response == llm_response_always_searches 
    assert mock_search_web.call_count == 2 # Called twice due to MAX_SEARCH_ITERATIONS = 2

@pytest.mark.asyncio
@patch('open_llm_vtuber.agent.agents.basic_memory_agent.search_web')
async def test_agent_search_fails(mock_search_web, basic_memory_agent_with_search, mock_llm):
    user_input = "Search for something that will fail."
    llm_asks_for_search = "Let me search. [SEARCH: failing query]"
    search_error_message = "Web search failed: Simulated error"
    llm_final_response = "Sorry, I couldn't complete the search due to an error."

    mock_search_web.return_value = search_error_message # search_web returns the error message string

    llm_call_count = 0
    async def multi_step_llm_response(messages, system_prompt):
        nonlocal llm_call_count
        llm_call_count += 1
        if llm_call_count == 1: # LLM asks to search
            for char_token in llm_asks_for_search: yield char_token
        elif llm_call_count == 2: # LLM gets search error message
            assert search_error_message in messages[-1]["content"]
            for char_token in llm_final_response: yield char_token
        else:
            for char_token in "Unexpected LLM call": yield char_token
                
    mock_llm.chat_completion = multi_step_llm_response

    response = await collect_response_from_agent(basic_memory_agent_with_search, user_input)

    assert response == llm_final_response
    mock_search_web.assert_called_once_with("failing query", num_results=3)

@pytest.mark.asyncio
async def test_agent_no_search_command_with_search_enabled(basic_memory_agent_with_search, mock_llm):
    """Test that if search is enabled but LLM doesn't issue command, it works normally."""
    user_input = "Just a normal question."
    llm_response = "Here's a normal answer."

    async def specific_response_iterator():
        for char_token in llm_response:
            yield char_token
    mock_llm.chat_completion.custom_response_iterator = specific_response_iterator()
    
    with patch('open_llm_vtuber.agent.agents.basic_memory_agent.search_web') as mock_search_web_not_called:
        response = await collect_response_from_agent(basic_memory_agent_with_search, user_input)
        assert response == llm_response
        mock_search_web_not_called.assert_not_called()

    assert len(basic_memory_agent_with_search._memory) == 3 # System, User, Assistant
    assert basic_memory_agent_with_search._memory[2]["content"] == llm_response
