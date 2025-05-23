from typing import AsyncIterator, List, Dict, Any, Callable, Literal
from loguru import logger

from .agent_interface import AgentInterface
from ..output_types import SentenceOutput, DisplayText
import re # For parsing search command
from ..stateless_llm.stateless_llm_interface import StatelessLLMInterface
from ...chat_history_manager import get_history
from ..tools.web_search_tool import search_web # Import the search tool
from ..transformers import (
    sentence_divider,
    actions_extractor,
    tts_filter,
    display_processor,
)
from ...config_manager import TTSPreprocessorConfig
from ..input_types import BatchInput, TextSource, ImageSource
from prompts import prompt_loader


class BasicMemoryAgent(AgentInterface):
    """
    Agent with basic chat memory using a list to store messages.
    Implements text-based responses with sentence processing pipeline.
    """

    _system: str = """You are an error message repeater. 
        Your job is repeating this error message: 
        'No system prompt set. Please set a system prompt'. 
        Don't say anything else.
        """

    def __init__(
        self,
        llm: StatelessLLMInterface,
        system: str,
        live2d_model,
        tts_preprocessor_config: TTSPreprocessorConfig = None,
        faster_first_response: bool = True,
        segment_method: str = "pysbd",
        interrupt_method: Literal["system", "user"] = "user",
        enable_web_search: bool = False, # Added enable_web_search
    ):
        """
        Initialize the agent with LLM, system prompt and configuration

        Args:
            llm: `StatelessLLMInterface` - The LLM to use
            system: `str` - System prompt
            live2d_model: `Live2dModel` - Model for expression extraction
            tts_preprocessor_config: `TTSPreprocessorConfig` - Configuration for TTS preprocessing
            faster_first_response: `bool` - Whether to enable faster first response
            segment_method: `str` - Method for sentence segmentation
            interrupt_method: `Literal["system", "user"]` -
                Methods for writing interruptions signal in chat history.

        """
        super().__init__()
        self._memory = []
        self._live2d_model = live2d_model
        self._tts_preprocessor_config = tts_preprocessor_config
        self._faster_first_response = faster_first_response
        self._segment_method = segment_method
        self.interrupt_method = interrupt_method
        self._enable_web_search = enable_web_search # Store the flag
        # Flag to ensure a single interrupt handling per conversation
        self._interrupt_handled = False
        self._set_llm(llm)
        self.set_system(system)
        logger.info("BasicMemoryAgent initialized.")

    def _set_llm(self, llm: StatelessLLMInterface):
        """
        Set the (stateless) LLM to be used for chat completion.
        Instead of assigning directly to `self.chat`, store it to `_chat_function`
        so that the async method chat remains intact.

        Args:
            llm: StatelessLLMInterface - the LLM instance.
        """
        self._llm = llm
        self.chat = self._chat_function_factory(llm.chat_completion)

    def set_system(self, system: str):
        """
        Set the system prompt
        system: str
            the system prompt
        """
        logger.debug(f"Memory Agent: Setting system prompt: '''{system}'''")

        if self.interrupt_method == "user":
            system = f"{system}\n\nIf you received `[interrupted by user]` signal, you were interrupted."

        self._system = system

    def _add_message(
        self,
        message: str | List[Dict[str, Any]],
        role: str,
        display_text: DisplayText | None = None,
    ):
        """
        Add a message to the memory

        Args:
            message: Message content (string or list of content items)
            role: Message role
            display_text: Optional display information containing name and avatar
        """
        if isinstance(message, list):
            text_content = ""
            for item in message:
                if item.get("type") == "text":
                    text_content += item["text"]
        else:
            text_content = message

        message_data = {
            "role": role,
            "content": text_content,
        }

        # Add display information if provided
        if display_text:
            if display_text.name:
                message_data["name"] = display_text.name
            if display_text.avatar:
                message_data["avatar"] = display_text.avatar

        self._memory.append(message_data)

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        """Load the memory from chat history"""
        messages = get_history(conf_uid, history_uid)

        self._memory = []
        self._memory.append(
            {
                "role": "system",
                "content": self._system,
            }
        )

        for msg in messages:
            self._memory.append(
                {
                    "role": "user" if msg["role"] == "human" else "assistant",
                    "content": msg["content"],
                }
            )

    def handle_interrupt(self, heard_response: str) -> None:
        """
        Handle an interruption by the user.

        Args:
            heard_response: str - The part of the AI response heard by the user before interruption
        """
        if self._interrupt_handled:
            return

        self._interrupt_handled = True

        if self._memory and self._memory[-1]["role"] == "assistant":
            self._memory[-1]["content"] = heard_response + "..."
        else:
            if heard_response:
                self._memory.append(
                    {
                        "role": "assistant",
                        "content": heard_response + "...",
                    }
                )
        self._memory.append(
            {
                "role": "system" if self.interrupt_method == "system" else "user",
                "content": "[Interrupted by user]",
            }
        )

    def _to_text_prompt(self, input_data: BatchInput) -> str:
        """
        Format BatchInput into a prompt string for the LLM.

        Args:
            input_data: BatchInput - The input data containing texts and images

        Returns:
            str - Formatted message string
        """
        message_parts = []

        # Process text inputs in order
        for text_data in input_data.texts:
            if text_data.source == TextSource.INPUT:
                message_parts.append(text_data.content)
            elif text_data.source == TextSource.CLIPBOARD:
                message_parts.append(f"[Clipboard content: {text_data.content}]")

        # Process images in order
        if input_data.images:
            message_parts.append("\nImages in this message:")
            for i, img_data in enumerate(input_data.images, 1):
                source_desc = {
                    ImageSource.CAMERA: "captured from camera",
                    ImageSource.SCREEN: "screenshot",
                    ImageSource.CLIPBOARD: "from clipboard",
                    ImageSource.UPLOAD: "uploaded",
                }[img_data.source]
                message_parts.append(f"- Image {i} ({source_desc})")

        return "\n".join(message_parts)

    def _to_messages(self, input_data: BatchInput) -> List[Dict[str, Any]]:
        """
        Prepare messages list with image support.
        """
        messages = self._memory.copy()

        if input_data.images:
            content = []
            text_content = self._to_text_prompt(input_data)
            content.append({"type": "text", "text": text_content})

            for img_data in input_data.images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": img_data.data, "detail": "auto"},
                    }
                )

            user_message = {"role": "user", "content": content}
        else:
            user_message = {"role": "user", "content": self._to_text_prompt(input_data)}

        messages.append(user_message)
        self._add_message(user_message["content"], "user")
        return messages

    def _chat_function_factory(
        self, chat_func: Callable[[List[Dict[str, Any]], str], AsyncIterator[str]]
    ) -> Callable[..., AsyncIterator[SentenceOutput]]:
        """
        Create the chat pipeline with transformers

        The pipeline:
        LLM tokens -> sentence_divider -> actions_extractor -> display_processor -> tts_filter
        """

        @tts_filter(self._tts_preprocessor_config)
        @display_processor()
        @actions_extractor(self._live2d_model)
        @sentence_divider(
            faster_first_response=self._faster_first_response,
            segment_method=self._segment_method,
            valid_tags=["think"],
        )
        async def chat_with_memory(input_data: BatchInput) -> AsyncIterator[str]:
            """
            Chat implementation with memory and processing pipeline.
            Includes logic for web search if enabled and triggered by the LLM.

            Args:
                input_data: BatchInput

            Returns:
                AsyncIterator[str] - Token stream from LLM
            """

            messages = self._to_messages(input_data) # This also adds the user message to self._memory

            MAX_SEARCH_ITERATIONS = 2 # Prevent infinite search loops
            current_search_iteration = 0
            
            while current_search_iteration < MAX_SEARCH_ITERATIONS:
                current_search_iteration += 1
                
                # Get token stream from LLM
                token_stream = chat_func(messages, self._system)
                complete_response = ""

                # Accumulate the full response from the LLM first
                # The decorators expect to process the full response for things like sentence division
                raw_llm_output_for_search_check = ""
                async for token in token_stream:
                    raw_llm_output_for_search_check += token
                
                # Check for search command
                search_match = None
                if self._enable_web_search:
                    search_match = re.search(r"\[SEARCH:\s*(.*?)\]", raw_llm_output_for_search_check)

                if search_match:
                    search_query = search_match.group(1).strip()
                    logger.info(f"LLM requested search: '{search_query}'")
                    
                    # Remove the [SEARCH: query] part from the raw output before yielding
                    # and before potentially showing it to the user if search fails
                    # However, the current pipeline yields tokens directly.
                    # This means the [SEARCH:] tag might be visible briefly.
                    # For a cleaner approach, we might need to buffer the output if a search tag is found,
                    # perform search, then regenerate response.
                    # For now, we'll proceed with the search and then feed results for a new LLM call.

                    # Inform user/system that search is happening (optional, could be a yield)
                    # yield f"[Performing search for: {search_query}]" # This would need to fit SentenceOutput

                    search_results_str = search_web(search_query, num_results=3)
                    logger.info(f"Search results for '{search_query}':\n{search_results_str}")

                    # Prepare new context for the LLM
                    # The original user message is already in self._memory
                    # The LLM's message that contained [SEARCH:] should NOT be added to memory.
                    # Instead, we add a system message with search results.
                    
                    # Remove the user message that triggered the LLM to search if it's the last one
                    # The self._to_messages call adds the user message.
                    # The LLM's response (raw_llm_output_for_search_check) is not yet in memory.

                    search_context_message = (
                        f"Web search for query '{search_query}' yielded the following results:\n"
                        f"{search_results_str}\n"
                        "Please use these results to answer the user's request."
                    )
                    
                    # Add this as a new "user" turn to guide the LLM.
                    # Or, potentially, re-run with a modified system prompt or insert into messages.
                    # For simplicity, let's append it as a new user message to guide the next turn.
                    # This means the original user query is in memory, then the LLM's search command (which we intercept),
                    # then we inject search results as if the user provided them.
                    
                    # We need to reconstruct `messages` for the next LLM call.
                    # The `self._memory` already contains the original user query.
                    # We should not add raw_llm_output_for_search_check to memory.
                    messages.append({"role": "assistant", "content": raw_llm_output_for_search_check}) # Add LLM's attempt
                    messages.append({"role": "user", "content": search_context_message}) # Add search results as user guidance
                    
                    # Loop back to call chat_func again with the new messages list
                    # The original user input is still in self._memory from the first _to_messages call
                    # We add the LLM's attempt (with search tag) and our search results to messages for the next LLM call
                    # We don't add the LLM's attempt or our search results to self._memory here,
                    # as the next iteration's complete_response will be the one stored.
                    logger.info("Feeding search results back to LLM.")
                    continue # Go to the next iteration of the while loop for a new LLM call
                
                else:
                    # No search command, or search disabled, or max iterations reached.
                    # This is the final response. Yield it token by token.
                    # The `raw_llm_output_for_search_check` contains the full response here.
                    # We need to yield it in a way the decorators can process.
                    # The original design yields token by token from chat_func.
                    # We have already consumed the token_stream to build raw_llm_output_for_search_check.
                    # This requires a slight refactor in how tokens are yielded if we pre-buffer.

                    # For now, let's simulate the stream yield for the decorators.
                    # This ensures the decorators see the tokens as they would have.
                    for char_token in raw_llm_output_for_search_check:
                        yield char_token
                    complete_response = raw_llm_output_for_search_check
                    break # Exit search loop

            # Store final complete response (after potential search)
            self._add_message(complete_response, "assistant")

        return chat_with_memory

    async def chat(self, input_data: BatchInput) -> AsyncIterator[SentenceOutput]:
        """Placeholder chat method that will be replaced at runtime"""
        return self.chat(input_data)

    def reset_interrupt(self) -> None:
        """
        Reset the interrupt handled flag for a new conversation.
        """
        self._interrupt_handled = False

    def start_group_conversation(
        self, human_name: str, ai_participants: List[str]
    ) -> None:
        """
        Start a group conversation by adding a system message that informs the AI about
        the conversation participants.

        Args:
            human_name: str - Name of the human participant
            ai_participants: List[str] - Names of other AI participants in the conversation
        """
        other_ais = ", ".join(name for name in ai_participants)

        # Load and format the group conversation prompt
        group_context = prompt_loader.load_util("group_conversation_prompt").format(
            human_name=human_name, other_ais=other_ais
        )

        self._memory.append({"role": "user", "content": group_context})

        logger.debug(f"Added group conversation context: '''{group_context}'''")
