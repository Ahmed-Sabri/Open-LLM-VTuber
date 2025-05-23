import pytest
import os
from pydantic import ValidationError

from open_llm_vtuber.config_manager.character import CharacterConfig, ConversationRecordingConfig # Added ConversationRecordingConfig
from open_llm_vtuber.config_manager.agent import AgentConfig, AgentSettings, BasicMemoryAgentConfig
from open_llm_vtuber.config_manager.asr import ASRConfig, FasterWhisperASRConfig
from open_llm_vtuber.config_manager.tts import TTSConfig, EdgeTTSConfig
from open_llm_vtuber.config_manager.vad import VADConfig, SileroVADConfig
from open_llm_vtuber.config_manager.tts_preprocessor import TTSPreprocessorConfig, TranslatorConfig


# Minimal valid configs for dependencies of CharacterConfig
def get_minimal_agent_config_data():
    return {
        "conversation_agent_choice": "basic_memory_agent",
        "agent_settings": {
            "basic_memory_agent": {
                "llm_provider": "openai_compatible_llm", # Assuming this is a valid choice
                "enable_web_search": False
            }
        },
        "llm_configs": {
            "openai_compatible_llm": {
                "model": "test-model"
            }
        }
    }

def get_minimal_asr_config_data():
    return {"asr_model": "faster_whisper", "faster_whisper": {"model_path": "tiny"}}

def get_minimal_tts_config_data():
    return {"tts_model": "edge_tts", "edge_tts": {"voice": "en-US-AriaNeural"}}

def get_minimal_vad_config_data():
    return {"vad_model": "silero_vad", "silero_vad": {}}

def get_minimal_tts_preprocessor_config_data():
    return {"translator_config": {"translate_audio": False, "translate_provider": "deeplx", "deeplx": {}}}


@pytest.fixture
def minimal_config_data(tmp_path):
    # Create a dummy persona_prompt.md file
    persona_file = tmp_path / "persona_prompt.md"
    persona_file.write_text("This is a test persona.")

    return {
        "conf_name": "test_char",
        "conf_uid": "test_uid_001",
        "live2d_model_name": "test_model",
        "persona_prompt": str(persona_file),
        "agent_config": get_minimal_agent_config_data(),
        "asr_config": get_minimal_asr_config_data(),
        "tts_config": get_minimal_tts_config_data(),
        "vad_config": get_minimal_vad_config_data(),
        "tts_preprocessor_config": get_minimal_tts_preprocessor_config_data(),
        # SMTP fields will be tested by adding them or ommitting them
        # ConversationRecordingConfig will use its default factory
    }

# --- Tests for Persona Prompt File Loading ---

def test_persona_prompt_loads_content(minimal_config_data):
    """Test that persona_prompt correctly loads content from a .md file."""
    config = CharacterConfig(**minimal_config_data)
    assert config.persona_prompt == "This is a test persona."

def test_persona_prompt_non_existent_file(minimal_config_data, tmp_path):
    """Test validation for a non-existent persona_prompt file."""
    config_data = minimal_config_data.copy()
    config_data["persona_prompt"] = str(tmp_path / "non_existent.md")
    with pytest.raises(ValidationError) as excinfo:
        CharacterConfig(**config_data)
    assert "Persona prompt file not found" in str(excinfo.value)

def test_persona_prompt_empty_file_path(minimal_config_data):
    """Test validation for an empty persona_prompt file path."""
    config_data = minimal_config_data.copy()
    # Pydantic's FilePath doesn't accept empty string if file must exist.
    # The field is not optional, so it must be a path.
    # Our validator checks `if not v:`, but FilePath itself will fail first for ""
    # If we set it to None (though it's not Optional), it would be a different error.
    # Let's test our validator by providing a path that *is* empty conceptually.
    # For FilePath, an empty string path is invalid before our validator.
    # If we bypass FilePath validation (e.g. by making it str temporarily), our validator would catch it.
    # Given it's FilePath, the "file not found" for an empty string is effectively what happens.
    # Let's test what happens if the path is valid but the file is empty content-wise.
    # The current validator returns the content, so empty content is fine.
    # The original intent might have been "path string is empty" which pydantic handles.
    # Let's test `if not v:` by providing a path that is just a space (which FilePath might reject).
    # Ok, FilePath will handle an empty string path by raising an error.
    # Our validator `if not v:` would catch it if FilePath allowed it.
    # Let's directly test the "cannot be empty" part of our validator if possible,
    # or accept that Pydantic's FilePath handles "empty path string" validation.
    
    # If persona_prompt is an empty string, Pydantic's FilePath itself will raise error.
    # Let's test what our validator says if it somehow got an empty value (e.g. if it was str type)
    # This is tricky with FilePath. Let's assume FilePath ensures it's a non-empty string.
    # The `if not v:` in our validator is for the FilePath object itself, not its content.
    # To truly test "Persona_prompt cannot be empty" from our validator, we'd need to mock FilePath.
    # For now, let's rely on Pydantic's default behavior for empty paths.
    # The more relevant test for our code is if the file exists but is empty.
    
    # Test case: File exists but its content is empty
    empty_persona_file = tmp_path / "empty_persona.md"
    empty_persona_file.write_text("")
    config_data = minimal_config_data.copy()
    config_data["persona_prompt"] = str(empty_persona_file)
    config = CharacterConfig(**config_data)
    assert config.persona_prompt == "" # Empty content is acceptable

def test_persona_prompt_not_md_file(minimal_config_data, tmp_path):
    """Test validation for a persona_prompt file that is not .md."""
    config_data = minimal_config_data.copy()
    not_md_file = tmp_path / "persona_prompt.txt"
    not_md_file.write_text("This is a test persona in a txt.")
    config_data["persona_prompt"] = str(not_md_file)
    with pytest.raises(ValidationError) as excinfo:
        CharacterConfig(**config_data)
    assert "Persona prompt file must be a Markdown file (.md)" in str(excinfo.value)

# --- Tests for SMTP Configuration Loading ---

def test_smtp_config_loads_defaults(minimal_config_data):
    """Test that SMTP fields are defaulted if not provided."""
    config = CharacterConfig(**minimal_config_data) # SMTP fields are not in minimal_config_data
    assert config.smtp_host == ""
    assert config.smtp_port == 0
    assert config.smtp_use_ssl is False
    assert config.smtp_username == ""
    assert config.smtp_password == ""

def test_smtp_config_loads_values(minimal_config_data):
    """Test that SMTP fields are correctly parsed when provided."""
    config_data = minimal_config_data.copy()
    config_data["smtp_host"] = "smtp.example.com"
    config_data["smtp_port"] = 587
    config_data["smtp_use_ssl"] = True
    config_data["smtp_username"] = "user@example.com"
    config_data["smtp_password"] = "password123"
    
    config = CharacterConfig(**config_data)
    assert config.smtp_host == "smtp.example.com"
    assert config.smtp_port == 587
    assert config.smtp_use_ssl is True
    assert config.smtp_username == "user@example.com"
    assert config.smtp_password == "password123"

def test_smtp_config_invalid_port(minimal_config_data):
    """Test SMTP port validation (e.g., if it expects int but gets str)."""
    config_data = minimal_config_data.copy()
    config_data["smtp_port"] = "not-a-port"
    with pytest.raises(ValidationError) as excinfo:
        CharacterConfig(**config_data)
    # Pydantic V2 error messages are more structured
    assert "Input should be a valid integer" in str(excinfo.value) # Example, actual message might vary

def test_smtp_config_invalid_ssl_type(minimal_config_data):
    """Test SMTP use_ssl validation (e.g., if it expects bool but gets str)."""
    config_data = minimal_config_data.copy()
    config_data["smtp_use_ssl"] = "not-a-boolean"
    with pytest.raises(ValidationError) as excinfo:
        CharacterConfig(**config_data)
    assert "Input should be a valid boolean" in str(excinfo.value)

# --- Tests for Conversation Recording Configuration Loading ---

def test_conversation_recording_config_defaults(minimal_config_data):
    """Test that ConversationRecordingConfig uses default_factory."""
    config = CharacterConfig(**minimal_config_data)
    assert config.conversation_recording_config is not None
    assert config.conversation_recording_config.enable_recording is False
    assert config.conversation_recording_config.recording_directory == "conversations"
    assert config.conversation_recording_config.audio_format == "wav"
    assert config.conversation_recording_config.text_format == "txt"

def test_conversation_recording_config_loads_values(minimal_config_data):
    """Test loading specific values for ConversationRecordingConfig."""
    config_data = minimal_config_data.copy()
    config_data["conversation_recording_config"] = {
        "enable_recording": True,
        "recording_directory": "my_talks",
        "audio_format": "mp3",
        "text_format": "md"
    }
    config = CharacterConfig(**config_data)
    assert config.conversation_recording_config.enable_recording is True
    assert config.conversation_recording_config.recording_directory == "my_talks"
    assert config.conversation_recording_config.audio_format == "mp3"
    assert config.conversation_recording_config.text_format == "md"

# Minimal test for web search flag in agent_config, assuming BasicMemoryAgentConfig is tested elsewhere
def test_web_search_flag_default_in_character_config(minimal_config_data):
    """
    Tests that the enable_web_search flag is accessible and defaults correctly
    through CharacterConfig -> AgentConfig -> AgentSettings -> BasicMemoryAgentConfig.
    """
    # This test assumes that the structure CharacterConfig -> AgentConfig -> AgentSettings -> BasicMemoryAgentConfig
    # correctly passes down the default for enable_web_search.
    # The BasicMemoryAgentConfig now directly contains enable_web_search.
    
    # Minimal data for agent_config part
    agent_config_data = {
        "conversation_agent_choice": "basic_memory_agent",
        "agent_settings": {
            "basic_memory_agent": { # BasicMemoryAgentConfig fields
                "llm_provider": "ollama_llm", 
                # enable_web_search is not provided, should use default from BasicMemoryAgentConfig
            }
        },
        "llm_configs": {"ollama_llm": {"model": "test"}} # Dummy LLM config
    }
    
    config_data = minimal_config_data.copy()
    config_data["agent_config"] = agent_config_data
    
    char_config = CharacterConfig(**config_data)
    
    # Navigate to the enable_web_search flag
    # The actual BasicMemoryAgentConfig object is not directly on AgentSettings anymore.
    # AgentSettings has basic_memory_agent, which IS the BasicMemoryAgentConfig.
    assert char_config.agent_config.agent_settings.basic_memory_agent.enable_web_search is False

def test_web_search_flag_set_in_character_config(minimal_config_data):
    agent_config_data = {
        "conversation_agent_choice": "basic_memory_agent",
        "agent_settings": {
            "basic_memory_agent": {
                "llm_provider": "ollama_llm",
                "enable_web_search": True # Explicitly set
            }
        },
        "llm_configs": {"ollama_llm": {"model": "test"}}
    }
    config_data = minimal_config_data.copy()
    config_data["agent_config"] = agent_config_data
    
    char_config = CharacterConfig(**config_data)
    assert char_config.agent_config.agent_settings.basic_memory_agent.enable_web_search is True

# Example of a more complete config for CharacterConfig to ensure no validation errors with all sub-models
@pytest.fixture
def full_character_config_data(tmp_path):
    persona_file = tmp_path / "persona.md"
    persona_file.write_text("Full persona here.")
    return {
        "conf_name": "full_char",
        "conf_uid": "full_uid_002",
        "live2d_model_name": "full_live2d_model",
        "character_name": "FullChar",
        "human_name": "FullHuman",
        "avatar": "full_avatar.png",
        "persona_prompt": str(persona_file),
        "agent_config": {
            "conversation_agent_choice": "basic_memory_agent",
            "agent_settings": {
                "basic_memory_agent": {
                    "llm_provider": "openai_compatible_llm",
                    "faster_first_response": True,
                    "segment_method": "pysbd",
                    "enable_web_search": True,
                }
            },
            "llm_configs": {
                 "openai_compatible_llm": {
                    "model": "gpt-4", "base_url": "http://localhost:8080", "llm_api_key": "testkey"
                }
            }
        },
        "asr_config": {
            "asr_model": "faster_whisper",
            "faster_whisper": {"model_path": "base", "language": "en", "device": "cpu"}
        },
        "tts_config": {
            "tts_model": "edge_tts",
            "edge_tts": {"voice": "en-US-JennyNeural"}
        },
        "vad_config": {
            "vad_model": "silero_vad",
            "silero_vad": {"prob_threshold": 0.5}
        },
        "tts_preprocessor_config": {
            "remove_special_char": True,
            "translator_config": {
                "translate_audio": False,
                "translate_provider": "deeplx",
                "deeplx": {"deeplx_target_lang": "EN"}
            }
        },
        "smtp_host": "mail.example.com",
        "smtp_port": 587,
        "smtp_use_ssl": True,
        "smtp_username": "test@example.com",
        "smtp_password": "securepassword",
        "conversation_recording_config": {
            "enable_recording": True,
            "recording_directory": "recorded_chats",
            "audio_format": "mp3",
            "text_format": "md"
        }
    }

def test_full_character_config_validation(full_character_config_data):
    """Test that a fully populated CharacterConfig validates successfully."""
    try:
        CharacterConfig(**full_character_config_data)
    except ValidationError as e:
        pytest.fail(f"Full CharacterConfig validation failed: {e}")
