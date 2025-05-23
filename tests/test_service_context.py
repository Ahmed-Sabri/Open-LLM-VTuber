import pytest
from unittest.mock import MagicMock, patch

from open_llm_vtuber.service_context import ServiceContext
from open_llm_vtuber.config_manager import Config, CharacterConfig, SystemConfig
from open_llm_vtuber.email_sender import EmailSender # To check its instance

# Minimal config fixtures, similar to test_character_config
@pytest.fixture
def minimal_system_config_data():
    return {"tool_prompts": {}} # Assuming tool_prompts is part of SystemConfig

@pytest.fixture
def minimal_character_config_data_no_smtp(tmp_path):
    persona_file = tmp_path / "persona.md"
    persona_file.write_text("Test persona")
    return {
        "conf_name": "test_char_no_smtp",
        "conf_uid": "test_uid_no_smtp",
        "live2d_model_name": "test_model",
        "persona_prompt": str(persona_file),
        "agent_config": {"conversation_agent_choice": "basic_memory_agent", "agent_settings": {"basic_memory_agent": {"llm_provider":"test"}}, "llm_configs": {"test":{}}},
        "asr_config": {"asr_model": "faster_whisper", "faster_whisper": {}},
        "tts_config": {"tts_model": "edge_tts", "edge_tts": {}},
        "vad_config": {"vad_model": "silero_vad", "silero_vad": {}},
        "tts_preprocessor_config": {"translator_config": {"translate_audio": False, "translate_provider":"deeplx", "deeplx":{}}},
        # No SMTP fields
    }

@pytest.fixture
def minimal_character_config_data_with_smtp(minimal_character_config_data_no_smtp):
    config_data = minimal_character_config_data_no_smtp.copy()
    config_data["smtp_host"] = "smtp.example.com"
    config_data["smtp_port"] = 587
    config_data["smtp_use_ssl"] = False
    config_data["smtp_username"] = "user@example.com"
    config_data["smtp_password"] = "password"
    return config_data

@pytest.fixture
def full_config_no_smtp(minimal_system_config_data, minimal_character_config_data_no_smtp):
    # Wrap CharacterConfig data in the top-level Config structure
    return Config(
        system_config=SystemConfig(**minimal_system_config_data),
        character_config=CharacterConfig(**minimal_character_config_data_no_smtp)
    )

@pytest.fixture
def full_config_with_smtp(minimal_system_config_data, minimal_character_config_data_with_smtp):
    return Config(
        system_config=SystemConfig(**minimal_system_config_data),
        character_config=CharacterConfig(**minimal_character_config_data_with_smtp)
    )


@patch('open_llm_vtuber.service_context.Live2dModel')
@patch('open_llm_vtuber.service_context.ASRFactory')
@patch('open_llm_vtuber.service_context.TTSFactory')
@patch('open_llm_vtuber.service_context.VADFactory')
@patch('open_llm_vtuber.service_context.AgentFactory')
@patch('open_llm_vtuber.service_context.TranslateFactory')
@patch('open_llm_vtuber.email_sender.EmailSender') # Mock EmailSender itself for these tests
def test_service_context_email_sender_initialized_with_smtp_config(
    MockEmailSenderConstructor, MockTranslateFactory, MockAgentFactory, MockVADFactory, 
    MockTTSFactory, MockASRFactory, MockLive2dModel, 
    full_config_with_smtp
):
    """Test EmailSender is initialized when SMTP config is present."""
    mock_email_sender_instance = MagicMock(spec=EmailSender)
    MockEmailSenderConstructor.return_value = mock_email_sender_instance
    
    # Mock return values for other factories to avoid deeper errors
    MockLive2dModel.return_value = MagicMock()
    MockASRFactory.get_asr_system.return_value = MagicMock()
    MockTTSFactory.get_tts_engine.return_value = MagicMock()
    MockVADFactory.get_vad_engine.return_value = MagicMock()
    MockAgentFactory.create_agent.return_value = MagicMock()
    MockTranslateFactory.get_translator.return_value = MagicMock()

    context = ServiceContext()
    context.load_from_config(full_config_with_smtp)

    assert context.email_sender is not None
    assert context.email_sender == mock_email_sender_instance
    MockEmailSenderConstructor.assert_called_once_with(
        host="smtp.example.com",
        port=587,
        use_ssl=False,
        username="user@example.com",
        password="password"
    )

@patch('open_llm_vtuber.service_context.Live2dModel')
@patch('open_llm_vtuber.service_context.ASRFactory')
@patch('open_llm_vtuber.service_context.TTSFactory')
@patch('open_llm_vtuber.service_context.VADFactory')
@patch('open_llm_vtuber.service_context.AgentFactory')
@patch('open_llm_vtuber.service_context.TranslateFactory')
@patch('open_llm_vtuber.email_sender.EmailSender')
def test_service_context_email_sender_none_without_smtp_config(
    MockEmailSenderConstructor, MockTranslateFactory, MockAgentFactory, MockVADFactory, 
    MockTTSFactory, MockASRFactory, MockLive2dModel,
    full_config_no_smtp
):
    """Test EmailSender is None when SMTP host is not configured."""
    MockLive2dModel.return_value = MagicMock()
    MockASRFactory.get_asr_system.return_value = MagicMock()
    MockTTSFactory.get_tts_engine.return_value = MagicMock()
    MockVADFactory.get_vad_engine.return_value = MagicMock()
    MockAgentFactory.create_agent.return_value = MagicMock()
    MockTranslateFactory.get_translator.return_value = MagicMock()
    
    context = ServiceContext()
    context.load_from_config(full_config_no_smtp)

    assert context.email_sender is None
    MockEmailSenderConstructor.assert_not_called()

@patch('open_llm_vtuber.service_context.Live2dModel')
@patch('open_llm_vtuber.service_context.ASRFactory')
@patch('open_llm_vtuber.service_context.TTSFactory')
@patch('open_llm_vtuber.service_context.VADFactory')
@patch('open_llm_vtuber.service_context.AgentFactory')
@patch('open_llm_vtuber.service_context.TranslateFactory')
@patch('open_llm_vtuber.email_sender.EmailSender')
def test_service_context_email_sender_none_if_smtp_host_empty(
    MockEmailSenderConstructor, MockTranslateFactory, MockAgentFactory, MockVADFactory,
    MockTTSFactory, MockASRFactory, MockLive2dModel,
    minimal_system_config_data, minimal_character_config_data_with_smtp, tmp_path
):
    """Test EmailSender is None if smtp_host is present but empty."""
    # Create a character config where smtp_host is an empty string
    char_config_data = minimal_character_config_data_with_smtp.copy()
    char_config_data["smtp_host"] = "" # Empty host
    
    config = Config(
        system_config=SystemConfig(**minimal_system_config_data),
        character_config=CharacterConfig(**char_config_data)
    )

    MockLive2dModel.return_value = MagicMock()
    MockASRFactory.get_asr_system.return_value = MagicMock()
    MockTTSFactory.get_tts_engine.return_value = MagicMock()
    MockVADFactory.get_vad_engine.return_value = MagicMock()
    MockAgentFactory.create_agent.return_value = MagicMock()
    MockTranslateFactory.get_translator.return_value = MagicMock()

    context = ServiceContext()
    context.load_from_config(config)

    assert context.email_sender is None
    MockEmailSenderConstructor.assert_not_called()

@patch('open_llm_vtuber.service_context.Live2dModel')
@patch('open_llm_vtuber.service_context.ASRFactory')
@patch('open_llm_vtuber.service_context.TTSFactory')
@patch('open_llm_vtuber.service_context.VADFactory')
@patch('open_llm_vtuber.service_context.AgentFactory')
@patch('open_llm_vtuber.service_context.TranslateFactory')
@patch('open_llm_vtuber.email_sender.EmailSender', side_effect=Exception("EmailSender init failed"))
def test_service_context_email_sender_init_exception(
    MockEmailSenderConstructor, MockTranslateFactory, MockAgentFactory, MockVADFactory,
    MockTTSFactory, MockASRFactory, MockLive2dModel,
    full_config_with_smtp
):
    """Test that if EmailSender fails to initialize, context.email_sender is None."""
    MockLive2dModel.return_value = MagicMock()
    MockASRFactory.get_asr_system.return_value = MagicMock()
    MockTTSFactory.get_tts_engine.return_value = MagicMock()
    MockVADFactory.get_vad_engine.return_value = MagicMock()
    MockAgentFactory.create_agent.return_value = MagicMock()
    MockTranslateFactory.get_translator.return_value = MagicMock()

    context = ServiceContext()
    # load_from_config should catch the exception and set email_sender to None
    context.load_from_config(full_config_with_smtp) 
    
    assert context.email_sender is None
    MockEmailSenderConstructor.assert_called_once() # Attempted to initialize
    # Check logs for "Failed to initialize EmailSender" if logging is part of the test setup
    # For now, just checking the outcome.
