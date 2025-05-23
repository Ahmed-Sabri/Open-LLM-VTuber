import pytest
from unittest.mock import patch, mock_open, call
import os
from datetime import datetime

from open_llm_vtuber.conversations.recording_utils import save_conversation_message
from open_llm_vtuber.config_manager.character import ConversationRecordingConfig

@pytest.fixture
def recording_config_enabled():
    return ConversationRecordingConfig(
        enable_recording=True,
        recording_directory="test_recordings",
        audio_format="mp3",
        text_format="md"
    )

@pytest.fixture
def recording_config_disabled():
    return ConversationRecordingConfig(enable_recording=False)

@pytest.fixture
def mock_datetime_now():
    # Fixed datetime for consistent filenames
    return datetime(2023, 10, 26, 12, 30, 0, 123456)


@patch("os.makedirs")
@patch("builtins.open", new_callable=mock_open)
@patch("shutil.copy")
@patch("open_llm_vtuber.conversations.recording_utils.datetime") # Mock datetime within the module
def test_save_user_text_message(
    mock_dt, mock_shutil_copy, mock_file_open, mock_makedirs, 
    recording_config_enabled, mock_datetime_now
):
    mock_dt.now.return_value = mock_datetime_now
    session_id = "session_abc123"
    char_name = "TestChar"
    
    save_conversation_message(
        recording_config=recording_config_enabled,
        character_name_or_human=char_name,
        session_id=session_id,
        message_type="user",
        text_content="Hello AI!"
    )

    expected_timestamp_str = "20231026_123000_123"
    expected_dir = os.path.join("test_recordings", char_name, session_id)
    mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)
    
    expected_filename = f"{expected_timestamp_str}_user.md"
    expected_filepath = os.path.join(expected_dir, expected_filename)
    mock_file_open.assert_called_once_with(expected_filepath, "w", encoding="utf-8")
    mock_file_open().write.assert_called_once_with("Hello AI!")

@patch("os.makedirs")
@patch("builtins.open", new_callable=mock_open)
@patch("shutil.copy")
@patch("open_llm_vtuber.conversations.recording_utils.datetime")
def test_save_ai_text_and_audio_path_message(
    mock_dt, mock_shutil_copy, mock_file_open, mock_makedirs,
    recording_config_enabled, mock_datetime_now
):
    mock_dt.now.return_value = mock_datetime_now
    session_id = "session_xyz789"
    char_name = "AIAssistant"
    original_audio_path = "/path/to/original/audio.mp3"

    with patch("os.path.exists", return_value=True): # Assume source audio file exists
        save_conversation_message(
            recording_config=recording_config_enabled,
            character_name_or_human=char_name,
            session_id=session_id,
            message_type="ai",
            text_content="This is the AI response.",
            audio_content_or_path=original_audio_path
        )

    expected_timestamp_str = "20231026_123000_123"
    expected_dir = os.path.join("test_recordings", char_name, session_id)
    mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)

    # Check text saving
    expected_text_filename = f"{expected_timestamp_str}_ai.md"
    expected_text_filepath = os.path.join(expected_dir, expected_text_filename)
    
    # Check audio saving (copy)
    expected_audio_filename = f"{expected_timestamp_str}_ai.mp3"
    expected_audio_filepath = os.path.join(expected_dir, expected_audio_filename)

    # Calls to open will be for text, then potentially for audio if it were bytes
    # Since it's a path, shutil.copy is used for audio.
    mock_file_open.assert_called_once_with(expected_text_filepath, "w", encoding="utf-8")
    mock_file_open().write.assert_called_once_with("This is the AI response.")
    
    mock_shutil_copy.assert_called_once_with(original_audio_path, expected_audio_filepath)


@patch("os.makedirs")
@patch("builtins.open", new_callable=mock_open)
@patch("shutil.copy")
@patch("open_llm_vtuber.conversations.recording_utils.datetime")
def test_save_ai_audio_bytes_message(
    mock_dt, mock_shutil_copy, mock_file_open, mock_makedirs,
    recording_config_enabled, mock_datetime_now
):
    mock_dt.now.return_value = mock_datetime_now
    session_id = "session_bytes456"
    char_name = "AIBytes"
    audio_bytes = b"dummy_audio_bytes_content"

    save_conversation_message(
        recording_config=recording_config_enabled,
        character_name_or_human=char_name,
        session_id=session_id,
        message_type="ai",
        audio_content_or_path=audio_bytes
    )

    expected_timestamp_str = "20231026_123000_123"
    expected_dir = os.path.join("test_recordings", char_name, session_id)
    mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)

    expected_audio_filename = f"{expected_timestamp_str}_ai.mp3" # .mp3 from fixture
    expected_audio_filepath = os.path.join(expected_dir, expected_audio_filename)
    
    mock_file_open.assert_called_once_with(expected_audio_filepath, "wb")
    mock_file_open().write.assert_called_once_with(audio_bytes)
    mock_shutil_copy.assert_not_called() # shutil.copy should not be called for bytes

def test_recording_disabled(recording_config_disabled):
    with patch("os.makedirs") as mock_makedirs, \
         patch("builtins.open") as mock_file_open, \
         patch("shutil.copy") as mock_shutil_copy:
        
        save_conversation_message(
            recording_config=recording_config_disabled,
            character_name_or_human="AnyChar",
            session_id="any_session",
            message_type="user",
            text_content="Should not be saved."
        )
        
        mock_makedirs.assert_not_called()
        mock_file_open.assert_not_called()
        mock_shutil_copy.assert_not_called()

@patch("os.makedirs")
@patch("builtins.open", new_callable=mock_open)
@patch("open_llm_vtuber.conversations.recording_utils.datetime")
def test_save_different_formats(mock_dt, mock_file_open, mock_makedirs, mock_datetime_now):
    mock_dt.now.return_value = mock_datetime_now
    config = ConversationRecordingConfig(
        enable_recording=True,
        recording_directory="varied_formats",
        audio_format="ogg",
        text_format="txt"
    )
    session_id = "format_test_sess"
    char_name = "FormatTester"

    # Test text format
    save_conversation_message(
        recording_config=config,
        character_name_or_human=char_name,
        session_id=session_id,
        message_type="user",
        text_content="Testing .txt format"
    )
    expected_text_path = os.path.join("varied_formats", char_name, session_id, "20231026_123000_123_user.txt")
    
    # Test audio format (with bytes)
    save_conversation_message(
        recording_config=config,
        character_name_or_human=char_name,
        session_id=session_id,
        message_type="ai",
        audio_content_or_path=b"ogg_bytes"
    )
    expected_audio_path = os.path.join("varied_formats", char_name, session_id, "20231026_123000_123_ai.ogg")

    # Check calls to open
    # The first call is for text, the second for audio bytes
    calls = [
        call(expected_text_path, "w", encoding="utf-8"),
        call(expected_audio_path, "wb")
    ]
    mock_file_open.assert_has_calls(calls, any_order=False) # Order matters here based on calls above

@patch("os.makedirs")
@patch("open_llm_vtuber.conversations.recording_utils.logger") # Mock logger
def test_save_message_no_session_id(mock_logger, mock_makedirs, recording_config_enabled):
    save_conversation_message(
        recording_config=recording_config_enabled,
        character_name_or_human="TestChar",
        session_id="", # Empty session_id
        message_type="user",
        text_content="Test content"
    )
    mock_makedirs.assert_not_called()
    mock_logger.warning.assert_called_once_with("Conversation recording skipped: session_id is missing.")

@patch("os.makedirs")
@patch("shutil.copy")
@patch("os.path.exists", return_value=False) # Source audio file does not exist
@patch("open_llm_vtuber.conversations.recording_utils.logger")
@patch("open_llm_vtuber.conversations.recording_utils.datetime")
def test_save_audio_path_not_exists(
    mock_dt, mock_logger, mock_os_path_exists, mock_shutil_copy, mock_makedirs,
    recording_config_enabled, mock_datetime_now
):
    mock_dt.now.return_value = mock_datetime_now
    session_id = "audio_not_exist_sess"
    char_name = "AudioFailChar"
    non_existent_audio_path = "/path/to/non_existent/audio.mp3"

    save_conversation_message(
        recording_config=recording_config_enabled,
        character_name_or_human=char_name,
        session_id=session_id,
        message_type="ai",
        audio_content_or_path=non_existent_audio_path
    )
    
    expected_dir = os.path.join("test_recordings", char_name, session_id)
    mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)
    mock_shutil_copy.assert_not_called()
    mock_logger.error.assert_called_once_with(f"Audio file not found at path: {non_existent_audio_path}")

@patch("os.makedirs", side_effect=OSError("Disk full"))
@patch("open_llm_vtuber.conversations.recording_utils.logger")
@patch("open_llm_vtuber.conversations.recording_utils.datetime")
def test_save_message_os_error_makedirs(mock_dt, mock_logger, mock_makedirs, recording_config_enabled, mock_datetime_now):
    mock_dt.now.return_value = mock_datetime_now
    save_conversation_message(
        recording_config=recording_config_enabled,
        character_name_or_human="ErrorChar",
        session_id="error_session",
        message_type="user",
        text_content="Content that fails"
    )
    mock_logger.error.assert_called_once_with("Error saving conversation message: Disk full")

@patch("os.makedirs") # Succeeds
@patch("builtins.open", mock_open()) # Mock open to allow entering the 'with' block
@patch("shutil.copy", side_effect=OSError("Permission denied on copy"))
@patch("open_llm_vtuber.conversations.recording_utils.logger")
@patch("open_llm_vtuber.conversations.recording_utils.datetime")
@patch("os.path.exists", return_value=True)
def test_save_message_os_error_shutil_copy(
    mock_os_path_exists, mock_dt, mock_logger, mock_shutil_copy, mock_file_open, mock_makedirs,
    recording_config_enabled, mock_datetime_now
):
    mock_dt.now.return_value = mock_datetime_now
    save_conversation_message(
        recording_config=recording_config_enabled,
        character_name_or_human="ErrorCharCopy",
        session_id="error_copy_session",
        message_type="ai",
        audio_content_or_path="/path/to/source.mp3" # Path to trigger shutil.copy
    )
    mock_logger.error.assert_called_once_with("Error saving conversation message: Permission denied on copy")

@patch("os.makedirs") # Succeeds
@patch("builtins.open", mock_open(side_effect=OSError("Permission denied on open")))
@patch("open_llm_vtuber.conversations.recording_utils.logger")
@patch("open_llm_vtuber.conversations.recording_utils.datetime")
def test_save_message_os_error_open_text(
    mock_dt, mock_logger, mock_file_open, mock_makedirs,
    recording_config_enabled, mock_datetime_now
):
    mock_dt.now.return_value = mock_datetime_now
    save_conversation_message(
        recording_config=recording_config_enabled,
        character_name_or_human="ErrorCharOpen",
        session_id="error_open_session",
        message_type="user",
        text_content="Text content" # To trigger text file open
    )
    mock_logger.error.assert_called_once_with("Error saving conversation message: Permission denied on open")
