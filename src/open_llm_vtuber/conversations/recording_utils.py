import os
import shutil
from datetime import datetime
from loguru import logger
from typing import Optional

from ..config_manager.character import ConversationRecordingConfig

def save_conversation_message(
    recording_config: ConversationRecordingConfig,
    character_name_or_human: str, # Name of the character or "Human"
    session_id: str, # e.g., history_uid for single, group_id for group
    message_type: str, # "user" or "ai"
    text_content: Optional[str] = None,
    audio_content_or_path: Optional[Union[bytes, str]] = None, # Can be bytes or path to audio file
    timestamp: Optional[datetime] = None,
):
    if not recording_config.enable_recording:
        return

    if not session_id:
        logger.warning("Conversation recording skipped: session_id is missing.")
        return

    try:
        ts = timestamp or datetime.now()
        formatted_timestamp = ts.strftime("%Y%m%d_%H%M%S_%f")[:-3] # Milliseconds

        # Sanitize character_name_or_human and session_id for directory/filename usage
        safe_character_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in character_name_or_human)
        safe_session_id = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in session_id)

        base_path = os.path.join(
            recording_config.recording_directory,
            safe_character_name, # Sub-directory for the character/human
            safe_session_id      # Sub-directory for the session
        )
        os.makedirs(base_path, exist_ok=True)

        if text_content is not None:
            text_filename = f"{formatted_timestamp}_{message_type}.{recording_config.text_format}"
            text_filepath = os.path.join(base_path, text_filename)
            with open(text_filepath, "w", encoding="utf-8") as f:
                f.write(text_content)
            logger.info(f"Saved {message_type} text to: {text_filepath}")

        if audio_content_or_path is not None:
            audio_filename = f"{formatted_timestamp}_{message_type}.{recording_config.audio_format}"
            audio_filepath = os.path.join(base_path, audio_filename)
            if isinstance(audio_content_or_path, str): # It's a path
                if os.path.exists(audio_content_or_path):
                    shutil.copy(audio_content_or_path, audio_filepath)
                    logger.info(f"Copied {message_type} audio from {audio_content_or_path} to: {audio_filepath}")
                else:
                    logger.error(f"Audio file not found at path: {audio_content_or_path}")
            elif isinstance(audio_content_or_path, bytes): # It's raw bytes
                with open(audio_filepath, "wb") as f:
                    f.write(audio_content_or_path)
                logger.info(f"Saved {message_type} audio bytes to: {audio_filepath}")
            else:
                logger.warning(f"Unsupported audio content type: {type(audio_content_or_path)}")

    except Exception as e:
        logger.error(f"Error saving conversation message: {e}")

# Example usage (for testing purposes, will be removed or commented out)
if __name__ == "__main__":
    class MockRecordingConfig:
        enable_recording = True
        recording_directory = "test_conversations"
        audio_format = "wav"
        text_format = "txt"

    mock_config = MockRecordingConfig()
    now = datetime.now()

    # Test user text
    save_conversation_message(
        recording_config=mock_config,
        character_name_or_human="TestChar",
        session_id="session123",
        message_type="user",
        text_content="Hello, this is a user message.",
        timestamp=now
    )

    # Test AI text and audio (path)
    # Create a dummy audio file
    dummy_audio_path = "dummy_audio.wav"
    with open(dummy_audio_path, "wb") as f:
        f.write(b"dummy audio data")

    save_conversation_message(
        recording_config=mock_config,
        character_name_or_human="TestChar",
        session_id="session123",
        message_type="ai",
        text_content="Hello, this is an AI response.",
        audio_content_or_path=dummy_audio_path,
        timestamp=now
    )
    os.remove(dummy_audio_path) # Clean up dummy file

    # Test AI audio (bytes)
    save_conversation_message(
        recording_config=mock_config,
        character_name_or_human="TestChar",
        session_id="session123",
        message_type="ai",
        audio_content_or_path=b"more dummy audio data as bytes",
        timestamp=now
    )
    logger.info("Example usage complete. Check the 'test_conversations' directory.")
    # You might want to manually clean up 'test_conversations' directory after testing
    # import shutil
    # if os.path.exists(mock_config.recording_directory):
    #     shutil.rmtree(mock_config.recording_directory)
