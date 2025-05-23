# config_manager/character.py
from pydantic import BaseModel, Field, field_validator, FilePath
from typing import Dict, ClassVar
from .i18n import I18nMixin, Description
from .asr import ASRConfig
from .tts import TTSConfig
from .vad import VADConfig
from .tts_preprocessor import TTSPreprocessorConfig

from .agent import AgentConfig


class ConversationRecordingConfig(BaseModel):
    """Conversation recording settings."""
    enable_recording: bool = Field(default=False, alias="enable_recording")
    recording_directory: str = Field(default="conversations", alias="recording_directory")
    audio_format: str = Field(default="wav", alias="audio_format")  # e.g., wav, mp3
    text_format: str = Field(default="txt", alias="text_format")    # e.g., txt, md

    class Config:
        # Pydantic V2 allows this for aliasing during model_dump
        populate_by_name = True


class CharacterConfig(I18nMixin):
    """Character configuration settings."""

    conf_name: str = Field(..., alias="conf_name")
    conf_uid: str = Field(..., alias="conf_uid")
    live2d_model_name: str = Field(..., alias="live2d_model_name")
    character_name: str = Field(default="", alias="character_name")
    human_name: str = Field(default="Human", alias="human_name")
    avatar: str = Field(default="", alias="avatar")
    persona_prompt: FilePath = Field(..., alias="persona_prompt")
    agent_config: AgentConfig = Field(..., alias="agent_config")
    asr_config: ASRConfig = Field(..., alias="asr_config")
    tts_config: TTSConfig = Field(..., alias="tts_config")
    vad_config: VADConfig = Field(..., alias="vad_config")
    tts_preprocessor_config: TTSPreprocessorConfig = Field(
        ..., alias="tts_preprocessor_config"
    )
    smtp_host: str = Field(default="", alias="smtp_host")
    smtp_port: int = Field(default=0, alias="smtp_port")
    smtp_use_ssl: bool = Field(default=False, alias="smtp_use_ssl")
    smtp_username: str = Field(default="", alias="smtp_username")
    smtp_password: str = Field(default="", alias="smtp_password")
    conversation_recording_config: ConversationRecordingConfig = Field(default_factory=ConversationRecordingConfig, alias="conversation_recording_config")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "conf_name": Description(
            en="Name of the character configuration", zh="角色配置名称"
        ),
        "conf_uid": Description(
            en="Unique identifier for the character configuration",
            zh="角色配置唯一标识符",
        ),
        "live2d_model_name": Description(
            en="Name of the Live2D model to use", zh="使用的Live2D模型名称"
        ),
        "character_name": Description(
            en="Name of the AI character in conversation", zh="对话中AI角色的名字"
        ),
        "persona_prompt": Description(
            en="Path to a Markdown file containing the persona prompt for the character.", zh="角色人设提示词Markdown文件路径"
        ),
        "agent_config": Description(
            en="Configuration for the conversation agent", zh="对话代理配置"
        ),
        "asr_config": Description(
            en="Configuration for Automatic Speech Recognition", zh="语音识别配置"
        ),
        "tts_config": Description(
            en="Configuration for Text-to-Speech", zh="语音合成配置"
        ),
        "vad_config": Description(
            en="Configuration for Voice Activity Detection", zh="语音活动检测配置"
        ),
        "tts_preprocessor_config": Description(
            en="Configuration for Text-to-Speech Preprocessor",
            zh="语音合成预处理器配置",
        ),
        "human_name": Description(
            en="Name of the human user in conversation", zh="对话中人类用户的名字"
        ),
        "avatar": Description(
            en="Avatar image path for the character", zh="角色头像图片路径"
        ),
        "smtp_host": Description(en="SMTP server host", zh="SMTP服务器主机"),
        "smtp_port": Description(en="SMTP server port", zh="SMTP服务器端口"),
        "smtp_use_ssl": Description(
            en="Whether to use SSL/TLS for SMTP connection",
            zh="SMTP连接是否使用SSL/TLS",
        ),
        "smtp_username": Description(en="SMTP username", zh="SMTP用户名"),
        "smtp_password": Description(en="SMTP password", zh="SMTP密码"),
        "conversation_recording_config": Description(
            en="Configuration for recording conversations", zh="对话录制配置"
        ),
    }

    @field_validator("persona_prompt")
    def check_persona_prompt_file(cls, v: FilePath):
        if not v:
            raise ValueError("Persona_prompt cannot be empty. Please provide a file path.")
        if not v.exists():
            raise ValueError(f"Persona prompt file not found: {v}")
        if v.suffix != ".md":
            raise ValueError(f"Persona prompt file must be a Markdown file (.md): {v}")
        return v.read_text(encoding="utf-8")

    @field_validator("character_name")
    def set_default_character_name(cls, v, values):
        if not v and "conf_name" in values:
            return values["conf_name"]
        return v
