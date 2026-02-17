import os
import sys
import tempfile

import pytest

from config import VoicePipelineConfig
from domain.conversation import ConversationHistory
from domain.pipeline import VoicePipeline
from domain.speech_detector import SpeechDetector
from domain.state import PipelineState
from domain.wake_word import WakeWordDetector
from ports.transcriber import TranscriptEvent

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import (
    FRAME_DURATION_MS,
    FakeAudioCapture,
    FakeAudioPlayback,
    FakeCompletion,
    FakeSynthesizer,
    FakeTranscriber,
    FakeVad,
)


def create_pipeline_with_languages(
    agent_language_map: dict[str, str] | None = None,
    system_prompt: str = "Be concise.",
    wake_words: list[str] | None = None,
    default_agent: str = "jarvis",
    agent_voice_map: dict[str, str] | None = None,
    completion: FakeCompletion | None = None,
) -> VoicePipeline:
    vad = FakeVad()
    detector = SpeechDetector(
        vad=vad,
        threshold=0.5,
        min_silence_ms=300,
        frame_duration_ms=FRAME_DURATION_MS,
    )
    return VoicePipeline(
        capture=FakeAudioCapture(),
        playback=FakeAudioPlayback(),
        transcriber=FakeTranscriber(),
        completion=completion or FakeCompletion(),
        synthesizer=FakeSynthesizer(),
        speech_detector=detector,
        wake_word_detector=WakeWordDetector(wake_words or ["jarvis", "robson"]),
        conversation=ConversationHistory(max_turns=20),
        default_agent=default_agent,
        agent_language_map=agent_language_map,
        agent_voice_map=agent_voice_map,
        system_prompt=system_prompt,
    )


class TestAgentLanguageSystemPrompt:
    def test_system_prompt_without_language_map_returns_base_prompt(self):
        pipeline = create_pipeline_with_languages(
            system_prompt="Be concise.",
            agent_language_map={},
        )
        assert pipeline._get_system_prompt() == "Be concise."

    def test_system_prompt_with_language_appends_language_instruction(self):
        pipeline = create_pipeline_with_languages(
            system_prompt="Be concise.",
            agent_language_map={"jarvis": "English", "robson": "Portuguese"},
        )
        assert pipeline._get_system_prompt() == "Be concise. Always respond in English."

    def test_system_prompt_changes_when_agent_switches(self):
        pipeline = create_pipeline_with_languages(
            system_prompt="Be concise.",
            agent_language_map={"jarvis": "English", "robson": "Portuguese"},
        )
        assert "English" in pipeline._get_system_prompt()

        pipeline._agent = "robson"
        assert "Portuguese" in pipeline._get_system_prompt()

    def test_system_prompt_for_agent_without_language_returns_base(self):
        pipeline = create_pipeline_with_languages(
            system_prompt="Be concise.",
            agent_language_map={"robson": "Portuguese"},
        )
        assert pipeline._get_system_prompt() == "Be concise."

    def test_none_language_map_treated_as_empty(self):
        pipeline = create_pipeline_with_languages(
            system_prompt="Be concise.",
            agent_language_map=None,
        )
        assert pipeline._get_system_prompt() == "Be concise."


class TestAgentSwitchOnWakeWord:
    @pytest.mark.asyncio
    async def test_wake_word_switches_agent(self):
        pipeline = create_pipeline_with_languages(default_agent="jarvis")
        pipeline._state = PipelineState.AMBIENT

        await pipeline._handle_transcript(
            TranscriptEvent(text="Hey Robson, what time is it?", is_final=False)
        )

        assert pipeline._agent == "robson"

    @pytest.mark.asyncio
    async def test_same_agent_wake_word_does_not_clear_conversation(self):
        pipeline = create_pipeline_with_languages(default_agent="jarvis")
        pipeline._state = PipelineState.AMBIENT
        pipeline._conversation.add_user_message("previous question")
        pipeline._conversation.add_assistant_message("previous answer")

        await pipeline._handle_transcript(
            TranscriptEvent(text="Hey Jarvis, another question", is_final=False)
        )

        assert pipeline._agent == "jarvis"
        assert len(pipeline._conversation._messages) == 2

    @pytest.mark.asyncio
    async def test_switching_agent_clears_conversation_history(self):
        pipeline = create_pipeline_with_languages(default_agent="jarvis")
        pipeline._state = PipelineState.AMBIENT
        pipeline._conversation.add_user_message("previous question")
        pipeline._conversation.add_assistant_message("previous answer")

        await pipeline._handle_transcript(
            TranscriptEvent(text="Hey Robson, new topic", is_final=False)
        )

        assert pipeline._agent == "robson"
        assert len(pipeline._conversation._messages) == 0

    @pytest.mark.asyncio
    async def test_agent_switch_updates_voice(self):
        pipeline = create_pipeline_with_languages(
            default_agent="jarvis",
            agent_voice_map={"jarvis": "onyx", "robson": "pt-BR-AntonioNeural"},
        )
        assert pipeline._get_voice() == "onyx"

        pipeline._state = PipelineState.AMBIENT
        await pipeline._handle_transcript(
            TranscriptEvent(text="Hey Robson", is_final=False)
        )
        assert pipeline._get_voice() == "pt-BR-AntonioNeural"


class TestAgentSwitchUpdatesSystemPrompt:
    @pytest.mark.asyncio
    async def test_language_changes_after_wake_word_switch(self):
        pipeline = create_pipeline_with_languages(
            default_agent="jarvis",
            agent_language_map={"jarvis": "English", "robson": "Portuguese"},
        )
        assert "English" in pipeline._get_system_prompt()

        pipeline._state = PipelineState.AMBIENT
        await pipeline._handle_transcript(
            TranscriptEvent(text="Hey Robson, como vai?", is_final=False)
        )
        assert "Portuguese" in pipeline._get_system_prompt()


@pytest.fixture(autouse=True)
def clean_voice_pipeline_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("VOICE_PIPELINE_"):
            monkeypatch.delenv(key, raising=False)


class TestGatewayTokenConfig:
    def test_direct_token_takes_precedence_over_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("file-token-value")
            token_file = f.name

        try:
            config = VoicePipelineConfig(
                gateway_token="direct-token-value",
                gateway_token_file=token_file,
            )
            resolved_token = config.gateway_token or config.read_secret(
                config.gateway_token_file
            )
            assert resolved_token == "direct-token-value"
        finally:
            os.unlink(token_file)

    def test_falls_back_to_file_when_direct_token_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("file-token-value")
            token_file = f.name

        try:
            config = VoicePipelineConfig(
                gateway_token="",
                gateway_token_file=token_file,
            )
            resolved_token = config.gateway_token or config.read_secret(
                config.gateway_token_file
            )
            assert resolved_token == "file-token-value"
        finally:
            os.unlink(token_file)

    def test_empty_token_when_both_missing(self):
        config = VoicePipelineConfig(
            gateway_token="",
            gateway_token_file="/nonexistent/path",
        )
        resolved_token = config.gateway_token or config.read_secret(
            config.gateway_token_file
        )
        assert resolved_token == ""

    def test_agent_languages_config_parses_json(self):
        config = VoicePipelineConfig(
            agent_languages={"jarvis": "English", "robson": "Portuguese"},
        )
        assert config.agent_languages == {
            "jarvis": "English",
            "robson": "Portuguese",
        }


class TestEdgeTtsSynthesizerConfig:
    def test_tts_engine_defaults_to_openai(self):
        config = VoicePipelineConfig()
        assert config.tts_engine == "openai"

    def test_tts_engine_can_be_set_to_edge_tts(self):
        config = VoicePipelineConfig(tts_engine="edge-tts")
        assert config.tts_engine == "edge-tts"

    def test_agent_voices_config_parses_dict(self):
        config = VoicePipelineConfig(
            agent_voices={
                "jarvis": "en-GB-RyanNeural",
                "robson": "pt-BR-AntonioNeural",
            },
        )
        assert config.agent_voices["robson"] == "pt-BR-AntonioNeural"
