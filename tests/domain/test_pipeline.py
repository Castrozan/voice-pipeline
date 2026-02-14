import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from domain.pipeline import VoicePipeline
from domain.state import PipelineState
from domain.speech_detector import SpeechDetector
from domain.wake_word import WakeWordDetector
from domain.conversation import ConversationHistory
from ports.transcriber import TranscriptEvent

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import (
    FakeAudioCapture,
    FakeAudioPlayback,
    FakeTranscriber,
    FakeCompletion,
    FakeSynthesizer,
    FakeVad,
    FRAME_DURATION_MS,
)


@pytest.fixture
def pipeline():
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
        completion=FakeCompletion(),
        synthesizer=FakeSynthesizer(),
        speech_detector=detector,
        wake_word_detector=WakeWordDetector(["jarvis"]),
        conversation=ConversationHistory(max_turns=20),
    )


class TestSmartUtteranceProcessing:
    @pytest.mark.asyncio
    async def test_punctuated_final_processes_immediately(self, pipeline):
        pipeline._state = PipelineState.LISTENING
        pipeline._process_utterance = AsyncMock()

        await pipeline._handle_transcript(TranscriptEvent(text="What's the weather?", is_final=True))

        pipeline._process_utterance.assert_called_once()

    @pytest.mark.asyncio
    async def test_unpunctuated_final_does_not_process_immediately(self, pipeline):
        pipeline._state = PipelineState.LISTENING
        pipeline._process_utterance = AsyncMock()

        await pipeline._handle_transcript(TranscriptEvent(text="what's the weather", is_final=True))

        pipeline._process_utterance.assert_not_called()
        assert pipeline._utterance_buffer == ["what's the weather"]

    @pytest.mark.asyncio
    async def test_unpunctuated_final_flushes_on_speech_end(self, pipeline):
        pipeline._state = PipelineState.LISTENING
        original_process = pipeline._process_utterance

        await pipeline._handle_transcript(TranscriptEvent(text="what's the weather", is_final=True))

        assert pipeline._utterance_flush_task is not None
        pipeline._speech_ended_event.set()
        await asyncio.sleep(0.05)

        assert pipeline._state == PipelineState.THINKING or pipeline._state == PipelineState.SPEAKING

    @pytest.mark.asyncio
    async def test_multiple_finals_accumulate_until_punctuation(self, pipeline):
        pipeline._state = PipelineState.LISTENING
        pipeline._process_utterance = AsyncMock()

        await pipeline._handle_transcript(TranscriptEvent(text="I was wondering", is_final=True))
        pipeline._process_utterance.assert_not_called()
        assert pipeline._utterance_buffer == ["I was wondering"]

        await pipeline._handle_transcript(TranscriptEvent(text="about the weather today.", is_final=True))
        pipeline._process_utterance.assert_called_once()

    @pytest.mark.asyncio
    async def test_exclamation_mark_triggers_processing(self, pipeline):
        pipeline._state = PipelineState.LISTENING
        pipeline._process_utterance = AsyncMock()

        await pipeline._handle_transcript(TranscriptEvent(text="Stop right there!", is_final=True))
        pipeline._process_utterance.assert_called_once()

    @pytest.mark.asyncio
    async def test_period_triggers_processing(self, pipeline):
        pipeline._state = PipelineState.LISTENING
        pipeline._process_utterance = AsyncMock()

        await pipeline._handle_transcript(TranscriptEvent(text="Turn on the lights.", is_final=True))
        pipeline._process_utterance.assert_called_once()

    @pytest.mark.asyncio
    async def test_interim_does_not_trigger_processing(self, pipeline):
        pipeline._state = PipelineState.LISTENING
        pipeline._process_utterance = AsyncMock()

        await pipeline._handle_transcript(TranscriptEvent(text="what's the weather?", is_final=False))
        pipeline._process_utterance.assert_not_called()

    @pytest.mark.asyncio
    async def test_duplicate_interim_skipped(self, pipeline):
        pipeline._state = PipelineState.LISTENING

        await pipeline._handle_transcript(TranscriptEvent(text="hello", is_final=False))
        assert pipeline._utterance_buffer == ["hello"]

        await pipeline._handle_transcript(TranscriptEvent(text="hello", is_final=False))
        assert pipeline._utterance_buffer == ["hello"]


class TestWakeWordWithSmartProcessing:
    @pytest.mark.asyncio
    async def test_wake_word_with_punctuated_command_processes(self, pipeline):
        pipeline._state = PipelineState.AMBIENT
        pipeline._process_utterance = AsyncMock()

        await pipeline._handle_transcript(
            TranscriptEvent(text="Jarvis, what time is it?", is_final=True)
        )

        pipeline._process_utterance.assert_called_once()

    @pytest.mark.asyncio
    async def test_wake_word_with_unpunctuated_command_waits(self, pipeline):
        pipeline._state = PipelineState.AMBIENT
        pipeline._process_utterance = AsyncMock()

        await pipeline._handle_transcript(
            TranscriptEvent(text="Jarvis tell me", is_final=True)
        )

        pipeline._process_utterance.assert_not_called()
        assert pipeline._state == PipelineState.LISTENING


class TestConversationWindowSmartProcessing:
    @pytest.mark.asyncio
    async def test_conversing_punctuated_processes(self, pipeline):
        pipeline._state = PipelineState.CONVERSING
        pipeline._conversation_window_task = asyncio.create_task(asyncio.sleep(100))
        pipeline._process_utterance = AsyncMock()

        await pipeline._handle_transcript(
            TranscriptEvent(text="And one more thing.", is_final=True)
        )

        pipeline._process_utterance.assert_called_once()

    @pytest.mark.asyncio
    async def test_conversing_unpunctuated_waits(self, pipeline):
        pipeline._state = PipelineState.CONVERSING
        pipeline._conversation_window_task = asyncio.create_task(asyncio.sleep(100))
        pipeline._process_utterance = AsyncMock()

        await pipeline._handle_transcript(
            TranscriptEvent(text="And also", is_final=True)
        )

        pipeline._process_utterance.assert_not_called()
        assert pipeline._state == PipelineState.LISTENING


class TestTranscriptDeduplication:
    @pytest.mark.asyncio
    async def test_final_transcript_clears_interim_tracker(self, pipeline):
        pipeline._state = PipelineState.AMBIENT
        pipeline._last_interim_text = "some old text"

        await pipeline._handle_transcript(TranscriptEvent(text="no wake word here.", is_final=True))

        assert pipeline._last_interim_text == ""

    @pytest.mark.asyncio
    async def test_new_interim_updates_tracker(self, pipeline):
        pipeline._state = PipelineState.AMBIENT

        await pipeline._handle_transcript(TranscriptEvent(text="first", is_final=False))
        assert pipeline._last_interim_text == "first"

        await pipeline._handle_transcript(TranscriptEvent(text="first second", is_final=False))
        assert pipeline._last_interim_text == "first second"
