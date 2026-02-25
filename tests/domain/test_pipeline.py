import asyncio
from unittest.mock import AsyncMock

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


class TestUtteranceProcessingWaitsForSpeechEnd:
    @pytest.mark.asyncio
    async def test_final_with_punctuation_schedules_flush_not_immediate(self, pipeline):
        pipeline._state = PipelineState.LISTENING
        pipeline._process_utterance = AsyncMock()

        await pipeline._handle_transcript(
            TranscriptEvent(text="What's the weather?", is_final=True)
        )

        pipeline._process_utterance.assert_not_called()
        assert pipeline._utterance_buffer == ["What's the weather?"]
        assert pipeline._utterance_flush_task is not None

    @pytest.mark.asyncio
    async def test_final_without_punctuation_schedules_flush(self, pipeline):
        pipeline._state = PipelineState.LISTENING
        pipeline._process_utterance = AsyncMock()

        await pipeline._handle_transcript(
            TranscriptEvent(text="what's the weather", is_final=True)
        )

        pipeline._process_utterance.assert_not_called()
        assert pipeline._utterance_buffer == ["what's the weather"]
        assert pipeline._utterance_flush_task is not None

    @pytest.mark.asyncio
    async def test_flush_fires_after_speech_end_then_final(self, pipeline):
        pipeline._state = PipelineState.LISTENING

        await pipeline._handle_transcript(
            TranscriptEvent(text="what's the weather", is_final=False)
        )
        pipeline._schedule_utterance_flush()

        pipeline._speech_ended_event.set()
        await asyncio.sleep(0.15)
        assert pipeline._waiting_for_final_after_speech_end is True

        await pipeline._handle_transcript(
            TranscriptEvent(text="what's the weather?", is_final=True)
        )
        await asyncio.sleep(0.05)

        assert pipeline._state in (
            PipelineState.THINKING,
            PipelineState.SPEAKING,
            PipelineState.CONVERSING,
        )

    @pytest.mark.asyncio
    async def test_multiple_finals_accumulate_in_buffer(self, pipeline):
        pipeline._state = PipelineState.LISTENING
        pipeline._process_utterance = AsyncMock()

        await pipeline._handle_transcript(
            TranscriptEvent(text="I was wondering", is_final=True)
        )
        pipeline._process_utterance.assert_not_called()
        assert pipeline._utterance_buffer == ["I was wondering"]

        await pipeline._handle_transcript(
            TranscriptEvent(text="about the weather today.", is_final=True)
        )
        pipeline._process_utterance.assert_not_called()
        assert pipeline._utterance_buffer == [
            "I was wondering",
            "about the weather today.",
        ]

    @pytest.mark.asyncio
    async def test_final_after_speech_end_triggers_processing(self, pipeline):
        pipeline._state = PipelineState.LISTENING

        await pipeline._handle_transcript(
            TranscriptEvent(text="I was wondering", is_final=False)
        )
        pipeline._schedule_utterance_flush()

        pipeline._speech_ended_event.set()
        await asyncio.sleep(0.15)

        await pipeline._handle_transcript(
            TranscriptEvent(
                text="I was wondering about the weather today.", is_final=True
            )
        )
        await asyncio.sleep(0.05)

        assert pipeline._state in (
            PipelineState.THINKING,
            PipelineState.SPEAKING,
            PipelineState.CONVERSING,
        )

    @pytest.mark.asyncio
    async def test_interim_does_not_trigger_processing(self, pipeline):
        pipeline._state = PipelineState.LISTENING
        pipeline._process_utterance = AsyncMock()

        await pipeline._handle_transcript(
            TranscriptEvent(text="what's the weather?", is_final=False)
        )
        pipeline._process_utterance.assert_not_called()

    @pytest.mark.asyncio
    async def test_duplicate_interim_skipped(self, pipeline):
        pipeline._state = PipelineState.LISTENING

        await pipeline._handle_transcript(TranscriptEvent(text="hello", is_final=False))
        assert pipeline._utterance_buffer == ["hello"]

        await pipeline._handle_transcript(TranscriptEvent(text="hello", is_final=False))
        assert pipeline._utterance_buffer == ["hello"]


class TestWakeWordSchedulesFlush:
    @pytest.mark.asyncio
    async def test_wake_word_with_command_schedules_flush(self, pipeline):
        pipeline._state = PipelineState.AMBIENT
        pipeline._process_utterance = AsyncMock()

        await pipeline._handle_transcript(
            TranscriptEvent(text="Jarvis, what time is it?", is_final=True)
        )

        pipeline._process_utterance.assert_not_called()
        assert pipeline._state == PipelineState.LISTENING
        assert pipeline._utterance_flush_task is not None

    @pytest.mark.asyncio
    async def test_wake_word_flushes_on_speech_end(self, pipeline):
        pipeline._state = PipelineState.AMBIENT

        await pipeline._handle_transcript(
            TranscriptEvent(text="Jarvis, what time is it?", is_final=True)
        )

        assert pipeline._state == PipelineState.LISTENING
        pipeline._speech_ended_event.set()
        await asyncio.sleep(0.05)

        assert pipeline._state in (
            PipelineState.THINKING,
            PipelineState.SPEAKING,
            PipelineState.CONVERSING,
        )

    @pytest.mark.asyncio
    async def test_wake_word_without_command_waits(self, pipeline):
        pipeline._state = PipelineState.AMBIENT
        pipeline._process_utterance = AsyncMock()

        await pipeline._handle_transcript(
            TranscriptEvent(text="Jarvis tell me", is_final=True)
        )

        pipeline._process_utterance.assert_not_called()
        assert pipeline._state == PipelineState.LISTENING


class TestConversationWindowSchedulesFlush:
    @pytest.mark.asyncio
    async def test_conversing_schedules_flush_not_immediate(self, pipeline):
        pipeline._state = PipelineState.CONVERSING
        pipeline._conversation_window_task = asyncio.create_task(asyncio.sleep(100))
        pipeline._process_utterance = AsyncMock()

        await pipeline._handle_transcript(
            TranscriptEvent(text="And one more thing.", is_final=True)
        )

        pipeline._process_utterance.assert_not_called()
        assert pipeline._state == PipelineState.LISTENING
        assert pipeline._utterance_flush_task is not None

    @pytest.mark.asyncio
    async def test_conversing_flushes_on_speech_end(self, pipeline):
        pipeline._state = PipelineState.CONVERSING
        pipeline._conversation_window_task = asyncio.create_task(asyncio.sleep(100))

        await pipeline._handle_transcript(
            TranscriptEvent(text="And one more thing.", is_final=True)
        )

        pipeline._speech_ended_event.set()
        await asyncio.sleep(0.05)

        assert pipeline._state in (
            PipelineState.THINKING,
            PipelineState.SPEAKING,
            PipelineState.CONVERSING,
        )

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

        await pipeline._handle_transcript(
            TranscriptEvent(text="no wake word here.", is_final=True)
        )

        assert pipeline._last_interim_text == ""

    @pytest.mark.asyncio
    async def test_new_interim_updates_tracker(self, pipeline):
        pipeline._state = PipelineState.AMBIENT

        await pipeline._handle_transcript(TranscriptEvent(text="first", is_final=False))
        assert pipeline._last_interim_text == "first"

        await pipeline._handle_transcript(
            TranscriptEvent(text="first second", is_final=False)
        )
        assert pipeline._last_interim_text == "first second"
