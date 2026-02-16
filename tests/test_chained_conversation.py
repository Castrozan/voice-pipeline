import asyncio
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest

from adapters.silero_vad import SileroVad
from domain.conversation import ConversationHistory
from domain.pipeline import VoicePipeline
from domain.speech_detector import SpeechDetector, SpeechEvent
from domain.state import PipelineState
from domain.wake_word import WakeWordDetector
from ports.transcriber import TranscriptEvent

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 16
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
BYTES_PER_FRAME = FRAME_SIZE * 2
RECORDINGS_DIR = Path(__file__).parent / "recordings" / "headset"


def generate_silence(duration_ms: int = 16) -> bytes:
    num_samples = int(SAMPLE_RATE * duration_ms / 1000)
    return np.zeros(num_samples, dtype=np.int16).tobytes()


def generate_sine_wave(duration_ms: int = 100) -> bytes:
    num_samples = int(SAMPLE_RATE * duration_ms / 1000)
    t = np.arange(num_samples) / SAMPLE_RATE
    signal = np.sin(2 * np.pi * 440.0 * t) * 0.8
    return (signal * 32767).astype(np.int16).tobytes()


def load_recording_as_frames(wav_path: Path) -> list[bytes]:
    with wave.open(str(wav_path), "rb") as wf:
        pcm_data = wf.readframes(wf.getnframes())
    frames = []
    for i in range(0, len(pcm_data), BYTES_PER_FRAME):
        chunk = pcm_data[i : i + BYTES_PER_FRAME]
        if len(chunk) == BYTES_PER_FRAME:
            frames.append(chunk)
    return frames


def generate_silence_frames(duration_ms: int) -> list[bytes]:
    frame_count = duration_ms // FRAME_DURATION_MS
    return [generate_silence(FRAME_DURATION_MS) for _ in range(frame_count)]


def concatenate_recordings_with_silence(
    recording_paths: list[Path],
    gap_ms: int = 2000,
    trailing_silence_ms: int = 2000,
) -> list[bytes]:
    gap_frames = generate_silence_frames(gap_ms)
    all_frames: list[bytes] = []
    for i, path in enumerate(recording_paths):
        if i > 0:
            all_frames.extend(gap_frames)
        all_frames.extend(load_recording_as_frames(path))
    all_frames.extend(generate_silence_frames(trailing_silence_ms))
    return all_frames


def has_required_recordings(paths: list[Path]) -> bool:
    return all(p.exists() for p in paths)


class YieldingFakeAudioCapture:
    def __init__(self, frames: list[bytes], frame_delay_seconds: float = 0) -> None:
        self._frames = frames
        self._sample_rate = SAMPLE_RATE
        self._frame_size = FRAME_SIZE
        self._frame_delay_seconds = frame_delay_seconds

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_size(self) -> int:
        return self._frame_size

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def read_frames(self) -> AsyncIterator[bytes]:
        for frame in self._frames:
            yield frame
            await asyncio.sleep(self._frame_delay_seconds)


class QueueBasedFakeTranscriber:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        self._session_active = False
        self._audio_received: list[bytes] = []
        self.start_session_count = 0

    async def start_session(self) -> None:
        self.start_session_count += 1
        self._session_active = True

    async def send_audio(self, frame: bytes) -> None:
        self._audio_received.append(frame)

    async def get_transcripts(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    async def close_session(self) -> None:
        self._session_active = False

    def push_transcript(self, text: str, is_final: bool = True) -> None:
        self._queue.put_nowait(TranscriptEvent(text=text, is_final=is_final))

    def stop(self) -> None:
        self._queue.put_nowait(None)


class FakeCompletion:
    def __init__(self) -> None:
        self._default_response = ["Hello", " there!"]
        self._cancelled = False
        self._call_count = 0

    async def stream(
        self,
        messages: list[dict[str, str]],
        agent: str,
    ) -> AsyncIterator[str]:
        self._cancelled = False
        self._call_count += 1
        for chunk in self._default_response:
            if self._cancelled:
                break
            yield chunk
            await asyncio.sleep(0)

    async def cancel(self) -> None:
        self._cancelled = True


class FakeSynthesizer:
    def __init__(self) -> None:
        self._audio_chunk = generate_sine_wave(duration_ms=100)
        self._cancelled = False
        self._synthesize_count = 0

    async def synthesize(self, text: str, voice: str = "onyx") -> AsyncIterator[bytes]:
        self._cancelled = False
        self._synthesize_count += 1
        for _ in range(3):
            if self._cancelled:
                break
            yield self._audio_chunk
            await asyncio.sleep(0)

    async def cancel(self) -> None:
        self._cancelled = True


class FakeAudioPlayback:
    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._cancelled = False

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def play_chunk(self, audio_data: bytes) -> None:
        if not self._cancelled:
            self._chunks.append(audio_data)

    async def drain(self) -> None:
        pass

    async def cancel(self) -> None:
        self._cancelled = True
        self._chunks.clear()


class FakeVad:
    def __init__(self) -> None:
        self._probabilities: list[float] = []
        self._call_count = 0

    def process_frame(self, audio_frame: bytes) -> float:
        if self._call_count < len(self._probabilities):
            prob = self._probabilities[self._call_count]
        else:
            prob = 0.0
        self._call_count += 1
        return prob

    def reset(self) -> None:
        self._call_count = 0

    def set_probabilities(self, probabilities: list[float]) -> None:
        self._probabilities = probabilities
        self._call_count = 0


async def wait_for_pipeline_state(
    pipeline: VoicePipeline,
    target_state: PipelineState,
    timeout_seconds: float = 5.0,
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_seconds
    while asyncio.get_event_loop().time() < deadline:
        if pipeline._state == target_state:
            return
        await asyncio.sleep(0)
    raise AssertionError(
        f"Pipeline did not reach {target_state.name} within {timeout_seconds}s, "
        f"stuck at {pipeline._state.name}"
    )


def build_pipeline_with_fake_vad(
    conversation_window_seconds: float = 15.0,
) -> tuple[VoicePipeline, FakeCompletion, FakeSynthesizer]:
    vad = FakeVad()
    transcriber = QueueBasedFakeTranscriber()
    completion = FakeCompletion()
    synthesizer = FakeSynthesizer()

    speech_detector = SpeechDetector(
        vad=vad,
        threshold=0.5,
        min_silence_ms=300,
        frame_duration_ms=FRAME_DURATION_MS,
    )

    pipeline = VoicePipeline(
        capture=YieldingFakeAudioCapture([]),
        playback=FakeAudioPlayback(),
        transcriber=transcriber,
        completion=completion,
        synthesizer=synthesizer,
        speech_detector=speech_detector,
        wake_word_detector=WakeWordDetector(["jarvis"]),
        conversation=ConversationHistory(max_turns=20),
        conversation_window_seconds=conversation_window_seconds,
        frame_duration_ms=FRAME_DURATION_MS,
    )
    return pipeline, completion, synthesizer


def build_pipeline_with_real_vad(
    real_vad: SileroVad,
    frames: list[bytes],
    transcriber: QueueBasedFakeTranscriber | None = None,
) -> tuple[VoicePipeline, QueueBasedFakeTranscriber, FakeCompletion, FakeSynthesizer]:
    transcriber = transcriber or QueueBasedFakeTranscriber()
    completion = FakeCompletion()
    synthesizer = FakeSynthesizer()

    speech_detector = SpeechDetector(
        vad=real_vad,
        threshold=0.5,
        min_silence_ms=800,
        frame_duration_ms=FRAME_DURATION_MS,
    )

    pipeline = VoicePipeline(
        capture=YieldingFakeAudioCapture(frames, frame_delay_seconds=0.001),
        playback=FakeAudioPlayback(),
        transcriber=transcriber,
        completion=completion,
        synthesizer=synthesizer,
        speech_detector=speech_detector,
        wake_word_detector=WakeWordDetector(["jarvis"]),
        conversation=ConversationHistory(max_turns=20),
        conversation_window_seconds=15.0,
        frame_duration_ms=FRAME_DURATION_MS,
    )
    return pipeline, transcriber, completion, synthesizer


class TestChainedRecordingsDetectSpeechSegments:
    @pytest.fixture
    def real_vad(self):
        return SileroVad(sample_rate=SAMPLE_RATE)

    def _count_speech_segments(
        self,
        real_vad: SileroVad,
        frames: list[bytes],
    ) -> int:
        speech_detector = SpeechDetector(
            vad=real_vad,
            threshold=0.5,
            min_silence_ms=800,
            frame_duration_ms=FRAME_DURATION_MS,
        )
        speech_end_count = 0
        for frame in frames:
            event = speech_detector.process_frame(frame)
            if event == SpeechEvent.SPEECH_END:
                speech_end_count += 1
        return speech_end_count

    async def test_two_recordings_produce_two_speech_segments(self, real_vad):
        paths = [
            RECORDINGS_DIR / "wake_jarvis_weather.wav",
            RECORDINGS_DIR / "loud_speech.wav",
        ]
        if not has_required_recordings(paths):
            pytest.skip("Required recordings not found")

        frames = concatenate_recordings_with_silence(paths)
        segments = self._count_speech_segments(real_vad, frames)

        assert segments >= 2, (
            f"Expected >= 2 speech segments for 2 recordings, got {segments}"
        )

    async def test_three_recordings_produce_three_speech_segments(self, real_vad):
        paths = [
            RECORDINGS_DIR / "wake_jarvis_weather.wav",
            RECORDINGS_DIR / "continuous_ramble.wav",
            RECORDINGS_DIR / "loud_speech.wav",
        ]
        if not has_required_recordings(paths):
            pytest.skip("Required recordings not found")

        frames = concatenate_recordings_with_silence(paths)
        segments = self._count_speech_segments(real_vad, frames)

        assert segments >= 3, (
            f"Expected >= 3 speech segments for 3 recordings, got {segments}"
        )


class TestChainedConversationFlow:
    async def _simulate_utterance_and_wait_for_response(
        self,
        pipeline: VoicePipeline,
        text: str,
        timeout_seconds: float = 3.0,
    ) -> None:
        pipeline._stt_generation += 1
        await pipeline._handle_transcript(
            TranscriptEvent(text=text, is_final=True)
        )
        pipeline._speech_ended_event.set()
        await wait_for_pipeline_state(
            pipeline, PipelineState.CONVERSING, timeout_seconds
        )

    async def test_wake_word_then_follow_up(self):
        pipeline, completion, _ = build_pipeline_with_fake_vad()
        pipeline._running = True
        pipeline._enabled = True

        await self._simulate_utterance_and_wait_for_response(
            pipeline, "Jarvis, what's the weather?"
        )
        assert completion._call_count == 1

        await self._simulate_utterance_and_wait_for_response(
            pipeline, "And what about tomorrow?"
        )
        assert completion._call_count == 2

        messages = pipeline._conversation.to_api_messages()
        user_messages = [m for m in messages if m["role"] == "user"]
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(user_messages) == 2
        assert len(assistant_messages) == 2

    async def test_three_turn_conversation_accumulates_history(self):
        pipeline, completion, synthesizer = build_pipeline_with_fake_vad()
        pipeline._running = True
        pipeline._enabled = True

        utterances = [
            "Jarvis, tell me about Mars.",
            "How far is it from Earth?",
            "Can humans live there?",
        ]

        for i, text in enumerate(utterances):
            await self._simulate_utterance_and_wait_for_response(pipeline, text)
            assert completion._call_count == i + 1
            assert synthesizer._synthesize_count >= i + 1

        messages = pipeline._conversation.to_api_messages()
        user_messages = [m for m in messages if m["role"] == "user"]
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(user_messages) == 3
        assert len(assistant_messages) == 3

    async def test_no_duplicate_processing_across_chained_utterances(self):
        pipeline, completion, _ = build_pipeline_with_fake_vad()
        pipeline._running = True
        pipeline._enabled = True

        await self._simulate_utterance_and_wait_for_response(
            pipeline, "Jarvis, hello!"
        )
        assert completion._call_count == 1

        await asyncio.sleep(0.3)
        assert completion._call_count == 1

        await self._simulate_utterance_and_wait_for_response(
            pipeline, "How are you?"
        )
        assert completion._call_count == 2

        await asyncio.sleep(0.3)
        assert completion._call_count == 2

    async def test_conversation_timeout_requires_new_wake_word(self):
        pipeline, completion, _ = build_pipeline_with_fake_vad(
            conversation_window_seconds=0.2,
        )
        pipeline._running = True
        pipeline._enabled = True

        await self._simulate_utterance_and_wait_for_response(
            pipeline, "Jarvis, hello!"
        )
        assert pipeline._state == PipelineState.CONVERSING

        await wait_for_pipeline_state(pipeline, PipelineState.AMBIENT, timeout_seconds=2.0)

        pipeline._stt_generation += 1
        await pipeline._handle_transcript(
            TranscriptEvent(text="How are you?", is_final=True)
        )
        await asyncio.sleep(0.1)
        assert pipeline._state == PipelineState.AMBIENT
        assert completion._call_count == 1

        await self._simulate_utterance_and_wait_for_response(
            pipeline, "Jarvis, I'm back!"
        )
        assert completion._call_count == 2

    async def test_five_turn_conversation_no_state_leak(self):
        pipeline, completion, _ = build_pipeline_with_fake_vad()
        pipeline._running = True
        pipeline._enabled = True

        utterances = [
            "Jarvis, start the timer.",
            "Set it for 10 minutes.",
            "Actually make it 15.",
            "And play some music.",
            "Something relaxing.",
        ]

        for i, text in enumerate(utterances):
            await self._simulate_utterance_and_wait_for_response(pipeline, text)
            assert completion._call_count == i + 1
            assert pipeline._state == PipelineState.CONVERSING

        messages = pipeline._conversation.to_api_messages()
        user_messages = [m for m in messages if m["role"] == "user"]
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(user_messages) == 5
        assert len(assistant_messages) == 5


async def wait_for_completion_count(
    completion: FakeCompletion,
    target_count: int,
    timeout_seconds: float = 5.0,
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_seconds
    while asyncio.get_event_loop().time() < deadline:
        if completion._call_count >= target_count:
            return
        await asyncio.sleep(0)
    raise AssertionError(
        f"Completion count did not reach {target_count} within {timeout_seconds}s, "
        f"stuck at {completion._call_count}"
    )


class TestEndToEndChainedRecordings:
    @pytest.fixture
    def real_vad(self):
        return SileroVad(sample_rate=SAMPLE_RATE)

    async def test_two_recordings_produce_two_responses(self, real_vad):
        paths = [
            RECORDINGS_DIR / "wake_jarvis_weather.wav",
            RECORDINGS_DIR / "loud_speech.wav",
        ]
        if not has_required_recordings(paths):
            pytest.skip("Required recordings not found")

        frames = concatenate_recordings_with_silence(paths, gap_ms=2000, trailing_silence_ms=2000)
        transcriber = QueueBasedFakeTranscriber()
        pipeline, _, completion, synthesizer = build_pipeline_with_real_vad(
            real_vad, frames, transcriber=transcriber,
        )
        pipeline._running = True
        pipeline._enabled = True

        async def director():
            sessions_handled = 0
            completions_expected = 0

            while transcriber.start_session_count <= sessions_handled:
                await asyncio.sleep(0)
            sessions_handled = transcriber.start_session_count

            transcriber.push_transcript("Jarvis, what's the weather?", is_final=True)
            completions_expected += 1
            await wait_for_completion_count(completion, completions_expected)

            while transcriber.start_session_count <= sessions_handled:
                await asyncio.sleep(0)
            sessions_handled = transcriber.start_session_count

            transcriber.push_transcript("Tell me more about that.", is_final=True)
            completions_expected += 1
            await wait_for_completion_count(completion, completions_expected)

            pipeline._running = False
            transcriber.stop()

        await asyncio.gather(
            pipeline._audio_loop(),
            pipeline._transcript_loop(),
            director(),
        )

        assert completion._call_count >= 2, (
            f"Expected at least 2 completion calls, got {completion._call_count}"
        )

        messages = pipeline._conversation.to_api_messages()
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) >= 2, (
            f"Expected at least 2 user messages, got {len(user_messages)}"
        )

    async def test_three_recordings_full_conversation(self, real_vad):
        paths = [
            RECORDINGS_DIR / "wake_jarvis_weather.wav",
            RECORDINGS_DIR / "continuous_ramble.wav",
            RECORDINGS_DIR / "loud_speech.wav",
        ]
        if not has_required_recordings(paths):
            pytest.skip("Required recordings not found")

        frames = concatenate_recordings_with_silence(paths, gap_ms=2000, trailing_silence_ms=2000)
        transcriber = QueueBasedFakeTranscriber()
        pipeline, _, completion, synthesizer = build_pipeline_with_real_vad(
            real_vad, frames, transcriber=transcriber,
        )
        pipeline._running = True
        pipeline._enabled = True

        transcript_texts = [
            "Jarvis, what's the weather?",
            "And what about tomorrow?",
            "Thanks, that's all.",
        ]

        async def director():
            sessions_handled = 0
            completions_expected = 0
            for text in transcript_texts:
                while transcriber.start_session_count <= sessions_handled:
                    await asyncio.sleep(0)
                sessions_handled = transcriber.start_session_count

                transcriber.push_transcript(text, is_final=True)
                completions_expected += 1
                await wait_for_completion_count(completion, completions_expected)

            pipeline._running = False
            transcriber.stop()

        await asyncio.gather(
            pipeline._audio_loop(),
            pipeline._transcript_loop(),
            director(),
        )

        assert completion._call_count >= 3, (
            f"Expected at least 3 completion calls, got {completion._call_count}"
        )

        messages = pipeline._conversation.to_api_messages()
        user_messages = [m for m in messages if m["role"] == "user"]
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(user_messages) >= 3, (
            f"Expected at least 3 user messages, got {len(user_messages)}"
        )
        assert len(assistant_messages) >= 3, (
            f"Expected at least 3 assistant messages, got {len(assistant_messages)}"
        )
