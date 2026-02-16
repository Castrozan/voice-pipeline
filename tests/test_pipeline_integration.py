import asyncio
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest

from adapters.silero_vad import SileroVad
from domain.conversation import ConversationHistory
from domain.pipeline import VoicePipeline
from domain.speech_detector import SpeechDetector
from domain.state import PipelineState
from domain.wake_word import WakeWordDetector
from ports.transcriber import TranscriptEvent

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 16
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)


def generate_silence(duration_ms: int = 16, sample_rate: int = SAMPLE_RATE) -> bytes:
    num_samples = int(sample_rate * duration_ms / 1000)
    return np.zeros(num_samples, dtype=np.int16).tobytes()


def generate_sine_wave(
    duration_ms: int = 100,
    frequency: float = 440.0,
    amplitude: float = 0.8,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    num_samples = int(sample_rate * duration_ms / 1000)
    t = np.arange(num_samples) / sample_rate
    signal = np.sin(2 * np.pi * frequency * t) * amplitude
    return (signal * 32767).astype(np.int16).tobytes()


class FakeAudioCapture:
    def __init__(self, frames: list[bytes] | None = None) -> None:
        self._frames = frames or []
        self._sample_rate = SAMPLE_RATE
        self._frame_size = FRAME_SIZE
        self._started = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_size(self) -> int:
        return self._frame_size

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def read_frames(self) -> AsyncIterator[bytes]:
        for frame in self._frames:
            yield frame


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
    def __init__(self, probabilities: list[float] | None = None) -> None:
        self._probabilities = probabilities or []
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


class FakeTranscriber:
    def __init__(self) -> None:
        self._events: list[TranscriptEvent] = []
        self._session_active = False
        self._audio_received: list[bytes] = []
        self.start_session_count = 0

    async def start_session(self) -> None:
        self.start_session_count += 1
        self._session_active = True

    async def send_audio(self, frame: bytes) -> None:
        self._audio_received.append(frame)

    async def get_transcripts(self) -> AsyncIterator[TranscriptEvent]:
        for event in self._events:
            yield event

    async def close_session(self) -> None:
        self._session_active = False

    def queue_events(self, events: list[TranscriptEvent]) -> None:
        self._events.extend(events)


class FakeCompletion:
    def __init__(self, responses: dict[str, list[str]] | None = None) -> None:
        self._responses = responses or {}
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
        chunks = self._responses.get(agent, self._default_response)
        for chunk in chunks:
            if self._cancelled:
                break
            yield chunk
            await asyncio.sleep(0)

    async def cancel(self) -> None:
        self._cancelled = True


class FakeSynthesizer:
    def __init__(self, audio_chunk: bytes | None = None) -> None:
        self._audio_chunk = audio_chunk or generate_sine_wave(duration_ms=100)
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

RECORDINGS_DIR = Path(__file__).parent / "recordings" / "headset"
BYTES_PER_FRAME = FRAME_SIZE * 2


def load_recording_as_frames(wav_path: Path) -> list[bytes]:
    with wave.open(str(wav_path), "rb") as wf:
        pcm_data = wf.readframes(wf.getnframes())
    frames = []
    for i in range(0, len(pcm_data), BYTES_PER_FRAME):
        chunk = pcm_data[i : i + BYTES_PER_FRAME]
        if len(chunk) == BYTES_PER_FRAME:
            frames.append(chunk)
    return frames


def has_headset_recordings() -> bool:
    return RECORDINGS_DIR.exists() and any(RECORDINGS_DIR.glob("*.wav"))


def find_speech_recording() -> Path | None:
    for name in ["wake_jarvis_weather.wav", "loud_speech.wav", "continuous_ramble.wav"]:
        path = RECORDINGS_DIR / name
        if path.exists():
            return path
    return None


class TestRealVadWithRecordings:
    @pytest.fixture
    def real_vad(self):
        return SileroVad(sample_rate=SAMPLE_RATE)

    def _make_pipeline_with_real_vad(
        self,
        real_vad,
        frames: list[bytes],
        transcriber: FakeTranscriber | None = None,
    ) -> tuple[VoicePipeline, FakeTranscriber, FakeAudioCapture]:
        transcriber = transcriber or FakeTranscriber()
        capture = FakeAudioCapture(frames)
        playback = FakeAudioPlayback()
        completion = FakeCompletion()
        synthesizer = FakeSynthesizer()
        speech_detector = SpeechDetector(
            vad=real_vad,
            threshold=0.5,
            min_silence_ms=300,
            frame_duration_ms=FRAME_DURATION_MS,
        )
        pipeline = VoicePipeline(
            capture=capture,
            playback=playback,
            transcriber=transcriber,
            completion=completion,
            synthesizer=synthesizer,
            speech_detector=speech_detector,
            wake_word_detector=WakeWordDetector(["jarvis"]),
            conversation=ConversationHistory(max_turns=20),
            pre_buffer_ms=300,
            frame_duration_ms=FRAME_DURATION_MS,
        )
        return pipeline, transcriber, capture

    async def test_speech_recording_triggers_stt_session(self, real_vad):
        recording_path = find_speech_recording()
        if recording_path is None:
            pytest.skip("No speech recordings found in tests/recordings/headset/")

        frames = load_recording_as_frames(recording_path)
        pipeline, transcriber, capture = self._make_pipeline_with_real_vad(real_vad, frames)

        pipeline._running = True
        pipeline._enabled = True
        await pipeline._audio_loop()

        assert transcriber.start_session_count >= 1, (
            f"Expected STT session to start for {recording_path.name}, "
            f"but start_session was called {transcriber.start_session_count} times"
        )

    async def test_silence_recording_does_not_trigger_stt(self, real_vad):
        silence_path = RECORDINGS_DIR / "silence_3s.wav"
        if not silence_path.exists():
            pytest.skip("No silence recording found")

        frames = load_recording_as_frames(silence_path)
        pipeline, transcriber, capture = self._make_pipeline_with_real_vad(real_vad, frames)

        pipeline._running = True
        pipeline._enabled = True
        await pipeline._audio_loop()

        assert transcriber.start_session_count == 0, (
            f"Expected no STT session for silence, "
            f"but start_session was called {transcriber.start_session_count} times"
        )

    async def test_pre_buffer_frames_sent_before_live_audio(self, real_vad):
        recording_path = find_speech_recording()
        if recording_path is None:
            pytest.skip("No speech recordings found in tests/recordings/headset/")

        frames = load_recording_as_frames(recording_path)
        transcriber = FakeTranscriber()
        pipeline, _, capture = self._make_pipeline_with_real_vad(
            real_vad, frames, transcriber=transcriber,
        )

        pipeline._running = True
        pipeline._enabled = True
        await pipeline._audio_loop()

        assert transcriber.start_session_count >= 1
        assert len(transcriber._audio_received) > 0, (
            "Expected audio frames sent to transcriber after session start"
        )

    async def test_audio_received_count_exceeds_pre_buffer(self, real_vad):
        recording_path = find_speech_recording()
        if recording_path is None:
            pytest.skip("No speech recordings found in tests/recordings/headset/")

        frames = load_recording_as_frames(recording_path)
        pre_buffer_ms = 300
        pre_buffer_frame_count = int(pre_buffer_ms / FRAME_DURATION_MS)

        pipeline, transcriber, capture = self._make_pipeline_with_real_vad(real_vad, frames)

        pipeline._running = True
        pipeline._enabled = True
        await pipeline._audio_loop()

        assert len(transcriber._audio_received) > pre_buffer_frame_count, (
            f"Expected more than {pre_buffer_frame_count} frames sent to transcriber "
            f"(pre-buffer + live), got {len(transcriber._audio_received)}"
        )


class TestFullPipelineFlow:
    def _build_pipeline(
        self,
        vad_probabilities: list[float],
        transcript_events: list[TranscriptEvent],
    ) -> tuple[VoicePipeline, FakeVad, FakeTranscriber, FakeCompletion, FakeSynthesizer]:
        vad = FakeVad(vad_probabilities)
        transcriber = FakeTranscriber()
        transcriber.queue_events(transcript_events)
        completion = FakeCompletion()
        synthesizer = FakeSynthesizer()

        speech_detector = SpeechDetector(
            vad=vad,
            threshold=0.5,
            min_silence_ms=300,
            frame_duration_ms=FRAME_DURATION_MS,
        )

        capture_frames = [generate_silence(duration_ms=FRAME_DURATION_MS) for _ in range(len(vad_probabilities))]

        pipeline = VoicePipeline(
            capture=FakeAudioCapture(capture_frames),
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
        return pipeline, vad, transcriber, completion, synthesizer

    async def test_full_state_machine_ambient_to_conversing(self):
        vad_probs = [0.0] * 10

        pipeline, vad, transcriber, completion, synthesizer = self._build_pipeline(
            vad_probs, [],
        )

        observed_states: list[PipelineState] = []
        original_transition = pipeline._transition_to

        def tracking_transition(target: PipelineState) -> None:
            original_transition(target)
            observed_states.append(target)

        pipeline._transition_to = tracking_transition

        pipeline._running = True
        pipeline._enabled = True

        await pipeline._handle_transcript(
            TranscriptEvent(text="Jarvis, what is the weather?", is_final=True)
        )
        pipeline._speech_ended_event.set()

        for _ in range(20):
            await asyncio.sleep(0.1)
            if PipelineState.CONVERSING in observed_states:
                break

        assert PipelineState.LISTENING in observed_states, (
            f"Expected LISTENING in states, got {observed_states}"
        )
        assert PipelineState.THINKING in observed_states, (
            f"Expected THINKING in states, got {observed_states}"
        )
        assert PipelineState.SPEAKING in observed_states, (
            f"Expected SPEAKING in states, got {observed_states}"
        )
        assert PipelineState.CONVERSING in observed_states, (
            f"Expected CONVERSING in states, got {observed_states}"
        )

    async def test_completion_and_synthesizer_called(self):
        vad_probs = [0.9] * 10 + [0.0] * 25

        pipeline, vad, transcriber, completion, synthesizer = self._build_pipeline(
            vad_probs, [],
        )

        pipeline._running = True
        pipeline._enabled = True
        pipeline._state = PipelineState.AMBIENT

        await pipeline._handle_transcript(
            TranscriptEvent(text="Jarvis, tell me a joke.", is_final=True)
        )
        pipeline._speech_ended_event.set()

        await asyncio.sleep(0.2)

        assert completion._call_count >= 1, (
            f"Expected completion to be called, got {completion._call_count} calls"
        )
        assert synthesizer._synthesize_count >= 1, (
            f"Expected synthesizer to be called, got {synthesizer._synthesize_count} calls"
        )

    async def test_conversation_history_updated(self):
        vad_probs = [0.9] * 10

        pipeline, vad, transcriber, completion, synthesizer = self._build_pipeline(
            vad_probs, [],
        )

        pipeline._running = True
        pipeline._enabled = True

        await pipeline._handle_transcript(
            TranscriptEvent(text="Jarvis, what time is it?", is_final=True)
        )
        pipeline._speech_ended_event.set()

        await asyncio.sleep(0.2)

        messages = pipeline._conversation.to_api_messages()
        user_messages = [m for m in messages if m["role"] == "user"]
        assistant_messages = [m for m in messages if m["role"] == "assistant"]

        assert len(user_messages) >= 1, "Expected user message in conversation history"
        assert len(assistant_messages) >= 1, "Expected assistant message in conversation history"


class TestBargeInMinimumSpeechDuration:
    def _build_speaking_pipeline(
        self,
        barge_in_min_speech_ms: int = 200,
    ) -> tuple[VoicePipeline, FakeVad, FakeCompletion, FakeSynthesizer, FakeAudioPlayback]:
        vad = FakeVad()
        transcriber = FakeTranscriber()
        completion = FakeCompletion()
        synthesizer = FakeSynthesizer()
        playback = FakeAudioPlayback()

        speech_detector = SpeechDetector(
            vad=vad,
            threshold=0.5,
            min_silence_ms=300,
            frame_duration_ms=FRAME_DURATION_MS,
        )

        pipeline = VoicePipeline(
            capture=FakeAudioCapture(),
            playback=playback,
            transcriber=transcriber,
            completion=completion,
            synthesizer=synthesizer,
            speech_detector=speech_detector,
            wake_word_detector=WakeWordDetector(["jarvis"]),
            conversation=ConversationHistory(max_turns=20),
            barge_in_enabled=True,
            barge_in_min_speech_ms=barge_in_min_speech_ms,
            frame_duration_ms=FRAME_DURATION_MS,
        )
        return pipeline, vad, completion, synthesizer, playback

    async def test_single_speech_frame_does_not_barge_in(self):
        pipeline, vad, completion, synthesizer, playback = self._build_speaking_pipeline()
        pipeline._state = PipelineState.SPEAKING
        pipeline._running = True
        pipeline._enabled = True

        vad.set_probabilities([0.9] + [0.0] * 5)

        frame = generate_silence(duration_ms=FRAME_DURATION_MS)
        for i in range(6):
            event = pipeline._speech_detector.process_frame(frame)
            if event.is_speech and pipeline._barge_in_enabled:
                pipeline._barge_in_speech_frames += 1
                if pipeline._barge_in_speech_frames >= pipeline._barge_in_min_frames:
                    await pipeline._handle_barge_in()
                    pipeline._barge_in_speech_frames = 0
            else:
                pipeline._barge_in_speech_frames = 0

        assert pipeline._state == PipelineState.SPEAKING, (
            f"Expected SPEAKING after single speech frame, got {pipeline._state}"
        )

    async def test_few_speech_frames_does_not_barge_in(self):
        pipeline, vad, completion, synthesizer, playback = self._build_speaking_pipeline()
        pipeline._state = PipelineState.SPEAKING
        pipeline._running = True
        pipeline._enabled = True

        min_frames_needed = int(200 / FRAME_DURATION_MS)
        too_few = min_frames_needed - 2
        vad.set_probabilities([0.9] * too_few + [0.0] * 5)

        frame = generate_silence(duration_ms=FRAME_DURATION_MS)
        for i in range(too_few + 5):
            event = pipeline._speech_detector.process_frame(frame)
            if event.is_speech and pipeline._barge_in_enabled:
                pipeline._barge_in_speech_frames += 1
                if pipeline._barge_in_speech_frames >= pipeline._barge_in_min_frames:
                    await pipeline._handle_barge_in()
                    pipeline._barge_in_speech_frames = 0
            else:
                pipeline._barge_in_speech_frames = 0

        assert pipeline._state == PipelineState.SPEAKING, (
            f"Expected SPEAKING after {too_few} speech frames, got {pipeline._state}"
        )

    async def test_sustained_speech_triggers_barge_in(self):
        pipeline, vad, completion, synthesizer, playback = self._build_speaking_pipeline()
        pipeline._state = PipelineState.SPEAKING
        pipeline._running = True
        pipeline._enabled = True

        min_frames_needed = int(200 / FRAME_DURATION_MS)
        enough_frames = min_frames_needed + 2
        vad.set_probabilities([0.9] * enough_frames)

        frame = generate_silence(duration_ms=FRAME_DURATION_MS)
        barge_in_triggered = False
        for i in range(enough_frames):
            event = pipeline._speech_detector.process_frame(frame)
            if event.is_speech and pipeline._barge_in_enabled:
                pipeline._barge_in_speech_frames += 1
                if pipeline._barge_in_speech_frames >= pipeline._barge_in_min_frames:
                    await pipeline._handle_barge_in()
                    pipeline._barge_in_speech_frames = 0
                    barge_in_triggered = True
                    break
            else:
                pipeline._barge_in_speech_frames = 0

        assert barge_in_triggered, (
            f"Expected barge-in after {enough_frames} speech frames, "
            f"but it was not triggered"
        )
        assert pipeline._state == PipelineState.LISTENING, (
            f"Expected LISTENING after barge-in, got {pipeline._state}"
        )

    async def test_barge_in_frame_count_matches_config(self):
        for min_speech_ms in [100, 200, 400]:
            pipeline, vad, _, _, _ = self._build_speaking_pipeline(
                barge_in_min_speech_ms=min_speech_ms,
            )
            expected_frames = int(min_speech_ms / FRAME_DURATION_MS)
            assert pipeline._barge_in_min_frames == expected_frames, (
                f"For {min_speech_ms}ms min speech, expected {expected_frames} frames, "
                f"got {pipeline._barge_in_min_frames}"
            )

    async def test_interrupted_speech_resets_barge_in_counter(self):
        pipeline, vad, completion, synthesizer, playback = self._build_speaking_pipeline()
        pipeline._state = PipelineState.SPEAKING
        pipeline._running = True
        pipeline._enabled = True

        min_frames_needed = int(200 / FRAME_DURATION_MS)
        almost_enough = min_frames_needed - 1
        speech_then_silence_then_speech = (
            [0.9] * almost_enough + [0.0] * 3 + [0.9] * almost_enough
        )
        vad.set_probabilities(speech_then_silence_then_speech)

        frame = generate_silence(duration_ms=FRAME_DURATION_MS)
        for i in range(len(speech_then_silence_then_speech)):
            event = pipeline._speech_detector.process_frame(frame)
            if event.is_speech and pipeline._barge_in_enabled:
                pipeline._barge_in_speech_frames += 1
                if pipeline._barge_in_speech_frames >= pipeline._barge_in_min_frames:
                    await pipeline._handle_barge_in()
                    pipeline._barge_in_speech_frames = 0
            else:
                pipeline._barge_in_speech_frames = 0

        assert pipeline._state == PipelineState.SPEAKING, (
            f"Expected SPEAKING (silence gap should reset counter), got {pipeline._state}"
        )
