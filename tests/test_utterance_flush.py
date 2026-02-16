import asyncio
import collections
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
BUILTIN_RECORDINGS_DIR = Path(__file__).parent / "recordings" / "builtin"
HEADSET_RECORDINGS_DIR = Path(__file__).parent / "recordings" / "headset"


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
    silence_frame = np.zeros(FRAME_SIZE, dtype=np.int16).tobytes()
    return [silence_frame for _ in range(frame_count)]


def generate_sine_wave(duration_ms: int = 100) -> bytes:
    num_samples = int(SAMPLE_RATE * duration_ms / 1000)
    t = np.arange(num_samples) / SAMPLE_RATE
    signal = np.sin(2 * np.pi * 440.0 * t) * 0.8
    return (signal * 32767).astype(np.int16).tobytes()


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


def count_speech_events(
    vad: SileroVad,
    frames: list[bytes],
    min_silence_ms: int = 800,
) -> tuple[int, int]:
    speech_detector = SpeechDetector(
        vad=vad,
        threshold=0.5,
        min_silence_ms=min_silence_ms,
        frame_duration_ms=FRAME_DURATION_MS,
    )
    starts = 0
    ends = 0
    for frame in frames:
        event = speech_detector.process_frame(frame)
        if event == SpeechEvent.SPEECH_START:
            starts += 1
        elif event == SpeechEvent.SPEECH_END:
            ends += 1
    return starts, ends


def skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Recording not found: {path}")


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
        self._default_response = ["Hello", " there."]
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


def build_pipeline_with_real_vad(
    real_vad: SileroVad,
    frames: list[bytes],
    transcriber: QueueBasedFakeTranscriber | None = None,
    wake_words: list[str] | None = None,
    barge_in_enabled: bool = True,
    barge_in_min_speech_ms: int = 200,
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
        wake_word_detector=WakeWordDetector(wake_words or ["jarvis", "robson"]),
        conversation=ConversationHistory(max_turns=20),
        conversation_window_seconds=15.0,
        barge_in_enabled=barge_in_enabled,
        barge_in_min_speech_ms=barge_in_min_speech_ms,
        frame_duration_ms=FRAME_DURATION_MS,
    )
    return pipeline, transcriber, completion, synthesizer


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


async def wait_for_session_count(
    transcriber: QueueBasedFakeTranscriber,
    target_count: int,
    timeout_seconds: float = 10.0,
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_seconds
    while asyncio.get_event_loop().time() < deadline:
        if transcriber.start_session_count >= target_count:
            return
        await asyncio.sleep(0)
    raise AssertionError(
        f"Session count did not reach {target_count} within {timeout_seconds}s, "
        f"stuck at {transcriber.start_session_count}"
    )


class TestSpeechSegmentDetectionOnBuiltinRecordings:
    @pytest.fixture
    def real_vad(self):
        return SileroVad(sample_rate=SAMPLE_RATE)

    async def test_short_complete_sentence_produces_speech(self, real_vad):
        path = BUILTIN_RECORDINGS_DIR / "short_complete_sentence.wav"
        skip_if_missing(path)
        frames = load_recording_as_frames(path) + generate_silence_frames(1500)
        starts, ends = count_speech_events(real_vad, frames)
        assert starts >= 1, f"Expected at least 1 speech start, got {starts}"
        assert ends >= 1, f"Expected at least 1 speech end, got {ends}"

    async def test_long_sentence_natural_pauses_produces_multiple_segments(self, real_vad):
        path = BUILTIN_RECORDINGS_DIR / "long_sentence_natural_pauses.wav"
        skip_if_missing(path)
        frames = load_recording_as_frames(path) + generate_silence_frames(1500)
        starts, ends = count_speech_events(real_vad, frames)
        assert starts >= 2, f"Expected at least 2 speech starts for long sentence with pauses, got {starts}"
        assert ends >= 2, f"Expected at least 2 speech ends for long sentence with pauses, got {ends}"

    async def test_incomplete_sentence_produces_speech(self, real_vad):
        path = BUILTIN_RECORDINGS_DIR / "incomplete_sentence_stops.wav"
        skip_if_missing(path)
        frames = load_recording_as_frames(path) + generate_silence_frames(1500)
        starts, ends = count_speech_events(real_vad, frames)
        assert starts >= 1, f"Expected at least 1 speech start, got {starts}"
        assert ends >= 1, f"Expected at least 1 speech end, got {ends}"

    async def test_chained_sentences_produce_multiple_segments(self, real_vad):
        path = BUILTIN_RECORDINGS_DIR / "chained_sentences_one_thought.wav"
        skip_if_missing(path)
        frames = load_recording_as_frames(path) + generate_silence_frames(1500)
        starts, ends = count_speech_events(real_vad, frames)
        assert starts >= 2, f"Expected at least 2 speech starts for chained sentences, got {starts}"
        assert ends >= 2, f"Expected at least 2 speech ends for chained sentences, got {ends}"

    async def test_mid_sentence_pause_produces_two_segments(self, real_vad):
        path = BUILTIN_RECORDINGS_DIR / "mid_sentence_pause_then_continue.wav"
        skip_if_missing(path)
        frames = load_recording_as_frames(path) + generate_silence_frames(1500)
        starts, ends = count_speech_events(real_vad, frames)
        assert starts >= 2, f"Expected at least 2 speech starts for mid-sentence pause, got {starts}"
        assert ends >= 2, f"Expected at least 2 speech ends for mid-sentence pause, got {ends}"

    async def test_two_quick_follow_ups_produce_two_segments(self, real_vad):
        path = BUILTIN_RECORDINGS_DIR / "two_quick_follow_ups.wav"
        skip_if_missing(path)
        frames = load_recording_as_frames(path) + generate_silence_frames(1500)
        starts, ends = count_speech_events(real_vad, frames)
        assert starts >= 2, f"Expected at least 2 speech starts for follow-ups, got {starts}"
        assert ends >= 2, f"Expected at least 2 speech ends for follow-ups, got {ends}"

    async def test_barge_in_speech_produces_sustained_speech(self, real_vad):
        path = BUILTIN_RECORDINGS_DIR / "barge_in_speech.wav"
        skip_if_missing(path)
        frames = load_recording_as_frames(path) + generate_silence_frames(1500)
        starts, ends = count_speech_events(real_vad, frames)
        assert starts >= 1, f"Expected at least 1 speech start for barge-in speech, got {starts}"


class TestUtteranceFlushWithRealVad:
    @pytest.fixture
    def real_vad(self):
        return SileroVad(sample_rate=SAMPLE_RATE)

    async def test_wake_word_plus_short_sentence_produces_single_response(self, real_vad):
        path = BUILTIN_RECORDINGS_DIR / "short_complete_sentence.wav"
        skip_if_missing(path)

        frames = load_recording_as_frames(path) + generate_silence_frames(3000)
        transcriber = QueueBasedFakeTranscriber()
        pipeline, _, completion, _ = build_pipeline_with_real_vad(
            real_vad, frames, transcriber=transcriber,
        )
        pipeline._running = True
        pipeline._enabled = True

        async def director():
            await wait_for_session_count(transcriber, 1)
            transcriber.push_transcript("Hey Robson, what time is it?", is_final=True)
            await wait_for_completion_count(completion, 1)
            pipeline._running = False
            transcriber.stop()

        await asyncio.gather(
            pipeline._audio_loop(),
            pipeline._transcript_loop(),
            director(),
        )

        assert completion._call_count == 1
        user_messages = [
            m for m in pipeline._conversation.to_api_messages()
            if m["role"] == "user"
        ]
        assert len(user_messages) == 1
        assert "what time is it" in user_messages[0]["content"].lower()

    async def test_long_sentence_with_pauses_collected_as_single_utterance(self, real_vad):
        path = BUILTIN_RECORDINGS_DIR / "long_sentence_natural_pauses.wav"
        skip_if_missing(path)

        frames = load_recording_as_frames(path) + generate_silence_frames(5000)
        transcriber = QueueBasedFakeTranscriber()
        pipeline, _, completion, _ = build_pipeline_with_real_vad(
            real_vad, frames, transcriber=transcriber,
        )
        pipeline._running = True
        pipeline._enabled = True

        async def director():
            await wait_for_session_count(transcriber, 1)
            transcriber.push_transcript(
                "Jarvis, I woke up today very late", is_final=False,
            )
            await asyncio.sleep(0.3)
            transcriber.push_transcript(
                "Jarvis, I woke up today very late, but now I'm not feeling very great.",
                is_final=True,
            )
            await asyncio.sleep(0.5)
            transcriber.push_transcript(
                "What do you think it is? Maybe sick or something.",
                is_final=True,
            )
            await wait_for_completion_count(completion, 1, timeout_seconds=10.0)
            pipeline._running = False
            transcriber.stop()

        await asyncio.gather(
            pipeline._audio_loop(),
            pipeline._transcript_loop(),
            director(),
        )

        assert completion._call_count == 1
        user_messages = [
            m for m in pipeline._conversation.to_api_messages()
            if m["role"] == "user"
        ]
        assert len(user_messages) == 1

    async def test_incomplete_sentence_still_gets_processed_after_grace_period(self, real_vad):
        path = BUILTIN_RECORDINGS_DIR / "incomplete_sentence_stops.wav"
        skip_if_missing(path)

        frames = load_recording_as_frames(path) + generate_silence_frames(6000)
        transcriber = QueueBasedFakeTranscriber()
        pipeline, _, completion, _ = build_pipeline_with_real_vad(
            real_vad, frames, transcriber=transcriber,
        )
        pipeline._running = True
        pipeline._enabled = True

        async def director():
            await wait_for_session_count(transcriber, 1)
            transcriber.push_transcript(
                "Jarvis, if you were to choose a color for your perfect",
                is_final=True,
            )
            await wait_for_completion_count(completion, 1, timeout_seconds=10.0)
            pipeline._running = False
            transcriber.stop()

        await asyncio.gather(
            pipeline._audio_loop(),
            pipeline._transcript_loop(),
            director(),
        )

        assert completion._call_count == 1
        user_messages = [
            m for m in pipeline._conversation.to_api_messages()
            if m["role"] == "user"
        ]
        assert len(user_messages) == 1
        assert "color" in user_messages[0]["content"].lower()

    async def test_chained_sentences_collected_as_single_utterance(self, real_vad):
        path = BUILTIN_RECORDINGS_DIR / "chained_sentences_one_thought.wav"
        skip_if_missing(path)

        frames = load_recording_as_frames(path) + generate_silence_frames(5000)
        transcriber = QueueBasedFakeTranscriber()
        pipeline, _, completion, _ = build_pipeline_with_real_vad(
            real_vad, frames, transcriber=transcriber,
        )
        pipeline._running = True
        pipeline._enabled = True

        async def director():
            await wait_for_session_count(transcriber, 1)
            transcriber.push_transcript(
                "Jarvis, I find it really interesting to be doing this",
                is_final=False,
            )
            await asyncio.sleep(0.3)
            transcriber.push_transcript(
                "Jarvis, I find it really interesting to be doing this. Now we have just some minor changes to do",
                is_final=False,
            )
            await asyncio.sleep(0.3)
            transcriber.push_transcript(
                "Jarvis, I find it really interesting to be doing this. Now we have just some minor changes to do, like running this at a CLI. So yeah, we are getting there.",
                is_final=True,
            )
            await wait_for_completion_count(completion, 1, timeout_seconds=10.0)
            pipeline._running = False
            transcriber.stop()

        await asyncio.gather(
            pipeline._audio_loop(),
            pipeline._transcript_loop(),
            director(),
        )

        assert completion._call_count == 1
        user_messages = [
            m for m in pipeline._conversation.to_api_messages()
            if m["role"] == "user"
        ]
        assert len(user_messages) == 1
        assert "getting there" in user_messages[0]["content"].lower()

    async def test_mid_sentence_pause_then_continue_collected_as_single_utterance(self, real_vad):
        path = BUILTIN_RECORDINGS_DIR / "mid_sentence_pause_then_continue.wav"
        skip_if_missing(path)

        frames = load_recording_as_frames(path) + generate_silence_frames(5000)
        transcriber = QueueBasedFakeTranscriber()
        pipeline, _, completion, _ = build_pipeline_with_real_vad(
            real_vad, frames, transcriber=transcriber,
        )
        pipeline._running = True
        pipeline._enabled = True

        async def director():
            await wait_for_session_count(transcriber, 1)
            transcriber.push_transcript(
                "Jarvis, I was thinking about", is_final=False,
            )
            await asyncio.sleep(0.5)
            transcriber.push_transcript(
                "Jarvis, I was thinking about a really interesting feature.",
                is_final=True,
            )
            await wait_for_completion_count(completion, 1, timeout_seconds=10.0)
            pipeline._running = False
            transcriber.stop()

        await asyncio.gather(
            pipeline._audio_loop(),
            pipeline._transcript_loop(),
            director(),
        )

        assert completion._call_count == 1
        user_messages = [
            m for m in pipeline._conversation.to_api_messages()
            if m["role"] == "user"
        ]
        assert len(user_messages) == 1
        assert "interesting feature" in user_messages[0]["content"].lower()


class TestChainedConversationWithBuiltinRecordings:
    @pytest.fixture
    def real_vad(self):
        return SileroVad(sample_rate=SAMPLE_RATE)

    async def test_wake_word_then_follow_up_produces_two_responses(self, real_vad):
        wake_path = BUILTIN_RECORDINGS_DIR / "wake_robson_command.wav"
        followup_path = BUILTIN_RECORDINGS_DIR / "two_quick_follow_ups.wav"
        skip_if_missing(wake_path)
        skip_if_missing(followup_path)

        frames = concatenate_recordings_with_silence(
            [wake_path, followup_path], gap_ms=2000, trailing_silence_ms=3000,
        )
        transcriber = QueueBasedFakeTranscriber()
        pipeline, _, completion, _ = build_pipeline_with_real_vad(
            real_vad, frames, transcriber=transcriber,
        )
        pipeline._running = True
        pipeline._enabled = True

        async def director():
            await wait_for_session_count(transcriber, 1)
            transcriber.push_transcript(
                "Robson, turn off the lights.", is_final=True,
            )
            await wait_for_completion_count(completion, 1)

            await wait_for_session_count(transcriber, 2, timeout_seconds=15.0)
            transcriber.push_transcript(
                "Hello there, now tell me a joke.", is_final=True,
            )
            await wait_for_completion_count(completion, 2)

            pipeline._running = False
            transcriber.stop()

        await asyncio.gather(
            pipeline._audio_loop(),
            pipeline._transcript_loop(),
            director(),
        )

        assert completion._call_count >= 2
        messages = pipeline._conversation.to_api_messages()
        user_messages = [m for m in messages if m["role"] == "user"]
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(user_messages) >= 2
        assert len(assistant_messages) >= 2

    async def test_wake_word_then_long_sentence_then_follow_up(self, real_vad):
        wake_path = BUILTIN_RECORDINGS_DIR / "wake_jarvis_weather.wav"
        long_path = BUILTIN_RECORDINGS_DIR / "long_sentence_natural_pauses.wav"
        followup_path = BUILTIN_RECORDINGS_DIR / "short_complete_sentence.wav"
        skip_if_missing(wake_path)
        skip_if_missing(long_path)
        skip_if_missing(followup_path)

        frames = concatenate_recordings_with_silence(
            [wake_path, long_path, followup_path],
            gap_ms=2000,
            trailing_silence_ms=3000,
        )
        transcriber = QueueBasedFakeTranscriber()
        pipeline, _, completion, _ = build_pipeline_with_real_vad(
            real_vad, frames, transcriber=transcriber,
        )
        pipeline._running = True
        pipeline._enabled = True

        async def director():
            await wait_for_session_count(transcriber, 1)
            transcriber.push_transcript(
                "Jarvis, what's the weather like today?", is_final=True,
            )
            await wait_for_completion_count(completion, 1)

            await wait_for_session_count(transcriber, 2, timeout_seconds=15.0)
            transcriber.push_transcript(
                "I woke up today very late. What do you think?", is_final=True,
            )
            await wait_for_completion_count(completion, 2)

            await wait_for_session_count(transcriber, 3, timeout_seconds=15.0)
            transcriber.push_transcript(
                "What time is it?", is_final=True,
            )
            await wait_for_completion_count(completion, 3)

            pipeline._running = False
            transcriber.stop()

        await asyncio.gather(
            pipeline._audio_loop(),
            pipeline._transcript_loop(),
            director(),
        )

        assert completion._call_count >= 3
        messages = pipeline._conversation.to_api_messages()
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) >= 3


class TestBargeInWithRealVad:
    @pytest.fixture
    def real_vad(self):
        return SileroVad(sample_rate=SAMPLE_RATE)

    async def test_barge_in_speech_triggers_state_change_to_listening(self, real_vad):
        path = BUILTIN_RECORDINGS_DIR / "barge_in_speech.wav"
        skip_if_missing(path)

        frames = load_recording_as_frames(path) + generate_silence_frames(1000)
        transcriber = QueueBasedFakeTranscriber()
        pipeline, _, completion, synthesizer = build_pipeline_with_real_vad(
            real_vad, frames, transcriber=transcriber,
            barge_in_enabled=True, barge_in_min_speech_ms=200,
        )

        pipeline._state = PipelineState.SPEAKING
        pipeline._running = True
        pipeline._enabled = True
        pipeline._conversation.add_user_message("previous question")
        pipeline._conversation.add_assistant_message("previous response being spoken")
        pipeline._spoken_text_buffer = "previous response"

        barge_in_triggered = False
        original_handle_barge_in = pipeline._handle_barge_in

        async def tracking_handle_barge_in():
            nonlocal barge_in_triggered
            barge_in_triggered = True
            await original_handle_barge_in()

        pipeline._handle_barge_in = tracking_handle_barge_in

        await pipeline._audio_loop()

        assert barge_in_triggered, (
            "Expected barge-in to be triggered by sustained speech in barge_in_speech.wav"
        )
        assert pipeline._state == PipelineState.LISTENING

    async def test_barge_in_recording_exceeds_sliding_window_speech_ratio(self, real_vad):
        path = BUILTIN_RECORDINGS_DIR / "barge_in_speech.wav"
        skip_if_missing(path)

        frames = load_recording_as_frames(path) + generate_silence_frames(1500)
        speech_detector = SpeechDetector(
            vad=real_vad,
            threshold=0.5,
            min_silence_ms=800,
            frame_duration_ms=FRAME_DURATION_MS,
        )

        barge_in_window_size = int(200 / FRAME_DURATION_MS)
        barge_in_speech_ratio_threshold = 0.7
        window: collections.deque[bool] = collections.deque(maxlen=barge_in_window_size)
        window_triggered = False

        for frame in frames:
            event = speech_detector.process_frame(frame)
            window.append(event.is_speech)
            if (
                len(window) >= barge_in_window_size
                and sum(window) / len(window) >= barge_in_speech_ratio_threshold
            ):
                window_triggered = True
                break

        assert window_triggered, (
            f"Expected sliding window ({barge_in_window_size} frames, "
            f"{barge_in_speech_ratio_threshold:.0%} threshold) to trigger "
            f"on barge_in_speech.wav"
        )

    async def test_barge_in_during_chained_conversation_with_recordings(self, real_vad):
        wake_path = BUILTIN_RECORDINGS_DIR / "short_complete_sentence.wav"
        barge_path = BUILTIN_RECORDINGS_DIR / "barge_in_speech.wav"
        skip_if_missing(wake_path)
        skip_if_missing(barge_path)

        wake_frames = load_recording_as_frames(wake_path)
        silence_gap = generate_silence_frames(3000)
        barge_frames = load_recording_as_frames(barge_path)
        trailing_silence = generate_silence_frames(2000)
        all_frames = wake_frames + silence_gap + barge_frames + trailing_silence

        class SlowFakeCompletion:
            def __init__(self) -> None:
                self._cancelled = False
                self._call_count = 0

            async def stream(self, messages, agent) -> AsyncIterator[str]:
                self._cancelled = False
                self._call_count += 1
                chunks = ["Well, ", "the current ", "time is ", "about ", "three ", "o'clock."]
                for chunk in chunks:
                    if self._cancelled:
                        break
                    yield chunk
                    await asyncio.sleep(0.2)

            async def cancel(self) -> None:
                self._cancelled = True

        transcriber = QueueBasedFakeTranscriber()
        slow_completion = SlowFakeCompletion()
        synthesizer = FakeSynthesizer()

        speech_detector = SpeechDetector(
            vad=real_vad,
            threshold=0.5,
            min_silence_ms=800,
            frame_duration_ms=FRAME_DURATION_MS,
        )

        pipeline = VoicePipeline(
            capture=YieldingFakeAudioCapture(all_frames, frame_delay_seconds=0.001),
            playback=FakeAudioPlayback(),
            transcriber=transcriber,
            completion=slow_completion,
            synthesizer=synthesizer,
            speech_detector=speech_detector,
            wake_word_detector=WakeWordDetector(["jarvis", "robson"]),
            conversation=ConversationHistory(max_turns=20),
            conversation_window_seconds=15.0,
            barge_in_enabled=True,
            barge_in_min_speech_ms=200,
            frame_duration_ms=FRAME_DURATION_MS,
        )
        pipeline._running = True
        pipeline._enabled = True

        barge_in_count = 0
        original_handle_barge_in = pipeline._handle_barge_in

        async def counting_handle_barge_in():
            nonlocal barge_in_count
            barge_in_count += 1
            await original_handle_barge_in()

        pipeline._handle_barge_in = counting_handle_barge_in

        async def director():
            await wait_for_session_count(transcriber, 1)
            transcriber.push_transcript(
                "Hey Robson, what time is it?", is_final=True,
            )

            deadline = asyncio.get_event_loop().time() + 20.0
            while asyncio.get_event_loop().time() < deadline:
                if barge_in_count > 0 or slow_completion._call_count >= 1:
                    break
                await asyncio.sleep(0.01)

            deadline = asyncio.get_event_loop().time() + 20.0
            while asyncio.get_event_loop().time() < deadline:
                if barge_in_count > 0:
                    break
                await asyncio.sleep(0.01)

            await asyncio.sleep(0.2)
            pipeline._running = False
            transcriber.stop()

        await asyncio.gather(
            pipeline._audio_loop(),
            pipeline._transcript_loop(),
            director(),
        )

        assert slow_completion._call_count >= 1
        assert barge_in_count >= 1, (
            "Expected barge-in when barge_in_speech.wav plays during slow response"
        )


class TestHeadsetRecordingsStillWork:
    @pytest.fixture
    def real_vad(self):
        return SileroVad(sample_rate=SAMPLE_RATE)

    async def test_headset_wake_word_triggers_stt_session(self, real_vad):
        path = HEADSET_RECORDINGS_DIR / "wake_jarvis_weather.wav"
        skip_if_missing(path)

        frames = load_recording_as_frames(path) + generate_silence_frames(1500)
        transcriber = QueueBasedFakeTranscriber()
        pipeline, _, _, _ = build_pipeline_with_real_vad(
            real_vad, frames, transcriber=transcriber,
        )
        pipeline._running = True
        pipeline._enabled = True

        await pipeline._audio_loop()

        assert transcriber.start_session_count >= 1, (
            f"Expected at least 1 STT session from headset wake_jarvis_weather.wav, "
            f"got {transcriber.start_session_count}"
        )

    async def test_headset_continuous_ramble_produces_speech(self, real_vad):
        path = HEADSET_RECORDINGS_DIR / "continuous_ramble.wav"
        skip_if_missing(path)

        frames = load_recording_as_frames(path) + generate_silence_frames(1500)
        starts, ends = count_speech_events(real_vad, frames)
        assert starts >= 1, f"Expected at least 1 speech start from headset continuous_ramble, got {starts}"

    async def test_headset_silence_does_not_trigger_stt(self, real_vad):
        path = HEADSET_RECORDINGS_DIR / "silence_3s.wav"
        skip_if_missing(path)

        frames = load_recording_as_frames(path) + generate_silence_frames(500)
        transcriber = QueueBasedFakeTranscriber()
        pipeline, _, _, _ = build_pipeline_with_real_vad(
            real_vad, frames, transcriber=transcriber,
        )
        pipeline._running = True
        pipeline._enabled = True

        await pipeline._audio_loop()

        assert transcriber.start_session_count == 0, (
            f"Expected no STT session for headset silence, "
            f"got {transcriber.start_session_count}"
        )

    async def test_headset_loud_speech_produces_speech(self, real_vad):
        path = HEADSET_RECORDINGS_DIR / "loud_speech.wav"
        skip_if_missing(path)

        frames = load_recording_as_frames(path) + generate_silence_frames(1500)
        starts, ends = count_speech_events(real_vad, frames)
        assert starts >= 1, f"Expected at least 1 speech start from headset loud_speech, got {starts}"
        assert ends >= 1, f"Expected at least 1 speech end from headset loud_speech, got {ends}"

    async def test_headset_two_recordings_chained_produce_multiple_sessions(self, real_vad):
        paths = [
            HEADSET_RECORDINGS_DIR / "wake_jarvis_weather.wav",
            HEADSET_RECORDINGS_DIR / "loud_speech.wav",
        ]
        for p in paths:
            skip_if_missing(p)

        frames = concatenate_recordings_with_silence(
            paths, gap_ms=2000, trailing_silence_ms=2000,
        )
        transcriber = QueueBasedFakeTranscriber()
        pipeline, _, completion, _ = build_pipeline_with_real_vad(
            real_vad, frames, transcriber=transcriber,
        )
        pipeline._running = True
        pipeline._enabled = True

        async def director():
            await wait_for_session_count(transcriber, 1)
            transcriber.push_transcript(
                "Jarvis, what's the weather?", is_final=True,
            )
            await wait_for_completion_count(completion, 1)

            await wait_for_session_count(transcriber, 2, timeout_seconds=15.0)
            transcriber.push_transcript(
                "Tell me more about that.", is_final=True,
            )
            await wait_for_completion_count(completion, 2)

            pipeline._running = False
            transcriber.stop()

        await asyncio.gather(
            pipeline._audio_loop(),
            pipeline._transcript_loop(),
            director(),
        )

        assert completion._call_count >= 2
