import asyncio
import sys
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cli import _load_env_file

_load_env_file()

from config import VoicePipelineConfig
from adapters.silero_vad import SileroVad
from adapters.sounddevice_audio import SounddevicePlayback
from adapters.deepgram_stt import DeepgramStreamingTranscriber
from adapters.openai_whisper_stt import OpenAIWhisperTranscriber
from adapters.openclaw_llm import OpenClawCompletion
from adapters.openai_tts import OpenAITtsSynthesizer
from domain.conversation import ConversationHistory
from domain.wake_word import WakeWordDetector
from domain.speech_detector import SpeechDetector
from domain.pipeline import VoicePipeline
from domain.state import PipelineState

RECORDINGS_DIR = Path(__file__).parent / "recordings"
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 32
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
BYTES_PER_FRAME = FRAME_SIZE * 2


class WavFileCapture:
    def __init__(
        self,
        wav_path: Path,
        frame_duration_ms: int = 32,
        post_silence_seconds: float = 5.0,
    ) -> None:
        self._wav_path = wav_path
        self._sample_rate = SAMPLE_RATE
        self._frame_size = FRAME_SIZE
        self._frame_duration_ms = frame_duration_ms
        self._post_silence_seconds = post_silence_seconds
        self._frame_delay = frame_duration_ms / 1000.0

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
        with wave.open(str(self._wav_path), "rb") as wf:
            pcm = wf.readframes(wf.getnframes())

        for i in range(0, len(pcm), BYTES_PER_FRAME):
            frame = pcm[i : i + BYTES_PER_FRAME]
            if len(frame) < BYTES_PER_FRAME:
                break
            yield frame
            await asyncio.sleep(self._frame_delay)

        silence_frame = b"\x00" * BYTES_PER_FRAME
        silence_frame_count = int(
            self._post_silence_seconds * 1000 / self._frame_duration_ms
        )
        for _ in range(silence_frame_count):
            yield silence_frame
            await asyncio.sleep(self._frame_delay)


class BufferPlayback:
    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._cancelled = False

    @property
    def played_chunks(self) -> list[bytes]:
        return self._chunks

    @property
    def total_bytes(self) -> int:
        return sum(len(c) for c in self._chunks)

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

    def save_wav(self, path: Path, sample_rate: int = 24000) -> None:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(self._chunks))


class SpeakerPlaybackWithTracking:
    def __init__(self, sample_rate: int = 24000) -> None:
        self._speaker = SounddevicePlayback(sample_rate=sample_rate)
        self._total_bytes_played = 0
        self._chunk_count = 0

    @property
    def total_bytes(self) -> int:
        return self._total_bytes_played

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    async def start(self) -> None:
        await self._speaker.start()

    async def stop(self) -> None:
        await self._speaker.stop()

    async def play_chunk(self, audio_data: bytes) -> None:
        self._total_bytes_played += len(audio_data)
        self._chunk_count += 1
        await self._speaker.play_chunk(audio_data)

    async def drain(self) -> None:
        await self._speaker.drain()

    async def cancel(self) -> None:
        await self._speaker.cancel()


def load_config() -> VoicePipelineConfig:
    config = VoicePipelineConfig()
    config.conversation_window_seconds = 2.0
    return config


def find_recording(clip_name: str) -> Path | None:
    for mic_dir in sorted(RECORDINGS_DIR.iterdir()):
        if not mic_dir.is_dir():
            continue
        wav = mic_dir / f"{clip_name}.wav"
        if wav.exists():
            return wav
    return None


def require_recording(clip_name: str) -> Path:
    path = find_recording(clip_name)
    if not path:
        pytest.skip(f"No recording found for {clip_name}")
    return path


def require_api_keys(config: VoicePipelineConfig) -> tuple[str, str, str]:
    gateway_token = config.read_secret(config.gateway_token_file)
    openai_api_key = config.read_secret(config.openai_api_key_file)
    deepgram_api_key = config.read_secret(config.deepgram_api_key_file)

    if not gateway_token:
        pytest.skip("No gateway token")
    if not openai_api_key:
        pytest.skip("No OpenAI API key")
    if not deepgram_api_key and config.stt_engine == "deepgram":
        pytest.skip("No Deepgram API key")

    return gateway_token, openai_api_key, deepgram_api_key


def build_e2e_pipeline(
    wav_path: Path,
    config: VoicePipelineConfig,
    gateway_token: str,
    openai_api_key: str,
    deepgram_api_key: str,
) -> tuple[VoicePipeline, BufferPlayback]:
    capture = WavFileCapture(wav_path)
    playback = BufferPlayback()
    vad = SileroVad(model_path=config.vad_model_path, sample_rate=config.sample_rate)

    if config.stt_engine == "deepgram":
        transcriber = DeepgramStreamingTranscriber(
            api_key=deepgram_api_key,
            sample_rate=config.sample_rate,
        )
    else:
        transcriber = OpenAIWhisperTranscriber(
            api_key=openai_api_key,
            sample_rate=config.sample_rate,
        )

    completion = OpenClawCompletion(
        gateway_url=config.gateway_url,
        token=gateway_token,
        model=config.model,
    )
    synthesizer = OpenAITtsSynthesizer(api_key=openai_api_key)
    speech_detector = SpeechDetector(
        vad=vad,
        threshold=config.vad_threshold,
        min_silence_ms=config.vad_min_silence_ms,
        frame_duration_ms=config.frame_duration_ms,
    )

    pipeline = VoicePipeline(
        capture=capture,
        playback=playback,
        transcriber=transcriber,
        completion=completion,
        synthesizer=synthesizer,
        speech_detector=speech_detector,
        wake_word_detector=WakeWordDetector(config.wake_words),
        conversation=ConversationHistory(max_turns=config.max_history_turns),
        default_agent=config.default_agent,
        conversation_window_seconds=config.conversation_window_seconds,
        barge_in_enabled=False,
        agent_voice_map=config.agent_voices,
        barge_in_min_speech_ms=config.barge_in_min_speech_ms,
        frame_duration_ms=config.frame_duration_ms,
    )

    return pipeline, playback


async def run_pipeline_with_timeout(
    pipeline: VoicePipeline, timeout: float = 45.0
) -> None:
    task = asyncio.create_task(pipeline.run())
    try:
        await asyncio.wait_for(
            _wait_for_pipeline_idle(pipeline, task),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        pass
    finally:
        if not task.done():
            await pipeline.stop()
            try:
                await task
            except asyncio.CancelledError:
                pass


async def _wait_for_pipeline_idle(
    pipeline: VoicePipeline, run_task: asyncio.Task
) -> None:
    saw_processing = False
    while not run_task.done():
        await asyncio.sleep(0.1)
        if pipeline.state in (
            PipelineState.LISTENING,
            PipelineState.THINKING,
            PipelineState.SPEAKING,
        ):
            saw_processing = True
        if saw_processing and pipeline.state in (
            PipelineState.CONVERSING,
            PipelineState.AMBIENT,
        ):
            await asyncio.sleep(0.5)
            return


@pytest.mark.slow
class TestE2EWithRecordings:
    @pytest.mark.asyncio
    async def test_wake_word_triggers_full_response(self):
        config = load_config()
        gateway_token, openai_api_key, deepgram_api_key = require_api_keys(config)
        wav_path = require_recording("wake_jarvis_weather")

        pipeline, playback = build_e2e_pipeline(
            wav_path,
            config,
            gateway_token,
            openai_api_key,
            deepgram_api_key,
        )

        await run_pipeline_with_timeout(pipeline)

        assert pipeline.state in (
            PipelineState.SPEAKING,
            PipelineState.CONVERSING,
            PipelineState.AMBIENT,
        ), (
            f"Expected SPEAKING/CONVERSING/AMBIENT after response, got {pipeline.state.name}"
        )

        assert playback.total_bytes > 0, "Expected TTS audio output but got nothing"
        assert len(playback.played_chunks) >= 1, "Expected at least one TTS chunk"

    @pytest.mark.asyncio
    async def test_robson_wake_word_triggers_response(self):
        config = load_config()
        config.wake_words = ["jarvis", "robson"]
        gateway_token, openai_api_key, deepgram_api_key = require_api_keys(config)
        wav_path = require_recording("wake_robson_command")

        pipeline, playback = build_e2e_pipeline(
            wav_path,
            config,
            gateway_token,
            openai_api_key,
            deepgram_api_key,
        )

        await run_pipeline_with_timeout(pipeline)

        assert playback.total_bytes > 0, "Expected TTS audio output for Robson command"

    @pytest.mark.asyncio
    async def test_silence_produces_no_response(self):
        config = load_config()
        gateway_token, openai_api_key, deepgram_api_key = require_api_keys(config)
        wav_path = require_recording("silence_3s")

        pipeline, playback = build_e2e_pipeline(
            wav_path,
            config,
            gateway_token,
            openai_api_key,
            deepgram_api_key,
        )

        await run_pipeline_with_timeout(pipeline, timeout=15.0)

        assert pipeline.state == PipelineState.AMBIENT, (
            f"Expected AMBIENT for silence, got {pipeline.state.name}"
        )
        assert playback.total_bytes == 0, "Expected no TTS output for silence"

    @pytest.mark.asyncio
    async def test_conversation_history_populated_after_response(self):
        config = load_config()
        gateway_token, openai_api_key, deepgram_api_key = require_api_keys(config)
        wav_path = require_recording("wake_jarvis_weather")

        pipeline, playback = build_e2e_pipeline(
            wav_path,
            config,
            gateway_token,
            openai_api_key,
            deepgram_api_key,
        )

        await run_pipeline_with_timeout(pipeline)

        messages = pipeline._conversation.to_api_messages()
        user_messages = [m for m in messages if m["role"] == "user"]
        assistant_messages = [m for m in messages if m["role"] == "assistant"]

        if playback.total_bytes > 0:
            assert len(user_messages) >= 1, (
                "Expected at least one user message in history"
            )
            assert len(assistant_messages) >= 1, (
                "Expected at least one assistant message in history"
            )

    @pytest.mark.asyncio
    async def test_tts_output_is_valid_pcm(self):
        config = load_config()
        gateway_token, openai_api_key, deepgram_api_key = require_api_keys(config)
        wav_path = require_recording("wake_jarvis_weather")

        pipeline, playback = build_e2e_pipeline(
            wav_path,
            config,
            gateway_token,
            openai_api_key,
            deepgram_api_key,
        )

        await run_pipeline_with_timeout(pipeline)

        if playback.total_bytes > 0:
            output_wav = Path("/tmp/e2e_tts_output.wav")
            playback.save_wav(output_wav)
            with wave.open(str(output_wav), "rb") as wf:
                assert wf.getnchannels() == 1
                assert wf.getsampwidth() == 2
                assert wf.getframerate() == 24000
                assert wf.getnframes() > 0

    @pytest.mark.asyncio
    async def test_loud_speech_without_wake_word_stays_ambient(self):
        config = load_config()
        gateway_token, openai_api_key, deepgram_api_key = require_api_keys(config)
        wav_path = require_recording("continuous_ramble")

        pipeline, playback = build_e2e_pipeline(
            wav_path,
            config,
            gateway_token,
            openai_api_key,
            deepgram_api_key,
        )

        await run_pipeline_with_timeout(pipeline, timeout=20.0)

        assert playback.total_bytes == 0, (
            "Expected no TTS output for speech without wake word"
        )


def build_audible_e2e_pipeline(
    wav_path: Path,
    config: VoicePipelineConfig,
    gateway_token: str,
    openai_api_key: str,
    deepgram_api_key: str,
) -> tuple[VoicePipeline, SpeakerPlaybackWithTracking]:
    capture = WavFileCapture(wav_path, post_silence_seconds=60.0)
    playback = SpeakerPlaybackWithTracking()
    vad = SileroVad(model_path=config.vad_model_path, sample_rate=config.sample_rate)

    if config.stt_engine == "deepgram":
        transcriber = DeepgramStreamingTranscriber(
            api_key=deepgram_api_key,
            sample_rate=config.sample_rate,
        )
    else:
        transcriber = OpenAIWhisperTranscriber(
            api_key=openai_api_key,
            sample_rate=config.sample_rate,
        )

    completion = OpenClawCompletion(
        gateway_url=config.gateway_url,
        token=gateway_token,
        model=config.model,
    )
    synthesizer = OpenAITtsSynthesizer(api_key=openai_api_key)
    speech_detector = SpeechDetector(
        vad=vad,
        threshold=config.vad_threshold,
        min_silence_ms=config.vad_min_silence_ms,
        frame_duration_ms=config.frame_duration_ms,
    )

    pipeline = VoicePipeline(
        capture=capture,
        playback=playback,
        transcriber=transcriber,
        completion=completion,
        synthesizer=synthesizer,
        speech_detector=speech_detector,
        wake_word_detector=WakeWordDetector(config.wake_words),
        conversation=ConversationHistory(max_turns=config.max_history_turns),
        default_agent=config.default_agent,
        conversation_window_seconds=config.conversation_window_seconds,
        barge_in_enabled=False,
        agent_voice_map=config.agent_voices,
        barge_in_min_speech_ms=config.barge_in_min_speech_ms,
        frame_duration_ms=config.frame_duration_ms,
    )

    return pipeline, playback


@pytest.mark.slow
class TestAudibleE2E:
    @pytest.mark.asyncio
    async def test_wake_word_plays_response_through_speakers(self):
        config = load_config()
        config.default_agent = "robson"
        config.wake_words = ["jarvis", "robson"]
        gateway_token, openai_api_key, deepgram_api_key = require_api_keys(config)
        wav_path = require_recording("wake_robson_command")

        pipeline, playback = build_audible_e2e_pipeline(
            wav_path,
            config,
            gateway_token,
            openai_api_key,
            deepgram_api_key,
        )

        await run_pipeline_with_timeout(pipeline, timeout=90.0)

        assert playback.total_bytes > 0, "No audio was played through speakers"
        assert playback.chunk_count >= 1, "Expected at least one audio chunk played"

        audio_duration_seconds = playback.total_bytes / (24000 * 2)
        assert audio_duration_seconds > 0.5, (
            f"Audio too short ({audio_duration_seconds:.1f}s), expected at least 0.5s of speech"
        )
