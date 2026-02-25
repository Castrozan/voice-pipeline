import io
import os
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cli import _load_env_file

_load_env_file()

import pytest

from adapters.silero_vad import SileroVad, DEFAULT_MODEL_PATH
from domain.speech_detector import SpeechDetector, SpeechEvent

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 16
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
BYTES_PER_FRAME = FRAME_SIZE * 2
PREBUFFER_MS = 300
PREBUFFER_FRAMES = int(PREBUFFER_MS / FRAME_DURATION_MS)
VAD_THRESHOLD = 0.5
MIN_SILENCE_MS = 300

RECORDINGS_DIR = Path(__file__).parent / "recordings"


def _resolve_openai_api_key() -> str:
    key_file = os.environ.get("VOICE_PIPELINE_OPENAI_API_KEY_FILE", "")
    if key_file and Path(key_file).exists():
        return Path(key_file).read_text().strip()
    raw_key = os.environ.get("OPENAI_API_KEY", "")
    if raw_key:
        return raw_key
    return ""


def _load_wav_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE, (
            f"Expected {SAMPLE_RATE}Hz, got {wf.getframerate()}Hz"
        )
        assert wf.getnchannels() == 1, (
            f"Expected mono, got {wf.getnchannels()} channels"
        )
        return wf.readframes(wf.getnframes())


def _split_pcm_into_frames(pcm_data: bytes) -> list[bytes]:
    frames = []
    for i in range(0, len(pcm_data), BYTES_PER_FRAME):
        chunk = pcm_data[i : i + BYTES_PER_FRAME]
        if len(chunk) == BYTES_PER_FRAME:
            frames.append(chunk)
    return frames


def _find_speech_start_frame(frames: list[bytes]) -> int:
    vad = SileroVad(sample_rate=SAMPLE_RATE)
    detector = SpeechDetector(
        vad=vad,
        threshold=VAD_THRESHOLD,
        min_silence_ms=MIN_SILENCE_MS,
        frame_duration_ms=FRAME_DURATION_MS,
    )
    for idx, frame in enumerate(frames):
        event = detector.process_frame(frame)
        if event == SpeechEvent.SPEECH_START:
            return idx
    return -1


def _pcm_frames_to_wav_bytes(frames: list[bytes]) -> bytes:
    pcm_data = b"".join(frames)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)
    return buffer.getvalue()


def _transcribe_with_whisper(wav_bytes: bytes, api_key: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    wav_file = io.BytesIO(wav_bytes)
    wav_file.name = "audio.wav"
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=wav_file,
        language="en",
    )
    return response.text.strip()


def transcribe_recording(
    wav_path: Path,
    api_key: str,
    use_prebuffer: bool,
) -> str:
    pcm_data = _load_wav_pcm(wav_path)
    frames = _split_pcm_into_frames(pcm_data)
    speech_start = _find_speech_start_frame(frames)
    assert speech_start >= 0, f"No speech detected in {wav_path.name}"

    if use_prebuffer:
        start_frame = max(0, speech_start - PREBUFFER_FRAMES)
    else:
        start_frame = speech_start

    selected_frames = frames[start_frame:]
    wav_bytes = _pcm_frames_to_wav_bytes(selected_frames)
    return _transcribe_with_whisper(wav_bytes, api_key)


requires_openai_key = pytest.mark.skipif(
    not _resolve_openai_api_key(),
    reason="OpenAI API key not available",
)

requires_vad_model = pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(),
    reason="Silero VAD model not cached",
)


def _recording_exists(relative_path: str) -> bool:
    return (RECORDINGS_DIR / relative_path).exists()


@pytest.mark.slow
@requires_openai_key
@requires_vad_model
class TestHeadsetTranscriptionWithoutPrebuffer:
    @pytest.mark.skipif(
        not _recording_exists("headset/wake_jarvis_weather.wav"),
        reason="Recording not found",
    )
    def test_headset_jarvis_weather_without_prebuffer(self):
        api_key = _resolve_openai_api_key()
        wav_path = RECORDINGS_DIR / "headset" / "wake_jarvis_weather.wav"
        transcript = transcribe_recording(wav_path, api_key, use_prebuffer=False)
        transcript_lower = transcript.lower()
        print(f"WITHOUT prebuffer: '{transcript}'")
        assert "jarvis" in transcript_lower or "weather" in transcript_lower, (
            f"Transcript too degraded without prebuffer: '{transcript}'"
        )

    @pytest.mark.skipif(
        not _recording_exists("headset/wake_robson_command.wav"),
        reason="Recording not found",
    )
    def test_headset_robson_command_without_prebuffer(self):
        api_key = _resolve_openai_api_key()
        wav_path = RECORDINGS_DIR / "headset" / "wake_robson_command.wav"
        transcript = transcribe_recording(wav_path, api_key, use_prebuffer=False)
        transcript_lower = transcript.lower()
        print(f"WITHOUT prebuffer: '{transcript}'")
        assert "light" in transcript_lower or "robson" in transcript_lower, (
            f"Transcript too degraded without prebuffer: '{transcript}'"
        )


@pytest.mark.slow
@requires_openai_key
@requires_vad_model
class TestHeadsetTranscriptionWithPrebuffer:
    @pytest.mark.skipif(
        not _recording_exists("headset/wake_jarvis_weather.wav"),
        reason="Recording not found",
    )
    def test_headset_jarvis_weather_with_prebuffer(self):
        api_key = _resolve_openai_api_key()
        wav_path = RECORDINGS_DIR / "headset" / "wake_jarvis_weather.wav"
        transcript = transcribe_recording(wav_path, api_key, use_prebuffer=True)
        transcript_lower = transcript.lower()
        print(f"WITH prebuffer: '{transcript}'")
        assert "jarvis" in transcript_lower, (
            f"Expected 'jarvis' in transcript with prebuffer: '{transcript}'"
        )
        assert "weather" in transcript_lower, (
            f"Expected 'weather' in transcript with prebuffer: '{transcript}'"
        )

    @pytest.mark.skipif(
        not _recording_exists("headset/wake_robson_command.wav"),
        reason="Recording not found",
    )
    def test_headset_robson_command_with_prebuffer(self):
        api_key = _resolve_openai_api_key()
        wav_path = RECORDINGS_DIR / "headset" / "wake_robson_command.wav"
        transcript = transcribe_recording(wav_path, api_key, use_prebuffer=True)
        transcript_lower = transcript.lower()
        print(f"WITH prebuffer: '{transcript}'")
        assert "robson" in transcript_lower, (
            f"Expected 'robson' in transcript with prebuffer: '{transcript}'"
        )
        assert "light" in transcript_lower, (
            f"Expected 'light' in transcript with prebuffer: '{transcript}'"
        )


@pytest.mark.slow
@requires_openai_key
@requires_vad_model
class TestBuiltinTranscriptionWithPrebuffer:
    @pytest.mark.skipif(
        not _recording_exists("builtin/wake_jarvis_weather.wav"),
        reason="Recording not found",
    )
    def test_builtin_jarvis_weather_with_prebuffer(self):
        api_key = _resolve_openai_api_key()
        wav_path = RECORDINGS_DIR / "builtin" / "wake_jarvis_weather.wav"
        transcript = transcribe_recording(wav_path, api_key, use_prebuffer=True)
        transcript_lower = transcript.lower()
        print(f"WITH prebuffer (builtin): '{transcript}'")
        assert "jarvis" in transcript_lower, (
            f"Expected 'jarvis' in transcript with prebuffer: '{transcript}'"
        )
        assert "weather" in transcript_lower, (
            f"Expected 'weather' in transcript with prebuffer: '{transcript}'"
        )


@pytest.mark.slow
@requires_openai_key
@requires_vad_model
class TestPrebufferComparison:
    @pytest.mark.skipif(
        not _recording_exists("headset/wake_jarvis_weather.wav"),
        reason="Recording not found",
    )
    def test_prebuffer_preserves_more_words(self):
        api_key = _resolve_openai_api_key()
        wav_path = RECORDINGS_DIR / "headset" / "wake_jarvis_weather.wav"

        transcript_without = transcribe_recording(
            wav_path, api_key, use_prebuffer=False
        )
        transcript_with = transcribe_recording(wav_path, api_key, use_prebuffer=True)

        print(f"WITHOUT prebuffer: '{transcript_without}'")
        print(f"WITH    prebuffer: '{transcript_with}'")

        words_without = len(transcript_without.split())
        words_with = len(transcript_with.split())

        assert words_with >= words_without, (
            f"Prebuffer transcript ({words_with} words) should have at least as many "
            f"words as no-prebuffer ({words_without} words). "
            f"Without: '{transcript_without}' | With: '{transcript_with}'"
        )
