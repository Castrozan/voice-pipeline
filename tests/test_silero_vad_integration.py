import urllib.request
import wave
from pathlib import Path

import numpy as np
import pytest

from adapters.silero_vad import SileroVad, DEFAULT_MODEL_PATH
from tests.conftest import SAMPLE_RATE, FRAME_DURATION_MS

SILERO_SPEECH_SAMPLE_URL = "https://models.silero.ai/vad_models/en.wav"
SPEECH_SAMPLE_CACHE = Path.home() / ".cache" / "voice-pipeline" / "test_speech_en.wav"
REQUIRED_FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)


def _download_speech_sample() -> Path:
    if SPEECH_SAMPLE_CACHE.exists():
        return SPEECH_SAMPLE_CACHE
    SPEECH_SAMPLE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SILERO_SPEECH_SAMPLE_URL, str(SPEECH_SAMPLE_CACHE))
    return SPEECH_SAMPLE_CACHE


def _load_speech_pcm(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE
        assert wf.getnchannels() == 1
        pcm = wf.readframes(wf.getnframes())
    return np.frombuffer(pcm, dtype=np.int16)


@pytest.fixture(scope="module")
def silero_vad():
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("Silero VAD model not cached, run pipeline once to download")
    return SileroVad(sample_rate=SAMPLE_RATE)


@pytest.fixture(scope="module")
def speech_pcm():
    try:
        path = _download_speech_sample()
    except Exception as exc:
        pytest.skip(f"Cannot download speech sample: {exc}")
    return _load_speech_pcm(path)


class TestSileroVadWithRealSpeech:
    def test_frame_size_must_be_256_samples(self):
        assert REQUIRED_FRAME_SIZE == 256

    def test_detects_speech_in_real_audio(self, silero_vad, speech_pcm):
        num_frames = len(speech_pcm) // REQUIRED_FRAME_SIZE
        assert num_frames > 100

        speech_detected_count = 0
        max_probability = 0.0

        for i in range(num_frames):
            frame = speech_pcm[i * REQUIRED_FRAME_SIZE : (i + 1) * REQUIRED_FRAME_SIZE]
            probability = silero_vad.process_frame(frame.tobytes())
            max_probability = max(max_probability, probability)
            if probability >= 0.5:
                speech_detected_count += 1

        assert max_probability > 0.3, (
            f"VAD max probability {max_probability:.4f} is too low for real speech. "
            f"This likely means the frame size ({REQUIRED_FRAME_SIZE} samples / "
            f"{FRAME_DURATION_MS}ms) is wrong for Silero VAD v5."
        )
        assert speech_detected_count > 0, (
            f"VAD detected 0 speech frames in {num_frames} frames of known speech audio"
        )

    def test_silence_produces_low_probabilities(self, silero_vad):
        silero_vad.reset()
        silence_frame = np.zeros(REQUIRED_FRAME_SIZE, dtype=np.int16).tobytes()
        max_probability = 0.0
        for _ in range(50):
            probability = silero_vad.process_frame(silence_frame)
            max_probability = max(max_probability, probability)

        assert max_probability < 0.1, (
            f"VAD gave {max_probability:.4f} for silence, expected < 0.1"
        )

    def test_reset_clears_state(self, silero_vad, speech_pcm):
        for i in range(20):
            frame = speech_pcm[i * REQUIRED_FRAME_SIZE : (i + 1) * REQUIRED_FRAME_SIZE]
            silero_vad.process_frame(frame.tobytes())

        silero_vad.reset()

        silence = np.zeros(REQUIRED_FRAME_SIZE, dtype=np.int16).tobytes()
        probability = silero_vad.process_frame(silence)
        assert probability < 0.1


class TestSpeechDetectorWithRealVad:
    def test_detects_speech_events(self, silero_vad, speech_pcm):
        from domain.speech_detector import SpeechDetector, SpeechEvent

        silero_vad.reset()
        detector = SpeechDetector(
            vad=silero_vad,
            threshold=0.5,
            min_silence_ms=300,
            frame_duration_ms=FRAME_DURATION_MS,
        )

        num_frames = len(speech_pcm) // REQUIRED_FRAME_SIZE
        events = []
        for i in range(min(num_frames, 200)):
            frame = speech_pcm[i * REQUIRED_FRAME_SIZE : (i + 1) * REQUIRED_FRAME_SIZE]
            event = detector.process_frame(frame.tobytes())
            events.append(event)

        speech_starts = [e for e in events if e == SpeechEvent.SPEECH_START]
        speech_continues = [e for e in events if e == SpeechEvent.SPEECH_CONTINUE]

        assert len(speech_starts) > 0, (
            "SpeechDetector never detected SPEECH_START in real audio"
        )
        assert len(speech_continues) > 0, (
            "SpeechDetector never detected SPEECH_CONTINUE in real audio"
        )
