import wave
from pathlib import Path

import numpy as np
import pytest

from adapters.silero_vad import SileroVad
from domain.speech_detector import SpeechDetector, SpeechEvent
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 16
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

RECORDINGS_DIR = Path(__file__).parent / "recordings"

VAD_THRESHOLD = 0.5
MIN_SILENCE_MS = 800


def find_recordings(clip_name: str) -> list[Path]:
    return sorted(RECORDINGS_DIR.glob(f"{clip_name}_*.wav"))


def load_recording(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        return wf.readframes(wf.getnframes())


def run_vad_on_pcm(pcm_data: bytes) -> dict:
    vad = SileroVad(sample_rate=SAMPLE_RATE)
    detector = SpeechDetector(
        vad=vad,
        threshold=VAD_THRESHOLD,
        min_silence_ms=MIN_SILENCE_MS,
        frame_duration_ms=FRAME_DURATION_MS,
    )

    bytes_per_frame = FRAME_SIZE * 2
    speech_frames = 0
    total_frames = 0
    max_prob = 0.0
    speech_starts = 0
    speech_ends = 0

    for i in range(0, len(pcm_data), bytes_per_frame):
        frame = pcm_data[i : i + bytes_per_frame]
        if len(frame) < bytes_per_frame:
            break
        total_frames += 1
        event = detector.process_frame(frame)

        prob = vad.process_frame(frame)
        max_prob = max(max_prob, prob)

        if event.is_speech:
            speech_frames += 1
        if event == SpeechEvent.SPEECH_START:
            speech_starts += 1
        if event == SpeechEvent.SPEECH_END:
            speech_ends += 1

    samples = np.frombuffer(pcm_data, dtype=np.int16)
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    peak = int(np.max(np.abs(samples)))

    return {
        "rms": rms,
        "peak": peak,
        "total_frames": total_frames,
        "speech_frames": speech_frames,
        "speech_pct": speech_frames / max(total_frames, 1) * 100,
        "max_prob": max_prob,
        "speech_starts": speech_starts,
        "speech_ends": speech_ends,
    }


def parametrize_recordings(clip_name: str):
    paths = find_recordings(clip_name)
    if not paths:
        return pytest.mark.parametrize(
            "recording_path",
            [pytest.param(clip_name, marks=pytest.mark.skip(reason=f"No recordings for {clip_name}"))],
        )
    return pytest.mark.parametrize(
        "recording_path",
        [pytest.param(p, id=p.stem) for p in paths],
    )


class TestSilenceRecordings:
    @parametrize_recordings("silence_3s")
    def test_silence_has_no_speech(self, recording_path):
        pcm = load_recording(recording_path)
        result = run_vad_on_pcm(pcm)
        assert result["speech_frames"] == 0, f"Expected no speech in silence, got {result['speech_pct']:.0f}%"
        assert result["max_prob"] < 0.3, f"Expected low prob in silence, got {result['max_prob']:.4f}"

    @parametrize_recordings("silence_3s")
    def test_silence_amplitude_is_low(self, recording_path):
        pcm = load_recording(recording_path)
        result = run_vad_on_pcm(pcm)
        assert result["rms"] < 500, f"Expected low RMS in silence, got {result['rms']:.0f}"


class TestWakeWordRecordings:
    @parametrize_recordings("wake_jarvis_weather")
    def test_jarvis_weather_detected(self, recording_path):
        pcm = load_recording(recording_path)
        result = run_vad_on_pcm(pcm)
        assert result["speech_starts"] >= 1, f"Expected speech detection, got {result['speech_starts']} starts"
        assert result["max_prob"] >= 0.5, f"Expected high VAD prob, got {result['max_prob']:.4f}"

    @parametrize_recordings("wake_robson_command")
    def test_robson_command_detected(self, recording_path):
        pcm = load_recording(recording_path)
        result = run_vad_on_pcm(pcm)
        assert result["speech_starts"] >= 1, f"Expected speech detection, got {result['speech_starts']} starts"
        assert result["max_prob"] >= 0.5, f"Expected high VAD prob, got {result['max_prob']:.4f}"


class TestEdgeCaseRecordings:
    @parametrize_recordings("thinking_pause")
    def test_thinking_pause_has_speech(self, recording_path):
        pcm = load_recording(recording_path)
        result = run_vad_on_pcm(pcm)
        assert result["speech_starts"] >= 1, f"Expected speech, got {result['speech_starts']} starts"

    @parametrize_recordings("whisper")
    def test_whisper_detected(self, recording_path):
        pcm = load_recording(recording_path)
        result = run_vad_on_pcm(pcm)
        assert result["max_prob"] > 0.1, f"Whisper not detected at all, max_prob={result['max_prob']:.4f}"

    @parametrize_recordings("loud_speech")
    def test_loud_speech_detected(self, recording_path):
        pcm = load_recording(recording_path)
        result = run_vad_on_pcm(pcm)
        assert result["speech_starts"] >= 1, f"Loud speech not detected"
        assert result["max_prob"] >= 0.5, f"Expected high prob for loud speech, got {result['max_prob']:.4f}"

    @parametrize_recordings("background_music")
    def test_background_music_wake_word(self, recording_path):
        pcm = load_recording(recording_path)
        result = run_vad_on_pcm(pcm)
        assert result["speech_starts"] >= 1, f"Speech with music not detected"

    @parametrize_recordings("background_music_long")
    def test_background_music_long_sentence(self, recording_path):
        pcm = load_recording(recording_path)
        result = run_vad_on_pcm(pcm)
        assert result["speech_starts"] >= 1, f"Long speech with music not detected"
        assert result["speech_pct"] > 1, f"Expected >1% speech, got {result['speech_pct']:.0f}%"

    @parametrize_recordings("background_music_loud")
    def test_background_music_loud(self, recording_path):
        pcm = load_recording(recording_path)
        result = run_vad_on_pcm(pcm)
        assert result["max_prob"] > 0.3, f"Speech with loud music not detected at all, max_prob={result['max_prob']:.4f}"

    @parametrize_recordings("continuous_ramble")
    def test_continuous_speech_detected(self, recording_path):
        pcm = load_recording(recording_path)
        result = run_vad_on_pcm(pcm)
        assert result["speech_starts"] >= 1, f"Continuous speech not detected"
        assert result["speech_pct"] > 1, f"Expected >1% speech, got {result['speech_pct']:.0f}%"
