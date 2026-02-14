#!/usr/bin/env python3
import os
import sys
import time
import wave
import struct
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import VoicePipelineConfig
from adapters.silero_vad import SileroVad
from domain.speech_detector import SpeechDetector

RECORDINGS_DIR = Path(__file__).parent / "recordings"
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 16
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

SCRIPT = [
    ("silence_3s", "Stay quiet for 3 seconds", 3),
    ("wake_jarvis_weather", "Say: 'Jarvis, what's the weather like today?'", 5),
    ("silence_2s", "Stay quiet for 2 seconds", 2),
    ("wake_robson_command", "Say: 'Robson, turn off the lights.'", 5),
    ("silence_2s_2", "Stay quiet for 2 seconds", 2),
    ("thinking_pause", "Say: 'Jarvis... I was wondering...' (pause 2s) '...about the weather.'", 8),
    ("whisper", "Whisper: 'Jarvis, can you hear me?'", 5),
    ("loud_speech", "Say loudly: 'JARVIS! WHAT TIME IS IT?'", 5),
    ("background_music", "Play some music/video nearby and say: 'Jarvis, hello.'", 8),
    ("continuous_ramble", "Talk continuously for 10 seconds about anything", 12),
]


def save_wav(path: Path, pcm_data: bytes, sample_rate: int = SAMPLE_RATE) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


def record_clip(duration_seconds: float, gain: float) -> bytes:
    frames_needed = int(duration_seconds * SAMPLE_RATE)
    audio = sd.rec(frames_needed, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    amplified = np.clip(audio[:, 0] * gain, -1.0, 1.0)
    return (amplified * 32767).astype(np.int16).tobytes()


def analyze_recording(pcm_data: bytes, config: VoicePipelineConfig) -> dict:
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    peak = int(np.max(np.abs(samples)))

    vad = SileroVad(model_path=config.vad_model_path, sample_rate=SAMPLE_RATE)
    detector = SpeechDetector(
        vad=vad,
        threshold=config.vad_threshold,
        min_silence_ms=config.vad_min_silence_ms,
        frame_duration_ms=FRAME_DURATION_MS,
    )

    bytes_per_frame = FRAME_SAMPLES * 2
    speech_frames = 0
    total_frames = 0
    max_prob = 0.0
    speech_starts = 0
    speech_ends = 0
    probs = []

    for i in range(0, len(pcm_data), bytes_per_frame):
        frame = pcm_data[i : i + bytes_per_frame]
        if len(frame) < bytes_per_frame:
            break
        total_frames += 1
        event = detector.process_frame(frame)
        prob = vad.process_frame(frame)
        probs.append(prob)

        if event.is_speech:
            speech_frames += 1
        if event.name == "SPEECH_START":
            speech_starts += 1
        if event.name == "SPEECH_END":
            speech_ends += 1
        max_prob = max(max_prob, prob)

    return {
        "rms": rms,
        "peak": peak,
        "total_frames": total_frames,
        "speech_frames": speech_frames,
        "speech_pct": speech_frames / max(total_frames, 1) * 100,
        "max_prob": max_prob,
        "speech_starts": speech_starts,
        "speech_ends": speech_ends,
        "duration_s": len(samples) / SAMPLE_RATE,
    }


def main() -> None:
    config = VoicePipelineConfig()

    RECORDINGS_DIR.mkdir(exist_ok=True)

    os.environ["PIPEWIRE_NODE"] = config.capture_device

    print("=" * 60)
    print("VOICE PIPELINE - AUDIO RECORDING TEST")
    print("=" * 60)
    print(f"Device: {config.capture_device}")
    print(f"Gain: {config.capture_gain}x")
    print(f"VAD threshold: {config.vad_threshold}")
    print(f"Frame: {FRAME_DURATION_MS}ms ({FRAME_SAMPLES} samples)")
    print(f"Recordings saved to: {RECORDINGS_DIR}")
    print("=" * 60)
    print()

    results = []

    for clip_name, instruction, duration in SCRIPT:
        print(f"[{clip_name}] {instruction}")
        print(f"  Recording {duration}s in 3...")
        time.sleep(1)
        print(f"  2...")
        time.sleep(1)
        print(f"  1...")
        time.sleep(1)
        print(f"  >>> RECORDING <<<")

        pcm = record_clip(duration, config.capture_gain)

        wav_path = RECORDINGS_DIR / f"{clip_name}.wav"
        save_wav(wav_path, pcm)

        analysis = analyze_recording(pcm, config)
        results.append((clip_name, analysis))

        print(f"  rms={analysis['rms']:.0f} peak={analysis['peak']} "
              f"speech={analysis['speech_pct']:.0f}% "
              f"max_prob={analysis['max_prob']:.4f} "
              f"starts={analysis['speech_starts']} ends={analysis['speech_ends']}")
        print()

    os.environ.pop("PIPEWIRE_NODE", None)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for clip_name, analysis in results:
        status = "OK" if analysis["speech_frames"] > 0 or "silence" in clip_name else "FAIL"
        if "silence" in clip_name and analysis["speech_frames"] > 0:
            status = "FAIL (false positive)"
        print(f"  [{status:20s}] {clip_name}: "
              f"rms={analysis['rms']:.0f} peak={analysis['peak']} "
              f"speech={analysis['speech_pct']:.0f}% max_prob={analysis['max_prob']:.4f}")

    print()
    print(f"Recordings saved in {RECORDINGS_DIR}/")
    print("Re-run VAD analysis without recording: uv run pytest tests/test_recorded_audio.py -v")


if __name__ == "__main__":
    from cli import _load_env_file
    _load_env_file()
    main()
