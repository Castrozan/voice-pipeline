#!/usr/bin/env python3
import os
import sys
import time
import wave
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

CLIPS = {
    "silence_3s": ("Stay quiet for 3 seconds", 3),
    "wake_jarvis_weather": ("Say: 'Jarvis, what's the weather like today?'", 5),
    "silence_2s": ("Stay quiet for 2 seconds", 2),
    "wake_robson_command": ("Say: 'Robson, turn off the lights.'", 5),
    "thinking_pause": ("Say: 'Jarvis... I was wondering...' (pause 2s) '...about the weather.'", 8),
    "whisper": ("Whisper: 'Jarvis, can you hear me?'", 5),
    "loud_speech": ("Say loudly: 'JARVIS! WHAT TIME IS IT?'", 5),
    "background_music": ("Play some music/video nearby and say: 'Jarvis, hello.'", 8),
    "continuous_ramble": ("Talk continuously for 10 seconds about anything", 12),
}


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

    for i in range(0, len(pcm_data), bytes_per_frame):
        frame = pcm_data[i : i + bytes_per_frame]
        if len(frame) < bytes_per_frame:
            break
        total_frames += 1
        event = detector.process_frame(frame)
        prob = vad.process_frame(frame)

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


def record_one(clip_name: str, config: VoicePipelineConfig) -> None:
    if clip_name not in CLIPS:
        print(f"Unknown clip: {clip_name}")
        print(f"Available: {', '.join(CLIPS.keys())}")
        sys.exit(1)

    instruction, duration = CLIPS[clip_name]
    RECORDINGS_DIR.mkdir(exist_ok=True)
    os.environ["PIPEWIRE_NODE"] = config.capture_device

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

    os.environ.pop("PIPEWIRE_NODE", None)

    analysis = analyze_recording(pcm, config)
    print(f"  Saved: {wav_path}")
    print(f"  rms={analysis['rms']:.0f} peak={analysis['peak']} "
          f"speech={analysis['speech_pct']:.0f}% "
          f"max_prob={analysis['max_prob']:.4f} "
          f"starts={analysis['speech_starts']} ends={analysis['speech_ends']}")


def record_all(config: VoicePipelineConfig) -> None:
    RECORDINGS_DIR.mkdir(exist_ok=True)
    os.environ["PIPEWIRE_NODE"] = config.capture_device

    print("=" * 60)
    print("VOICE PIPELINE - AUDIO RECORDING TEST")
    print("=" * 60)
    print(f"Device: {config.capture_device} | Gain: {config.capture_gain}x")
    print(f"Recordings: {RECORDINGS_DIR}")
    print("=" * 60)
    print()

    for clip_name, (instruction, duration) in CLIPS.items():
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
        print(f"  rms={analysis['rms']:.0f} peak={analysis['peak']} "
              f"speech={analysis['speech_pct']:.0f}% max_prob={analysis['max_prob']:.4f}")
        print()

    os.environ.pop("PIPEWIRE_NODE", None)
    print("Done. Run: uv run pytest tests/test_recorded_audio.py -v")


def transcribe_clip(clip_name: str) -> None:
    wav_path = RECORDINGS_DIR / f"{clip_name}.wav"
    if not wav_path.exists():
        print(f"Recording not found: {wav_path}")
        sys.exit(1)

    import asyncio
    from openai import AsyncOpenAI

    async def _transcribe() -> str:
        openai_api_key_file = os.environ.get(
            "VOICE_PIPELINE_OPENAI_API_KEY_FILE", ""
        )
        if openai_api_key_file and Path(openai_api_key_file).exists():
            api_key = Path(openai_api_key_file).read_text().strip()
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "")

        if not api_key:
            print("No OpenAI API key found")
            sys.exit(1)

        client = AsyncOpenAI(api_key=api_key)
        with open(wav_path, "rb") as f:
            result = await client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            )
        return result.text

    text = asyncio.run(_transcribe())
    print(f"[{clip_name}] Whisper: {text}")


def list_clips() -> None:
    print("Available clips:")
    for clip_name, (instruction, duration) in CLIPS.items():
        exists = (RECORDINGS_DIR / f"{clip_name}.wav").exists()
        status = "recorded" if exists else "missing"
        print(f"  [{status:8s}] {clip_name} ({duration}s) — {instruction}")


def main() -> None:
    config = VoicePipelineConfig()

    if len(sys.argv) < 2 or sys.argv[1] == "--help":
        print("Usage:")
        print(f"  {sys.argv[0]} <clip_name>    Record a single clip")
        print(f"  {sys.argv[0]} all            Record all clips")
        print(f"  {sys.argv[0]} list           List clips and status")
        print(f"  {sys.argv[0]} transcribe <n> Transcribe a recorded clip")
        print()
        list_clips()
        sys.exit(0)

    if sys.argv[1] == "list":
        list_clips()
    elif sys.argv[1] == "all":
        record_all(config)
    elif sys.argv[1] == "transcribe":
        if len(sys.argv) < 3:
            print("Usage: transcribe <clip_name>")
            sys.exit(1)
        transcribe_clip(sys.argv[2])
    else:
        record_one(sys.argv[1], config)


if __name__ == "__main__":
    from cli import _load_env_file
    _load_env_file()
    main()
