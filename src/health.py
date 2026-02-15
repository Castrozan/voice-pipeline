import logging
import struct
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from config import VoicePipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    name: str
    passed: bool
    detail: str


def run_startup_checks(config: VoicePipelineConfig) -> list[HealthCheckResult]:
    results = [
        _check_audio_device(config),
        _check_pipewire_source(config),
        _check_vad_model(config),
        _check_vad_with_audio(config),
        _check_api_keys(config),
        _check_gateway_reachable(config),
    ]

    passed = sum(1 for r in results if r.passed)
    failed = [r for r in results if not r.passed]

    logger.info("Health check: %d/%d passed", passed, len(results))
    for result in results:
        level = logging.INFO if result.passed else logging.WARNING
        symbol = "OK" if result.passed else "FAIL"
        logger.log(level, "  [%s] %s: %s", symbol, result.name, result.detail)

    return results


def has_critical_failures(results: list[HealthCheckResult]) -> bool:
    critical_checks = {"audio_device", "vad_model", "api_keys"}
    return any(not r.passed and r.name in critical_checks for r in results)


def _check_audio_device(config: VoicePipelineConfig) -> HealthCheckResult:
    name = "audio_device"
    try:
        devices = sd.query_devices()
        device_found = False
        device_name = config.capture_device

        for dev in devices:
            if device_name.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
                device_found = True
                break

        if not device_found:
            try:
                default = sd.query_devices(kind="input")
                return HealthCheckResult(
                    name=name,
                    passed=True,
                    detail=f"'{device_name}' not in PortAudio (will use PIPEWIRE_NODE), default input: {default['name']}",
                )
            except sd.PortAudioError:
                return HealthCheckResult(name=name, passed=False, detail="No input devices available")

        return HealthCheckResult(name=name, passed=True, detail=f"Device '{device_name}' found")
    except Exception as exc:
        return HealthCheckResult(name=name, passed=False, detail=str(exc))


def _check_pipewire_source(config: VoicePipelineConfig) -> HealthCheckResult:
    name = "pipewire_source"
    try:
        import subprocess
        result = subprocess.run(
            ["pw-cli", "info", config.capture_device],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0:
            return HealthCheckResult(name=name, passed=False, detail=f"'{config.capture_device}' not found in PipeWire")

        output = result.stdout
        props = {}
        for line in output.splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("*"):
                key, _, val = line.partition("=")
                props[key.strip()] = val.strip().strip('"')

        node_desc = props.get("node.description", config.capture_device)
        media_class = props.get("media.class", "unknown")

        link_result = subprocess.run(
            ["pw-link", "-io"],
            capture_output=True, text=True, timeout=3,
        )
        hw_source = "unknown"
        if link_result.returncode == 0:
            lines = link_result.stdout.splitlines()
            for i, line in enumerate(lines):
                if "echo-cancel-capture" in line:
                    for j in range(max(0, i - 5), i):
                        candidate = lines[j].strip()
                        if candidate and not candidate.startswith("|") and ":" not in candidate:
                            hw_source = candidate
                            break
                    break

        return HealthCheckResult(
            name=name,
            passed=True,
            detail=f"{node_desc} ({media_class}), hw source: {hw_source}",
        )
    except FileNotFoundError:
        return HealthCheckResult(name=name, passed=True, detail="pw-cli not available, skipping PipeWire check")
    except Exception as exc:
        return HealthCheckResult(name=name, passed=False, detail=str(exc))


def _check_vad_model(config: VoicePipelineConfig) -> HealthCheckResult:
    name = "vad_model"
    try:
        from adapters.silero_vad import SileroVad
        vad = SileroVad(model_path=config.vad_model_path, sample_rate=config.sample_rate)

        frame_samples = int(config.sample_rate * config.frame_duration_ms / 1000)
        silence = np.zeros(frame_samples, dtype=np.int16).tobytes()
        prob = vad.process_frame(silence)

        if prob > 0.3:
            return HealthCheckResult(name=name, passed=False, detail=f"Silence gave prob={prob:.4f}, model may be broken")

        return HealthCheckResult(name=name, passed=True, detail=f"Loaded, silence prob={prob:.4f}")
    except Exception as exc:
        return HealthCheckResult(name=name, passed=False, detail=str(exc))


def _check_vad_with_audio(config: VoicePipelineConfig) -> HealthCheckResult:
    name = "vad_live_audio"
    try:
        frame_samples = int(config.sample_rate * config.frame_duration_ms / 1000)
        captured_frames: list[np.ndarray] = []
        frames_needed = int(0.5 * config.sample_rate / frame_samples)

        import threading
        done = threading.Event()

        def callback(indata, frames, time_info, status):
            captured_frames.append(indata[:, 0].copy())
            if len(captured_frames) >= frames_needed:
                done.set()

        device = None
        for i, dev in enumerate(sd.query_devices()):
            if config.capture_device.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
                device = i
                break

        import os
        if device is None:
            os.environ["PIPEWIRE_NODE"] = config.capture_device

        try:
            stream = sd.InputStream(
                device=device,
                samplerate=config.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=frame_samples,
                callback=callback,
            )
            stream.start()
            done.wait(timeout=2.0)
            stream.stop()
            stream.close()
        finally:
            if device is None:
                os.environ.pop("PIPEWIRE_NODE", None)

        if not captured_frames:
            return HealthCheckResult(name=name, passed=False, detail="No audio frames captured")

        all_audio = np.concatenate(captured_frames)
        pcm = (all_audio * config.capture_gain * 32767).astype(np.int16)
        rms = np.sqrt(np.mean(pcm.astype(np.float64) ** 2))
        peak = int(np.max(np.abs(pcm)))

        if rms < 1.0:
            return HealthCheckResult(name=name, passed=False, detail=f"Audio silent (rms={rms:.0f}, peak={peak}), mic may be muted")

        return HealthCheckResult(
            name=name,
            passed=True,
            detail=f"Capturing audio (rms={rms:.0f}, peak={peak}, gain={config.capture_gain}x)",
        )
    except Exception as exc:
        return HealthCheckResult(name=name, passed=False, detail=str(exc))


def _check_api_keys(config: VoicePipelineConfig) -> HealthCheckResult:
    name = "api_keys"
    missing = []

    if config.stt_engine == "deepgram":
        key = config.read_secret(config.deepgram_api_key_file)
        if not key:
            missing.append(f"deepgram ({config.deepgram_api_key_file or 'not configured'})")

    openai_key = config.read_secret(config.openai_api_key_file)
    if not openai_key:
        missing.append(f"openai ({config.openai_api_key_file or 'not configured'})")

    if config.completion_engine == "anthropic":
        anthropic_key = config.read_secret(config.anthropic_api_key_file)
        if not anthropic_key:
            missing.append(f"anthropic ({config.anthropic_api_key_file or 'not configured'})")
    else:
        gateway_token = config.read_secret(config.gateway_token_file)
        if not gateway_token:
            missing.append(f"gateway ({config.gateway_token_file or 'not configured'})")

    if missing:
        return HealthCheckResult(name=name, passed=False, detail=f"Missing: {', '.join(missing)}")

    return HealthCheckResult(name=name, passed=True, detail="All API keys loaded")


def _check_gateway_reachable(config: VoicePipelineConfig) -> HealthCheckResult:
    name = "gateway"
    if config.completion_engine != "openclaw":
        return HealthCheckResult(name=name, passed=True, detail=f"Skipped (engine={config.completion_engine})")
    try:
        import urllib.request
        req = urllib.request.Request(f"{config.gateway_url}/health", method="GET")
        req.add_header("User-Agent", "voice-pipeline/healthcheck")
        response = urllib.request.urlopen(req, timeout=3)
        return HealthCheckResult(name=name, passed=True, detail=f"Reachable ({response.status})")
    except urllib.error.URLError as exc:
        reason = str(exc.reason) if hasattr(exc, "reason") else str(exc)
        return HealthCheckResult(name=name, passed=False, detail=f"Unreachable: {reason}")
    except Exception as exc:
        return HealthCheckResult(name=name, passed=False, detail=f"Unreachable: {exc}")
