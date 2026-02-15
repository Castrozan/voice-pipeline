import logging
import shutil
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
        _check_pipewire_audio_environment(config),
        _check_vad_model(config),
        _check_vad_with_audio(config),
        _check_api_keys(config),
        _check_completion_engine_reachable(config),
    ]

    passed = sum(1 for r in results if r.passed)

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
        device_name = config.capture_device

        for dev in devices:
            if device_name.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
                return HealthCheckResult(name=name, passed=True, detail=f"Device '{device_name}' found")

        try:
            default = sd.query_devices(kind="input")
            return HealthCheckResult(
                name=name,
                passed=True,
                detail=f"'{device_name}' not in PortAudio (will use PIPEWIRE_NODE), default input: {default['name']}",
            )
        except sd.PortAudioError:
            return HealthCheckResult(name=name, passed=False, detail="No input devices available")
    except Exception as exc:
        return HealthCheckResult(name=name, passed=False, detail=str(exc))


def _check_pipewire_audio_environment(config: VoicePipelineConfig) -> HealthCheckResult:
    name = "pipewire_source"
    try:
        from audio_env import discover

        environment = discover()
        if environment is None:
            return HealthCheckResult(name=name, passed=True, detail="wpctl not available, skipping PipeWire check")

        capture_device_name = config.capture_device
        source_found = any(
            capture_device_name.lower() in source.name.lower()
            for source in environment.sources
        )

        if not source_found:
            return HealthCheckResult(
                name=name,
                passed=False,
                detail=f"'{capture_device_name}' not found in PipeWire sources",
            )

        if environment.echo_cancel:
            echo = environment.echo_cancel
            return HealthCheckResult(
                name=name,
                passed=True,
                detail=f"capture={echo.capture_physical_source}, playback={echo.playback_physical_sink}",
            )

        return HealthCheckResult(
            name=name,
            passed=True,
            detail=f"Source '{capture_device_name}' found (no echo-cancel detected)",
        )
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
        import os
        anthropic_key = config.read_secret(config.anthropic_api_key_file) or os.environ.get("ANTHROPIC_API_KEY", "")
        if not anthropic_key:
            missing.append("anthropic (no key file and ANTHROPIC_API_KEY not set)")
    elif config.completion_engine == "cli":
        cli_executable = config.completion_cli_command.split()[0]
        if not shutil.which(cli_executable):
            missing.append(f"cli command '{cli_executable}' not found in PATH")
    else:
        gateway_token = config.read_secret(config.gateway_token_file)
        if not gateway_token:
            missing.append(f"gateway ({config.gateway_token_file or 'not configured'})")

    if missing:
        return HealthCheckResult(name=name, passed=False, detail=f"Missing: {', '.join(missing)}")

    return HealthCheckResult(name=name, passed=True, detail="All API keys loaded")


def _check_completion_engine_reachable(config: VoicePipelineConfig) -> HealthCheckResult:
    name = "completion_engine"

    if config.completion_engine == "cli":
        cli_executable = config.completion_cli_command.split()[0]
        path = shutil.which(cli_executable)
        if path:
            return HealthCheckResult(name=name, passed=True, detail=f"CLI command found: {path}")
        return HealthCheckResult(name=name, passed=False, detail=f"CLI command '{cli_executable}' not found in PATH")

    if config.completion_engine != "openclaw":
        return HealthCheckResult(name=name, passed=True, detail=f"Skipped (engine={config.completion_engine})")

    try:
        import urllib.request
        req = urllib.request.Request(f"{config.gateway_url}/health", method="GET")
        req.add_header("User-Agent", "voice-pipeline/healthcheck")
        response = urllib.request.urlopen(req, timeout=3)
        return HealthCheckResult(name=name, passed=True, detail=f"Gateway reachable ({response.status})")
    except urllib.error.URLError as exc:
        reason = str(exc.reason) if hasattr(exc, "reason") else str(exc)
        return HealthCheckResult(name=name, passed=False, detail=f"Gateway unreachable: {reason}")
    except Exception as exc:
        return HealthCheckResult(name=name, passed=False, detail=f"Gateway unreachable: {exc}")
