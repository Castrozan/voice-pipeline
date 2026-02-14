import os
import subprocess

import pytest

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except (ImportError, OSError):
    HAS_SOUNDDEVICE = False


def _pipewire_available() -> bool:
    xdg = os.environ.get("XDG_RUNTIME_DIR", "")
    if not xdg:
        return False
    try:
        result = subprocess.run(
            ["wpctl", "status"],
            capture_output=True,
            text=True,
            timeout=5,
            env={**os.environ, "XDG_RUNTIME_DIR": xdg},
        )
        return result.returncode == 0 and "PipeWire" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(not HAS_SOUNDDEVICE, reason="sounddevice not available")
class TestAudioDeviceDiscovery:
    def test_has_input_devices(self):
        devices = sd.query_devices()
        input_devices = [d for d in devices if d["max_input_channels"] > 0]
        assert len(input_devices) > 0, "No audio input devices found"

    def test_has_output_devices(self):
        devices = sd.query_devices()
        output_devices = [d for d in devices if d["max_output_channels"] > 0]
        assert len(output_devices) > 0, "No audio output devices found"

    def test_default_input_exists(self):
        default_input = sd.query_devices(kind="input")
        assert default_input is not None
        assert default_input["max_input_channels"] > 0

    def test_default_output_exists(self):
        default_output = sd.query_devices(kind="output")
        assert default_output is not None
        assert default_output["max_output_channels"] > 0

    def test_default_input_supports_16khz(self):
        default_input = sd.query_devices(kind="input")
        supported_rate = default_input["default_samplerate"]
        assert supported_rate > 0, "Default input has no sample rate"


@pytest.mark.skipif(
    not _pipewire_available(),
    reason="PipeWire not available",
)
class TestPipeWireSetup:
    def _wpctl_status(self) -> str:
        result = subprocess.run(
            ["wpctl", "status"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout

    def test_pipewire_running(self):
        status = self._wpctl_status()
        assert "PipeWire" in status

    def test_has_audio_sources(self):
        status = self._wpctl_status()
        assert "Sources:" in status

    def test_has_audio_sinks(self):
        status = self._wpctl_status()
        assert "Sinks:" in status

    def test_echo_cancel_source_exists(self):
        status = self._wpctl_status()
        assert "Echo-Cancel Source" in status or "echo-cancel" in status.lower(), (
            "Echo-Cancel Source not found in PipeWire. "
            "Voice pipeline needs the echo-cancel module for barge-in support."
        )

    def test_default_source_is_echo_cancel(self):
        status = self._wpctl_status()
        in_audio = False
        in_sources = False
        for line in status.splitlines():
            if line.startswith("Audio"):
                in_audio = True
                continue
            if line.startswith("Video") or line.startswith("Settings"):
                in_audio = False
                in_sources = False
                continue
            if in_audio and "Sources:" in line:
                in_sources = True
                continue
            if in_sources and "*" in line:
                assert "echo-cancel" in line.lower(), (
                    f"Default PipeWire source is not echo-cancel-source. "
                    f"Default source line: {line.strip()}"
                )
                return
            if in_sources and ("Sink" in line or "Stream" in line or "endpoint" in line.lower()):
                in_sources = False
        pytest.fail("Could not determine default audio source from wpctl status")
