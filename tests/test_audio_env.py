import pytest

from audio_env import (
    AudioDevice,
    AudioEnvironment,
    AudioNode,
    AudioStream,
    EchoCancelTopology,
    StreamPort,
    _parse_wpctl_status,
    format_environment_report,
)

SAMPLE_WPCTL_OUTPUT = """\
PipeWire 'pipewire-0' [0.3.48, user@host, cookie:123456]
 └─ Clients:
        32. pipewire                            [0.3.48, user@host, pid:1084]
        35. WirePlumber                         [0.3.48, user@host, pid:1086]

Audio
 ├─ Devices:
 │      51. Built-in Audio                      [alsa]
 │      85. BT Headphones                       [bluez5]
 │
 ├─ Sinks:
 │      38. Echo-Cancel Sink                    [vol: 1.00]
 │      54. Built-in Audio Analog Stereo        [vol: 0.60]
 │  *   86. BT Headphones                       [vol: 0.65]
 │
 ├─ Sink endpoints:
 │
 ├─ Sources:
 │  *   37. Echo-Cancel Source                  [vol: 0.73]
 │      55. Built-in Audio Analog Stereo        [vol: 1.00]
 │
 ├─ Source endpoints:
 │
 └─ Streams:
        36. echo-cancel-capture
             64. input_FL        < HDA Intel PCH:capture_FL
             65. monitor_FL
             66. input_FR        < HDA Intel PCH:capture_FR
             67. monitor_FR
        39. echo-cancel-playback
             62. output_FL       > BT Headphones:playback_FL
             63. output_FR       > BT Headphones:playback_FR

Video
 ├─ Devices:
 │      49. Webcam                              [v4l2]
 │
 ├─ Sinks:
 │
 ├─ Sink endpoints:
 │
 ├─ Sources:
 │  *   52. Webcam
 │
 ├─ Source endpoints:
 │
 └─ Streams:

Settings
 └─ Default Configured Node Names:
         0. Audio/Sink    bluez_output.XX_XX.a2dp-sink
         1. Audio/Source  echo-cancel-source"""


class TestParsePipewireVersion:
    def test_extracts_version_from_header(self):
        environment = _parse_wpctl_status(SAMPLE_WPCTL_OUTPUT)
        assert environment.pipewire_version == "0.3.48"


class TestParseDevices:
    def test_finds_all_audio_devices(self):
        environment = _parse_wpctl_status(SAMPLE_WPCTL_OUTPUT)
        assert len(environment.devices) == 2

    def test_parses_alsa_device(self):
        environment = _parse_wpctl_status(SAMPLE_WPCTL_OUTPUT)
        alsa_device = next(d for d in environment.devices if d.device_type == "alsa")
        assert alsa_device.node_id == 51
        assert alsa_device.name == "Built-in Audio"

    def test_parses_bluetooth_device(self):
        environment = _parse_wpctl_status(SAMPLE_WPCTL_OUTPUT)
        bt_device = next(d for d in environment.devices if d.device_type == "bluez5")
        assert bt_device.node_id == 85
        assert bt_device.name == "BT Headphones"


class TestParseSinks:
    def test_finds_all_sinks(self):
        environment = _parse_wpctl_status(SAMPLE_WPCTL_OUTPUT)
        assert len(environment.sinks) == 3

    def test_identifies_default_sink(self):
        environment = _parse_wpctl_status(SAMPLE_WPCTL_OUTPUT)
        default_sinks = [s for s in environment.sinks if s.is_default]
        assert len(default_sinks) == 1
        assert default_sinks[0].name == "BT Headphones"

    def test_parses_volume(self):
        environment = _parse_wpctl_status(SAMPLE_WPCTL_OUTPUT)
        echo_sink = next(s for s in environment.sinks if "Echo-Cancel" in s.name)
        assert echo_sink.volume == 1.00


class TestParseSources:
    def test_finds_all_sources(self):
        environment = _parse_wpctl_status(SAMPLE_WPCTL_OUTPUT)
        assert len(environment.sources) == 2

    def test_identifies_default_source(self):
        environment = _parse_wpctl_status(SAMPLE_WPCTL_OUTPUT)
        default_sources = [s for s in environment.sources if s.is_default]
        assert len(default_sources) == 1
        assert default_sources[0].name == "Echo-Cancel Source"


class TestParseStreams:
    def test_finds_echo_cancel_streams(self):
        environment = _parse_wpctl_status(SAMPLE_WPCTL_OUTPUT)
        stream_names = [s.name for s in environment.streams]
        assert "echo-cancel-capture" in stream_names
        assert "echo-cancel-playback" in stream_names

    def test_capture_stream_has_input_ports(self):
        environment = _parse_wpctl_status(SAMPLE_WPCTL_OUTPUT)
        capture = next(s for s in environment.streams if "capture" in s.name)
        input_ports = [p for p in capture.ports if p.direction == "input"]
        assert len(input_ports) == 2
        assert all("HDA Intel PCH" in p.linked_node for p in input_ports)

    def test_playback_stream_has_output_ports(self):
        environment = _parse_wpctl_status(SAMPLE_WPCTL_OUTPUT)
        playback = next(s for s in environment.streams if "playback" in s.name)
        output_ports = [p for p in playback.ports if p.direction == "output"]
        assert len(output_ports) == 2
        assert all("BT Headphones" in p.linked_node for p in output_ports)


class TestEchoCancelTopology:
    def test_detects_echo_cancel(self):
        environment = _parse_wpctl_status(SAMPLE_WPCTL_OUTPUT)
        assert environment.echo_cancel is not None

    def test_identifies_physical_capture_source(self):
        environment = _parse_wpctl_status(SAMPLE_WPCTL_OUTPUT)
        assert environment.echo_cancel.capture_physical_source == "HDA Intel PCH"

    def test_identifies_physical_playback_sink(self):
        environment = _parse_wpctl_status(SAMPLE_WPCTL_OUTPUT)
        assert environment.echo_cancel.playback_physical_sink == "BT Headphones"

    def test_identifies_echo_cancel_nodes(self):
        environment = _parse_wpctl_status(SAMPLE_WPCTL_OUTPUT)
        assert environment.echo_cancel.source_node == "echo-cancel-source"
        assert environment.echo_cancel.sink_node == "echo-cancel-sink"


class TestDefaultConfiguredNodes:
    def test_parses_default_sink(self):
        environment = _parse_wpctl_status(SAMPLE_WPCTL_OUTPUT)
        assert environment.default_configured_sink == "bluez_output.XX_XX.a2dp-sink"

    def test_parses_default_source(self):
        environment = _parse_wpctl_status(SAMPLE_WPCTL_OUTPUT)
        assert environment.default_configured_source == "echo-cancel-source"


class TestNoEchoCancel:
    MINIMAL_WPCTL_OUTPUT = """\
PipeWire 'pipewire-0' [1.0.0, user@host, cookie:999]

Audio
 ├─ Devices:
 │      10. Built-in Audio                      [alsa]
 │
 ├─ Sinks:
 │  *   20. Built-in Audio Analog Stereo        [vol: 0.50]
 │
 ├─ Sink endpoints:
 │
 ├─ Sources:
 │  *   30. Built-in Audio Analog Stereo        [vol: 1.00]
 │
 ├─ Source endpoints:
 │
 └─ Streams:

Video
 └─ Streams:

Settings
 └─ Default Configured Node Names:"""

    def test_no_echo_cancel_when_not_present(self):
        environment = _parse_wpctl_status(self.MINIMAL_WPCTL_OUTPUT)
        assert environment.echo_cancel is None


class TestFormatEnvironmentReport:
    def test_report_contains_key_sections(self):
        environment = _parse_wpctl_status(SAMPLE_WPCTL_OUTPUT)
        report = format_environment_report(environment)
        assert "PipeWire 0.3.48" in report
        assert "Devices:" in report
        assert "Sinks:" in report
        assert "Sources:" in report
        assert "Streams:" in report
        assert "Echo Cancel Topology:" in report
        assert "Default Configured Nodes:" in report

    def test_report_shows_echo_cancel_routing(self):
        environment = _parse_wpctl_status(SAMPLE_WPCTL_OUTPUT)
        report = format_environment_report(environment)
        assert "HDA Intel PCH" in report
        assert "BT Headphones" in report
