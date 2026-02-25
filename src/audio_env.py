import logging
import re
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AudioDevice:
    node_id: int
    name: str
    device_type: str


@dataclass
class AudioNode:
    node_id: int
    name: str
    is_default: bool
    volume: float | None
    linked_ports: list[str] = field(default_factory=list)


@dataclass
class AudioStream:
    node_id: int
    name: str
    ports: list["StreamPort"] = field(default_factory=list)


@dataclass
class StreamPort:
    port_id: int
    port_name: str
    direction: str
    linked_node: str


@dataclass
class EchoCancelTopology:
    capture_physical_source: str
    playback_physical_sink: str
    source_node: str
    sink_node: str


@dataclass
class AudioEnvironment:
    pipewire_version: str
    devices: list[AudioDevice]
    sinks: list[AudioNode]
    sources: list[AudioNode]
    streams: list[AudioStream]
    echo_cancel: EchoCancelTopology | None
    default_configured_sink: str
    default_configured_source: str


def discover() -> AudioEnvironment | None:
    wpctl_output = _run_wpctl_status()
    if wpctl_output is None:
        return None
    return _parse_wpctl_status(wpctl_output)


def format_environment_report(environment: AudioEnvironment) -> str:
    lines = []
    lines.append(f"PipeWire {environment.pipewire_version}")
    lines.append("")

    lines.append("Devices:")
    for device in environment.devices:
        lines.append(f"  {device.node_id}. {device.name} [{device.device_type}]")
    lines.append("")

    lines.append("Sinks:")
    for sink in environment.sinks:
        default_marker = " *" if sink.is_default else ""
        volume_info = f" [vol: {sink.volume:.2f}]" if sink.volume is not None else ""
        lines.append(f"  {sink.node_id}. {sink.name}{volume_info}{default_marker}")
    lines.append("")

    lines.append("Sources:")
    for source in environment.sources:
        default_marker = " *" if source.is_default else ""
        volume_info = (
            f" [vol: {source.volume:.2f}]" if source.volume is not None else ""
        )
        lines.append(f"  {source.node_id}. {source.name}{volume_info}{default_marker}")
    lines.append("")

    lines.append("Streams:")
    for stream in environment.streams:
        lines.append(f"  {stream.node_id}. {stream.name}")
        for port in stream.ports:
            direction_symbol = "<" if port.direction == "input" else ">"
            lines.append(f"    {port.port_name} {direction_symbol} {port.linked_node}")
    lines.append("")

    if environment.echo_cancel:
        echo = environment.echo_cancel
        lines.append("Echo Cancel Topology:")
        lines.append(f"  Physical mic    -> {echo.capture_physical_source}")
        lines.append(f"  Echo-cancel src -> {echo.source_node}")
        lines.append(f"  Echo-cancel snk -> {echo.sink_node}")
        lines.append(f"  Playback sink   -> {echo.playback_physical_sink}")
    else:
        lines.append("Echo Cancel: not detected")

    lines.append("")
    lines.append("Default Configured Nodes:")
    lines.append(f"  Sink:   {environment.default_configured_sink or 'none'}")
    lines.append(f"  Source: {environment.default_configured_source or 'none'}")

    return "\n".join(lines)


def _run_wpctl_status() -> str | None:
    try:
        result = subprocess.run(
            ["wpctl", "status"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.warning("wpctl status failed: %s", result.stderr.strip())
            return None
        return result.stdout
    except FileNotFoundError:
        logger.warning("wpctl not found in PATH")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("wpctl status timed out")
        return None


def _parse_wpctl_status(output: str) -> AudioEnvironment:
    pipewire_version = _extract_pipewire_version(output)
    lines = output.splitlines()

    audio_section_lines = _extract_section(lines, "Audio")
    settings_section_lines = _extract_section(lines, "Settings")

    devices = _parse_devices(audio_section_lines)
    sinks = _parse_nodes(audio_section_lines, "Sinks:")
    sources = _parse_nodes(audio_section_lines, "Sources:")
    filter_sources, filter_sinks = _parse_filter_nodes(audio_section_lines)
    sources.extend(filter_sources)
    sinks.extend(filter_sinks)
    streams = _parse_streams(audio_section_lines)

    echo_cancel = _detect_echo_cancel_topology(streams)

    default_configured_sink, default_configured_source = (
        _parse_default_configured_nodes(settings_section_lines)
    )

    return AudioEnvironment(
        pipewire_version=pipewire_version,
        devices=devices,
        sinks=sinks,
        sources=sources,
        streams=streams,
        echo_cancel=echo_cancel,
        default_configured_sink=default_configured_sink,
        default_configured_source=default_configured_source,
    )


def _extract_pipewire_version(output: str) -> str:
    first_line = output.splitlines()[0] if output.splitlines() else ""
    version_match = re.search(r"\[([^,\]]+)", first_line)
    return version_match.group(1) if version_match else "unknown"


def _extract_section(lines: list[str], section_name: str) -> list[str]:
    section_lines = []
    inside_section = False

    for line in lines:
        stripped = line.strip()
        if stripped == section_name:
            inside_section = True
            continue

        if inside_section:
            if (
                stripped
                and not stripped.startswith(("├", "│", "└", " "))
                and not line.startswith(" ")
            ):
                break
            section_lines.append(line)

    return section_lines


def _parse_devices(section_lines: list[str]) -> list[AudioDevice]:
    devices = []
    inside_devices = False

    for line in section_lines:
        cleaned = _clean_tree_prefix(line)

        if cleaned.strip() == "Devices:":
            inside_devices = True
            continue

        if inside_devices:
            if cleaned.strip() == "":
                inside_devices = False
                continue
            if cleaned.strip().endswith(":"):
                inside_devices = False
                continue

            device = _parse_device_line(cleaned.strip())
            if device:
                devices.append(device)

    return devices


def _parse_device_line(line: str) -> AudioDevice | None:
    match = re.match(r"(\d+)\.\s+(.+?)\s+\[(\w+)\]", line)
    if not match:
        return None
    return AudioDevice(
        node_id=int(match.group(1)),
        name=match.group(2).strip(),
        device_type=match.group(3),
    )


def _parse_nodes(section_lines: list[str], header: str) -> list[AudioNode]:
    nodes = []
    inside_section = False

    for line in section_lines:
        cleaned = _clean_tree_prefix(line)

        if cleaned.strip() == header:
            inside_section = True
            continue

        if inside_section:
            if cleaned.strip() == "":
                inside_section = False
                continue
            if cleaned.strip().endswith(":"):
                inside_section = False
                continue

            node = _parse_node_line(cleaned.strip())
            if node:
                nodes.append(node)

    return nodes


def _parse_node_line(line: str) -> AudioNode | None:
    is_default = line.startswith("*")
    cleaned = line.lstrip("* ")

    match = re.match(r"(\d+)\.\s+(.+?)(?:\s+\[vol:\s+([\d.]+)\])?$", cleaned)
    if not match:
        return None

    volume = float(match.group(3)) if match.group(3) else None

    return AudioNode(
        node_id=int(match.group(1)),
        name=match.group(2).strip(),
        is_default=is_default,
        volume=volume,
    )


def _parse_filter_nodes(
    section_lines: list[str],
) -> tuple[list[AudioNode], list[AudioNode]]:
    filter_sources: list[AudioNode] = []
    filter_sinks: list[AudioNode] = []
    inside_filters = False

    for line in section_lines:
        cleaned = _clean_tree_prefix(line)
        stripped = cleaned.strip()

        if stripped == "Filters:":
            inside_filters = True
            continue

        if inside_filters:
            if stripped == "":
                continue
            if (
                stripped.endswith(":")
                and not stripped[0].isdigit()
                and "[" not in stripped
            ):
                inside_filters = False
                continue

            node_with_role_match = re.match(
                r"(\d+)\.\s+(.+?)\s+\[(Audio/Source|Audio/Sink)\]", stripped
            )
            if node_with_role_match:
                node = AudioNode(
                    node_id=int(node_with_role_match.group(1)),
                    name=node_with_role_match.group(2).strip(),
                    is_default=False,
                    volume=None,
                )
                if node_with_role_match.group(3) == "Audio/Source":
                    filter_sources.append(node)
                else:
                    filter_sinks.append(node)

    return filter_sources, filter_sinks


def _parse_streams(section_lines: list[str]) -> list[AudioStream]:
    streams = []
    inside_streams = False
    current_stream: AudioStream | None = None
    stream_indent: int | None = None

    for line in section_lines:
        cleaned = _clean_tree_prefix(line)
        stripped = cleaned.strip()

        if stripped == "Streams:":
            inside_streams = True
            continue

        if not inside_streams:
            continue

        if stripped == "":
            continue

        if stripped.endswith(":") and not stripped[0].isdigit():
            break

        number_match = re.match(r"(\d+)\.\s+", stripped)
        if not number_match:
            continue

        line_indent = len(cleaned) - len(cleaned.lstrip())

        linked_port_match = re.match(r"(\d+)\.\s+(\S+)\s+([<>])\s+(.+)", stripped)

        if (
            linked_port_match
            and current_stream is not None
            and stream_indent is not None
            and line_indent > stream_indent
        ):
            direction = "input" if linked_port_match.group(3) == "<" else "output"
            port = StreamPort(
                port_id=int(linked_port_match.group(1)),
                port_name=linked_port_match.group(2),
                direction=direction,
                linked_node=linked_port_match.group(4).strip(),
            )
            current_stream.ports.append(port)
        elif stream_indent is None or line_indent <= stream_indent:
            stream_name_match = re.match(r"(\d+)\.\s+(.+)", stripped)
            if stream_name_match:
                current_stream = AudioStream(
                    node_id=int(stream_name_match.group(1)),
                    name=stream_name_match.group(2).strip(),
                )
                streams.append(current_stream)
                stream_indent = line_indent

    return streams


def _detect_echo_cancel_topology(
    streams: list[AudioStream],
) -> EchoCancelTopology | None:
    capture_source = ""
    playback_sink = ""
    source_node = ""
    sink_node = ""

    for stream in streams:
        if "echo-cancel-capture" in stream.name:
            source_node = "echo-cancel-source"
            for port in stream.ports:
                if port.direction == "input" and port.port_name.startswith("input_"):
                    capture_source = port.linked_node.split(":")[0]
                    break

        if "echo-cancel-playback" in stream.name:
            sink_node = "echo-cancel-sink"
            for port in stream.ports:
                if port.direction == "output" and port.port_name.startswith("output_"):
                    playback_sink = port.linked_node.split(":")[0]
                    break

    if not source_node and not sink_node:
        return None

    return EchoCancelTopology(
        capture_physical_source=capture_source or "unknown",
        playback_physical_sink=playback_sink or "unknown",
        source_node=source_node or "unknown",
        sink_node=sink_node or "unknown",
    )


def _parse_default_configured_nodes(settings_lines: list[str]) -> tuple[str, str]:
    default_sink = ""
    default_source = ""

    for line in settings_lines:
        cleaned = _clean_tree_prefix(line).strip()
        if "Audio/Sink" in cleaned:
            parts = cleaned.split(None, 2)
            if len(parts) >= 3:
                default_sink = parts[2]
        elif "Audio/Source" in cleaned:
            parts = cleaned.split(None, 2)
            if len(parts) >= 3:
                default_source = parts[2]

    return default_sink, default_source


def _clean_tree_prefix(line: str) -> str:
    cleaned = line
    for char in "├│└─":
        cleaned = cleaned.replace(char, " ")
    return cleaned
