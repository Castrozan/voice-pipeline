import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from config import VoicePipelineConfig

ENV_FILE_PATH = Path.home() / ".config" / "voice-pipeline" / "env"


def _load_env_file() -> None:
    if not ENV_FILE_PATH.exists():
        return
    with open(ENV_FILE_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip("'\"")
            if key not in os.environ:
                os.environ[key] = value


def main() -> None:
    _load_env_file()
    parser = argparse.ArgumentParser(description="Real-time voice pipeline")
    parser.add_argument("--agent", help="Default agent to use")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("toggle", help="Toggle pipeline on/off")

    agent_parser = subparsers.add_parser("agent", help="Switch agent")
    agent_parser.add_argument("name", help="Agent name")

    subparsers.add_parser("status", help="Query pipeline status")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.verbose:
        logging.getLogger("websockets").setLevel(logging.INFO)
        logging.getLogger("httpcore").setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.INFO)

    config = VoicePipelineConfig()
    if args.agent:
        config.default_agent = args.agent

    if args.command in ("toggle", "agent", "status"):
        asyncio.run(_run_client_command(args, config))
    else:
        asyncio.run(_run_daemon(config))


async def _run_client_command(args: argparse.Namespace, config: VoicePipelineConfig) -> None:
    from adapters.unix_control import UnixSocketControlClient

    client = UnixSocketControlClient(socket_path=config.socket_path)

    try:
        if args.command == "toggle":
            result = await client.send_command("toggle")
        elif args.command == "agent":
            result = await client.send_command("agent", {"name": args.name})
        elif args.command == "status":
            result = await client.send_command("status")
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            sys.exit(1)

        print(f"{result}")
    except ConnectionRefusedError:
        print("Voice pipeline is not running", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Voice pipeline is not running", file=sys.stderr)
        sys.exit(1)


async def _run_daemon(config: VoicePipelineConfig) -> None:
    from health import run_startup_checks, has_critical_failures
    from factory import create_pipeline

    results = run_startup_checks(config)
    if has_critical_failures(results):
        logging.error("Critical health check failures, aborting startup")
        sys.exit(1)

    pipeline, control = create_pipeline(config)

    shutdown_event = asyncio.Event()
    shutdown_triggered = False

    def handle_signal() -> None:
        nonlocal shutdown_triggered
        if shutdown_triggered:
            logging.warning("Forced exit")
            sys.exit(1)
        shutdown_triggered = True
        logging.info("Shutting down...")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    await control.start()

    async def control_loop() -> None:
        async for cmd in control.commands():
            if cmd.action == "toggle":
                pipeline.toggle()
            elif cmd.action == "agent" and cmd.payload:
                pipeline.switch_agent(cmd.payload.get("name", config.default_agent))
            elif cmd.action == "status":
                logging.info(
                    "Status: state=%s agent=%s enabled=%s",
                    pipeline.state.name,
                    pipeline.agent,
                    pipeline.enabled,
                )

    pipeline_task = asyncio.create_task(pipeline.run())
    control_task = asyncio.create_task(control_loop())

    try:
        await shutdown_event.wait()
    finally:
        pipeline_task.cancel()
        control_task.cancel()
        try:
            await asyncio.wait_for(pipeline_task, timeout=3.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        try:
            await asyncio.wait_for(control_task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        await control.stop()
