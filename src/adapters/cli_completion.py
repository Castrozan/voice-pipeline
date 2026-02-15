import asyncio
import logging
import os
from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

STDOUT_READ_CHUNK_SIZE = 4096


class CliCompletion:
    def __init__(self, command: str) -> None:
        self._command = command
        self._cancel_event = asyncio.Event()
        self._process: asyncio.subprocess.Process | None = None

    async def stream(
        self,
        messages: list[dict[str, str]],
        agent: str,
    ) -> AsyncIterator[str]:
        self._cancel_event.clear()

        prompt_text = _format_messages_as_plain_text(messages)
        environment_without_claudecode = _build_environment_without_claudecode()

        try:
            self._process = await asyncio.create_subprocess_shell(
                self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=environment_without_claudecode,
            )

            self._process.stdin.write(prompt_text.encode())
            self._process.stdin.close()

            while True:
                if self._cancel_event.is_set():
                    break

                chunk = await self._process.stdout.read(STDOUT_READ_CHUNK_SIZE)
                if not chunk:
                    break

                yield chunk.decode(errors="replace")

            await self._process.wait()

            if self._process.returncode and self._process.returncode != 0 and not self._cancel_event.is_set():
                stderr_output = await self._process.stderr.read()
                logger.error(
                    "CLI command exited with code %d: %s",
                    self._process.returncode,
                    stderr_output.decode(errors="replace").strip(),
                )
        except Exception:
            if not self._cancel_event.is_set():
                logger.exception("CLI completion error")
        finally:
            self._process = None

    async def cancel(self) -> None:
        self._cancel_event.set()
        if self._process:
            try:
                self._process.kill()
            except ProcessLookupError:
                pass


def _format_messages_as_plain_text(messages: list[dict[str, str]]) -> str:
    formatted_lines = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        label = role.capitalize()
        formatted_lines.append(f"{label}: {content}")
    return "\n\n".join(formatted_lines)


def _build_environment_without_claudecode() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("CLAUDECODE", None)
    return environment
