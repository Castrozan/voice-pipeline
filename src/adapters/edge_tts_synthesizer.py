import asyncio
import logging
from collections.abc import AsyncIterator

from edge_tts import Communicate

logger = logging.getLogger(__name__)

FFMPEG_DECODE_COMMAND = [
    "ffmpeg",
    "-i",
    "pipe:0",
    "-f",
    "s16le",
    "-acodec",
    "pcm_s16le",
    "-ar",
    "24000",
    "-ac",
    "1",
    "-loglevel",
    "error",
    "pipe:1",
]


class EdgeTtsSynthesizer:
    def __init__(self) -> None:
        self._cancelled = False
        self._process: asyncio.subprocess.Process | None = None

    async def synthesize(
        self, text: str, voice: str = "en-US-GuyNeural"
    ) -> AsyncIterator[bytes]:
        self._cancelled = False

        try:
            self._process = await asyncio.create_subprocess_exec(
                *FFMPEG_DECODE_COMMAND,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            feed_task = asyncio.create_task(
                self._feed_mp3_to_ffmpeg(text, voice),
            )

            try:
                while not self._cancelled:
                    chunk = await self._process.stdout.read(4096)
                    if not chunk:
                        break
                    yield chunk
            finally:
                feed_task.cancel()
                try:
                    await feed_task
                except asyncio.CancelledError:
                    pass
                if self._process and self._process.returncode is None:
                    self._process.kill()
                    await self._process.wait()
                self._process = None

        except Exception:
            if not self._cancelled:
                logger.exception("Edge TTS synthesis failed for: %s", text[:50])

    async def _feed_mp3_to_ffmpeg(self, text: str, voice: str) -> None:
        try:
            communicate = Communicate(text, voice=voice)
            async for chunk in communicate.stream():
                if self._cancelled:
                    break
                if chunk["type"] == "audio" and chunk["data"]:
                    self._process.stdin.write(chunk["data"])
                    await self._process.stdin.drain()
        except asyncio.CancelledError:
            pass
        except Exception:
            if not self._cancelled:
                logger.exception("Edge TTS stream failed")
        finally:
            try:
                if self._process and self._process.stdin:
                    self._process.stdin.close()
                    await self._process.stdin.wait_closed()
            except Exception:
                pass

    async def cancel(self) -> None:
        self._cancelled = True
        if self._process and self._process.returncode is None:
            self._process.kill()
