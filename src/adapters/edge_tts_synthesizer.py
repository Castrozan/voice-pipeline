import asyncio
import logging
from collections.abc import AsyncIterator

import miniaudio
from edge_tts import Communicate

logger = logging.getLogger(__name__)

PLAYBACK_SAMPLE_RATE = 24000
PLAYBACK_CHANNELS = 1
PCM_CHUNK_SIZE = 4096


class EdgeTtsSynthesizer:
    def __init__(self) -> None:
        self._cancelled = False

    async def synthesize(
        self, text: str, voice: str = "en-US-GuyNeural"
    ) -> AsyncIterator[bytes]:
        self._cancelled = False

        try:
            mp3_chunks: list[bytes] = []
            communicate = Communicate(text, voice=voice)
            async for chunk in communicate.stream():
                if self._cancelled:
                    return
                if chunk["type"] == "audio" and chunk["data"]:
                    mp3_chunks.append(chunk["data"])

            if self._cancelled or not mp3_chunks:
                return

            mp3_data = b"".join(mp3_chunks)
            decoded = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: miniaudio.decode(
                    mp3_data,
                    output_format=miniaudio.SampleFormat.SIGNED16,
                    nchannels=PLAYBACK_CHANNELS,
                    sample_rate=PLAYBACK_SAMPLE_RATE,
                ),
            )
            pcm_data = decoded.samples.tobytes()

            for offset in range(0, len(pcm_data), PCM_CHUNK_SIZE):
                if self._cancelled:
                    return
                yield pcm_data[offset : offset + PCM_CHUNK_SIZE]

        except Exception:
            if not self._cancelled:
                logger.exception("Edge TTS synthesis failed for: %s", text[:50])

    async def cancel(self) -> None:
        self._cancelled = True
