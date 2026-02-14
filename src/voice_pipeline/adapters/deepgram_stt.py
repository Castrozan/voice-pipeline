import asyncio
import logging
from collections.abc import AsyncIterator

from deepgram import AsyncDeepgramClient
from deepgram.extensions.types.sockets.listen_v1_results_event import ListenV1ResultsEvent
from deepgram.listen.v1.socket_client import EventType

from voice_pipeline.ports.transcriber import TranscriptEvent

logger = logging.getLogger(__name__)


class DeepgramStreamingTranscriber:
    def __init__(self, api_key: str, sample_rate: int = 16000) -> None:
        self._api_key = api_key
        self._sample_rate = sample_rate
        self._socket = None
        self._context_manager = None
        self._transcript_queue: asyncio.Queue[TranscriptEvent] = asyncio.Queue()
        self._session_active = False
        self._listener_task: asyncio.Task | None = None

    async def start_session(self) -> None:
        if self._session_active:
            await self.close_session()

        client = AsyncDeepgramClient(api_key=self._api_key)
        self._context_manager = client.listen.v1.connect(
            model="nova-2",
            language="multi",
            encoding="linear16",
            sample_rate=str(self._sample_rate),
            channels="1",
            interim_results="true",
            utterance_end_ms="1500",
            vad_events="true",
            endpointing="300",
            smart_format="true",
        )
        self._socket = await self._context_manager.__aenter__()
        self._socket.on(EventType.MESSAGE, self._on_message)
        self._socket.on(EventType.ERROR, self._on_error)
        self._listener_task = asyncio.create_task(self._socket.start_listening())
        self._session_active = True
        logger.info("Deepgram session started")

    async def send_audio(self, frame: bytes) -> None:
        if self._socket and self._session_active:
            try:
                await self._socket._send(frame)
            except Exception:
                logger.warning("Failed to send audio to Deepgram")

    async def get_transcripts(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            event = await self._transcript_queue.get()
            yield event

    async def close_session(self) -> None:
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except (asyncio.CancelledError, Exception):
                pass
        self._listener_task = None

        if self._context_manager:
            try:
                await self._context_manager.__aexit__(None, None, None)
            except Exception:
                pass
        self._context_manager = None
        self._socket = None
        self._session_active = False
        logger.info("Deepgram session closed")

    async def _on_message(self, message) -> None:
        if not isinstance(message, ListenV1ResultsEvent):
            return
        try:
            transcript = message.channel.alternatives[0].transcript
            if not transcript:
                return

            is_final = message.is_final
            speech_final = message.speech_final

            event = TranscriptEvent(
                text=transcript,
                is_final=is_final or speech_final,
            )
            await self._transcript_queue.put(event)
        except (IndexError, AttributeError):
            pass

    async def _on_error(self, error) -> None:
        logger.error("Deepgram error: %s", error)
