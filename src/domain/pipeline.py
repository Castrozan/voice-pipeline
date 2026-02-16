import asyncio
import collections
import logging

from domain.state import PipelineState, validate_transition
from domain.conversation import ConversationHistory
from domain.wake_word import WakeWordDetector
from domain.speech_detector import SpeechDetector, SpeechEvent
from ports.audio import AudioCapturePort, AudioPlaybackPort
from ports.transcriber import TranscriberPort, TranscriptEvent
from ports.completion import CompletionPort
from ports.synthesizer import SynthesizerPort

logger = logging.getLogger(__name__)

UTTERANCE_FLUSH_TIMEOUT_SECONDS = 3.0
UTTERANCE_FLUSH_MAX_WAIT_SECONDS = 15.0
INCOMPLETE_SENTENCE_GRACE_SECONDS = 2.0


class VoicePipeline:
    def __init__(
        self,
        capture: AudioCapturePort,
        playback: AudioPlaybackPort,
        transcriber: TranscriberPort,
        completion: CompletionPort,
        synthesizer: SynthesizerPort,
        speech_detector: SpeechDetector,
        wake_word_detector: WakeWordDetector,
        conversation: ConversationHistory,
        default_agent: str = "jarvis",
        conversation_window_seconds: float = 15.0,
        barge_in_enabled: bool = True,
        agent_voice_map: dict[str, str] | None = None,
        pre_buffer_ms: int = 300,
        barge_in_min_speech_ms: int = 200,
        frame_duration_ms: int = 16,
        system_prompt: str = "",
    ) -> None:
        self._capture = capture
        self._playback = playback
        self._transcriber = transcriber
        self._completion = completion
        self._synthesizer = synthesizer
        self._speech_detector = speech_detector
        self._wake_word_detector = wake_word_detector
        self._conversation = conversation

        self._agent = default_agent
        self._system_prompt = system_prompt
        self._conversation_window_seconds = conversation_window_seconds
        self._barge_in_enabled = barge_in_enabled
        self._agent_voice_map = agent_voice_map or {}
        self._pre_buffer_frames = int(pre_buffer_ms / frame_duration_ms)
        self._barge_in_window_size = int(barge_in_min_speech_ms / frame_duration_ms)
        self._barge_in_speech_ratio_threshold = 0.7

        self._state = PipelineState.AMBIENT
        self._enabled = True
        self._running = False
        self._stt_session_active = False
        self._stt_generation = 0
        self._processed_generation = -1
        self._waiting_for_final_after_speech_end = False
        self._flush_deadline: float = 0.0
        self._utterance_buffer: list[str] = []
        self._has_pending_interim: bool = False
        self._spoken_text_buffer: str = ""
        self._last_interim_text: str = ""
        self._speech_ended_event = asyncio.Event()
        self._conversation_window_task: asyncio.Task | None = None
        self._utterance_flush_task: asyncio.Task | None = None
        self._current_tasks: list[asyncio.Task] = []
        self._barge_in_window: collections.deque[bool] = collections.deque(
            maxlen=self._barge_in_window_size,
        )

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def agent(self) -> str:
        return self._agent

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _transition_to(self, target: PipelineState) -> None:
        validate_transition(self._state, target)
        logger.info("State: %s -> %s", self._state.name, target.name)
        self._state = target

    def toggle(self) -> bool:
        self._enabled = not self._enabled
        logger.info("Pipeline %s", "enabled" if self._enabled else "disabled")
        if not self._enabled:
            self._reset_to_ambient()
        return self._enabled

    def switch_agent(self, agent: str) -> None:
        self._agent = agent
        self._conversation.clear()
        logger.info("Switched to agent: %s", agent)

    def _get_voice(self) -> str:
        return self._agent_voice_map.get(self._agent, "onyx")

    async def run(self) -> None:
        self._running = True
        logger.info("Voice pipeline started (agent=%s)", self._agent)

        await self._capture.start()
        await self._playback.start()

        try:
            audio_task = asyncio.create_task(self._audio_loop())
            transcript_task = asyncio.create_task(self._transcript_loop())
            self._current_tasks = [audio_task, transcript_task]
            await asyncio.gather(audio_task, transcript_task)
        except asyncio.CancelledError:
            logger.info("Pipeline cancelled")
        finally:
            self._running = False
            await self._cleanup()

    async def stop(self) -> None:
        self._running = False
        for task in self._current_tasks:
            task.cancel()
        await self._cleanup()

    async def _cleanup(self) -> None:
        await self._transcriber.close_session()
        await self._capture.stop()
        await self._playback.stop()

    async def _close_stt_session(self) -> None:
        self._processed_generation = self._stt_generation
        if self._stt_session_active:
            await self._transcriber.close_session()
            self._stt_session_active = False

    async def _audio_loop(self) -> None:
        pre_buffer: collections.deque[bytes] = collections.deque(
            maxlen=self._pre_buffer_frames
        )

        async for frame in self._capture.read_frames():
            if not self._running or not self._enabled:
                await self._close_stt_session()
                pre_buffer.clear()
                continue

            event = self._speech_detector.process_frame(frame)

            if event == SpeechEvent.SPEECH_END:
                self._speech_ended_event.set()

            if event.is_speech:
                self._speech_ended_event.clear()

                if (
                    event == SpeechEvent.SPEECH_START
                    and self._state == PipelineState.LISTENING
                    and self._waiting_for_final_after_speech_end
                ):
                    logger.info("Flush: speech restarted while waiting, rescheduling")
                    self._waiting_for_final_after_speech_end = False
                    self._schedule_utterance_flush()

                if self._state == PipelineState.SPEAKING and self._barge_in_enabled:
                    self._barge_in_window.append(True)
                    if (
                        len(self._barge_in_window) >= self._barge_in_window_size
                        and sum(self._barge_in_window) / len(self._barge_in_window)
                        >= self._barge_in_speech_ratio_threshold
                    ):
                        await self._handle_barge_in()
                        self._barge_in_window.clear()

                if not self._stt_session_active and self._state in (
                    PipelineState.AMBIENT,
                    PipelineState.LISTENING,
                    PipelineState.CONVERSING,
                ):
                    self._stt_generation += 1
                    await self._transcriber.start_session()
                    self._stt_session_active = True
                    for buffered_frame in pre_buffer:
                        await self._transcriber.send_audio(buffered_frame)
                    pre_buffer.clear()

                if self._stt_session_active:
                    await self._transcriber.send_audio(frame)
                else:
                    pre_buffer.append(frame)
            else:
                if self._state == PipelineState.SPEAKING and self._barge_in_enabled:
                    self._barge_in_window.append(False)
                if self._stt_session_active:
                    await self._transcriber.send_audio(frame)
                else:
                    pre_buffer.append(frame)

    async def _transcript_loop(self) -> None:
        async for event in self._transcriber.get_transcripts():
            if not self._running or not self._enabled:
                continue
            await self._handle_transcript(event)

    async def _handle_transcript(self, event: TranscriptEvent) -> None:
        if self._stt_generation <= self._processed_generation:
            return

        if not event.text.strip():
            return

        if event.is_final:
            logger.info("Transcript: %s", event.text)
            self._last_interim_text = ""
        elif event.text != self._last_interim_text:
            logger.debug("Transcript (interim): %s", event.text)
            self._last_interim_text = event.text
        else:
            return

        if self._state == PipelineState.AMBIENT:
            detected_word = self._wake_word_detector.detect(event.text)
            if detected_word:
                logger.info(
                    "Wake word detected: '%s' in '%s'", detected_word, event.text
                )
                if detected_word != self._agent:
                    self._agent = detected_word
                    self._conversation.clear()
                    logger.info("Agent switched to: %s", self._agent)
                post_wake_text = self._wake_word_detector.extract_post_wake_word_text(
                    event.text
                )
                self._transition_to(PipelineState.LISTENING)
                self._utterance_buffer = [post_wake_text] if post_wake_text else []
                self._has_pending_interim = bool(post_wake_text) and not event.is_final
                self._schedule_utterance_flush()

        elif self._state == PipelineState.LISTENING:
            if event.is_final:
                if self._has_pending_interim:
                    self._utterance_buffer[-1] = event.text
                    self._has_pending_interim = False
                else:
                    self._utterance_buffer.append(event.text)

                if self._waiting_for_final_after_speech_end:
                    full_text = " ".join(self._utterance_buffer).strip()
                    if _transcript_ends_with_sentence_punctuation(full_text):
                        logger.info(
                            "Flush: final with sentence punctuation, processing now"
                        )
                        self._waiting_for_final_after_speech_end = False
                        self._cancel_utterance_flush()
                        await self._process_utterance()
                    else:
                        logger.info(
                            "Flush: final without punctuation '%s', grace %.1fs",
                            full_text[-40:],
                            INCOMPLETE_SENTENCE_GRACE_SECONDS,
                        )
                        self._flush_deadline = (
                            asyncio.get_event_loop().time()
                            + INCOMPLETE_SENTENCE_GRACE_SECONDS
                        )
                else:
                    self._schedule_utterance_flush()
            else:
                if self._has_pending_interim:
                    self._utterance_buffer[-1] = event.text
                else:
                    self._utterance_buffer.append(event.text)
                    self._has_pending_interim = True
                if self._waiting_for_final_after_speech_end:
                    logger.debug(
                        "Flush: interim extended deadline by %.1fs",
                        UTTERANCE_FLUSH_TIMEOUT_SECONDS,
                    )
                    self._flush_deadline = (
                        asyncio.get_event_loop().time()
                        + UTTERANCE_FLUSH_TIMEOUT_SECONDS
                    )

        elif self._state == PipelineState.CONVERSING:
            self._cancel_conversation_window()
            self._transition_to(PipelineState.LISTENING)
            self._utterance_buffer = [event.text]
            self._has_pending_interim = not event.is_final
            self._schedule_utterance_flush()

    async def _process_utterance(self) -> None:
        self._cancel_utterance_flush()
        full_text = " ".join(self._utterance_buffer).strip()
        self._utterance_buffer.clear()

        if not full_text:
            self._reset_to_ambient()
            return

        logger.info("Utterance: %s", full_text)
        self._transition_to(PipelineState.THINKING)
        await self._close_stt_session()
        self._conversation.add_user_message(full_text)

        try:
            api_messages = self._conversation.to_api_messages(
                system_prefix=self._system_prompt
            )
            response_chunks: list[str] = []

            self._transition_to(PipelineState.SPEAKING)
            self._spoken_text_buffer = ""

            sentence_buffer = ""
            async for chunk in self._completion.stream(api_messages, self._agent):
                response_chunks.append(chunk)
                sentence_buffer += chunk

                sentence_end_index = _find_sentence_boundary(sentence_buffer)
                if sentence_end_index > 0:
                    sentence_to_speak = sentence_buffer[:sentence_end_index].strip()
                    sentence_buffer = sentence_buffer[sentence_end_index:]

                    if sentence_to_speak:
                        await self._speak_sentence(sentence_to_speak)

                if self._state != PipelineState.SPEAKING:
                    break

            if sentence_buffer.strip() and self._state == PipelineState.SPEAKING:
                await self._speak_sentence(sentence_buffer.strip())

            await self._playback.drain()
            full_response = "".join(response_chunks)
            self._conversation.add_assistant_message(full_response)
            logger.info("Response complete (%d chars)", len(full_response))

            if self._state == PipelineState.SPEAKING:
                self._transition_to(PipelineState.CONVERSING)
                self._start_conversation_window()

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error during completion/synthesis")
            self._reset_to_ambient()

    async def _speak_sentence(self, text: str) -> None:
        voice = self._get_voice()
        async for audio_chunk in self._synthesizer.synthesize(text, voice):
            await self._playback.play_chunk(audio_chunk)
            if self._state != PipelineState.SPEAKING:
                break
        self._spoken_text_buffer += " " + text

    async def _handle_barge_in(self) -> None:
        logger.info("Barge-in detected")
        self._cancel_utterance_flush()
        await self._completion.cancel()
        await self._synthesizer.cancel()
        await self._playback.cancel()

        self._conversation.truncate_last_assistant_message(
            self._spoken_text_buffer.strip()
        )

        self._transition_to(PipelineState.LISTENING)
        self._utterance_buffer.clear()
        self._has_pending_interim = False
        self._schedule_utterance_flush()

    def _schedule_utterance_flush(self) -> None:
        if self._utterance_flush_task and not self._utterance_flush_task.done():
            self._utterance_flush_task.cancel()
        self._utterance_flush_task = asyncio.create_task(
            self._wait_for_speech_end_then_flush()
        )

    def _cancel_utterance_flush(self) -> None:
        if self._utterance_flush_task and not self._utterance_flush_task.done():
            if self._utterance_flush_task is not asyncio.current_task():
                self._utterance_flush_task.cancel()
            self._utterance_flush_task = None

    async def _wait_for_speech_end_then_flush(self) -> None:
        try:
            await self._speech_ended_event.wait()
            self._speech_ended_event.clear()
            if self._state != PipelineState.LISTENING:
                return

            full_text = " ".join(self._utterance_buffer).strip()
            if self._utterance_buffer and not self._has_pending_interim:
                if _transcript_ends_with_sentence_punctuation(full_text):
                    logger.info(
                        "Flush: complete sentence on speech end, processing now"
                    )
                    await self._process_utterance()
                    return
                logger.info(
                    "Flush: incomplete sentence on speech end '%s', waiting for more",
                    full_text[-40:],
                )

            self._waiting_for_final_after_speech_end = True
            loop = asyncio.get_event_loop()
            self._flush_deadline = loop.time() + UTTERANCE_FLUSH_TIMEOUT_SECONDS
            max_deadline = loop.time() + UTTERANCE_FLUSH_MAX_WAIT_SECONDS
            while loop.time() < self._flush_deadline and loop.time() < max_deadline:
                await asyncio.sleep(0.1)
                if self._state != PipelineState.LISTENING:
                    return
                if not self._waiting_for_final_after_speech_end:
                    return
            if self._state == PipelineState.LISTENING and self._utterance_buffer:
                self._waiting_for_final_after_speech_end = False
                logger.info("Flush: deadline expired, processing what we have")
                await self._process_utterance()
        except asyncio.CancelledError:
            pass

    def _start_conversation_window(self) -> None:
        self._cancel_conversation_window()
        self._conversation_window_task = asyncio.create_task(
            self._conversation_window_timer()
        )

    def _cancel_conversation_window(self) -> None:
        if self._conversation_window_task and not self._conversation_window_task.done():
            self._conversation_window_task.cancel()
            self._conversation_window_task = None

    async def _conversation_window_timer(self) -> None:
        try:
            await asyncio.sleep(self._conversation_window_seconds)
            if self._state == PipelineState.CONVERSING:
                logger.info("Conversation window expired")
                self._conversation.clear()
                self._transition_to(PipelineState.AMBIENT)
        except asyncio.CancelledError:
            pass

    def _reset_to_ambient(self) -> None:
        self._state = PipelineState.AMBIENT
        self._utterance_buffer.clear()
        self._has_pending_interim = False
        self._last_interim_text = ""
        self._waiting_for_final_after_speech_end = False
        self._flush_deadline = 0.0
        self._speech_detector.reset()
        self._speech_ended_event.clear()
        self._cancel_utterance_flush()
        self._cancel_conversation_window()


def _transcript_ends_with_sentence_punctuation(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in ".!?"


def _find_sentence_boundary(text: str) -> int:
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ".!?\n" and i < len(text) - 1:
            return i + 1
    return 0
