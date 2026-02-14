import logging
import struct
from enum import Enum, auto

from ports.vad import VadPort

logger = logging.getLogger(__name__)


class SpeechEvent(Enum):
    SILENCE = auto()
    SPEECH_START = auto()
    SPEECH_CONTINUE = auto()
    SPEECH_END = auto()

    @property
    def is_speech(self) -> bool:
        return self in (SpeechEvent.SPEECH_START, SpeechEvent.SPEECH_CONTINUE)


class SpeechDetector:
    def __init__(
        self,
        vad: VadPort,
        threshold: float,
        min_silence_ms: int,
        frame_duration_ms: int,
    ) -> None:
        self._vad = vad
        self._threshold = threshold
        self._silence_frames_required = int(min_silence_ms / frame_duration_ms)
        self._speech_active = False
        self._silence_frame_count = 0
        self._frame_count = 0
        self._log_interval_silence = int(500 / frame_duration_ms)
        self._log_interval_speech = int(200 / frame_duration_ms)

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    def process_frame(self, frame: bytes) -> SpeechEvent:
        self._frame_count += 1
        probability = self._vad.process_frame(frame)
        is_speech = probability >= self._threshold

        log_interval = self._log_interval_speech if self._speech_active else self._log_interval_silence
        if self._frame_count % log_interval == 0:
            amplitude = 0
            if len(frame) >= 2:
                samples = struct.unpack(f"<{len(frame) // 2}h", frame)
                amplitude = max(abs(s) for s in samples)
            logger.debug(
                "VAD prob=%.4f threshold=%.2f speech=%s amp=%d",
                probability, self._threshold, is_speech, amplitude,
            )

        if is_speech:
            self._silence_frame_count = 0
            if not self._speech_active:
                self._speech_active = True
                logger.info("Speech detected (prob=%.2f)", probability)
                return SpeechEvent.SPEECH_START
            return SpeechEvent.SPEECH_CONTINUE

        self._silence_frame_count += 1
        if self._speech_active and self._silence_frame_count >= self._silence_frames_required:
            self._speech_active = False
            logger.info("Speech ended (silence frames=%d)", self._silence_frame_count)
            return SpeechEvent.SPEECH_END

        return SpeechEvent.SILENCE

    def reset(self) -> None:
        self._speech_active = False
        self._silence_frame_count = 0
        self._frame_count = 0
        self._vad.reset()
