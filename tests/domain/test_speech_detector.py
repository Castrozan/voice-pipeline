from domain.speech_detector import SpeechEvent
from tests.conftest import generate_silence, FRAME_DURATION_MS


class TestSpeechDetectorEvents:
    def test_silence_returns_silence_event(self, fake_speech_detector, fake_vad):
        fake_vad.set_probabilities([0.1])
        event = fake_speech_detector.process_frame(generate_silence())
        assert event == SpeechEvent.SILENCE

    def test_speech_start_on_first_speech_frame(self, fake_speech_detector, fake_vad):
        fake_vad.set_probabilities([0.9])
        event = fake_speech_detector.process_frame(generate_silence())
        assert event == SpeechEvent.SPEECH_START

    def test_speech_continue_on_subsequent_speech(self, fake_speech_detector, fake_vad):
        fake_vad.set_probabilities([0.9, 0.8])
        fake_speech_detector.process_frame(generate_silence())
        event = fake_speech_detector.process_frame(generate_silence())
        assert event == SpeechEvent.SPEECH_CONTINUE

    def test_speech_end_after_enough_silence(self, fake_speech_detector, fake_vad):
        silence_frames_needed = int(300 / FRAME_DURATION_MS)
        probabilities = [0.9] + [0.1] * silence_frames_needed
        fake_vad.set_probabilities(probabilities)

        fake_speech_detector.process_frame(generate_silence())
        for _ in range(silence_frames_needed - 1):
            event = fake_speech_detector.process_frame(generate_silence())
            assert event == SpeechEvent.SILENCE

        event = fake_speech_detector.process_frame(generate_silence())
        assert event == SpeechEvent.SPEECH_END

    def test_no_speech_end_before_min_silence(self, fake_speech_detector, fake_vad):
        silence_frames_needed = int(300 / FRAME_DURATION_MS)
        probabilities = [0.9] + [0.1] * (silence_frames_needed - 1)
        fake_vad.set_probabilities(probabilities)

        fake_speech_detector.process_frame(generate_silence())
        for _ in range(silence_frames_needed - 1):
            event = fake_speech_detector.process_frame(generate_silence())
            assert event == SpeechEvent.SILENCE
        assert fake_speech_detector.speech_active


class TestSpeechDetectorState:
    def test_speech_active_after_speech_start(self, fake_speech_detector, fake_vad):
        fake_vad.set_probabilities([0.9])
        fake_speech_detector.process_frame(generate_silence())
        assert fake_speech_detector.speech_active

    def test_speech_inactive_after_speech_end(self, fake_speech_detector, fake_vad):
        silence_frames_needed = int(300 / FRAME_DURATION_MS)
        fake_vad.set_probabilities([0.9] + [0.1] * silence_frames_needed)
        for _ in range(1 + silence_frames_needed):
            fake_speech_detector.process_frame(generate_silence())
        assert not fake_speech_detector.speech_active

    def test_reset_clears_state(self, fake_speech_detector, fake_vad):
        fake_vad.set_probabilities([0.9])
        fake_speech_detector.process_frame(generate_silence())
        assert fake_speech_detector.speech_active

        fake_speech_detector.reset()
        assert not fake_speech_detector.speech_active


class TestSpeechEventProperties:
    def test_speech_start_is_speech(self):
        assert SpeechEvent.SPEECH_START.is_speech

    def test_speech_continue_is_speech(self):
        assert SpeechEvent.SPEECH_CONTINUE.is_speech

    def test_silence_is_not_speech(self):
        assert not SpeechEvent.SILENCE.is_speech

    def test_speech_end_is_not_speech(self):
        assert not SpeechEvent.SPEECH_END.is_speech
