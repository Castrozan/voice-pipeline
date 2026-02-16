from domain.wake_word import WakeWordDetector


class TestWakeWordDetector:
    def setup_method(self):
        self.detector = WakeWordDetector(["jarvis", "robson"])

    def test_detects_wake_word_at_start(self):
        assert self.detector.detect("jarvis what time is it") == "jarvis"

    def test_detects_wake_word_in_middle(self):
        assert self.detector.detect("hey jarvis please") == "jarvis"

    def test_detects_second_wake_word(self):
        assert self.detector.detect("hello robson") == "robson"

    def test_case_insensitive(self):
        assert self.detector.detect("JARVIS help me") == "jarvis"

    def test_no_wake_word(self):
        assert self.detector.detect("hello world") is None

    def test_empty_string(self):
        assert self.detector.detect("") is None

    def test_extract_post_wake_word_text(self):
        result = self.detector.extract_post_wake_word_text("jarvis what time is it")
        assert result == "what time is it"

    def test_extract_post_wake_word_no_trailing(self):
        result = self.detector.extract_post_wake_word_text("jarvis")
        assert result == ""

    def test_extract_post_wake_word_no_match(self):
        result = self.detector.extract_post_wake_word_text("hello world")
        assert result == "hello world"

    def test_wake_words_property(self):
        assert self.detector.wake_words == ["jarvis", "robson"]

    def test_partial_word_not_detected(self):
        assert self.detector.detect("jarvislike behavior") is None

    def test_phonetic_alternative_rabson_detects_as_robson(self):
        assert self.detector.detect("Hey, Rabson. Can you hear me?") == "robson"

    def test_phonetic_alternative_jarvus_detects_as_jarvis(self):
        assert self.detector.detect("jarvus what time is it") == "jarvis"

    def test_phonetic_alternative_jervis_detects_as_jarvis(self):
        assert self.detector.detect("hey jervis please") == "jarvis"

    def test_phonetic_alternative_robsen_detects_as_robson(self):
        assert self.detector.detect("hello robsen") == "robson"

    def test_phonetic_alternative_rabs_detects_as_robson(self):
        assert self.detector.detect("rabs turn off the lights") == "robson"

    def test_phonetic_alternative_extracts_post_wake_text(self):
        result = self.detector.extract_post_wake_word_text("Rabson, can you hear me?")
        assert result == ", can you hear me?"

    def test_phonetic_alternative_case_insensitive(self):
        assert self.detector.detect("RABSON what is up") == "robson"

    def test_unknown_word_without_alternatives_still_works(self):
        detector = WakeWordDetector(["customword"])
        assert detector.detect("hey customword do something") == "customword"

    def test_jenny_alternative_jenni_detects(self):
        detector = WakeWordDetector(["jenny"])
        assert detector.detect("hey jenni what's up") == "jenny"
