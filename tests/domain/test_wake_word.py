from domain.wake_word import WakeWordDetector

PHONETIC_ALTERNATIVES = {
    "jarvis": ["jarvus", "jarves", "jervis", "gervis", "jarvas"],
    "robson": ["rabson", "robsen", "robeson", "robs", "robsun", "rabs"],
    "jenny": ["jeni", "jeny", "jenni"],
}


class TestWakeWordDetector:
    def setup_method(self):
        self.detector = WakeWordDetector(
            ["jarvis", "robson"], phonetic_alternatives=PHONETIC_ALTERNATIVES
        )

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
        detector = WakeWordDetector(
            ["jenny"], phonetic_alternatives=PHONETIC_ALTERNATIVES
        )
        assert detector.detect("hey jenni what's up") == "jenny"


class TestWakeWordDetectorWithoutAlternatives:
    def test_no_alternatives_still_detects_exact_word(self):
        detector = WakeWordDetector(["jarvis", "robson"])
        assert detector.detect("hey jarvis") == "jarvis"

    def test_no_alternatives_does_not_match_misspellings(self):
        detector = WakeWordDetector(["jarvis"])
        assert detector.detect("hey jarvus") is None

    def test_no_alternatives_does_not_match_phonetic_variants(self):
        detector = WakeWordDetector(["robson"])
        assert detector.detect("hey rabson") is None


class TestWakeWordDetectorWithCustomAlternatives:
    def test_custom_alternatives_for_new_agent(self):
        detector = WakeWordDetector(
            ["silver"],
            phonetic_alternatives={"silver": ["silva", "sylver", "silber"]},
        )
        assert detector.detect("hey silva") == "silver"
        assert detector.detect("sylver do something") == "silver"
        assert detector.detect("silber help") == "silver"

    def test_custom_alternatives_do_not_affect_other_words(self):
        detector = WakeWordDetector(
            ["jarvis", "silver"],
            phonetic_alternatives={"silver": ["silva"]},
        )
        assert detector.detect("hey silva") == "silver"
        assert detector.detect("jarvus something") is None

    def test_empty_alternatives_dict_works(self):
        detector = WakeWordDetector(["jarvis"], phonetic_alternatives={})
        assert detector.detect("hey jarvis") == "jarvis"

    def test_none_alternatives_works(self):
        detector = WakeWordDetector(["jarvis"], phonetic_alternatives=None)
        assert detector.detect("hey jarvis") == "jarvis"

    def test_alternatives_for_nonexistent_wake_word_are_ignored(self):
        detector = WakeWordDetector(
            ["jarvis"],
            phonetic_alternatives={"robson": ["rabson"]},
        )
        assert detector.detect("hey rabson") is None
        assert detector.detect("hey jarvis") == "jarvis"

    def test_multiple_agents_with_different_alternatives(self):
        detector = WakeWordDetector(
            ["monster", "golden"],
            phonetic_alternatives={
                "monster": ["monstro", "munster"],
                "golden": ["goldie", "goalden"],
            },
        )
        assert detector.detect("hey monstro") == "monster"
        assert detector.detect("goldie help me") == "golden"
        assert detector.detect("munster do it") == "monster"


class TestWakeWordAlternativesConfig:
    def test_config_parses_alternatives_json(self):
        import os

        from config import VoicePipelineConfig

        for key in list(os.environ):
            if key.startswith("VOICE_PIPELINE_"):
                os.environ.pop(key)

        config = VoicePipelineConfig(
            wake_word_alternatives={
                "jarvis": ["jarvus", "jervis"],
                "robson": ["rabson"],
            },
        )
        assert config.wake_word_alternatives == {
            "jarvis": ["jarvus", "jervis"],
            "robson": ["rabson"],
        }

    def test_config_defaults_to_empty_alternatives(self):
        import os

        from config import VoicePipelineConfig

        for key in list(os.environ):
            if key.startswith("VOICE_PIPELINE_"):
                os.environ.pop(key)

        config = VoicePipelineConfig()
        assert config.wake_word_alternatives == {}
