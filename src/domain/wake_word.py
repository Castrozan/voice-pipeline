import re


class WakeWordDetector:
    def __init__(
        self,
        wake_words: list[str],
        phonetic_alternatives: dict[str, list[str]] | None = None,
    ) -> None:
        self._wake_words = [w.lower() for w in wake_words]
        self._alternative_to_canonical: dict[str, str] = {}
        alternatives = phonetic_alternatives or {}
        all_patterns: list[str] = []

        for word in self._wake_words:
            escaped_word = re.escape(word)
            all_patterns.append(escaped_word)
            self._alternative_to_canonical[word] = word

            for alt in alternatives.get(word, []):
                escaped_alt = re.escape(alt.lower())
                all_patterns.append(escaped_alt)
                self._alternative_to_canonical[alt.lower()] = word

        self._pattern = re.compile(
            r"\b(" + "|".join(all_patterns) + r")\b",
            re.IGNORECASE,
        )

    @property
    def wake_words(self) -> list[str]:
        return list(self._wake_words)

    def detect(self, transcript: str) -> str | None:
        match = self._pattern.search(transcript)
        if match:
            matched_text = match.group(1).lower()
            return self._alternative_to_canonical.get(matched_text, matched_text)
        return None

    def extract_post_wake_word_text(self, transcript: str) -> str:
        match = self._pattern.search(transcript)
        if not match:
            return transcript
        return transcript[match.end() :].strip()
