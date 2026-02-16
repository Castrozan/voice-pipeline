import logging

from config import VoicePipelineConfig
from adapters.sounddevice_audio import SounddeviceCapture, SounddevicePlayback
from adapters.silero_vad import SileroVad
from ports.completion import CompletionPort
from ports.synthesizer import SynthesizerPort
from adapters.openai_tts import OpenAITtsSynthesizer
from adapters.unix_control import UnixSocketControlServer
from domain.conversation import ConversationHistory
from domain.wake_word import WakeWordDetector
from domain.speech_detector import SpeechDetector
from domain.pipeline import VoicePipeline
from ports.transcriber import TranscriberPort

logger = logging.getLogger(__name__)


def create_capture(config: VoicePipelineConfig) -> SounddeviceCapture:
    return SounddeviceCapture(
        device=config.capture_device,
        sample_rate=config.sample_rate,
        frame_duration_ms=config.frame_duration_ms,
        gain=config.capture_gain,
    )


def create_transcriber(
    config: VoicePipelineConfig, deepgram_api_key: str, openai_api_key: str
) -> TranscriberPort:
    if config.stt_engine == "deepgram":
        from adapters.deepgram_stt import DeepgramStreamingTranscriber

        return DeepgramStreamingTranscriber(
            api_key=deepgram_api_key,
            sample_rate=config.sample_rate,
        )
    from adapters.openai_whisper_stt import OpenAIWhisperTranscriber

    return OpenAIWhisperTranscriber(
        api_key=openai_api_key,
        sample_rate=config.sample_rate,
    )


def create_completion(config: VoicePipelineConfig) -> CompletionPort:
    if config.completion_engine == "cli":
        from adapters.cli_completion import CliCompletion

        return CliCompletion(command=config.completion_cli_command)

    if config.completion_engine == "anthropic":
        from adapters.anthropic_llm import AnthropicCompletion

        anthropic_api_key = config.read_secret(config.anthropic_api_key_file)
        return AnthropicCompletion(api_key=anthropic_api_key, model=config.model)

    from adapters.openclaw_llm import OpenClawCompletion

    gateway_token = config.read_secret(config.gateway_token_file)
    return OpenClawCompletion(
        gateway_url=config.gateway_url,
        token=gateway_token,
        model=config.model,
    )


def create_synthesizer(
    config: VoicePipelineConfig, openai_api_key: str
) -> SynthesizerPort:
    if config.tts_engine == "edge-tts":
        from adapters.edge_tts_synthesizer import EdgeTtsSynthesizer

        return EdgeTtsSynthesizer()
    return OpenAITtsSynthesizer(api_key=openai_api_key)


def create_speech_detector(
    config: VoicePipelineConfig, vad: SileroVad
) -> SpeechDetector:
    return SpeechDetector(
        vad=vad,
        threshold=config.vad_threshold,
        min_silence_ms=config.vad_min_silence_ms,
        frame_duration_ms=config.frame_duration_ms,
    )


def create_pipeline(
    config: VoicePipelineConfig,
) -> tuple[VoicePipeline, UnixSocketControlServer]:
    openai_api_key = config.read_secret(config.openai_api_key_file)
    deepgram_api_key = config.read_secret(config.deepgram_api_key_file)

    capture = create_capture(config)
    playback = SounddevicePlayback(sample_rate=24000)
    vad = SileroVad(model_path=config.vad_model_path, sample_rate=config.sample_rate)
    transcriber = create_transcriber(config, deepgram_api_key, openai_api_key)
    completion = create_completion(config)
    synthesizer = create_synthesizer(config, openai_api_key)
    speech_detector = create_speech_detector(config, vad)
    control = UnixSocketControlServer(socket_path=config.socket_path)

    wake_detector = WakeWordDetector(config.wake_words)
    conversation = ConversationHistory(max_turns=config.max_history_turns)

    pipeline = VoicePipeline(
        capture=capture,
        playback=playback,
        transcriber=transcriber,
        completion=completion,
        synthesizer=synthesizer,
        speech_detector=speech_detector,
        wake_word_detector=wake_detector,
        conversation=conversation,
        default_agent=config.default_agent,
        conversation_window_seconds=config.conversation_window_seconds,
        barge_in_enabled=config.barge_in_enabled,
        agent_voice_map=config.agent_voices,
        barge_in_min_speech_ms=config.barge_in_min_speech_ms,
        frame_duration_ms=config.frame_duration_ms,
        system_prompt=config.system_prompt,
    )

    return pipeline, control
