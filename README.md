# Voice Pipeline

Real-time conversational AI voice pipeline with wake word detection, streaming STT/TTS, and barge-in support.

## Architecture

```
Mic (PipeWire) -> VAD (Silero ONNX) -> STT (Deepgram Nova-2) -> Wake Word Detection
                                                                        |
Speaker <- TTS (OpenAI tts-1) <- LLM (Claude via OpenClaw Gateway) <---+
    |
    +-> VAD (barge-in) -> interrupt & resume listening
```

### State Machine

```
AMBIENT -> LISTENING -> THINKING -> SPEAKING -> CONVERSING -> AMBIENT
   ^                                   |           |
   +--- timeout (no speech) ----------+           |
                  ^                                |
                  +---- barge-in ------------------+
```

### Hexagonal Architecture

```
src/
  __main__.py          CLI entry point (argparse)
  config.py            Pydantic Settings
  factory.py           Adapter creation, wiring

  domain/              Core logic, no external deps
    pipeline.py        State machine orchestrator
    speech_detector.py VAD + silence tracking
    conversation.py    Multi-turn history
    wake_word.py       Keyword detection on transcript
    state.py           State enum + transitions
    events.py          Domain events

  ports/               Interfaces only
    audio.py           AudioCapturePort, AudioPlaybackPort
    vad.py             VadPort
    transcriber.py     TranscriberPort
    completion.py      CompletionPort
    synthesizer.py     SynthesizerPort
    control.py         ControlPort

  adapters/            Concrete implementations
    sounddevice_audio.py
    silero_vad.py
    deepgram_stt.py
    openai_whisper_stt.py
    openclaw_llm.py
    openai_tts.py
    unix_control.py

tests/
  domain/              Pure unit tests
  adapters/            Integration tests
  conftest.py          Shared fixtures and fakes
```

## Usage

```bash
voice-pipeline                  # Start daemon (always-on)
voice-pipeline --agent jarvis   # Start with specific agent
voice-pipeline -v               # Verbose logging
voice-pipeline toggle           # Toggle on/off via socket
voice-pipeline agent <name>     # Switch agent via socket
voice-pipeline status           # Query state via socket
```

## Configuration

All settings via `VOICE_PIPELINE_` prefixed environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_URL` | `http://localhost:18789` | OpenClaw gateway URL |
| `GATEWAY_TOKEN_FILE` | `/run/agenix/openclaw-gateway-token` | Path to gateway token |
| `DEFAULT_AGENT` | `jarvis` | Default agent name |
| `STT_ENGINE` | `deepgram` | `deepgram` or `openai-whisper` |
| `DEEPGRAM_API_KEY_FILE` | | Path to Deepgram API key |
| `OPENAI_API_KEY_FILE` | | Path to OpenAI API key |
| `TTS_VOICE` | `onyx` | Default TTS voice |
| `VAD_THRESHOLD` | `0.5` | Speech detection threshold |
| `VAD_MIN_SILENCE_MS` | `300` | Silence duration to end speech |
| `FRAME_DURATION_MS` | `32` | Audio frame size (Silero optimal) |
| `WAKE_WORDS` | `["jarvis"]` | JSON list of wake words |
| `CAPTURE_DEVICE` | `echo-cancel-source` | PipeWire capture device |
| `BARGE_IN_ENABLED` | `true` | Allow interrupting AI speech |
| `CONVERSATION_WINDOW_SECONDS` | `15.0` | Follow-up window without wake word |
| `AGENT_VOICES` | `{}` | JSON map of agent to TTS voice |

## Development

```bash
uv sync
uv pip install pytest pytest-asyncio
uv run pytest tests/ -v
```

## NixOS Integration

Included as a flake input in dotfiles. Home Manager module at `nix/module.nix` provides `services.voice-pipeline` options and a systemd user service.
