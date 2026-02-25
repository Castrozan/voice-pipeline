# Voice Pipeline - Agent Instructions

---
description: Voice pipeline development rules
alwaysApply: true
---

## Project

Real-time voice pipeline for OpenClaw Claude agents. Full-duplex conversational AI with wake word detection, VAD-gated STT, streaming LLM completion, and streaming TTS playback with barge-in support.

Hexagonal architecture with ports/adapters pattern. Domain has zero external dependencies.

## Architecture

```
  Local                              Cloud
┌──────────────────────┐     ┌─────────────────────┐
│ Mic (PipeWire)       │     │ Deepgram Nova-2     │
│   ↓                  │     │ STT streaming       │
│ VAD (Silero ONNX)    │────▶│       ↓             │
│   ↓                  │     └──────┬──────────────┘
│ Wake Word Detect     │            │ transcript
│   (on transcript)    │     ┌──────▼──────────────┐
│                      │     │ Anthropic Claude    │
│ OpenClaw Gateway ────│────▶│ via OpenClaw Gateway │
│   (local daemon)     │     │ SSE streaming       │
│                      │     └──────┬──────────────┘
│                      │            │ text stream
│ Speaker ◀────────────│◀────┌──────▼──────────────┐
│   ↓                  │     │ TTS (OpenAI/edge)   │
│ VAD (barge-in) ──────│────▶│ streaming           │
│                      │     └─────────────────────┘
└──────────────────────┘
```

### Always-on Flow

1. Mic → VAD → speech detected → open Deepgram WebSocket → stream audio
2. Deepgram returns real-time transcription
3. Scan transcript for wake word
4. Wake word found → continue streaming until endpointing signals utterance complete
5. Send post-wake-word text to LLM
6. Stream LLM response → stream to TTS → play audio
7. After AI responds → conversation window: any speech = follow-up (no wake word needed)
8. During AI speech → VAD active on echo-cancelled source → user speaks → barge-in: cancel LLM + TTS + playback
9. Window expires with no speech → return to ambient wake word listening

No push-to-talk. Service is on or off.

### State Machine

```
AMBIENT → LISTENING → THINKING → SPEAKING → CONVERSING → AMBIENT
   ↑                                 │          │
   └─── timeout (no speech) ────────┘          │
                    ↑                            │
                    └──── barge-in ──────────────┘
```

States defined in `src/domain/state.py`. Transitions enforced by the pipeline orchestrator.

## Tech Stack

| Component | Technology | Location |
|-----------|-----------|----------|
| Audio I/O | `sounddevice` (PortAudio) | Local |
| VAD | Silero ONNX v5 (16ms frames) | Local |
| STT | Deepgram Nova-2 streaming WS | Cloud |
| LLM | OpenClaw Gateway → Anthropic Claude | Local daemon → Cloud |
| TTS | OpenAI `tts-1` or `edge-tts` | Cloud |
| Echo Cancel | PipeWire WebRTC AEC | Local |
| IPC | Unix domain socket (JSON) | Local |

## Structure

Flat package under `src/`. No `voice_pipeline` namespace — imports are direct:
```python
from config import VoicePipelineConfig
from domain.pipeline import VoicePipeline
from adapters.silero_vad import SileroVad
```

Entry point: `src/__main__.py:main()`
Factory wiring: `src/factory.py:create_pipeline()`

Ports are Protocol classes in `src/ports/`. Adapters are concrete implementations in `src/adapters/`. Domain logic in `src/domain/` has zero external dependencies.

## Ports (Protocol classes)

Defined in `src/ports/`. Read the actual Protocol definitions there for current signatures.

```python
class CompletionPort(Protocol):
    async def stream(self, messages: list[dict[str, str]], agent: str) -> AsyncIterator[str]: ...
    async def cancel(self) -> None: ...

class SynthesizerPort(Protocol):
    async def synthesize(self, text: str, voice: str) -> AsyncIterator[bytes]: ...
    async def cancel(self) -> None: ...
```

## Configuration

All settings via `VOICE_PIPELINE_` prefixed environment variables. `src/config.py` is the single source of truth for all config options, defaults, and types — read it directly instead of relying on a table here.

Key patterns:
- Secret files (API keys, tokens) use `*_file` suffix with `read_secret()` helper
- Per-agent overrides via JSON dict configs (voices, languages)
- Audio params (sample rate, frame duration, VAD thresholds) tuned for Silero VAD v5
- LLM completion supports multiple engines (openclaw, anthropic, cli)
- TTS supports multiple engines (openai, edge-tts)

## Testing

```bash
uv run pytest tests/ -v
```

All domain tests are pure (no I/O). Fakes in `tests/conftest.py`.
Silero VAD v5 requires exactly 256 samples (16ms at 16kHz) per frame — all test frame generation respects this.

### Audio Recording Workflow

Recordings live in `tests/recordings/{mic_type}/` and are committed to git for reproducible tests.
Tool: `tests/record_test_audio.py`

**Before recording:** Stop the voice pipeline (`Ctrl+C` in `vc-pipe-exec` tmux session) to free the audio device. Both the pipeline and the recording tool use the echo-cancel source.

**Adding new clips:** Define clips in `tests/record_test_audio.py` CLIPS dict with `(instruction, duration_seconds)`. Agent designs the exact script the user must say, including "Apple" prefix when needed to work around first-word-eating on echo-cancel source.

**Recording flow (one clip at a time):**
1. Agent presents ONE clip: name, exact script to say, duration
2. User says "ok" or "go" to start recording
3. Agent runs `uv run python tests/record_test_audio.py <clip_name> --mic <type>` (3s countdown then records)
4. Agent runs `uv run python tests/record_test_audio.py transcribe <mic/clip>` to verify with Whisper
5. **Transcription must match the script exactly** (word for word). If Whisper drops or changes words, re-record. VAD stats are informational but transcription accuracy is the gate.
6. If transcription matches → move to next clip. If not → re-record same clip.
7. After all clips recorded → commit to git → write/update tests.

**Mic types:** `builtin` (laptop mic, echo-cancel processed), `headset` (3.5mm jack, no first-word-eating).

After all clips are recorded, `uv run pytest tests/test_recorded_audio.py -v` runs VAD replay tests without user interaction.

## Conventions

- No comments in code
- No `voice_pipeline.` prefix in imports
- Ports are Protocol classes, adapters are concrete implementations
- Config via environment variables with `VOICE_PIPELINE_` prefix

## Dotfiles Integration

Separate repo, included as a flake input in dotfiles. Home Manager module at `nix/module.nix` provides `services.voice-pipeline` options and a systemd user service. Nix packaging uses venv + pip install pattern (see `nix/package.nix`).

Reference files in dotfiles repo:
- `home/modules/hey-bot.nix` — gateway interaction, systemd service, TTS flow
- `home/modules/voxtype.nix` — flake input + HM module pattern
- `home/modules/openclaw/config.nix` — agent schema
- `home/modules/openclaw/install.nix` — fallback uv packaging pattern
- `hosts/dellg15/configs/audio.nix` — PipeWire config
- `.config/hypr/conf.d/bindings.conf` — keybind format (`SUPER ALT, V` for toggle)
