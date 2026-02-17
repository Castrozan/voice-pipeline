{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.voice-pipeline;

  agentVoicesJson = builtins.toJSON (
    lib.mapAttrs (
      _name: agentCfg:
      if cfg.ttsEngine == "edge-tts" && agentCfg.edgeTtsVoice != "" then
        agentCfg.edgeTtsVoice
      else
        agentCfg.openaiVoice
    ) cfg.agents
  );

  agentLanguagesJson = builtins.toJSON (
    lib.filterAttrs (_: v: v != "") (lib.mapAttrs (_name: agentCfg: agentCfg.language) cfg.agents)
  );

  wakeWordsJson = builtins.toJSON cfg.wakeWords;
in
{
  options.services.voice-pipeline = {
    enable = lib.mkEnableOption "Voice Pipeline real-time conversational AI";

    gatewayUrl = lib.mkOption {
      type = lib.types.str;
      default = "http://localhost:18789";
    };

    gatewayTokenFile = lib.mkOption {
      type = lib.types.str;
      default = "/run/agenix/openclaw-gateway-token";
    };

    deepgramApiKeyFile = lib.mkOption {
      type = lib.types.str;
      default = "/run/agenix/deepgram-api-key";
    };

    openaiApiKeyFile = lib.mkOption {
      type = lib.types.str;
      default = "/run/agenix/openai-api-key";
    };

    defaultAgent = lib.mkOption {
      type = lib.types.str;
      default = "jarvis";
    };

    wakeWords = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ "jarvis" ];
    };

    ttsEngine = lib.mkOption {
      type = lib.types.enum [
        "openai"
        "edge-tts"
      ];
      default = "openai";
    };

    ttsVoice = lib.mkOption {
      type = lib.types.str;
      default = "onyx";
    };

    model = lib.mkOption {
      type = lib.types.str;
      default = "anthropic/claude-sonnet-4-5";
    };

    sttEngine = lib.mkOption {
      type = lib.types.enum [
        "deepgram"
        "openai-whisper"
      ];
      default = "deepgram";
    };

    captureDevice = lib.mkOption {
      type = lib.types.str;
      default = "echo-cancel-source";
    };

    conversationWindowSeconds = lib.mkOption {
      type = lib.types.float;
      default = 15.0;
    };

    maxHistoryTurns = lib.mkOption {
      type = lib.types.int;
      default = 20;
    };

    bargeInEnabled = lib.mkOption {
      type = lib.types.bool;
      default = true;
    };

    systemPrompt = lib.mkOption {
      type = lib.types.str;
      default = "This is a voice conversation via microphone and TTS. Respond concisely, max 3 sentences. Match the spoken language (English or Portuguese). Never include markdown, file paths, code blocks, URLs, or any formatting.";
      description = "System prompt prepended to LLM messages. Adds voice-specific formatting rules without overriding agent personality.";
    };

    agents = lib.mkOption {
      type = lib.types.attrsOf (
        lib.types.submodule {
          options.openaiVoice = lib.mkOption {
            type = lib.types.str;
            default = "onyx";
          };
          options.edgeTtsVoice = lib.mkOption {
            type = lib.types.str;
            default = "";
            description = "Edge-TTS voice name (e.g. pt-BR-AntonioNeural). When set and ttsEngine=edge-tts, overrides openaiVoice.";
          };
          options.language = lib.mkOption {
            type = lib.types.str;
            default = "";
            description = "Language the agent should respond in (e.g. Portuguese, English). Appended to system prompt.";
          };
        }
      );
      default = { };
    };
  };

  config = lib.mkIf cfg.enable {
    home.file.".config/voice-pipeline/env".text = ''
      VOICE_PIPELINE_GATEWAY_URL=${cfg.gatewayUrl}
      VOICE_PIPELINE_GATEWAY_TOKEN_FILE=${cfg.gatewayTokenFile}
      VOICE_PIPELINE_DEFAULT_AGENT=${cfg.defaultAgent}
      VOICE_PIPELINE_DEEPGRAM_API_KEY_FILE=${cfg.deepgramApiKeyFile}
      VOICE_PIPELINE_OPENAI_API_KEY_FILE=${cfg.openaiApiKeyFile}
      VOICE_PIPELINE_TTS_ENGINE=${cfg.ttsEngine}
      VOICE_PIPELINE_TTS_VOICE=${cfg.ttsVoice}
      VOICE_PIPELINE_WAKE_WORDS='${wakeWordsJson}'
      VOICE_PIPELINE_CONVERSATION_WINDOW_SECONDS=${toString cfg.conversationWindowSeconds}
      VOICE_PIPELINE_MAX_HISTORY_TURNS=${toString cfg.maxHistoryTurns}
      VOICE_PIPELINE_CAPTURE_DEVICE=${cfg.captureDevice}
      VOICE_PIPELINE_BARGE_IN_ENABLED=${if cfg.bargeInEnabled then "true" else "false"}
      VOICE_PIPELINE_STT_ENGINE=${cfg.sttEngine}
      VOICE_PIPELINE_AGENT_VOICES='${agentVoicesJson}'
      VOICE_PIPELINE_AGENT_LANGUAGES='${agentLanguagesJson}'
      VOICE_PIPELINE_MODEL=${cfg.model}
      VOICE_PIPELINE_SYSTEM_PROMPT="${cfg.systemPrompt}"
    '';

    systemd.user.services.voice-pipeline = {
      Unit = {
        Description = "Voice Pipeline - Real-time conversational AI";
        After = [ "pipewire.service" ];
      };
      Service = {
        Type = "simple";
        EnvironmentFile = "%h/.config/voice-pipeline/env";
        ExecStart = "${pkgs.lib.getExe (
          pkgs.writeShellScriptBin "voice-pipeline-start" ''
            OPENCLAW_TOKEN=$(openclaw config get gateway.auth.token 2>/dev/null || true)
            if [ -n "$OPENCLAW_TOKEN" ]; then
              export VOICE_PIPELINE_GATEWAY_TOKEN="$OPENCLAW_TOKEN"
            fi
            exec voice-pipeline
          ''
        )}";
        Restart = "on-failure";
        RestartSec = 5;
      };
      Install.WantedBy = [ ];
    };
  };
}
