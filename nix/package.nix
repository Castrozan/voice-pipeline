{ pkgs, ... }:
let
  python = pkgs.python312;
  uv = pkgs.uv;
  prefix = "$HOME/.local/share/voice-pipeline-venv";
  sourceStorePath = "${../.}";

  voicePipeline = pkgs.writeShellScriptBin "voice-pipeline" ''
    export PATH="${python}/bin:${pkgs.portaudio}/lib:''${PATH:+:$PATH}"
    export LD_LIBRARY_PATH="${pkgs.portaudio}/lib:${
      pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ]
    }:''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    VENV="${prefix}"
    CURRENT_SOURCE="${sourceStorePath}"

    if [ ! -d "$VENV" ]; then
      echo "[voice-pipeline] Creating virtualenv..." >&2
      ${uv}/bin/uv venv --python "${python}/bin/python" "$VENV" >&2
    fi

    INSTALLED_SOURCE=""
    if [ -f "$VENV/.installed-source" ]; then
      INSTALLED_SOURCE=$(cat "$VENV/.installed-source")
    fi

    if [ "$INSTALLED_SOURCE" != "$CURRENT_SOURCE" ]; then
      echo "[voice-pipeline] Installing dependencies (uv)..." >&2
      TMPBUILD=$(mktemp -d)
      cp -r "$CURRENT_SOURCE"/. "$TMPBUILD/"
      chmod -R u+w "$TMPBUILD"
      ${uv}/bin/uv pip install --python "$VENV/bin/python" "$TMPBUILD" >&2
      rm -rf "$TMPBUILD"
      echo "$CURRENT_SOURCE" > "$VENV/.installed-source"
    fi

    exec "$VENV/bin/voice-pipeline" "$@"
  '';
in
{
  home.packages = [
    voicePipeline
    pkgs.portaudio
  ];
}
