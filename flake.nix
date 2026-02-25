{
  description = "Real-time conversational voice pipeline";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
    {
      homeManagerModules.default = ./nix/module.nix;

      packages.${system}.default = pkgs.callPackage (
        {
          writeShellScriptBin,
          python312,
          uv,
          portaudio,
          alsa-lib,
          lib,
          stdenv,
        }:
        let
          python = python312;
          prefix = "$HOME/.local/share/voice-pipeline-venv";
          nativeLibs = [
            portaudio
            alsa-lib
            stdenv.cc.cc.lib
          ];
        in
        writeShellScriptBin "voice-pipeline" ''
          export LD_LIBRARY_PATH="${lib.makeLibraryPath nativeLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

          ENV_FILE="''${HOME}/.config/voice-pipeline/env"
          if [ -f "$ENV_FILE" ]; then
            set -a
            source "$ENV_FILE"
            set +a
          fi

          VENV="${prefix}"
          CURRENT_SOURCE="${self}"

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
            ${uv}/bin/uv pip install --python "$VENV/bin/python" "$CURRENT_SOURCE" >&2
            echo "$CURRENT_SOURCE" > "$VENV/.installed-source"
          fi

          exec "$VENV/bin/voice-pipeline" "$@"
        ''
      ) { };
    };
}
