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
          export LD_LIBRARY_PATH="${lib.makeLibraryPath nativeLibs}:''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          VENV="${prefix}"
          SRC="${self}"

          if [ ! -d "$VENV" ]; then
            echo "[voice-pipeline] Creating virtualenv..." >&2
            ${python}/bin/python -m venv "$VENV" >&2
          fi

          if [ ! -f "$VENV/.installed" ] || [ "$VENV/.installed" -ot "$SRC/pyproject.toml" ]; then
            echo "[voice-pipeline] Installing dependencies..." >&2
            TMPBUILD=$(mktemp -d)
            cp -r "$SRC"/. "$TMPBUILD/"
            chmod -R u+w "$TMPBUILD"
            "$VENV/bin/pip" install --quiet --upgrade pip >&2
            "$VENV/bin/pip" install --quiet "$TMPBUILD" >&2
            rm -rf "$TMPBUILD"
            touch "$VENV/.installed"
          fi

          exec "$VENV/bin/voice-pipeline" "$@"
        ''
      ) { };
    };
}
