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
          lib,
          stdenv,
        }:
        let
          python = python312;
          prefix = "$HOME/.local/share/voice-pipeline-venv";
        in
        writeShellScriptBin "voice-pipeline" ''
          export PATH="${python}/bin:${portaudio}/lib:''${PATH:+:$PATH}"
          export LD_LIBRARY_PATH="${portaudio}/lib:${
            lib.makeLibraryPath [ stdenv.cc.cc.lib ]
          }:''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          VENV="${prefix}"

          if [ ! -d "$VENV" ]; then
            echo "[voice-pipeline] Creating virtualenv..." >&2
            ${python}/bin/python -m venv "$VENV" >&2
          fi

          if [ ! -f "$VENV/.installed" ] || [ "$VENV/.installed" -ot "${self}/pyproject.toml" ]; then
            echo "[voice-pipeline] Installing dependencies..." >&2
            "$VENV/bin/pip" install --quiet --upgrade pip >&2
            "$VENV/bin/pip" install --quiet "${self}" >&2
            touch "$VENV/.installed"
          fi

          exec "$VENV/bin/voice-pipeline" "$@"
        ''
      ) { };
    };
}
