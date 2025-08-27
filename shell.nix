with import <nixpkgs> {};

mkShell {
  # The `buildInputs` attribute lists all packages to be made available in the shell.
  buildInputs = [
    # 1. Your existing Python environment with its packages
    (python3.withPackages (ps: with ps; [
      numpy
      matplotlib
      jax
      scipy
    ]))

    # 2. Add other system-level packages here
    ffmpeg
  ];
}
