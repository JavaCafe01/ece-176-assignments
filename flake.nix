{
  description = "ECE 176 Python Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/master";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        python = "python39";
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShell = pkgs.mkShell {
          buildInputs = [
            (pkgs.${python}.withPackages
              (ps: with ps; [ pip setuptools black numpy matplotlib urllib3 scikitimage jupyterlab ]))
            pkgs.nodePackages.pyright
          ];
        };
      });
}
