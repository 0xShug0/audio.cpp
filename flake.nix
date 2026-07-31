{
  description = "audio.cpp - High-performance C++ audio inference framework";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      forAllSystems = nixpkgs.lib.genAttrs [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      version = self.shortRev or self.dirtyShortRev or "dirty";

      pkgs = forAllSystems (system: import nixpkgs { inherit system; });

      pkgsCuda = forAllSystems (
        system:
        import nixpkgs {
          inherit system;
          config.cudaSupport = true;
          config.allowUnfreePredicate =
            p:
            builtins.all (
              l:
              l.free
              || builtins.elem l.shortName [
                "CUDA EULA"
                "cuDNN EULA"
                "cuSPARSELt EULA"
              ]
            ) (p.meta.licenses or (nixpkgs.lib.toList (p.meta.license or [ ])));
        }
      );

      audioPackages = forAllSystems (
        system:
        pkgs.${system}.callPackage .devops/nix/scope.nix {
          inherit version;
          python-scripts = pythonScripts.${system};
        }
      );
      audioPackagesCuda = forAllSystems (
        system:
        pkgsCuda.${system}.callPackage .devops/nix/scope.nix {
          inherit version;
          python-scripts = pythonScripts.${system};
        }
      );
      pythonScripts = forAllSystems (
        system: pkgs.${system}.callPackage .devops/nix/python-scripts.nix { }
      );
    in
    {
      packages = forAllSystems (
        system:
        let
          base = audioPackages.${system}.audiocpp;
          baseCuda = audioPackagesCuda.${system}.audiocpp;
        in
        {
          cpu = base;
          vulkan = base.override { vulkanSupport = true; };
          python-scripts = pythonScripts.${system};
          default =
            if pkgs.${system}.stdenv.isDarwin then
              self.packages.${system}.metal
            else
              self.packages.${system}.cpu;
        }
        // nixpkgs.lib.optionalAttrs pkgs.${system}.stdenv.isLinux {
          cuda = baseCuda.override { cudaSupport = true; };
          rocm = base.override { rocmSupport = true; };
          rocm-gfx1151 = base.override {
            rocmSupport = true;
            rocmGpuTargets = [ "gfx1151" ];
            strixHaloOptimizations = true;
          };
        }
        // nixpkgs.lib.optionalAttrs pkgs.${system}.stdenv.isDarwin {
          metal = base.override { metalSupport = true; };
        }
      );

      devShells = forAllSystems (
        system:
        {
          default = pkgs.${system}.mkShell {
            inputsFrom = [ self.packages.${system}.default ];
          };
        }
        // nixpkgs.lib.optionalAttrs pkgs.${system}.stdenv.isLinux {
          cuda = pkgsCuda.${system}.mkShell {
            inputsFrom = [ self.packages.${system}.cuda ];
          };
        }
        # ROCm's Nix toolchain is currently supported on x86_64 Linux only.
        # Keep the shell off aarch64 rather than exposing an unevaluable output.
        // nixpkgs.lib.optionalAttrs (system == "x86_64-linux") {
          rocm = pkgs.${system}.mkShell {
            inputsFrom = [ self.packages.${system}.rocm ];
          };
          rocm-gfx1151 = pkgs.${system}.mkShell {
            inputsFrom = [ self.packages.${system}.rocm-gfx1151 ];
          };
        }
      );
    };
}
