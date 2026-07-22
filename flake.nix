{
  description = "audio.cpp - High-performance C++ audio inference framework";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      pkgs = forAllSystems (system: import nixpkgs { 
        inherit system; 
      });
      pkgsCuda = forAllSystems (system: import nixpkgs { 
        inherit system; 
        config.cudaSupport = true;
        config.allowUnfreePredicate = p:
          builtins.all (
            license:
            license.free
            || builtins.elem license.shortName [
              "CUDA EULA"
              "cuDNN EULA"
              "cuSPARSELt EULA"
            ]
          ) (p.meta.licenses or (nixpkgs.lib.toList (p.meta.license or [])));
      });

      # Python environment for model_manager.py and test scripts
      mkPython = { pkgs, cudaSupport ? false, vulkanSupport ? false }: 
        pkgs.python3.withPackages (ps: with ps; [
          (if cudaSupport then ps.torchWithCuda else if vulkanSupport then ps.torchWithVulkan else ps.torch)
          ps.safetensors
          ps.pyyaml
          ps.librosa
          ps.numpy
          ps.soundfile
          ps.psutil
          ps.pynvml
        ]);

      # Python environment for development shell with TUI tools
      mkPythonShell = { pkgs, cudaSupport ? false, vulkanSupport ? false }: 
        pkgs.python3.withPackages (ps: with ps; [
          (if cudaSupport then ps.torchWithCuda else if vulkanSupport then ps.torchWithVulkan else ps.torch)
          ps.safetensors
          ps.pyyaml
          ps.librosa
          ps.numpy
          ps.soundfile
          ps.psutil
          ps.pynvml
          ps.rich
          ps.textual
        ]);

      # Shared build inputs for both packages and shells
      getNativeBuildInputs = { pkgs, cudaSupport ? false }:
        with pkgs; [
          cmake
          ninja
          pkg-config
        ] ++ pkgs.lib.optional cudaSupport pkgs.cudaPackages.cuda_nvcc;

      getBuildInputs = { pkgs, cudaSupport ? false, vulkanSupport ? false, metalSupport ? pkgs.stdenv.isDarwin, isShell ? false }:
        with pkgs; [
          (if isShell then mkPythonShell { inherit pkgs cudaSupport vulkanSupport; } else mkPython { inherit pkgs cudaSupport vulkanSupport; })
        ] ++ pkgs.lib.optionals isShell [
          openai-whisper
        ] ++ pkgs.lib.optionals vulkanSupport [
          vulkan-headers
          vulkan-loader
          vulkan-tools
          glslang
          shaderc
        ] ++ pkgs.lib.optionals cudaSupport [
          pkgs.cudaPackages.cudatoolkit
        ] ++ pkgs.lib.optionals metalSupport [
          pkgs.apple-sdk_14
        ];
    in
    {
      packages = forAllSystems (system:
        let
          # A function to build audio.cpp with any set of features
          mkAudioCpp = { pkgs, cudaSupport ? false, vulkanSupport ? false, metalSupport ? pkgs.stdenv.isDarwin }: 
            pkgs.stdenv.mkDerivation {
              pname = "audio.cpp";
              version = self.shortRev or self.dirtyShortRev or "dirty";

              src = ./.;

              nativeBuildInputs = getNativeBuildInputs { inherit pkgs cudaSupport; };
              buildInputs = getBuildInputs { inherit pkgs cudaSupport vulkanSupport metalSupport; };

              cmakeFlags = [
                "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
                "-DENGINE_ENABLE_NATIVE_CPU=ON"
                "-DENGINE_ENABLE_LLAMAFILE=ON"
              ] ++ pkgs.lib.optional vulkanSupport "-DENGINE_ENABLE_VULKAN=ON"
                ++ pkgs.lib.optional cudaSupport "-DENGINE_ENABLE_CUDA=ON"
                ++ pkgs.lib.optional metalSupport "-DENGINE_ENABLE_METAL=ON";

              installPhase = ''
                runHook preInstall

                mkdir -p $out/bin
                
                # Copy the built C++ executables directly from the bin directory
                cp bin/audiocpp_cli bin/audiocpp_server bin/audiocpp_gguf $out/bin/

                # Install the python model manager script
                cp $src/tools/model_manager.py $out/bin/audiocpp_model_manager
                chmod +x $out/bin/audiocpp_model_manager

                # Patch the shebang to use our python environment with torch/safetensors/pyyaml
                patchShebangs $out/bin/audiocpp_model_manager

                runHook postInstall
              '';

              meta = with pkgs.lib; {
                description = "A high-performance C++ audio inference framework";
                homepage = "https://github.com/0xShug0/audio.cpp";
                license = licenses.mit;
                platforms = platforms.unix;
                mainProgram = "audiocpp_cli";
              };
            };
        in
        {
          # Expose specific backend variants
          cpu = mkAudioCpp { pkgs = pkgs.${system}; cudaSupport = false; vulkanSupport = false; metalSupport = false; };
          vulkan = mkAudioCpp { pkgs = pkgs.${system}; vulkanSupport = true; cudaSupport = false; metalSupport = false; };
        } // pkgs.${system}.lib.optionalAttrs pkgs.${system}.stdenv.isLinux {
          cuda = mkAudioCpp { pkgs = pkgsCuda.${system}; cudaSupport = true; vulkanSupport = false; metalSupport = false; };
        } // pkgs.${system}.lib.optionalAttrs pkgs.${system}.stdenv.isDarwin {
          metal = mkAudioCpp { pkgs = pkgs.${system}; metalSupport = true; cudaSupport = false; vulkanSupport = false; };
        } // {
          # Automatically select best default for the current platform
          default = if pkgs.${system}.stdenv.isDarwin then self.packages.${system}.metal else self.packages.${system}.vulkan;
        }
      );

      devShells = forAllSystems (system:
        let
          p = pkgs.${system};
          pc = pkgsCuda.${system};
        in
        {
          default = p.mkShell {
            nativeBuildInputs = getNativeBuildInputs { pkgs = p; };
            buildInputs = getBuildInputs { pkgs = p; vulkanSupport = !p.stdenv.isDarwin; isShell = true; };
          };
        } // p.lib.optionalAttrs p.stdenv.isLinux {
          cuda = pc.mkShell {
            nativeBuildInputs = getNativeBuildInputs { pkgs = pc; cudaSupport = true; };
            buildInputs = getBuildInputs { pkgs = pc; cudaSupport = true; metalSupport = false; isShell = true; };
          };
        }
      );
    };
}
