{
  description = "audio.cpp - High-performance C++ audio inference framework";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      nixpkgsFor = forAllSystems (system: import nixpkgs { 
        inherit system; 
        config.allowUnfree = true;
        config.cudaSupport = true;
      });
    in
    {
      packages = forAllSystems (system:
        let
          pkgs = nixpkgsFor.${system};
          # Python environment for model_manager.py
          myPython = pkgs.python3.withPackages (ps: with ps; [
            torch
            safetensors
            pyyaml
          ]);

          # A function to build audio.cpp with any set of features
          mkAudioCpp = { cudaSupport ? false, vulkanSupport ? false, metalSupport ? pkgs.stdenv.isDarwin }: 
            pkgs.stdenv.mkDerivation {
              pname = "audio.cpp";
              version = self.shortRev or self.dirtyShortRev or "dirty";

              src = ./.;

              nativeBuildInputs = with pkgs; [
                cmake
                ninja
                pkg-config
                makeWrapper
              ] ++ pkgs.lib.optional cudaSupport pkgs.cudaPackages.cuda_nvcc;

              buildInputs = with pkgs; [
                libunwind
                myPython
              ] ++ pkgs.lib.optionals vulkanSupport [
                vulkan-headers
                vulkan-loader
                vulkan-tools
                glslang
                shaderc
              ] ++ pkgs.lib.optionals cudaSupport [
                pkgs.cudaPackages.cudatoolkit
              ] ++ pkgs.lib.optionals metalSupport [
                darwin.apple_sdk.frameworks.Metal
                darwin.apple_sdk.frameworks.Foundation
                darwin.apple_sdk.frameworks.MetalPerformanceShaders
                darwin.apple_sdk.frameworks.Accelerate
              ];

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
                
                # Find and copy the built C++ executables
                find . -type f -executable \( -name "audiocpp_cli" -o -name "audiocpp_server" -o -name "audiocpp_gguf" \) -exec cp {} $out/bin/ \;

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
          cpu = mkAudioCpp { cudaSupport = false; vulkanSupport = false; metalSupport = false; };
          vulkan = mkAudioCpp { vulkanSupport = true; cudaSupport = false; metalSupport = false; };
          cuda = mkAudioCpp { cudaSupport = true; vulkanSupport = false; metalSupport = false; };
          metal = mkAudioCpp { metalSupport = true; cudaSupport = false; vulkanSupport = false; };

          # Automatically select best default for the current platform
          default = if pkgs.stdenv.isDarwin then self.packages.${system}.metal else self.packages.${system}.vulkan;
        }
      );

      devShells = forAllSystems (system:
        let
          pkgs = nixpkgsFor.${system};
        in
        {
          default = pkgs.mkShell {
            inputsFrom = [ self.packages.${system}.default ];
          };
        }
      );
    };
}
