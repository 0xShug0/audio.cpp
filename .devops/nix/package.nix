{
  lib,
  stdenv,
  cmake,
  ninja,
  pkg-config,
  rocmPackages,
  cudaPackages,
  vulkan-headers,
  vulkan-loader,
  vulkan-tools,
  glslang,
  shaderc,
  darwin,
  python-scripts,
  config,
  version,

  # Overridable feature flags
  cudaSupport ? config.cudaSupport or false,
  vulkanSupport ? false,
  metalSupport ? stdenv.isDarwin,
  rocmSupport ? config.rocmSupport or false,
  rocmGpuTargets ? (lib.optionals rocmSupport rocmPackages.clr.gpuTargets),
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "audio.cpp";
  inherit version;
  src = lib.cleanSource ../..;

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
  ]
  ++ lib.optional cudaSupport cudaPackages.cuda_nvcc
  ++ lib.optional rocmSupport rocmPackages.clr;

  buildInputs = [
    python-scripts
  ]
  ++ lib.optionals vulkanSupport [
    vulkan-headers
    vulkan-loader
    vulkan-tools
    glslang
    shaderc
  ]
  ++ lib.optionals cudaSupport [
    cudaPackages.cudatoolkit
  ]
  ++ lib.optionals rocmSupport [
    rocmPackages.clr
    rocmPackages.hipblas
    rocmPackages.rocblas
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
    "-DENGINE_ENABLE_NATIVE_CPU=ON"
    "-DENGINE_ENABLE_LLAMAFILE=ON"
  ] ++ lib.optional stdenv.isDarwin "-DENGINE_ENABLE_OPENMP=OFF"
  ++ lib.optional vulkanSupport "-DENGINE_ENABLE_VULKAN=ON"
  ++ lib.optional cudaSupport "-DENGINE_ENABLE_CUDA=ON"
  ++ lib.optional metalSupport "-DENGINE_ENABLE_METAL=ON"
  ++ lib.optional rocmSupport "-DENGINE_ENABLE_HIP=ON"
  ++ lib.optional rocmSupport "-DCMAKE_HIP_COMPILER=${rocmPackages.llvm.clang}/bin/clang"
  ++ lib.optional rocmSupport "-DGPU_TARGETS=${lib.concatStringsSep ";" rocmGpuTargets}";

  env = lib.optionalAttrs rocmSupport {
    ROCM_PATH = "${rocmPackages.clr}";
  };

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

  meta = with lib; {
    description = "A high-performance C++ audio inference framework";
    homepage = "https://github.com/0xShug0/audio.cpp";
    license = licenses.mit;
    platforms = platforms.unix;
    mainProgram = "audiocpp_cli";
  };
})
