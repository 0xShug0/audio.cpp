{
  lib,
  stdenv,
  cmake,
  ninja,
  pkg-config,
  openssl,
  rocmPackages,
  cudaPackages,
  vulkan-headers,
  vulkan-loader,
  vulkan-tools,
  glslang,
  shaderc,
  python-scripts,
  config,
  version,
  autoAddDriverRunpath,

  # Overridable feature flags
  cudaSupport ? config.cudaSupport or false,
  vulkanSupport ? false,
  metalSupport ? stdenv.isDarwin,
  rocmSupport ? config.rocmSupport or false,
  rocmGpuTargets ? (lib.optionals rocmSupport rocmPackages.clr.gpuTargets),
  strixHaloOptimizations ? (rocmSupport && rocmGpuTargets == [ "gfx1151" ]),
  nativeModelManagerSupport ? true,
  # Model selection: if non-empty, only these model targets are built.
  # See CMakeLists.txt AUDIOCPP_MODEL_SET / AUDIOCPP_MODELS.
  models ? [ ],
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
  ++ lib.optional cudaSupport autoAddDriverRunpath
  ++ lib.optional rocmSupport rocmPackages.clr;

  buildInputs = [
    python-scripts
  ]
  ++ lib.optional nativeModelManagerSupport openssl
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
  ]
  ++ lib.optionals nativeModelManagerSupport [
    "-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON"
    "-DAUDIOCPP_USE_SYSTEM_OPENSSL=ON"
  ]
  ++ (
    if models != [ ] then
      [
        "-DAUDIOCPP_MODEL_SET=custom"
        "-DAUDIOCPP_MODELS=${lib.concatStringsSep "," models}"
      ]
    else
      [ "-DAUDIOCPP_MODEL_SET=full" ]
  )
  ++ lib.optional stdenv.isDarwin "-DENGINE_ENABLE_OPENMP=OFF"
  ++ lib.optional vulkanSupport "-DENGINE_ENABLE_VULKAN=ON"
  ++ lib.optional cudaSupport "-DENGINE_ENABLE_CUDA=ON"
  ++ lib.optional metalSupport "-DENGINE_ENABLE_METAL=ON"
  ++ lib.optional rocmSupport "-DENGINE_ENABLE_HIP=ON"
  ++ lib.optional rocmSupport "-DCMAKE_HIP_COMPILER=${rocmPackages.llvm.clang}/bin/clang"
  ++ lib.optional rocmSupport "-DGPU_TARGETS=${lib.concatStringsSep ";" rocmGpuTargets}"
  ++ lib.optional strixHaloOptimizations "-DENGINE_HIP_STRIX_HALO_OPTIMIZATIONS=ON";

  env = lib.optionalAttrs rocmSupport {
    ROCM_PATH = "${rocmPackages.clr}";
  };

  installPhase = ''
    runHook preInstall

    mkdir -p $out/bin

    # Copy the built C++ executables directly from the bin directory
    cp bin/audiocpp_cli bin/audiocpp_server bin/audiocpp_gguf $out/bin/
    ${lib.optionalString nativeModelManagerSupport ''
      cp bin/audiocpp_model_manager $out/bin/
    ''}

    # Keep the supported Python v2 manager available during migration without
    # overwriting the native audiocpp_model_manager executable copied above.
    install -Dm755 $src/tools/model_manager_v2.py $out/bin/audiocpp_model_manager_v2.py
    cp -R $src/model_specs $out/model_specs

    # Patch the shebang to use our python environment with torch/safetensors/pyyaml
    patchShebangs $out/bin/audiocpp_model_manager_v2.py

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
