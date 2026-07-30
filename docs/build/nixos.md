# Running audio.cpp on NixOS (and Nix)

## Prerequisites

- The [Nix package manager](https://nixos.org/download) must be installed.
- [Flakes](https://nixos.wiki/wiki/Flakes) must be enabled.

## Packages

The flake provides backend-specific builds:

| Package        | Backend                  | Platform |
| -------------- | ------------------------ | -------- |
| `cpu`          | CPU-only                 | All      |
| `vulkan`       | Vulkan                   | All      |
| `cuda`         | NVIDIA CUDA              | Linux    |
| `rocm`         | AMD ROCm                 | Linux    |
| `rocm-gfx1151` | AMD ROCm (single target) | Linux    |
| `metal`        | Apple Metal              | macOS    |

All packages include: **audiocpp_cli**, **audiocpp_server**, **audiocpp_gguf**, and **audiocpp_model_manager**.

## Build

```bash
nix build .#cpu
nix build .#vulkan
nix build .#cuda
nix build .#rocm
nix build .#rocm-gfx1151    # Single GPU target (faster)
```

## Run

```bash
nix run .#rocm -- --backend hip --task tts --family supertonic --model ./model --text "hello"
./result/bin/audiocpp_cli --backend hip --device 0
```

## Download Models

The `python-scripts` package includes all dependencies for `model_manager.py`:

```bash
nix shell .#python-scripts -c python3 tools/model_manager.py install supertonic_3
```

## Development Shell

```bash
nix develop              # Default backend
nix develop .#cuda       # CUDA dev environment
```

## Custom GPU Target

Override `hipGpuTargets` for a specific GPU:

```bash
nix build --impure --expr '(builtins.getFlake (toString ./.)).outputs.packages.x86_64-linux.rocm.override { rocmGpuTargets = ["gfx1151"]; }'
```

## NixOS Configuration

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    audiocpp.url = "github:0xShug0/audio.cpp";
    audiocpp.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, audiocpp, ... }: {
    nixosConfigurations.my-machine = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ({ ... }: {
          environment.systemPackages = [
            audiocpp.packages.x86_64-linux.default

            # Or explicitly select a backend flavor:
            # audiocpp.packages.x86_64-linux.vulkan
            # audiocpp.packages.x86_64-linux.cuda
            # audiocpp.packages.x86_64-linux.rocm

            # Custom GPU target:
            # audiocpp.packages.x86_64-linux.rocm.override {
            #   rocmGpuTargets = [ "gfx1151" ];
            # }
          ];
        })
      ];
    };
  };
}
```
