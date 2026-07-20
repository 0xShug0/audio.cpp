# Running audio.cpp on NixOS (and Nix)

## Prerequisites

- The [Nix package manager](https://nixos.org/download) must be installed.
- [Flakes](https://nixos.wiki/wiki/Flakes) must be enabled in your Nix configuration.

## Packages

The `flake.nix` provides multiple platform and hardware-specific builds. 
The following backends are available:
- **`vulkan`**: Hardware-accelerated build using Vulkan. (Default on Linux)
- **`cuda`**: Hardware-accelerated build using NVIDIA CUDA.
- **`metal`**: Hardware-accelerated build using Apple Metal. (Default on macOS)
- **`cpu`**: Portable CPU-only build.

All packages include the main tools: ***audiocpp_cli***, ***audiocpp_server***, ***audiocpp_gguf***, and the Python-based ***audiocpp_model_manager***.

## Build the package

### Default (Auto-detected OS)

Build the default package for your operating system:

```bash
nix build .
```

### Specific Backends

Build explicitly for a desired backend:

```bash
nix build .#vulkan
nix build .#cuda
nix build .#metal
nix build .#cpu
```

## Usage

After building, the compiled binaries will be available in the `result/bin/` directory.

### Running the CLI

You can execute the CLI directly via `nix run` (which uses `audiocpp_cli` as the main program):

```bash
nix run .#vulkan -- <arguments...>
```

### Running the Server

To run the server, use `nix shell` or execute from the build result:

```bash
nix shell .#vulkan -c audiocpp_server <arguments...>
# or
./result/bin/audiocpp_server <arguments...>
```

### Downloading Models (Model Manager)

The flake seamlessly bundles the `model_manager.py` Python script along with all its required dependencies (PyTorch, Safetensors, etc.) as `audiocpp_model_manager`.

```bash
nix shell .#vulkan -c audiocpp_model_manager install supertonic_3
```

## Development Shell

If you want to work on `audio.cpp` and need a development environment with `cmake`, `ninja`, and all the necessary C++ and Python dependencies pre-configured, simply run:

```bash
nix develop
```

## Using in a NixOS Configuration

You can easily include `audio.cpp` as a flake input in your own NixOS configuration or other Nix projects.

In your `flake.nix`, add it to your `inputs`, making sure to use `follows` to avoid duplicating `nixpkgs` versions:

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    
    audiocpp.url = "github:fedeizzo/audio.cpp";
    audiocpp.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, audiocpp, ... }: {
    nixosConfigurations.my-machine = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ({ pkgs, ... }: {
          environment.systemPackages = [
            # Add the default package (auto-detects Metal/Vulkan based on OS)
            audiocpp.packages.x86_64-linux.default
            
            # Or explicitly select a backend flavor:
            # audiocpp.packages.x86_64-linux.vulkan
            # audiocpp.packages.x86_64-linux.cuda
          ];
        })
      ];
    };
  };
}
```
