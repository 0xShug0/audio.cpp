{
  lib,
  python3,
}:

python3.withPackages (ps: with ps; [
  ps.torch
  ps.safetensors
  ps.pyyaml
  ps.numpy
])
