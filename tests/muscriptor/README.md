# MuScriptor Conversion

Use `convert_gguf.py` for the official Small, Medium and Large checkpoints.
It checks F32 dtypes and configuration/weight shapes, then delegates packaging
to `audiocpp_gguf`. Tensor names and values are preserved; the GGUF embeds
`config.json` and the model spec.

See [the model documentation](../../docs/models/muscriptor.md#convert-original-checkpoints)
for build, conversion and inference commands.

The Python dependencies are `numpy` and `safetensors`. Download access to each
upstream model requires accepting its license on Hugging Face.
