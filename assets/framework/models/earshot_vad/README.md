# Earshot VAD weights

`earshot_vad.safetensors` holds the network weights and front-end tables of
[Earshot](https://github.com/pykeio/earshot) 1.2.2 (`src/weights.bin` and the
`src/filters.rs` Hann window and mel filterbank), unchanged.

Earshot is dual-licensed under MIT OR Apache-2.0; the MIT notice is in
`LICENSE-MIT`. Regenerate the file with:

```bash
python tests/earshot_vad/convert_earshot.py \
  --output assets/framework/models/earshot_vad/earshot_vad.safetensors
```

See [docs/models/earshot_vad.md](../../../../docs/models/earshot_vad.md).
