# YuE2

YuE2 is a music generation model with symbolic ABC planning, semantic codec generation, NAR acoustic flow synthesis, and Oobleck VAE decode.

The audio.cpp port uses component GGUFs: one GGUF for the main YuE2 MoT model and one GGUF for the VAE. This keeps the main model and VAE independently quantizable.

Example:

```bash
./build/debug/bin/audiocpp_cli \
  --task gen \
  --family yue2 \
  --model models/YuE2-GGUF \
  --backend cuda \
  --threads 8 \
  --text "A hopeful chorus about sunrise over the city." \
  --request-option style="bright pop rock, energetic drums, clean guitars" \
  --request-option cot=off \
  --request-option seed=831001 \
  --out yue2.wav \
  --log
```

Use `cot=melody` or `cot=full` to run the symbolic route. `abc` or `abc_file` can be provided to use an external score.
