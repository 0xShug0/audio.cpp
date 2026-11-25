# Model tools

Model-output utilities and source-code generators live in this flat directory, with model-first descriptive names. Core and community models use the same layout.

| Tool | Purpose |
|---|---|
| [moss_codec_contexts_check.py](moss_codec_contexts_check.py) | Compare MOSS codec attention contexts with checkpoint configuration. |
| [moss_voicegen_audio_report.py](moss_voicegen_audio_report.py) | Report duration, silence, levels and estimated pitch. |
| [soulx_flashhead_video_render.py](soulx_flashhead_video_render.py) | Encode generated RGB video artifacts as MP4. |
| [tiny_voca_tables_generate.py](tiny_voca_tables_generate.py) | Generate frontend Unicode and pronunciation tables in the C++ source tree. |

GGUF converters, their related weight/asset preparation, inspection tools and supporting files live under [tools/gguf](../gguf/), grouped by model when there are multiple files. Reference runners, fixture generators and parity/lifecycle checks live under `tests/<model>/`.
