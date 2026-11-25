# MOSS-VoiceGenerator server soak

The soak retains one server and sends sequential requests through the actual
speech endpoint:

```bash
conda run -n qwen3-tts python tests/moss_voicegen/moss_voicegen_session_soak.py \
  --server build/debug/bin/audiocpp_server --model MODEL.gguf \
  --backend cuda --requests 12 --server-log soak-server.log
```

Server logging is enabled and saved. HTTP failures, readiness timeout and invalid
or empty audio cause a nonzero exit. Successful responses must contain complete
24 kHz mono PCM16 WAV audio; duration comes from the WAV header and frames.

Optional `--card /sys/class/drm/cardN/device` reads device-wide VRAM after requests.
An explicitly requested but unreadable/malformed telemetry source fails. Without
it, memory is reported as unavailable, never zero. These endpoint observations
are not an inference-time peak or a memory-stability acceptance test. The script
does not invent a memory-growth limit.
