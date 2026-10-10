# Streaming Audio Controller

`engine::runtime::StreamingAudioController<Scalar>` coordinates generated-frame
streaming. Store it in the model session; retain weights, decoder graphs, and
backend state in the existing model decoder. It does not select a backend or
change inference math.

## Lifecycle

Text-driven models can inherit `StreamingTtsSessionBase` instead of implementing
the streaming interface lifecycle repeatedly. It owns the event sink and final
result, rejects incoming audio chunks, and resets request state before generation,
after errors, and after finalization. Errors propagate to the caller. Sink
registration survives reset; it can be cleared with `set_stream_event_sink({})`.
Implement `generate_stream(request)` and `reset_stream_state()`; the latter clears
request state, not weights or cached graphs. Use `stream_event_sink()` when
starting the controller. This base is synchronous and does not apply to models
that require incoming audio chunks or asynchronous generation.

1. `begin(config, decode, sink, reference, reset_decoder, drain_decoder)` starts a
   request (or a text segment). The reference argument is optional, frame-major
   context, not generated audio. Only its requested left-context suffix is kept.
   An optional reset callback resets decoder state without discarding its graphs.
2. `push(frames)` accepts one or more complete frames. The controller buffers them
   until the next chunk and its required right context are available, calls the
   decoder, and emits an ordinary `StreamEvent` with `audio_output`.
3. `finish()` decodes the remaining frames, optionally drains a stateful decoder's
   waveform tail, and returns the concatenation of all emitted audio. Decoder and
   sink callbacks are released; cached model resources are unaffected.
4. `reset()` abandons request-local controller state. Start a new request with
   `begin`; the decoder-reset callback runs there. Exceptions from decoding or
   publishing propagate and discard controller state. Model generation failures
   outside `push` must also reset the controller before reuse.

All calls and callbacks are synchronous on the session thread. The controller
does not own threads, cancellation, weights, graph caches, or model loading.

## Policies

The default `Grow` policy emits 1, 2, 4, ... frames capped at `frames_per_chunk`.
`Fixed` emits that many frames from the first chunk. Fixed size 1 is single-frame
streaming, not a separate implementation. End-of-input can produce a shorter
chunk. These counts refer to generated frames, excluding reference and context.

`StreamingAudioConfig` defines frame width and left/right context sizes. The
controller uses bounded input buffering even when `push` receives an entire
diffusion segment. The accumulated final waveform still grows with output length,
as required by the existing final-result contract.

## Decoder Contract

The decode callback receives a borrowed `StreamingAudioWindow<Scalar>` valid only
during that call. Its frame-major data contains left context, new frames, then
right context. `first_frame` counts already emitted generated frames, excluding
reference context. Return only new waveform samples: trim context and graph
padding in the model decoder. Do not replay previous audio in the result.

`final` marks a last window flushed by `finish`. A full chunk may have been
published before end-of-input was known; use `drain_decoder` for mandatory
end-of-stream work rather than relying on the last decode having `final=true`.

The controller uses `append_audio_buffer` and the existing event contract; it
does not define a new server schema or replace `StreamingPolicy`.

## Adopters

- **Qwen3 TTS:** integer codec frames are pushed during AR generation. The
  controller retains reference/left context; the cached decoder trims it.

This helper does not make noncausal generation incremental. A model still needs
an explicit, correct partial-decoding boundary. LFM and PocketTTS are unchanged.
