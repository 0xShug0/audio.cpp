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

## Incremental ASR Sessions

`StreamingAsrSessionBase` provides one incoming-audio lifecycle for Parakeet TDT,
Phonon, SenseVoice, Higgs Audio STT, and Kroko ASR. Implement four hooks:

- `begin_stream(request)` configures a new request after the base resets it.
- `consume_audio(chunk)` processes incoming samples and submits recognition updates.
- `complete_stream(result)` flushes pending audio and attaches final-only metadata
  to the base-owned result, which already contains the committed transcript.
- `reset_stream_state()` clears request state while retaining reusable graphs
  and weights.

The base owns the accumulated result, UTF-8-safe delta publication, pending
events, active-stream checks, sink registration, and `session.wall_ms`.
`finish_stream()` and `finalize()` share the same completion path. A successful
finalization closes the stream; start another request before feeding more audio.
Sink registration survives reset. Exceptions propagate; a model must reset its
state before rethrowing if inference failure leaves it unusable. Validation
before mutation can leave the request active so the caller can retry a chunk.

Models use these publication methods without inspecting the sink:

- `publish_transcript(transcript, words, metadata)` submits the cumulative committed
  transcript and cumulative timestamps. The existing `PartialTextPublisher`
  derives the text delta; cumulative timestamp updates replace the pending list.
- `append_transcript(segment, words, include_snapshot)` submits newly committed
  text and disjoint word spans. The base accumulates the final result. The model
  supplies any separator appropriate to its text; the base does not guess language
  or punctuation rules. The optional snapshot includes the committed transcript.
- `publish_hypothesis(transcript)` replaces the provisional hypothesis without
  committing it or adding it to the final transcript.
- `publish_metadata(metadata)` delivers ASR metadata such as voice activity or
  an end-of-stream marker. `AsrStreamMetadata` cannot carry text or waveform
  outputs, so it cannot bypass transcript publication.

With a sink installed, updates are delivered immediately and the feed returns
an empty event. Otherwise the base combines committed deltas, retains the latest
snapshot, and collects metadata into the returned event. Models do not duplicate
this collection logic. Models can inspect `committed_transcript()` but cannot
mutate the accumulator during a feed. `complete_stream(result)` flushes through
the same publication methods and can finalize timestamps or attach model-specific
metadata. The base moves the complete result to the caller, without a copy.

`partial_text` contains append-only committed text. `partial_text_snapshot`
contains a revisable hypothesis; retain that distinction when a decoder revises
uncommitted words. Buffering, context windows, decoder caches, timestamp
publication, and commitment policy remain model-local. This base introduces no
new HTTP schema and does not make a noncausal encoder causal. Existing streaming
implementations are regression baselines, not proof that their chunking or event
behavior is correct.
