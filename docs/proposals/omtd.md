# Output Multimodal Transfer/Decoder

OMTD is a proposed output-side companion library for llama.cpp. It is not an
available audio.cpp library or frontend feature. The former placeholder code
and test were removed; this document retains the design proposal.

The prototype is preserved at the
[pre-cleanup checkpoint](https://github.com/0xShug0/audio.cpp/tree/4363a483002df5b0c3eb83ae93addf0073de388c/tools/omtd).

It mirrors MTMD conceptually, but runs in the opposite direction:

```text
MTMD: media input -> embeddings/tokens -> language model
OMTD: language model output -> output companion -> generated media
```

The proposed library would define the output boundary. Model-specific output
implementations, such as audio, image, or video generators, would be layered on
top in separate patches.

## Intended Frontend Support

The intended frontend path is:

```text
llama-cli
llama-server
llama-tts
    |
    v
  libomtd
    |
    v
concrete output backend
```

Planned frontend responsibilities:

- `llama-cli`: accept `--omtd FILE` for local output generation from a prompt.
- `llama-server`: accept `--omtd FILE` and expose output-generation endpoints.
- `llama-tts`: use OMTD as the shared TTS/audio-output backend path.

These flags are proposed, not implemented. A future implementation would add the
shared library and API first, followed by frontend wiring and concrete backends.

## Review Path

Recommended patch order:

```text
1. OMTD core library and docs
2. Frontend wiring for llama-cli, llama-server, and llama-tts
3. First concrete OMTD backend
```

This keeps the generic output-runtime boundary reviewable without tying it to a
specific model implementation.

## Responsibilities

The proposed OMTD library would own:

- A public C API for output companion detection.
- A public C API for structured output generation requests.
- Common status/error reporting for output generation backends.
- Placeholder modality and model-type enums for future backends.

It would not own:

- Input media processing. That remains MTMD.
- `--mmproj` handling.
- A concrete output model in the initial library-only patch.
- Application-specific file or HTTP response handling.

## Proposed API

Detection:

```text
omtd_is_output_companion_gguf(path)
omtd_get_model_type(path)
omtd_get_model_modality(path)
```

Audio generation boundary:

```text
omtd_audio_generate_file(params, error, error_size)
```

The removed prototype only validated required fields and returned
`OMTD_STATUS_UNSUPPORTED`; it had no concrete audio backend.

## Diagram

```text
llama.cpp frontend
        |
        v
  OMTD public API
        |
        +--> companion detection
        |
        +--> structured output request
        |
        v
 model-specific output backend
   added by later patches
        |
        v
generated media output
```

## Relationship To MTMD

| Area | MTMD | OMTD |
| --- | --- | --- |
| Direction | input -> model | model -> output |
| Media role | user-provided input media | generated output media |
| File role | input projector | output companion |
| API style | tokenize/encode input | generate/decode output |
