# Built-in recipes

Recipes are ready-to-run compositions of existing audio.cpp models. Unlike
`--pipeline`, a recipe does not require a JSON workflow file. The recipe defines
the stages and their data flow; the caller only selects one model for each role
and supplies the normal task inputs.

List the available recipes; the server response also includes their model roles:

```bash
audiocpp_cli --list-recipes
curl http://127.0.0.1:8080/v1/recipes
```

## Any TTS voice cloning

`any-tts` generates speech with an offline TTS model and converts each generated
chunk to the supplied reference voice with Tone Color VC. The two sessions are
created once and reused for every text chunk.

```bash
audiocpp_cli --recipe any-tts --backend cuda --log \
  --recipe-model tts=/path/to/kokoro.gguf \
  --recipe-family tts=kokoro_tts \
  --recipe-model converter=/path/to/tone-color-vc.gguf \
  --recipe-family converter=tone_color_vc \
  --voice-id am_michael \
  --voice-ref reference.wav \
  --text "The next train leaves in ten minutes." \
  --request-option converter.seed=42 \
  --out cloned.wav
```

`text_chunk_size` and `text_chunk_mode` control recipe-level text chunking.
Unprefixed request options and `tts.*` options go to the TTS stage;
`converter.*` options go to Tone Color VC. Role-prefixed options override shared
options with the same name.
Prefix CLI load and session option keys with their role too, retaining any
model-family prefix after the role prefix.

## Cross-language dubbing

`cross-language-dubbing` translates input speech and sends the translated text
directly to a target-language TTS model. Canary 180M Flash is a compact choice
for its supported English, German, Spanish, and French pairs. OWSM and OWSM-CTC
expose the same translation interface for broader language coverage.

```bash
audiocpp_cli --recipe cross-language-dubbing --backend cuda --log \
  --recipe-model translator=/path/to/canary-180m-flash-f32.gguf \
  --recipe-family translator=canary_asr \
  --recipe-model tts=/path/to/kokoro.gguf \
  --recipe-family tts=kokoro_tts \
  --audio german-speech.wav \
  --request-option translator.language=de \
  --request-option translator.target_language=en \
  --voice-id am_michael \
  --out english-dub.wav \
  --text-out translation.txt
```

Prefix translation options with `translator.*`. Unprefixed request options and
`tts.*` options go to the TTS stage. The translation target is required because
omitting it would turn the recipe back into same-language transcription and
re-speaking. Translator and TTS language identifiers are model-specific, so
the recipe does not infer one from the other.

## Server

Recipes use the default legacy server runtime, not the experimental parallel
runtime. They use model IDs from the normal server configuration, so model
loading, lazy loading, locking, and unloading retain their existing behavior.
The request maps recipe roles to those IDs:

```bash
curl http://127.0.0.1:8080/v1/recipes/run \
  -H "Content-Type: application/json" \
  -d '{
    "recipe": "cross-language-dubbing",
    "models": {"translator": "canary", "tts": "kokoro"},
    "request": {
      "audio": "input.wav",
      "voice_id": "am_michael",
      "options": {
        "translator.language": "de",
        "translator.target_language": "en"
      }
    }
  }'
```

Set `"stream": true` to receive recipe output events and the final result over
SSE. Recipe components must be configured for the offline task and mode shown by
`GET /v1/recipes`.

## Adding a recipe plugin

Recipe implementations are isolated source plugins under
`app/recipe/plugins/`. A plugin implements `IRecipe` and declares one
`RecipePluginRegistration` factory in its own translation unit. The build
discovers plugin sources in that directory, so adding one does not require
editing the recipe registry, CLI, server, or another recipe.

The `IRecipeModelExecutor` boundary gives every plugin the same model execution
surface. `roles()` declares the required task and mode for each component, and
`run()` connects their in-memory `TaskRequest` and `TaskResult` values. Use
`options_for_recipe_role()` to isolate role-prefixed options. Source plugins are
compiled into the CLI and server; loading arbitrary binary plugins at runtime
is intentionally outside this interface.
