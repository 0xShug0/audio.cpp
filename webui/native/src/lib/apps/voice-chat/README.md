# Voice chat

Choose an OWSM v4 Base 102M package and a built-in voice. The app automatically uses installed
LFM3-Audio for conversation and voice synthesis and SoulX-FlashHead for portrait
animation. The app uses its own model selections
and conversation state, independently of Studio.

LFM3 voice chat currently supports English only. Its multilingual ASR and TTS
capabilities do not imply multilingual conversation support.

## Conversation flow

Choose a built-in LFM3 voice and upload a portrait. The portrait preview shows
its square center crop. The optional **Personality** field sets how LFM3 should
respond; changing it starts a new conversation context. Press **Talk**, then **Finish** to send a question, or
upload an audio recording. Microphone recording requires localhost or HTTPS.

OWSM v4 Base 102M provides the user's transcript with automatic language detection.
LFM3 receives the original recording, not that transcript. The app restricts
recognition to this OWSM model.

Recognition runs alongside the `voice-chat-avatar` recipe:

1. Conversation text appears as LFM3 generates it.
2. A bounded text queue feeds phrases into TTS in the selected built-in voice.
   Phrase boundaries use the framework's UTF-8/tag-aware text chunker.
3. A bounded PCM queue and stateful resampler feed the synthesized speech into
   one continuous SoulX stream at 16 kHz.
4. The browser plays matching audio and animation blocks as they arrive, while
   conversation generation and synthesis continue.

Playback does not wait for the complete reply text or synthesized WAV. LFM3's
native S2S reply artifacts preserve conversation history; its native audio is
not played. The complete selected-voice audio remains available for replay.

**Speaking speed** defaults to 1×. Faster settings preserve pitch and process PCM
before both playback and avatar generation, so animation follows the adjusted
speech. Speed adjustment requires libsonic on the server; normal speed does not
load it. The processor keeps state across chunks and flushes once per reply.

Audio is the playback clock. If generation falls behind, audio and animation
wait together rather than drifting apart. Generation RTF includes recognition,
conversation, voice synthesis and avatar generation. It excludes recording time
and the time spent finishing playback after generation completes.

**New chat** clears messages and conversation artifacts. Changing the
conversation model starts a new context. Voice and recognition choices can
change between turns. **Stop** aborts pending browser requests and scheduled
playback; a server-side kernel already running may finish its current operation.
Failed turns are not added to subsequent model context. Conversations remain in
memory for the app instance; no new browser database is created.

## Server setup

Use [the example config](../../../../../../examples/server/voice-chat-avatar.json)
and replace its placeholder model paths with installed GGUFs. Register four
model entries: ASR, LFM3 S2S, LFM3 TTS and SoulX. Both LFM3 entries must point to
the same checkpoint. Conversation, TTS and avatar entries must use streaming
mode and separate model IDs. The server must allow all four sessions to remain
loaded. `lazy_load: false` loads them before serving requests.

Model loading uses the shared ModelRuntime. Managed roles have their own IDs;
configured server entries retain their declared IDs and modes. Different TTS
and avatar sample rates require libsoxr for stateful resampling.

Run from the repository root:

```bash
build/debug/bin/audiocpp_server --config examples/server/voice-chat-avatar.json \
  --log > voice-chat.log 2>&1
```

Open **Apps → Voice chat**, select recognition, voice and portrait, then press
**Talk**. LFM3-Audio and SoulX-FlashHead are fixed for this app and are selected
automatically.

The example selects SoulX's cuDNN VAE implementation and requires a binary
built with cuDNN Conv3D support. Follow
[the SoulX build instructions](../../../../../../docs/models/soulx_flashhead.md).
For a build without cuDNN, change the VAE implementation to `ggml` in the
`soulx_flashhead.exec_plan` session option.

## Browser checks

With the built UI served locally and Playwright installed, run from the
repository root:

```bash
UI_BASE_URL=http://127.0.0.1:8080 \
  node webui/native/src/lib/apps/voice-chat/tests/workflow.mjs
UI_MANAGED_CASE=1 UI_BASE_URL=http://127.0.0.1:8080 \
  node webui/native/src/lib/apps/voice-chat/tests/workflow.mjs
```

`UI_BROWSER_TOOLS` and `UI_BROWSER_CHANNEL` work as in the shared UI checks.
The fixtures cover configured and managed model routing, voice selection,
conversation artifacts, session reuse, new chat, failure recovery, microphone
input, cancellation and synchronized playback. They also verify that playback
starts while reply text is still incomplete. They do not load models or measure
model accuracy or performance.
