# YuE2: change style in the middle of a song

YuE2 continues what it has already sung far more than it follows its style
prompt, so changing the prompt part-way through a song does little. Editing the
song's *history* does work: render one score in two styles, hand the model style
B's rendition as its past with the last few seconds of the real song on the end,
and let it continue. It keeps the singer's place in the lyrics from those seconds
and moves into style B over the next ten seconds or so.

`style_change.py` does this with `audiocpp_cli` alone, using two YuE2 request
options: `stop_after` and `semantic_prefix_file`
(see [docs/models/yue2.md](../../docs/models/yue2.md)).

```bash
python3 examples/yue2_style_change/style_change.py --backend cuda
# -> yue2_style_change_out/4_song.wav
```

It needs only the Python standard library. `--cli` and `--model` default to
`build/bin/audiocpp_cli` and `models/Yue2-3B-GGUF`; options the script does not
know (`--backend`, `--device`, `--threads`, ...) are passed to `audiocpp_cli`.
The default run is a storybook musical number that turns into death metal at its
second chorus, about three minutes of audio.

## What it runs

| Step | Request | Output |
|---|---|---|
| `1_score` (only with `--new-score`) | `stop_after=abc` | `score.abc` |
| `2_take_a`, `2_take_b` | `abc_file`, `stop_after=semantic`, one per style | `semantic.json` each |
| `3_leg_0` | `semantic_prefix_file=3_history_0.json`, stops at the next edit | tokens up to the cut |
| `3_leg_1` | `semantic_prefix_file=3_history_1.json` | tokens to the end |
| `4_song` | `semantic_prefix_file=4_song.json`, `semantic_min_tokens = semantic_max_tokens = N` | `4_song.wav` |

An edited history and the splice are two list operations on the token arrays.
`shift` is how many frames later style B's take plays the same moment of the score:

```python
history = take_b[:at + shift - keep] + song[at - keep:at]
song = song[:at] + leg[at + shift:]
```

The script edits the history twice. Ten seconds before the cut it keeps 30 s of
the real song: the music stays in style A, but the model has seen where it is
going and tends to play a lead-in. At the cut it keeps 5 s, and the style changes.
Only the final step renders audio; the borrowed history is never heard.

## Notes

- **The score decides how far the second style can go.** Tempo, groove and how
  busy the vocal line is all come from the score; the style prompt dresses what
  the score allows. On many scores a second style only changes the intro and the
  instrumental breaks. `score.abc` here was picked because death metal is audibly
  death metal on it. `--new-score` writes a fresh one from `lyrics.txt`, which
  may or may not leave room for a second style: render `2_take_b` to audio and
  listen before judging the change.
- Two renders of one score do not keep the same clock; on the bundled score the
  second take runs almost two seconds behind the first. `lag()` measures that from
  the tokens (at the right lag two takes share a few percent of identical tokens,
  at any other lag almost none) and the splice is shifted by it.
- The cut defaults to the second chorus, read from the score's section comments
  and bar counts. `--cut SECONDS` overrides it. A section where the vocal leaves
  gaps takes a change better than a wordy verse.
- Results are deterministic for a given seed. Change `--seed` and delete the
  `3_*` and `4_song*` outputs to re-roll only the transition.
- To hear a take on its own, render its tokens: `semantic_prefix_file` set to its
  `semantic.json`, and both `semantic_min_tokens` and `semantic_max_tokens` set
  to the number of tokens in it.
