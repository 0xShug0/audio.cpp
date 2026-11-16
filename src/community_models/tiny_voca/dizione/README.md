# dizione (C++)

English text to phonemes: the C++ port of [dizione](https://github.com/n4ze3m/dizione),
the phonemizer of tiny-voca. For the same data files it gives the same phoneme
ids as the Python package.

This folder stands alone. It uses the C++17 standard library and nothing else,
and nothing in it includes audio.cpp code. The model's `frontend.cpp`, one
folder up, connects it to the framework.

```cpp
#include "dizione.h"

dizione::Phonemizer phonemizer(dizione::load_directory("/path/to/phonemizer"));
dizione::Phonemized result = phonemizer.phonemize("Hello, it costs $3.50.");
// result.symbols, result.ids, result.stress
```

`load_directory` reads `lexicon.tsv`, `align.json`, `g2p.json`, `g2p.npz` and
`homographs.json`, the files under `phonemizer/` in the
[tiny-voca model repository](https://huggingface.co/n4ze3m/tiny-voca). A
`Phonemizer` can also be made from the contents of those files (`dizione::Data`),
which is how audio.cpp gives it the files stored in a GGUF.

One `Phonemizer` serves one thread at a time. It keeps the dictionary as the
text of the file with an index of four bytes a word, so the 125,000 words
take 4 MB; a process that only phonemizes peaks at 12 MB.

## Files

| File | What it holds | Python file |
|---|---|---|
| `text.*`, `unicode.cpp` | Strings as Python has them: code points, `str` methods, NFKC/NFKD | |
| `pyre.*` | A matcher for the regular expressions the rules use, behaving like Python's `re` | |
| `json.*`, `data.cpp` | Readers for the data files | |
| `symbols.*` | The 76 symbols and the token sequence | `symbols.py` |
| `tables.*` | The hand-written tables | `tables.py`, `data/*.txt` |
| `normalize*.cpp` | Text rules: numbers, dates, money, units, codes, addresses | `normalize.py` |
| `convert.*`, `align.*` | CMUdict pronunciation to the model's phones | `convert.py`, `align.py` |
| `morphology.*` | Endings, prefixes, compounds | `morphology.py` |
| `homographs.*` | "read", "live", "record": which reading is meant | `homographs.py` |
| `g2p.*` | The model for unknown words | `g2p.py` |
| `sentence.*` | Sentence stress, linking | `sentence.py` |
| `engine.cpp`, `dizione.h` | The phonemizer | `engine.py` |

The patterns in `normalize*.cpp` are the Python ones, word for word. `pyre`
stops a match that takes more than five million steps (`StepLimit`), so that no
text can keep a server busy; Python has no such limit.

## Generated files

`unicode_tables.inc` and `tables_data.inc` are written by
`tools/community_models/tiny_voca_tables.py`. The first holds Unicode data read
from the Python that runs the script, so the C++ code answers as that Python
does; the second holds dizione's `data/*.txt` at one commit. Run the script
again after a change to dizione's tables.

## Checks

`tests/community_models/tiny_voca/reference.py` writes what Python gives and
`dizione_probe` compares the C++ code with it: string methods, every pattern,
and phoneme ids for any text file.
