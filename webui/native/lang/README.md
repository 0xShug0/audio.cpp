# Native WebUI languages

English is built into `src/lib/i18n.ts` and is always the fallback language.

Add a UTF-8 JSON file named `lang_<code>.json` to this directory to add another
language to the selector. The file is discovered automatically by Vite:

```json
{
  "code": "de",
  "name": "Deutsch",
  "translations": {
    "nav.models": "Modelle",
    "run.run": "Starten"
  }
}
```

Translation keys that are omitted fall back to English. Rebuild the native WebUI
and `audiocpp_server` after adding or changing a language file; the language
catalog is bundled into the embedded single-file interface.
