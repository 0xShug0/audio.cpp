#!/usr/bin/env python3
"""Golden numerical fixtures for the lfm3_audio C++ port (LFM3-Audio-2B, HF remote code).

  make_fixtures.py --model <model dir> --assets <dir with c.wav, speech.wav, a.wav> --out <fixture dir>

Reference = HF model in float32 on CUDA (sdpa, TF32 off). Key stages are repeated with a bf16 copy of the model
to measure the bf16-vs-fp32 noise floor (manifest["noise_floor"]).
Every array is a .npy file listed in <out>/manifest.json with shape, dtype, meaning and inputs.
The fixtures are derived from the model weights: keep them out of git.
"""
import argparse
import contextlib
import json
import os
import time

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, DynamicCache
from transformers.dynamic_module_utils import get_class_from_dynamic_module

_args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_args.add_argument("--model", required=True)
_args.add_argument("--assets", required=True)
_args.add_argument("--out", required=True)
ARGS = _args.parse_args()
MID, ASSETS, OUT = ARGS.model, ARGS.assets, ARGS.out
DEV = "cuda"
ORDER_SEED = 20261024  # numpy default_rng seed of the injected MaskGIT orders (reset per run)
S2S_ORDER_SEED = 3  # with this seed greedy S2S ends with an EOS frame + <|im_end|>; ORDER_SEED never emits EOS
MAX_LOGIT_FRAMES = 40  # depth logits are stored for the first frames of each generation only
# generate() takes the speech sampling it is not given from generation_config.json, which samples, so greedy
# speech is asked for.
GREEDY = {"audio_temperature": 0.0, "audio_repetition_penalty": 1.0}

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

AUDIO_TOKEN, AUDIO_OUT_TOKEN, EOS_CODE = 125031, 125032, 2016
IM_END, NEWLINE = 124900, 207
S2S_SYSTEM = "Respond with interleaved text and audio."
M = {"meta": {}, "inputs": {}, "prompts": {}, "arrays": {}, "noise_floor": {}, "results": {}}


def np_of(x):
    if torch.is_tensor(x):
        x = x.detach().cpu()
        x = x.float() if x.is_floating_point() else x
        x = x.numpy()
    x = np.asarray(x)
    if x.dtype.kind in "iub":
        return x.astype(np.int32)
    return x.astype(np.float32) if x.dtype != np.float32 else x


def save(name, x, meaning, **inputs):
    a = np.ascontiguousarray(np_of(x))
    path = os.path.join(OUT, name + ".npy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, a)
    M["arrays"][name] = {"file": name + ".npy", "shape": list(a.shape), "dtype": str(a.dtype), "meaning": meaning,
                         "inputs": inputs}
    return a


def rel_rms(a, b):
    a, b = np_of(a).astype(np.float64), np_of(b).astype(np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(b ** 2)), 1e-30))


def floor(stage, bf, fp, **extra):
    bf_, fp_ = np_of(bf).astype(np.float64), np_of(fp).astype(np.float64)
    d = {"rel_rms": rel_rms(bf_, fp_), "max_abs": float(np.abs(bf_ - fp_).max()), "ref_rms": float(np.sqrt(np.mean(fp_ ** 2)))}
    d.update(extra)
    M["noise_floor"][stage] = d
    print(f"  floor {stage}: {d}", flush=True)


def first(o):
    return o[0] if isinstance(o, (tuple, list)) else o


@contextlib.contextmanager
def capture(mods):
    """Records the first output of each named module."""
    got = {}
    hooks = [m.register_forward_hook(lambda mod, a, o, k=k: got.setdefault(k, first(o).detach().clone())) for k, m in mods.items()]
    try:
        yield got
    finally:
        for h in hooks:
            h.remove()


@contextlib.contextmanager
def frame_recorder(model, inject_seed):
    """Wraps sample_audio_frame: injects the MaskGIT order and records the depth decoder passes."""
    rec = {"hidden": [], "order": [], "codes": [], "pass_codes": [], "pass_masked": [], "pass_logits": []}
    rng = np.random.default_rng(inject_seed)
    orig_sample, orig_randperm = model.sample_audio_frame, torch.randperm
    passes = []

    def on_depth(mod, args, out):
        _, codes, masked = args
        passes.append((codes[0].detach().clone(), masked[0].detach().clone(), out[0].detach().float().clone()))

    def sample(hidden_state, *a, **kw):
        assert hidden_state.ndim == 1, "batch 1 only"
        order = torch.as_tensor(rng.permutation(12), dtype=torch.long)
        torch.randperm = lambda n, device=None, **_: order.to(device)
        passes.clear()
        try:
            codes = orig_sample(hidden_state, *a, **kw)
        finally:
            torch.randperm = orig_randperm
        rec["order"].append(order.clone())
        assert len(passes) == 3
        rec["hidden"].append(hidden_state.detach().float().cpu().clone())
        rec["codes"].append(codes.detach().cpu().clone())
        rec["pass_codes"].append(torch.stack([p[0] for p in passes]).cpu())
        rec["pass_masked"].append(torch.stack([p[1] for p in passes]).cpu())
        rec["pass_logits"].append(torch.stack([p[2] for p in passes]).cpu())
        return codes

    h = model.model.depth_decoder.register_forward_hook(on_depth)
    model.sample_audio_frame = sample
    try:
        yield rec
    finally:
        h.remove()
        del model.sample_audio_frame  # back to the class method
        torch.randperm = orig_randperm


@contextlib.contextmanager
def text_logit_recorder(model, k=20):
    rec = []
    h = model.lm_head.register_forward_hook(
        lambda mod, a, o: rec.append(torch.topk(o.detach().float().reshape(-1), k)) if o.ndim == 1 else None)
    try:
        yield rec
    finally:
        h.remove()


def slots_of(out, prompt_len):
    """Per generated slot: kind (0 text, 1 audio), token id (125032 for frames) and codes (-1 for text)."""
    seq = out.sequences[0, prompt_len:].tolist()
    codes = out.audio_codes[0].cpu()
    kinds, toks, frames, j = [], [], [], 0
    for t in seq:
        kinds.append(int(t == AUDIO_OUT_TOKEN))
        toks.append(t)
        if t == AUDIO_OUT_TOKEN:
            frames.append(codes[:, j].tolist())
            j += 1
        else:
            frames.append([-1] * 12)
    assert j == int(out.audio_lengths[0])
    pattern, i = [], 0
    while i < len(seq):
        e = i
        while e < len(seq) and kinds[e] == kinds[i]:
            e += 1
        pattern.append(f"A{e - i}" if kinds[i] else f"T{e - i}")
        i = e
    return np.array(kinds), np.array(toks), np.array(frames), " ".join(pattern)


def multiturn(model, proc):
    """Turn 2 of an S2S conversation (user audio a.wav) after turn 1 = the gen/s2s_c_greedy(_trunc) slots.

    The fixture is the re-prefill of the replayed history: turn-1 prompt + turn-1 slots exactly as
    generated (+ an EOS frame when the cut turn ends without one, + <|im_end|> when the turn was cut) + "\\n" + the
    turn-2 user block and <|mixed_start|>. It is checked against HF cache continuation (turn 1 regenerated with
    past_key_values, then only the suffix fed) and against the HF chat template rendering the same history."""
    tok = proc.tokenizer
    a = librosa.load(f"{ASSETS}/a.wav", sr=16000, mono=True)[0]
    sf.write(f"{OUT}/audio/a_16k.wav", a, 16000, subtype="FLOAT")
    assert np.array_equal(sf.read(f"{OUT}/audio/a_16k.wav", dtype="float32")[0], a)
    M["inputs"]["a"] = {"file": "audio/a_16k.wav", "samples": int(len(a)), "sample_rate": 16000,
                        "source": f"{ASSETS}/a.wav librosa.load(sr=16000, mono=True, soxr_hq)"}
    c_path = f"{OUT}/audio/c_16k.wav"
    c = sf.read(c_path, dtype="float32")[0]
    conv1 = [{"role": "system", "content": S2S_SYSTEM}, {"role": "user", "content": [{"type": "audio", "audio": c_path}]},
             {"role": "assistant", "content": "<|mixed_start|>"}]
    inp1 = proc.apply_chat_template(conv1, continue_final_message=True, tokenize=True, return_dict=True,
                                    return_tensors="pt").to(DEV)
    ids1 = inp1["input_ids"][0].tolist()
    assert ids1 == np.load(f"{OUT}/prompt/s2s_c_ids.npy").tolist(), "turn-1 prompt differs from prompt/s2s_c_ids"
    tail = [{"role": "user", "content": [{"type": "audio", "audio": f"{OUT}/audio/a_16k.wav"}]},
            {"role": "assistant", "content": "<|mixed_start|>"}]
    # the template's "<|im_end|>\n" after turn 1, then the turn-2 blocks (an empty system block renders nothing)
    suffix_text = "\n" + proc.apply_chat_template([{"role": "system", "content": ""}] + tail, continue_final_message=True,
                                                  tokenize=False)[len(tok.bos_token):]
    assert suffix_text.startswith("\n<|im_start|>user\n"), suffix_text[:40]

    def encode(text, clips):
        b = proc(text=text, audio=clips, return_tensors="pt")
        return {k: b[k].to(DEV) for k in ("input_ids", "input_features", "feature_attention_mask") if k in b}

    def fwd(ids, feats, codes, cache):
        past = cache.get_seq_length()
        L = ids.shape[1]
        with torch.no_grad():
            o = model.model(input_ids=ids, input_features=feats.get("input_features"),
                            feature_attention_mask=feats.get("feature_attention_mask"), audio_out_codes=codes,
                            attention_mask=torch.ones(1, past + L, dtype=torch.long, device=DEV),
                            position_ids=torch.arange(past, past + L, device=DEV)[None], past_key_values=cache, use_cache=True)
        return o.last_hidden_state[0, -1].float(), model.lm_head(o.last_hidden_state[:, -1])[0].float()

    suf = encode(suffix_text[1:], [a])["input_ids"][0].tolist()  # the "\n" is appended with [NEWLINE]
    both = encode("<|audio|><|audio|>", [c, a])  # features of both clips, batched as the processor does
    for tag, gen_tag, order_seed, max_new in (("multiturn_s2s_turn2", "s2s_c_greedy", S2S_ORDER_SEED, 1024),
                                              ("multiturn_s2s_trunc_turn2", "s2s_c_greedy_trunc", ORDER_SEED, 96)):
        gen = np.load(f"{OUT}/gen/{gen_tag}_slot_token.npy").tolist()
        kinds = np.load(f"{OUT}/gen/{gen_tag}_slot_kind.npy")
        frames = np.load(f"{OUT}/gen/{gen_tag}_slot_codes.npy")[kinds == 1].tolist()
        appended = []
        if gen[-1] != IM_END:
            if frames and frames[-1][0] != EOS_CODE:
                appended.append(AUDIO_OUT_TOKEN)
                frames.append([EOS_CODE] * 12)
            appended.append(IM_END)
        ids = ids1 + gen + appended + [NEWLINE] + suf
        codes = torch.tensor(frames, device=DEV)
        # (b) re-prefill of the whole conversation: the fixture
        hb, lb = fwd(torch.tensor([ids], device=DEV), both, codes, DynamicCache(config=model.config.text_config))
        # (a) cache continuation: regenerate turn 1 with a caller cache, then feed only what follows the slots
        cache = DynamicCache(config=model.config.text_config)
        with frame_recorder(model, order_seed), torch.no_grad():
            o = model.generate(**inp1, past_key_values=cache, return_audio_eos=True, max_new_tokens=max_new, **GREEDY)
        _, toks, fr, _ = slots_of(o, len(ids1))
        regenerated_equal = toks.tolist() == gen and np.array_equal(fr, np.load(f"{OUT}/gen/{gen_tag}_slot_codes.npy"))
        cache_len = cache.get_seq_length()
        sa = torch.tensor([appended + [NEWLINE] + suf], device=DEV)
        a_feats = encode(suffix_text[1:], [a])
        ha, la = fwd(sa, a_feats, codes[-1:] if AUDIO_OUT_TOKEN in appended else None, cache)
        # (c) chat-template re-render: assistant content = prefill + decoded slots (+ the EOS frame placeholder)
        body = gen[:-1] if gen[-1] == IM_END else gen + appended[:-1]
        content = "<|mixed_start|>" + tok.decode(body, skip_special_tokens=False)
        conv = conv1[:2] + [{"role": "assistant", "content": content}] + tail
        text = proc.apply_chat_template(conv, continue_final_message=True, tokenize=False)
        ids_tpl = encode(text, [c, a])["input_ids"][0].tolist()
        top = torch.topk(lb, 20)
        save(f"gen/{tag}_prompt_ids", np.array(ids), "turn-2 prompt: turn-1 prompt + turn-1 slots (+ EOS frame + <|im_end|> "
             "if cut) + \\n + user a.wav + <|mixed_start|>", turn1=f"gen/{gen_tag}_slot_*", audio=["audio/c_16k.wav", "audio/a_16k.wav"])
        save(f"gen/{tag}_last_logits", lb, "last-position logits of the turn-2 prompt, re-prefilled [128000]",
             prompt=f"gen/{tag}_prompt_ids")
        save(f"gen/{tag}_top20_ids", top.indices, "top-20 ids of last_logits", prompt=f"gen/{tag}_prompt_ids")
        r = {"turn1": gen_tag, "prompt_len": len(ids), "appended_after_turn1": appended + [NEWLINE],
             "turn1_regenerated_with_cache_equal_fixture": bool(regenerated_equal), "cache_len_after_turn1": int(cache_len),
             "cache_covers_prompt_plus_slots": cache_len == len(ids1) + len(gen),
             "cache_continuation_vs_reprefill": {"logits_max_abs": float((la - lb).abs().max()),
                                                 "hidden_max_abs": float((ha - hb).abs().max()),
                                                 "hidden_rel": float((ha - hb).norm() / hb.norm()),
                                                 "argmax_same": int(la.argmax()) == int(lb.argmax()),
                                                 "top20_same": torch.topk(la, 20).indices.tolist() == top.indices.tolist()},
             "template_render_ids_equal": ids_tpl == ids}
        if ids_tpl != ids:
            i = next((k for k in range(min(len(ids), len(ids_tpl))) if ids[k] != ids_tpl[k]), min(len(ids), len(ids_tpl)))
            r["template_first_diff"] = {"pos": i, "template": ids_tpl[i:i + 8], "replay": ids[i:i + 8]}
        M["results"][tag] = r
        print(f"  {tag}: {r}", flush=True)


def dump_probe(name, obj, indent=1):
    os.makedirs(f"{OUT}/probes", exist_ok=True)
    with open(f"{OUT}/probes/{name}", "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def tokenizer_cases(proc):
    """probes/tokenizer_cases.json: ids, tokens and decodes of HF's tokenizer (as the processor loads it) for a regression set."""
    tok = proc.tokenizer
    added = sorted(tok.added_tokens_decoder.items())  # (id, AddedToken)
    lang = {
        "arabic": ["مرحبا، كيف حالك اليوم؟", "القطار التالي يغادر خلال عشر دقائق.", "أنا أحب القراءة ١٢٣٤٥"],
        "danish": ["Hej, hvordan går det i dag?", "Rødgrød med fløde på ærøskøbing.", "Næste tog kører om ti minutter."],
        "german": ["Guten Tag, wie geht's?", "Die Straße ist größer als gedacht.", "Donaudampfschifffahrtsgesellschaftskapitän"],
        "greek": ["Γειά σου κόσμε!", "Το επόμενο τρένο φεύγει σε δέκα λεπτά.", "Ψυχή, ἀρετή, ὕδωρ"],
        "british_english": ["Colour, favour and organisation.", "I'll queue at the centre, mate.", "WHAT'S HE DOING? THEY'LL SEE."],
        "american_english": ["The quick brown fox jumps over the lazy dog", "I'm sure you've heard: it's 3:45 p.m.", "Don't, won't, can't, shouldn't've"],
        "finnish": ["Hyvää huomenta, mitä kuuluu?", "Lentokonesuihkuturbiinimoottoriapumekaanikkoaliupseerioppilas", "Seuraava juna lähtee kymmenen minuutin päästä."],
        "french": ["Bonjour, ça va ? C'est l'été !", "L'œuvre d'art coûte 1 234,56 €.", "Qu'est-ce que c'est ?"],
        "hebrew": ["שלום, מה שלומך?", "הרכבת הבאה יוצאת בעוד עשר דקות.", "בְּרֵאשִׁית בָּרָא אֱלֹהִים"],
        "italian": ["Buongiorno, come stai?", "Perché l'università è chiusa?", "Il prossimo treno parte tra dieci minuti."],
        "japanese": ["こんにちは、元気ですか？", "次の電車は十分後に出発します。", "カタカナとひらがなと漢字、ｶﾀｶﾅ半角", "東京タワーは333メートルです。"],
        "korean": ["안녕하세요, 잘 지내세요?", "다음 기차는 10분 후에 출발합니다.", "한국어 자모: ㄱㄴㄷ 가"],
        "dutch": ["Goedemorgen, hoe gaat het?", "De volgende trein vertrekt over tien minuten.", "'s Ochtends en 't is koud."],
        "polish": ["Dzień dobry, jak się masz?", "Zażółć gęślą jaźń.", "Następny pociąg odjeżdża za dziesięć minut."],
        "portuguese": ["Olá, tudo bem? Não sei.", "O próximo comboio parte em dez minutos.", "Ação, coração, pão."],
        "spanish": ["¡Hola! ¿Cómo estás?", "El niño pequeño comió piña.", "El próximo tren sale en diez minutos."],
        "swedish": ["Hej, hur mår du?", "Nästa tåg avgår om tio minuter.", "Smörgåsbord och blåbär."],
        "turkish": ["Merhaba, nasılsın?", "İstanbul'da ılık bir gün. IĞDIR ığdır", "Bir sonraki tren on dakika içinde kalkıyor."],
        "chinese": ["你好，世界！", "下一班火车十分钟后出发。", "繁體中文與简体中文混合", "数字12345 can't"],
    }
    cases = []
    def add(cat, s):
        cases.append({"category": cat, "text": s})

    for l, ss in lang.items():
        for s in ss:
            add("lang:" + l, s)

    for s in ["1", "12", "123", "1234", "12345", "1234567890", "3.14159", "1,000,000", "2026-10-24", "+1 (555) 010-9999",
              "٣٤٥٦", "１２３４５", "a1b22c333d4444", "007", "1st 2nd 3rd 4th", "0x1F600", "100%", "$5.99", "-273.15°C"]:
        add("digits", s)

    for s in ["IT'S", "it's", "It'S", "HE'LL", "he'll", "THEY'RE", "we've", "WE'VE", "I'M", "i'd", "I'D", "don't", "DON'T",
              "'s", "'S", "'ll", "'LL", "o'clock", "rock'n'roll", "The cat's toy's colour", "l’homme", "it’s (curly)", "''quoted''"]:
        add("contractions", s)

    for s in ["a\r\nb", "a\r\n\r\nb", "\r\n", "\r\n\r\n\r\n", "line1\nline2\n", "\n\n\n", "a\rb", "x \r\n y", "end.\r\n",
              "  \n  \n", "\n", " \n", "\n "]:
        add("newlines", s)

    for s in [" leading", "trailing ", "  two leading", "two trailing  ", " ", "  ", "   ", "a  b", "a   b", "a    b",
              " Hello  world\n\nfoo", "\u00a0nbsp\u00a0", "ideographic\u3000space", "zero\u200bwidth"]:
        add("spaces", s)

    for s in ["\t", "a\tb", "\t\tindented", "col1\tcol2\tcol3", "\t \n\t", "def f():\n\treturn 1"]:
        add("tabs", s)

    for s in ["😀", "👍🏽", "👨\u200d👩\u200d👧\u200d👦", "🇯🇵🇺🇸", "I ❤\ufe0f NY", "emoji😀in😀text", "✓ ✗ ★ ☆", "🙂🙂🙂", "©®™", "→←↑↓"]:
        add("emoji", s)

    for s in ["e\u0301", "café vs cafe\u0301", "n\u0303", "a\u0300\u0301\u0302", "Z\u0351\u0352\u0353algo", "क\u094dष",
              "ﬁ ligature", "Å vs A\u030a", "ǅ title"]:
        add("combining", s)

    for s in ["<function=x>", "<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>", "<think>", "</think>",
              "<think>I should answer.</think>The answer is 4.", "<think></think>", "<tool_call>\n<function=f>\n</function>\n</tool_call>",
              "<|im_start|>user\nhi<|im_end|>\n", "<|im_end|><|im_end|>", "text<|text_end|>", "<|audio|><|audio|>",
              "<|audio_out|>x", "a<|audio_start|>b", "<|mixed_start|>Hello<|text_end|>", "<|startoftext|>", "<|im_start|",
              "<|im_end", "< |im_end|>", "<|IM_END|>", "<think >", "<thinking>", "</think >", "<tools>[]</tools>", "<|pad|>"]:
        add("special_in_text", s)

    for _, t in added:
        add("added_token_embedded", f"pre {t.content} post")
        add("added_token_bare", t.content)

    for s in ["Pneumonoultramicroscopicsilicovolcanoconiosis", "a" * 100, "ab" * 60, "supercalifragilisticexpialidocious!",
              "https://example.com/path?query=1&b=2#frag", "user@example.com", "snake_case_identifier_name", "CamelCaseIdentifierName",
              "x" * 300, "ﷺ", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢", "\u0000nul", "\x7fdel", "\ufeffbom", "\U0010ffff"]:
        add("long_or_odd", s)

    for s in ["", "Hello.", "Hello there.", "Perform TTS in American English using a female voice.", "Perform ASR in English.",
              "Respond with interleaved text and audio.", "Perform TTS in Japanses using a male voice.", "Read this with a female voice in Korean."]:
        add("prompts", s)

    out = []
    for c in cases:
        ids = tok(c["text"], add_special_tokens=False)["input_ids"]
        ids_sp = tok(c["text"], add_special_tokens=True)["input_ids"]
        out.append({**c, "ids": ids,
                    "decode_keep_special": tok.decode(ids, skip_special_tokens=False),
                    "decode_skip_special": tok.decode(ids, skip_special_tokens=True),
                    "tokens": tok.convert_ids_to_tokens(ids),
                    "add_special_tokens_changes_ids": ids_sp != ids})
    meta = {"tokenizer_class": type(tok).__name__, "len_tokenizer": len(tok), "n_added": len(added),
            "added_tokens": [{"id": i, "content": t.content, "special": t.special} for i, t in added],
            "loaded_via": "AutoProcessor.from_pretrained(model, trust_remote_code=True).tokenizer; tok(text, add_special_tokens=False)",
            "n_cases": len(out)}
    dump_probe("tokenizer_cases.json", {"meta": meta, "cases": out})
    print(f"  tokenizer_cases: {len(out)} cases", flush=True)


def ignore_merges_cases():
    """probes/ignore_merges_cases.json: every base-vocab string whose ids differ between tokenizer.json's
    ignore_merges=true (as shipped) and false. Ids and tokens are the shipped ones."""
    from tokenizers import Tokenizer
    with open(f"{MID}/tokenizer.json") as f:
        j = json.load(f)
    shipped = Tokenizer.from_str(json.dumps(j))
    j["model"]["ignore_merges"] = False
    merges_only = Tokenizer.from_str(json.dumps(j))
    n = len(j["model"]["vocab"])
    cases = []
    for i in range(n):
        t = shipped.decode([i])
        ea, eb = shipped.encode(t, add_special_tokens=False), merges_only.encode(t, add_special_tokens=False)
        if ea.ids != eb.ids:
            cases.append({"text": t, "ids": ea.ids, "tokens": ea.tokens, "ids_without_ignore_merges": eb.ids})
    dump_probe("ignore_merges_cases.json", {"meta": {"vocab": n, "n_cases": len(cases), "source": "make_fixtures.py"}, "cases": cases}, None)
    print(f"  ignore_merges_cases: {len(cases)} of {n}", flush=True)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    M["meta"] = {"model": MID, "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                 "librosa": librosa.__version__, "device": torch.cuda.get_device_name(0),
                 "reference": "fp32 weights (upcast from bf16 checkpoint), attn sdpa, TF32 off",
                 "order_seed": ORDER_SEED, "max_logit_frames": MAX_LOGIT_FRAMES,
                 "rel_rms": "sqrt(mean((bf16-fp32)^2))/sqrt(mean(fp32^2))"}

    wavs = {}
    for name in ("c", "speech"):
        y, _ = librosa.load(f"{ASSETS}/{name}.wav", sr=16000, mono=True)  # res_type soxr_hq, like the HF processor
        wavs[name] = y
    sr = 16000
    rs = np.random.default_rng(7)
    for name, n in (("syn_1s", 16000), ("syn_037s", 5921)):
        t = np.arange(n) / sr
        y = 0.3 * np.sin(2 * np.pi * (200 * t + 900 * t ** 2)) + 0.05 * rs.standard_normal(n)
        wavs[name] = y.astype(np.float32)
    os.makedirs(f"{OUT}/audio", exist_ok=True)
    for name, y in wavs.items():
        p = f"{OUT}/audio/{name}_16k.wav"
        sf.write(p, y, sr, subtype="FLOAT")
        back, _ = sf.read(p, dtype="float32")
        assert np.array_equal(back, y)
        M["inputs"][name] = {"file": f"audio/{name}_16k.wav", "samples": int(len(y)), "sample_rate": sr,
                             "source": f"{ASSETS}/{name}.wav librosa.load(sr=16000, mono=True, soxr_hq)"
                             if not name.startswith("syn") else
                             "0.3*sin(2pi(200t+900t^2)) + 0.05*N(0,1) (numpy default_rng(7), syn_1s first)"}

    proc = AutoProcessor.from_pretrained(MID, trust_remote_code=True)
    fe = proc.feature_extractor
    feats = {}
    for name, y in wavs.items():
        b = fe(y, sampling_rate=16000)
        f = b["input_features"][0]
        feats[name] = f
        save(f"feat/{name}_mel", f, "normalized log-mel features [128, T] = (clamp(log10(mel),-12)+4)/4", wav=f"audio/{name}_16k.wav")

    print("loading fp32 model", flush=True)
    model = AutoModelForMultimodalLM.from_pretrained(MID, dtype=torch.float32, device_map=DEV, trust_remote_code=True,
                                                     attn_implementation="sdpa").eval()
    Codec = get_class_from_dynamic_module("modeling_liquid_codec.LiquidCodecModel", MID)
    SCache = get_class_from_dynamic_module("modeling_liquid_codec.LiquidCodecStreamingCache", MID)
    codec = Codec.from_pretrained(MID, subfolder="audio_tokenizer", dtype=torch.float32, device_map=DEV,
                                  attn_implementation="sdpa").eval()
    print("loading bf16 model", flush=True)
    model_bf = AutoModelForMultimodalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map=DEV, trust_remote_code=True,
                                                        attn_implementation="sdpa").eval()
    codec_bf = Codec.from_pretrained(MID, subfolder="audio_tokenizer", dtype=torch.bfloat16, device_map=DEV,
                                     attn_implementation="sdpa").eval()
    M["meta"]["attn_implementation"] = model.config._attn_implementation

    def enc_mods(m):
        e = m.model.audio_encoder
        return {"downsample": e.downsample, "final_norm": e.lfm.embedding_norm, "projector": m.model.audio_projector}

    print("encoder", flush=True)
    with torch.no_grad():
        for name, f in feats.items():
            ft = torch.as_tensor(f, device=DEV)[None]
            mask = torch.ones(1, ft.shape[-1], dtype=torch.long, device=DEV)
            with capture(enc_mods(model)) as g:
                model.model.get_audio_features(ft, mask)
            src = {"features": f"feat/{name}_mel"}
            save(f"enc/{name}_downsample", g["downsample"][0].T, "downsampler output (after lin_out) [N, 768]; "
                 "flatten index c*16+f before lin_out", **src)
            save(f"enc/{name}_final_norm", g["final_norm"][0], "encoder output after embedding_norm [N, 768]", **src)
            save(f"enc/{name}_projector", g["projector"], "projector output [N, 2048] (scattered into <|audio|> slots)", **src)
            if name == "c":
                with capture(enc_mods(model_bf)) as gb:
                    model_bf.model.get_audio_features(ft, mask)
                for k in g:
                    floor(f"encoder.{k}", gb[k], g[k])

    A = {n: f"{OUT}/audio/{n}_16k.wav" for n in wavs}
    convs = {
        "asr_en_c": ([{"role": "system", "content": "Perform ASR in English."},
                      {"role": "user", "content": [{"type": "audio", "audio": A["c"]}]}], False),
        "asr_plain_c": ([{"role": "system", "content": "Perform ASR."},
                         {"role": "user", "content": [{"type": "audio", "audio": A["c"]}]}], False),
        "asr_plain_speech": ([{"role": "system", "content": "Perform ASR."},
                              {"role": "user", "content": [{"type": "audio", "audio": A["speech"]}]}], False),
        "tts_fox": ([{"role": "system", "content": "Perform TTS in American English using a female voice."},
                     {"role": "user", "content": "The quick brown fox jumps over the lazy dog"},
                     {"role": "assistant", "content": "<|audio_start|>"}], True),
        "s2s_c": ([{"role": "system", "content": "Respond with interleaved text and audio."},
                   {"role": "user", "content": [{"type": "audio", "audio": A["c"]}]},
                   {"role": "assistant", "content": "<|mixed_start|>"}], True),
        "chat_nothink_c": ([{"role": "user", "content": [{"type": "audio", "audio": A["c"]}]},
                            {"role": "assistant", "content": "<think></think>"}], True),
    }
    inputs = {}
    for name, (conv, cont) in convs.items():
        kw = dict(add_generation_prompt=not cont, continue_final_message=cont)
        text = proc.apply_chat_template(conv, tokenize=False, **kw)
        inp = proc.apply_chat_template(conv, tokenize=True, return_dict=True, return_tensors="pt", **kw).to(DEV)
        inputs[name] = inp
        ids = inp["input_ids"][0]
        save(f"prompt/{name}_ids", ids, "prompt token ids (audio placeholders expanded)")
        wav = next((c["audio"] for m in conv if isinstance(m["content"], list) for c in m["content"]), None)
        if wav is not None:
            wn = os.path.basename(wav).replace("_16k.wav", "")
            same = np.array_equal(inp["input_features"][0].cpu().numpy(), feats[wn])
            assert same, f"processor features differ from feature extractor for {name}"
        M["prompts"][name] = {"rendered": text, "len": int(ids.numel()), "n_audio_tokens": int((ids == AUDIO_TOKEN).sum()),
                              "ids_file": f"prompt/{name}_ids.npy", "add_generation_prompt": not cont,
                              "continue_final_message": cont,
                              "conversation": [{**m, "content": [{**c, "audio": os.path.relpath(c["audio"], OUT)} if "audio" in c else c
                                                                 for c in m["content"]] if isinstance(m["content"], list) else m["content"]}
                                               for m in conv],
                              "ids": ids.tolist() if ids.numel() < 400 else None}

    print("backbone", flush=True)

    def backbone(m, inp):
        with capture({"final_norm": m.model.language_model.embedding_norm}) as g:
            out = m.model(**{k: v for k, v in inp.items() if k != "attention_mask"}, use_cache=False)
        return g["final_norm"], m.lm_head(out.last_hidden_state[0, -1]).float()

    with torch.no_grad():
        inp = inputs["asr_en_c"]
        hidden, logits = backbone(model, inp)
        src = {"prompt": "prompt/asr_en_c_ids", "audio_features": "enc/c_projector"}
        save("lm/asr_en_c_final_norm", hidden[0], "backbone output after embedding_norm [L, 2048]", **src)
        save("lm/asr_en_c_last_logits", logits, "last-position logits [128000] (tied lm_head)", **src)
        hidden_b, logits_b = backbone(model_bf, inp)
        floor("backbone.final_norm", hidden_b, hidden)
        floor("backbone.last_logits", logits_b, logits, argmax_fp32=int(logits.argmax()), argmax_bf16=int(logits_b.argmax()))
        del hidden, hidden_b

        for name in ("asr_en_c", "asr_plain_c"):
            inp = inputs[name]
            gen = model.generate(**inp, max_new_tokens=256, **GREEDY).sequences[0, inp["input_ids"].shape[1]:]
            text = proc.decode(gen, skip_special_tokens=True)
            M["results"][f"{name}_greedy_fp32"] = {"text": text, "n_tokens": int(gen.numel()), "ids": gen.tolist()}
            print(f"  {name}: {text!r}", flush=True)
        save("gen/asr_en_c_greedy_ids", M["results"]["asr_en_c_greedy_fp32"]["ids"], "greedy ASR generated ids (fp32), ends with <|im_end|>",
             prompt="prompt/asr_en_c_ids")

        # long sequence: speech.wav (~5.3k positions)
        inp = inputs["asr_plain_speech"]
        L = inp["input_ids"].shape[1]
        pos = sorted({0, 100, 255, 256, 511, 1000, 2047, 2048, 3000, 4096, L // 2, L - 2, L - 1})
        out = model.model(**{k: v for k, v in inp.items() if k != "attention_mask"}, use_cache=False)
        hl = out.last_hidden_state[0, pos]
        ll = model.lm_head(out.last_hidden_state[0, -1]).float()
        del out
        src = {"prompt": "prompt/asr_plain_speech_ids", "audio_features": "enc/speech_projector"}
        save("lm/long_speech_positions", np.array(pos), "positions sampled from the long ASR prompt", **src)
        save("lm/long_speech_final_norm_at_positions", hl, "backbone final-norm hidden at long_speech_positions [P, 2048]", **src)
        save("lm/long_speech_last_logits", ll, "last-position logits of the long ASR prompt [128000]", **src)
        out = model_bf.model(**{k: v for k, v in inp.items() if k != "attention_mask"}, use_cache=False)
        floor("backbone_long.final_norm_at_positions", out.last_hidden_state[0, pos], hl)
        # The C++ long prefill compares the last row only; its floor is about twice the 13-position average.
        floor("backbone_long.final_norm_last", out.last_hidden_state[0, -1], hl[-1])
        llb = model_bf.lm_head(out.last_hidden_state[0, -1]).float()
        floor("backbone_long.last_logits", llb, ll, argmax_fp32=int(ll.argmax()), argmax_bf16=int(llb.argmax()))
        del out
        torch.cuda.empty_cache()

    def run_gen(name, order_seed=ORDER_SEED, **kw):
        inp = inputs[name]
        L = inp["input_ids"].shape[1]
        with frame_recorder(model, order_seed) as fr, text_logit_recorder(model) as tl, torch.no_grad():
            o = model.generate(**inp, return_audio_eos=True, **GREEDY, **kw)
        kinds, toks, frames, pattern = slots_of(o, L)
        return o, fr, tl, kinds, toks, frames, pattern

    def dump_gen(tag, name, o, fr, tl, kinds, toks, frames, pattern, settings):
        src = {"prompt": f"prompt/{name}_ids", **settings}
        F = len(fr["codes"])
        save(f"gen/{tag}_slot_kind", kinds, "per generated slot: 0 = text token, 1 = audio frame", **src)
        save(f"gen/{tag}_slot_token", toks, "per slot token id (125032 for audio frames)", **src)
        save(f"gen/{tag}_slot_codes", frames, "per slot codes [S, 12] (-1 for text slots; EOS frame = all 2016)", **src)
        if F:
            save(f"gen/{tag}_frame_hidden", torch.stack(fr["hidden"]), "backbone final-norm hidden fed to the depth decoder per frame [F, 2048]", **src)
            save(f"gen/{tag}_frame_order", torch.stack(fr["order"]), "MaskGIT order per frame [F, 12]; passes reveal order[0:4], [4:8], [8:12]", **src)
            save(f"gen/{tag}_frame_codes_raw", torch.stack(fr["codes"]), "sampled codes per frame before EOS normalization [F, 12]", **src)
            save(f"gen/{tag}_frame_pass_codes", torch.stack(fr["pass_codes"]), "codes input to each depth pass [F, 3, 12] (masked entries irrelevant)", **src)
            save(f"gen/{tag}_frame_pass_masked", torch.stack(fr["pass_masked"]), "masked flags of each depth pass [F, 3, 12] (1 = masked)", **src)
            nl = min(F, MAX_LOGIT_FRAMES)
            save(f"gen/{tag}_frame_pass_logits", torch.stack(fr["pass_logits"][:nl]), f"depth logits of each pass, first {nl} frames [F', 3, 12, 2017]", **src)
        if tl:
            save(f"gen/{tag}_text_top20_vals", torch.stack([t.values for t in tl]), "per text step top-20 logits", **src)
        n_audio = int(o.audio_lengths[0])
        text = proc.decode([t for t in toks.tolist() if t != AUDIO_OUT_TOKEN], skip_special_tokens=True)
        M["results"][tag] = {"pattern": pattern, "n_slots": int(len(kinds)), "n_frames_incl_eos": n_audio, "text": text,
                             "settings": settings}
        print(f"  {tag}: {pattern} text={text!r}", flush=True)

    print("generation", flush=True)
    greedy = {**GREEDY, "text": "greedy", "orders": f"injected numpy default_rng({ORDER_SEED}).permutation(12) per frame"}
    o, fr, tl, kinds, toks, frames, pat = run_gen("tts_fox", max_new_tokens=400)
    dump_gen("tts_fox_greedy", "tts_fox", o, fr, tl, kinds, toks, frames, pat, {**greedy, "max_new_tokens": 400})
    tts_frames = frames[kinds == 1]
    tts_codes = torch.as_tensor(tts_frames[~(tts_frames == EOS_CODE).any(1)]).T.contiguous()  # [12, T] without EOS
    tts_fr = fr

    # depth decoder logits noise floor: replay the fp32 pass inputs through the bf16 depth decoder
    with torch.no_grad():
        nl = min(10, len(tts_fr["codes"]))
        lb, lf = [], []
        for i in range(nl):
            h = tts_fr["hidden"][i].to(DEV)
            for p in range(3):
                c, mk = tts_fr["pass_codes"][i][p].to(DEV), tts_fr["pass_masked"][i][p].to(DEV)
                lb.append(model_bf.model.depth_decoder(h[None].bfloat16(), c[None], mk[None])[0].float())
                lf.append(tts_fr["pass_logits"][i][p].to(DEV))
        lb, lf = torch.stack(lb), torch.stack(lf)
        floor("depth.logits", lb, lf, argmax_agree=float((lb.argmax(-1) == lf.argmax(-1)).float().mean()))

    o, fr, tl, kinds, toks, frames, pat = run_gen("s2s_c", order_seed=S2S_ORDER_SEED, max_new_tokens=1024)
    dump_gen("s2s_c_greedy", "s2s_c", o, fr, tl, kinds, toks, frames, pat,
             {**greedy, "orders": f"injected numpy default_rng({S2S_ORDER_SEED}).permutation(12) per frame", "max_new_tokens": 1024})
    # same prompt, default order seed, cut at max_new_tokens in the middle of the reply (it runs to about 160 slots)
    o, fr, tl, kinds, toks, frames, pat = run_gen("s2s_c", max_new_tokens=96)
    dump_gen("s2s_c_greedy_trunc", "s2s_c", o, fr, tl, kinds, toks, frames, pat, {**greedy, "max_new_tokens": 96})

    print("codec", flush=True)
    gen_ = torch.Generator(device="cpu").manual_seed(0)
    code_sets = {"tts": tts_codes, "rand200": torch.randint(0, 2016, (12, 200), generator=gen_)}
    for cname, codes in code_sets.items():
        codes = codes.to(DEV)
        save(f"codec/{cname}_codes", codes, "codes [12, T] in [0, 2016) " +
             ("(greedy TTS frames, EOS dropped)" if cname == "tts" else "(torch.randint, CPU Generator seed 0)"))
        src = {"codes": f"codec/{cname}_codes"}

        def codec_mods(c):
            return {"project_out": c.quantizer.project_out, "from_quantizer": c.from_quantizer,
                    "dec_final_norm": c.decoder.lfm.embedding_norm, "lin": c.decoder.lin}
        with torch.no_grad():
            with capture(codec_mods(codec)) as g:
                wav = codec.decode(codes[None]).audio_values[0]
            save(f"codec/{cname}_project_out", g["project_out"][0], "quantizer.project_out output [T, 256]", **src)
            save(f"codec/{cname}_from_quantizer", g["from_quantizer"][0], "from_quantizer output [T, 512]", **src)
            save(f"codec/{cname}_dec_final_norm", g["dec_final_norm"][0], "decoder LFM3 output after embedding_norm [6T, 512]", **src)
            nl = min(48, g["lin"].shape[1])
            save(f"codec/{cname}_lin_first", g["lin"][0, :nl], f"decoder.lin output, first {nl} positions [{nl}, 1282] "
                 "(log_mag = [:641], angle = [641:])", **src)
            save(f"codec/{cname}_wave", wav, "offline decode, 24 kHz float [1920*T]", **src)
            with capture(codec_mods(codec_bf)) as gb:
                wav_b = codec_bf.decode(codes[None]).audio_values[0]
            if cname == "tts":
                for k in g:
                    floor(f"codec.{k}", gb[k], g[k])
            floor(f"codec.{cname}_wave", wav_b, wav)
            T, s, i, parts, sizes = codes.shape[1], 0, 0, [], []
            cache = SCache(codec.config)
            while s < T:
                e_ = min(T, s + (7, 13, 20)[i % 3])
                parts.append(codec.decode(codes[None, :, s:e_], streaming_cache=cache, finalize=e_ == T).audio_values[0])
                sizes.append((e_ - s, parts[-1].numel()))
                s, i = e_, i + 1
            sw = torch.cat(parts)
            save(f"codec/{cname}_stream_chunks", np.array(sizes), "streaming chunks [n, 2] = (frames, output samples); "
                 "last chunk finalize=True", **src)
            save(f"codec/{cname}_stream_wave", sw, "concatenated streaming output", **src)

    print("multiturn", flush=True)
    multiturn(model, proc)
    print("probes", flush=True)
    tokenizer_cases(proc)
    ignore_merges_cases()
    M["meta"]["elapsed_s"] = round(time.time() - t0, 1)
    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump(M, f, indent=1)
    total = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(OUT) for fn in fns)
    print(f"done in {M['meta']['elapsed_s']} s, {len(M['arrays'])} arrays, {total / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
