# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers>=0.22,<0.23"]
# ///
"""Generate exact token IDs from Drama 3's unmodified official tokenizer.json."""
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tokenizer', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
source = json.loads(args.tokenizer.read_text(encoding='utf-8'))
assert source['normalizer'] == {'type': 'NFC'}, 'Expected the official NFC tokenizer'
tokenizer = Tokenizer.from_file(str(args.tokenizer))
texts = {
    'english': 'The careful narrator paused, then spoke clearly.',
    'contractions': "I'm happy. WE'RE ready! They'd already left, hadn't they?",
    'numbers': '1234567890 costs 3.14 dollars on 2026-10-14.',
    'emotion': '[happy] That is wonderful! [sighs] Well, perhaps.',
    'speakers': '<|speaker:0|> Hello. <|speaker:15|> Goodbye.',
    'whisper': '<whisper>Keep this between us.</whisper>',
    'polish': 'Ptaszki ćwierkają, a żółw spokojnie odpoczywa.',
    'french_nfc': "Élodie préfère le café près de la fenêtre.",
    'french_nfd': "E\u0301lodie pre\u0301fe\u0300re le cafe\u0301 pre\u0300s de la fene\u0302tre.",
    'spanish': 'El niño pidió café y habló con María.',
    'combining': 'Cafe\u0301 and nai\u0308ve. A\u030angstro\u0308m.',
    'combining_order': 'q\u0307\u0323 a\u0315\u0300',
    'hindi': 'नमस्ते दुनिया। आज मौसम अच्छा है।',
    'thai': 'สวัสดีครับ วันนี้อากาศดีมาก',
    'arabic': 'مَرْحَبًا بِالْعَالَمِ، كَيْفَ حَالُكَ؟',
    'chinese': '你好，世界。今天的天气很好。',
    'japanese': 'こんにちは、世界。今日はいい天気ですね。',
    'korean_nfc': '안녕하세요. 반갑습니다.',
    'korean_jamo': '안녕하세요.',
    'russian': 'Привет, мир. Сегодня прекрасный день.',
    'cyrillic_combining': 'И\u0306 и и\u0306, е\u0308 и Е\u0308.',
    'vietnamese': 'Tôi yêu tiếng Việt. Cảm ơn bạn.',
    'unicode_whitespace': '\u3000<|speaker:2|> Hello.\u00a0Next\u202fword.\n\tDone.',
    'unicode_punctuation': '“Hello…” she said—then paused. «Bonjour !»',
    'emoji': 'Hello 👩🏽\u200d💻! A family: 👨\u200d👩\u200d👧.',
    'nfc_not_nfkc': 'ﬃ ﬁ ① Ａ Å',
    'embedded_nul': 'Before\u0000after.',
    'empty': '',
    'controls': '\u001c\u001d\u001e\u001f',
    'system_literal': '<|im_start|>system\nconvert the provided text to speech',
    'generation_literal': '<|im_end|>\n<|im_start|>assistant\n<|voice|>',
    'reference_literal': '\n\n[Reference 1]\nText: <|speaker:0|> Cafe\u0301.\nSpeech: ',
}
rows = [{'id': name, 'text': text, 'tokens': tokenizer.encode(text).ids}
        for name, text in texts.items()]
for row in rows:
    if '\x00' in row['text']:
        # Preserve the direct tokenizer input independently of C-string JSON parsers.
        row['utf8'] = list(row['text'].encode('utf-8'))
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({'source': 'fishaudio/drama-3 official tokenizer.json',
    'normalizer': source['normalizer'], 'cases': rows}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'reference_cases': len(rows), 'output': str(args.output)}))
