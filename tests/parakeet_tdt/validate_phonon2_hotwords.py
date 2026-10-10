"""Compare native hotword transitions/bonuses with the original 0.2.11 wheel.

Requires numpy and an unpacked original fermion-research wheel, not its engine
dependencies. The original algorithm module is loaded without running a model.
"""
import argparse
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--original-hotwords', type=Path, required=True)
    parser.add_argument('--source-config', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('original_phonon_hotwords', args.original_hotwords)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    vocabulary = json.loads(args.source_config.read_text(encoding='utf-8'))['joint']['vocabulary']
    cases = [['Ada', 'Quillon'], ['Ada Lovelace', 'Neutrino'], ['RoFormer', 'Intel', 'Whisper'],
             ['Apple', 'server', 'CUDA'], ['Zoë', 'Łukasz', 'Émilie'],
             [{'word': 'Quillon', 'spoken': ['kwil on', 'quillon']}], [], ['!?#unlikely'],
             ['Ada', 'ada', ' Ada ']]
    count = 0
    for terms in cases:
        reference = original.HotwordAutomaton(terms, vocabulary, lam=2.0)
        prefixes = [[], [len(vocabulary)], [100], [100, len(vocabulary)]]
        for seq in reference.phrases[:12]:
            prefixes.extend([list(seq[:n]) for n in range(1, len(seq) + 1)])
        case = {'vocabulary': vocabulary + ['<blank>'], 'blank': len(vocabulary),
                'hotwords': terms, 'strength': 2, 'queries': [{'prefix': p} for p in prefixes]}
        with tempfile.TemporaryDirectory(prefix='phonon-hotwords-') as temp:
            path = Path(temp) / 'input.json'
            path.write_text(json.dumps(case, ensure_ascii=False), encoding='utf-8')
            flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            result = subprocess.run([str(args.probe.resolve()), str(path)], check=True,
                                    capture_output=True, encoding='utf-8', creationflags=flags)
        actual = json.loads(result.stdout)
        assert actual['states'] == reference.n_states
        assert len(actual['rows']) == len(prefixes)
        for prefix, row in zip(prefixes, actual['rows']):
            state = 0
            for token in prefix:
                state = reference.step(state, token)
            assert row['state'] == state and row['bonus'] == reference.bonus(state).tolist()
            count += 1
    print(f'PASS: {len(cases)} hotword lists, {count} original-policy checks')

if __name__ == '__main__':
    main()
