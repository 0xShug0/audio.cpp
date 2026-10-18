#!/usr/bin/env python3
"""Change a YuE2 song's style part-way through by editing its semantic history.

Run from the repository root; unknown options go to audiocpp_cli unchanged:

    python3 examples/yue2_style_change/style_change.py --backend cuda

Needs only the Python standard library. Steps whose output directory already
exists are skipped, so delete a directory under --out to redo that step.
"""
import argparse, fractions, json, pathlib, subprocess

HERE = pathlib.Path(__file__).resolve().parent
FPS = 25     # semantic frames per second of audio
LEAD = 35    # the audio runs about this many frames ahead of the written score

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--cli', default='build/bin/audiocpp_cli', help='audiocpp_cli binary (default: %(default)s)')
ap.add_argument('--model', default='models/Yue2-3B-GGUF', help='YuE2 model directory (default: %(default)s)')
ap.add_argument('--lyrics', default=str(HERE / 'lyrics.txt'))
ap.add_argument('--score', default=str(HERE / 'score.abc'), help='score both styles are sung from')
ap.add_argument('--new-score', action='store_true', help='write a fresh score instead (stop_after=abc)')
ap.add_argument('--style-a', default='animated film musical, storybook, minor key, string section answering the vocal, '
                'call and response, pizzicato strings, female vocal, dramatic, storytelling')
ap.add_argument('--style-b', default='brutal death metal, blast beats, down-tuned chugging guitars, '
                'guttural growled male vocal, shrieked backing screams, horror, relentless')
ap.add_argument('--cut', type=float, help='second at which the style changes (default: the second chorus)')
ap.add_argument('--seed', type=int, default=3)
ap.add_argument('--out', default='yue2_style_change_out')
args, cli_options = ap.parse_known_args()

out = pathlib.Path(args.out)
lyrics = pathlib.Path(args.lyrics).read_text(encoding='utf-8')


def run(name, style, **options):
    """One audiocpp_cli request. Returns the directory holding its artifacts."""
    where = out / name
    if not where.exists():
        cmd = [args.cli, '--task', 'gen', '--family', 'yue2', '--model', args.model, '--lyrics', lyrics,
               '--seed', str(args.seed), '--out-dir', str(where), '--request-option', 'style=' + style]
        if 'stop_after' not in options:
            cmd += ['--out', str(where) + '.wav']
        for key, value in options.items():
            cmd += ['--request-option', '%s=%s' % (key, value)]
        print('==', name, flush=True)
        subprocess.run(cmd + cli_options, check=True)
    return where


def sections(score):
    """(start second, name) of each section, from the bar counts of the vocal voice"""
    bar, quarter = 4.0, 0.5    # quarter notes per bar, seconds per quarter note
    found, seconds, name, vocal = [], 0.0, None, False
    for line in pathlib.Path(score).read_text().splitlines():
        if line.startswith('M:'):
            bar = float(fractions.Fraction(line[2:].strip())) * 4
        elif line.startswith('Q:'):
            unit, tempo = line[2:].split('=')
            quarter = 60 / (float(tempo) * float(fractions.Fraction(unit)) * 4)
        elif line.startswith('% '):
            name = line[2:]
        elif line.startswith('V:'):
            vocal = line.startswith('V: Vocal')
        elif vocal and '|' in line and not line.startswith('w:'):
            if name:
                found.append((seconds, name))
                name = None
            seconds += line.count('|') * bar * quarter
    return found


def tokens(where):
    return json.loads((where / 'semantic.json').read_text())


def save(name, frames):
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(json.dumps(frames))
    return path


def lag(song, take, at, span=30 * FPS, reach=4 * FPS):
    """How many frames later `take` plays what `song` plays before `at`.

    Two renders of one score do not keep the same clock. At the right lag they share
    a few percent of identical tokens, at any other lag almost none.
    """
    def hits(shift):
        return sum(1 for i in range(max(at - span, -shift, 0), at) if i + shift < len(take) and song[i] == take[i + shift])
    best = max(range(-reach, reach + 1), key=hits)
    return best if hits(best) >= 8 else 0


# 1. One score: the bundled one, or a fresh one. A score decides how far a second
#    style can move away from the first, so a fresh one is a gamble.
score = run('1_score', args.style_a, stop_after='abc') / 'score.abc' if args.new_score else pathlib.Path(args.score)
starts = sections(score)
print('sections:', ', '.join('%s %d s' % (name, second) for second, name in starts))
if args.cut is None:
    choruses = [second for second, name in starts if name == 'chorus']
    args.cut = choruses[1] if len(choruses) > 1 else starts[len(starts) // 2][0]
cut = round(args.cut * FPS) - LEAD

# 2. The same score sung in both styles; tokens only, no audio yet.
only_tokens = dict(abc_file=score, stop_after='semantic')
song = tokens(run('2_take_a', args.style_a, **only_tokens))
take_b = tokens(run('2_take_b', args.style_b, **only_tokens))

# (where the history is edited, how much of the real song is left at its end), frames.
# The first edit is early and long: the song stays in style A but plays a lead-in.
# The second is the change itself.
edits = [(cut - 10 * FPS, 30 * FPS), (cut, 5 * FPS)]
if edits[0][0] <= edits[0][1] + 4 * FPS or cut >= min(len(song), len(take_b)) - 4 * FPS:
    raise SystemExit('--cut must be later than 46 s and inside both takes (%d s and %d s)'
                     % (len(song) // FPS, len(take_b) // FPS))

# 3. At each edit the model is handed style B's take as its past, with the last
#    `keep` frames of the real song on the end, and asked what comes next. What it
#    writes replaces the song from there on. A leg stops where the next edit starts.
for index, (at, keep) in enumerate(edits):
    shift = lag(song, take_b, at)
    print('edit at %.1f s: style B runs %+d frames against the song' % (at / FPS, shift))
    history = save('3_history_%d.json' % index, take_b[:at + shift - keep] + song[at - keep:at])
    bounds = {}
    if index + 1 < len(edits):
        stop = edits[index + 1][0] + shift
        bounds = dict(semantic_min_tokens=stop, semantic_max_tokens=stop)
    leg = tokens(run('3_leg_%d' % index, args.style_b, semantic_prefix_file=history, **only_tokens, **bounds))
    song = song[:at] + leg[at + shift:]

# 4. Render the finished stream. The prefix is the whole song, so nothing is sampled.
final = run('4_song', args.style_a, abc_file=score, semantic_prefix_file=save('4_song.json', song),
            semantic_min_tokens=len(song), semantic_max_tokens=len(song))
print('done: %s.wav, %d s, style change at %.1f s' % (final, len(song) // FPS, cut / FPS))
