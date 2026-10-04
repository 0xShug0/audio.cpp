#!/usr/bin/env python3
"""Check session.wall_ms against the session probe's external run() timer.

Capture both stdout and stderr from kitten_tts2_session_probe with its reference
model argument. The first clone includes lazy encoder loading; the last changes
the reference. Neither cost should be missing from session.wall_ms.
"""
import argparse
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    args = parser.parse_args()
    wall_ms = None
    seen = []
    failures = []
    for line in args.log.read_text(encoding='utf-8', errors='replace').splitlines():
        wall = re.search(r'\bsession\.wall_ms\s+([\d.]+)', line)
        if wall:
            assert wall_ms is None, 'unmatched session.wall_ms entry'
            wall_ms = float(wall[1])
        request = re.match(r'(request|clone)=(\d+) seconds=([\d.]+)', line)
        if not request:
            continue
        label = f'{request[1]}={request[2]}'
        assert wall_ms is not None, f'missing session.wall_ms for {label}'
        external_ms = float(request[3]) * 1000
        # Permit logging/return overhead and the probe's printed-time rounding.
        allowance = max(5.0, external_ms * 0.02)
        delta = external_ms - wall_ms
        print(f'{label}: external_ms={external_ms:.3f} session.wall_ms={wall_ms:.3f} delta_ms={delta:.3f}')
        if not -1.0 <= delta <= allowance:
            failures.append(label)
        seen.append((request[1], int(request[2])))
        wall_ms = None
    assert seen == [('request', i) for i in range(4)] + [('clone', i) for i in range(3)], 'incomplete session probe log'
    assert wall_ms is None, 'unmatched session.wall_ms entry'
    assert not failures, f'session.wall_ms does not cover run(): {failures}'
    print('PASS: session timing includes reference conditioning and lazy encoder loading')


if __name__ == '__main__':
    main()
