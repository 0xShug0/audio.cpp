"""Verify the Stage 1 separation against the current upstream tree.

Run after integrating upstream: python tests/server/check_parallel_boundaries.py
--base origin/main. This checks source boundaries, not model quality/admission.
"""
import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, help='current upstream main commit/ref')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]

    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(root), *arguments])

    protected = [
        'app/server/runtime.h', 'app/server/busy_guard.h',
        'app/server/http.h', 'app/server/http.cpp',
        'app/server/frontend.h', 'app/server/frontend.cpp',
        'include/engine/framework/runtime/session.h',
        'app/cli', 'src/models', 'include/engine/models',
        'src/community_models', 'include/engine/community_models', 'external/ggml',
    ]
    changed = git('diff', '--name-only', args.base, '--', *protected).decode().strip()
    if changed:
        raise AssertionError('Protected upstream files changed:\n' + changed)
    original = git('show', args.base + ':app/server/runtime.cpp').decode()
    expected_addition = '''    if (body.find("slots") != nullptr) {
        return error_response(400,
            "slots requires --parallel-jobs at server startup",
            "invalid_request_error");
    }
'''
    candidate = (root / 'app/server/runtime.cpp').read_text(encoding='utf-8')
    if candidate.count(expected_addition) != 1:
        raise AssertionError('Legacy registration rejection missing or duplicated')
    if candidate.replace(expected_addition, '', 1) != original.replace('\r\n', '\n'):
        raise AssertionError('Legacy runtime has changes beyond registration rejection')
    print('PASS: protected files unchanged; legacy runtime only rejects explicit slots')


if __name__ == '__main__':
    main()
