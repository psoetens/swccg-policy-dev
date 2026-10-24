"""Plan dev-distribution: git holds code and small configuration, Git LFS everything large or binary.

    python3 scripts/check_lfs_policy.py [--staged]

Fails (exit 1) on any tracked (or, with --staged, staged) file that is not stored by LFS although it is larger than
512 KB or has a binary extension. The LFS patterns live in .gitattributes. Run by tests/test_lfs_policy.py; install as
a pre-commit hook with: ln -s ../../scripts/check_lfs_policy.py .git/hooks/pre-commit (it then checks --staged).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 512 * 1024
BINARY = {'.pt', '.ts', '.pt2', '.onnx', '.npz', '.npy', '.wasm', '.so', '.gz', '.zip', '.png', '.jfr'}


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], check=True, capture_output=True, text=True).stdout


def violations(staged=False):
    files = git('diff', '--cached', '--name-only', '--diff-filter=AM', '-z') if staged else git('ls-files', '-z')
    files = [f for f in files.split('\0') if f and (ROOT / f).is_file()]
    if not files:
        return []
    lfs = {}
    out = git('check-attr', 'filter', '-z', '--', *files).split('\0')
    for path, _, value in zip(out[0::3], out[1::3], out[2::3]):
        lfs[path] = value == 'lfs'
    bad = []
    for f in files:
        if lfs.get(f):
            continue
        size = (ROOT / f).stat().st_size
        if size > LIMIT or Path(f).suffix in BINARY:
            bad.append(f'{f} ({size / 1e6:.2f} MB) is not in Git LFS')
    return bad


def main():
    staged = '--staged' in sys.argv[1:] or Path(sys.argv[0]).name == 'pre-commit'
    bad = violations(staged)
    for line in bad:
        print(line, file=sys.stderr)
    if bad:
        print(f'{len(bad)} file(s) break the LFS policy: add a pattern to .gitattributes', file=sys.stderr)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
