"""Plan nvf-b-local A4: evaluations play greedy (temperature 0) by default; the temperature reaches the game JVM."""
import sys
from pathlib import Path

import evaluate_v31


def test_default_is_greedy_and_reaches_the_engine(monkeypatch):
    captured = {}
    monkeypatch.setattr(sys, 'argv', ['evaluate_v31', '--checkpoint', 'a.pt', '--reference', 'b.pt', '--tag', 't'])
    monkeypatch.setattr(evaluate_v31.argparse.ArgumentParser, 'parse_args',
                        lambda self, *a, **k: captured.setdefault('args', self.parse_known_args(*a, **k)[0]))
    try:
        evaluate_v31.main()
    except Exception:
        pass
    assert captured['args'].temperature == 0.0
    source = (Path(__file__).resolve().parents[1] / 'scripts/evaluate_v31.py').read_text()
    assert "'native.temperature': args.temperature" in source
