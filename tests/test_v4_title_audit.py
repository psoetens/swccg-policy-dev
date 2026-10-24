"""The audit must catch the title/quantity residuals identified by Amendment 2."""

import importlib.util
from pathlib import Path
import re
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("v4_title_audit", SCRIPTS / "audit_v4_titles.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
sys.path.pop(0)


def test_amendment_residuals():
    titles = re.compile(r"(?<!\w)(?:Han Solo|Ralltiir: Forest)(?!\w)")
    suffixes = [" (V)", " (AI)", " (AI) (V)"]
    expected = {
        "Deploy ◇Tatooine: <c>": "unresolved_title_marker",
        "Deploy <c> (V)": "variant_suffix_after_reference",
        "Deploy <c> (AI)": "variant_suffix_after_reference",
        "Lose 4 Force": "standalone_integer",
        "Deploy Han Solo": "card_title",
        "class Filters$548": "anonymous_class_number",
    }
    for key, reason in expected.items():
        assert reason in audit.residual_reasons(key, "q", titles, suffixes)
    for key in ["Deploy <c>", "unique (•) aliens", "◇<c>", "••<c>", "<> <not-a-marker-reference>"]:
        if key.startswith("<> "):
            assert audit.residual_reasons(key, "q", titles, suffixes)
        else:
            assert not audit.residual_reasons(key, "q", titles, suffixes)
    assert not audit.residual_reasons("SSA-306 OOM-9 2-1B", "o", titles, suffixes)
