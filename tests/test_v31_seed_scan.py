import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("seed_scan", Path(__file__).parents[1] / "scripts/scan_v31_seed.py")
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)


def test_bounded_static_extraction():
    source = '''
        // action.setText("not real");
        int force = 4;
        action.setText("Lose " + force + " Force with " + GameUtils.getFullName(card));
        action.setText("A + B (literal)");
        action.setText(dynamic.getText());
        action.setText("Ignore " + method(force));
        action.setText("Quoted \\\"title\\\"");
    '''
    results = list(scan.scan(source))
    assert [item[0] for item in results] == [
        "Lose <n> Force with <c>", "A + B (literal)", None, None, 'Quoted "title"'
    ]


def test_conflicting_variable_types_are_not_guessed():
    source = 'int amount; String amount; action.setText("Use " + amount);'
    assert next(scan.scan(source))[0] is None
