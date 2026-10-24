"""Plan dev-distribution: the core chain runs on any machine (no lab machine paths, hosts or rigs)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ['scripts/machine_env.sh', 'scripts/setup.sh', 'scripts/build_knowledge_engine.py', 'scripts/v31_stage.py',
        'scripts/v31_engine.sh', 'scripts/efficient_java_check.sh', 'scripts/lr2_arm.sh', 'scripts/train_arm.sh',
        'scripts/branch_decay.sh', 'scripts/evaluate_v31.py', 'scripts/round_robin_v31.py', 'scripts/v31_collect.py',
        'scripts/brainiac_server.sh', 'scripts/brainiac_overlay.py', 'scripts/brainiac_release.py',
        'scripts/export_card_database.py', 'scripts/export_torchscript.py', 'native/build.sh', 'native/setup.sh']
LAB = re.compile(r'/home/(kaltan|peter)|gemp-swccg-agent|gemp-swccg-private|im-server')


def test_core_chain_has_no_lab_paths():
    hits = [f'{name}:{number}: {line.strip()}' for name in CORE
            for number, line in enumerate((ROOT / name).read_text().splitlines(), 1) if LAB.search(line)]
    assert not hits, '\n'.join(hits)
