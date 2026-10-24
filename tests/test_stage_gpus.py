"""Plan host-migration M-3: GPU selection and the per-GPU memory guard; without the new flags nothing changes."""

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from stage_utils import gpu_table  # noqa: E402
from v31_stage import gpu_over_limit, rollout_environment, stage_gpus  # noqa: E402

# nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu on the 4-GPU server.
SERVER = """15, 24564, 0, 31
15, 24564, 0, 30
15, 32607, 0, 29
15, 32607, 0, 28
"""


def sample(table, gpus):
    rows = gpu_table(table)
    first = rows[gpus[0] if gpus else 0]
    out = {'gpu_used_mib': first[0], 'gpu_total_mib': first[1]}
    if gpus:
        out['gpus'] = [{'index': i, 'used_mib': rows[i][0], 'total_mib': rows[i][1], 'util_percent': rows[i][2]}
                       for i in gpus]
    return out


def with_used(table, index, used):
    lines = table.splitlines()
    fields = lines[index].split(', ')
    lines[index] = ', '.join([str(used), *fields[1:]])
    return '\n'.join(lines) + '\n'


def test_no_flags_keep_todays_behaviour():
    args = Namespace(learner_device=None, rollout_gpus=None)
    assert stage_gpus(args) is None
    environment = {'A': '1'}
    assert rollout_environment(environment, args) is environment
    # Today's guard: the first GPU only, at 28 GiB.
    assert not gpu_over_limit(sample(with_used(SERVER, 3, 31000), None))
    assert gpu_over_limit(sample(with_used(SERVER, 0, 28 * 1024), None))


def test_gpu_selection():
    args = Namespace(learner_device=3, rollout_gpus='2')
    assert stage_gpus(args) == [2, 3]
    assert rollout_environment({'A': '1'}, args) == {'A': '1', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                                                     'CUDA_VISIBLE_DEVICES': '2'}
    assert stage_gpus(Namespace(learner_device=3, rollout_gpus=None)) == [3]  # rollouts share the learner GPU
    assert stage_gpus(Namespace(learner_device=None, rollout_gpus='2')) == [0, 2]


def test_guard_trips_on_either_used_gpu_only():
    gpus = [2, 3]
    assert not gpu_over_limit(sample(SERVER, gpus))
    assert gpu_over_limit(sample(with_used(SERVER, 2, 28600), gpus))  # 5090: 28 GiB minus 4 GiB headroom rule
    assert gpu_over_limit(sample(with_used(SERVER, 3, 29000), gpus))
    assert not gpu_over_limit(sample(with_used(SERVER, 0, 24000), gpus))  # another user's 4090 is not ours
    assert gpu_over_limit(sample(with_used(SERVER, 0, 20500), [0]))  # a 4090: 4 GiB below its total
