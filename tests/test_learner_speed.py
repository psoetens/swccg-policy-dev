"""Plan learner-speed: the speed options are off by default, exact where they claim to be, and the harness is
deterministic."""

import collections
import copy
import json
import random

import numpy as np
import pytest
import torch

from test_ppo_step import fixed_batch
from swccg_policy import learner_speed, v4_rl
from swccg_policy.learner_speed import Prefetch, compile_tables, eager_tables, table_modules, uncompile_tables
from swccg_policy.ppo_step import PPOConfig, critic_precision, step
from swccg_policy.v4_data import stack

CUDA = torch.cuda.is_available()


def pass_inputs(seed=5):
    actor, critic, items, critics = fixed_batch()
    rng = np.random.default_rng(seed)
    count = len(items)
    return dict(actor=actor, critic=critic, items=items, critic_items=critics,
                sequences=[[index % 2] for index in range(count)], exclusions=[None] * count,
                old_logp=np.full(count, -0.7, np.float32), advantages=rng.normal(size=count).astype(np.float32),
                returns=rng.normal(size=count).astype(np.float32),
                wins=np.array([1, 0, np.nan, 1, 0, 1, 0], np.float32)[:count])


def optimizers(actor, critic):
    return [torch.optim.AdamW(actor.parameters(), lr=1e-3), torch.optim.AdamW(critic.parameters(), lr=1e-3)]


def old_inline_loop(lengths, minibatch):
    """The minibatch order exactly as learn() built it before plan learner-speed."""
    permutation = np.random.permutation(len(lengths))
    batches = []
    for offset in range(0, len(lengths), minibatch * 16):
        part = permutation[offset:offset + minibatch * 16]
        part = part[np.argsort(lengths[part], kind='stable')]
        batches.extend(part[index:index + minibatch] for index in range(0, len(part), minibatch))
    random.shuffle(batches)
    return batches


def run_pass(inputs, config, batches, accumulate=2):
    actor, critic = inputs['actor'], inputs['critic']
    opts = optimizers(actor, critic)
    statistics, counts = collections.defaultdict(list), collections.Counter()
    v4_rl.ppo_pass(actor, critic, opts, inputs['items'], inputs['critic_items'], batches, inputs['sequences'],
                   inputs['exclusions'], inputs['old_logp'], inputs['advantages'], inputs['returns'], inputs['wins'],
                   config, True, accumulate, statistics, counts)
    return statistics, counts


def test_ppo_batches_match_the_old_inline_order():
    lengths = np.random.default_rng(1).integers(10, 600, size=3000)
    for seed in (1, 2):
        np.random.seed(seed)
        random.seed(seed)
        expected = old_inline_loop(lengths, 64)
        np.random.seed(seed)
        random.seed(seed)
        actual = v4_rl.ppo_batches(lengths, 64)
        assert len(expected) == len(actual)
        assert all(np.array_equal(a, b) for a, b in zip(expected, actual))


def test_options_off_pass_is_byte_identical_to_manual_steps():
    """ppo_pass with the default config = the pre-plan per-minibatch step() calls, bit for bit."""
    config = PPOConfig(actor_tokens=35, critic_tokens=55, bf16=False, win_coef=0.0)
    batches = [np.array([0, 3, 5]), np.array([1, 2]), np.array([4, 6])]
    inputs = pass_inputs()
    reference = copy.deepcopy(inputs)
    run_pass(inputs, config, batches)
    opts = optimizers(reference['actor'], reference['critic'])
    for number, indices in enumerate(batches):
        first = number - number % 2
        size = min(2, len(batches) - first)
        step(reference['actor'], reference['critic'], opts, reference['items'], reference['critic_items'], indices,
             reference['sequences'], reference['exclusions'], reference['old_logp'], reference['advantages'],
             reference['returns'], stack=stack, config=config, policy_on=True, critic_zero=number == first,
             critic_step=number == first + size - 1, wins=reference['wins'], critic_scale=1.0 / size)
    for model in ('actor', 'critic'):
        for left, right in zip(inputs[model].parameters(), reference[model].parameters(), strict=True):
            assert torch.equal(left, right)


def test_critic_precision_is_scoped():
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        with critic_precision(PPOConfig()):
            assert not torch.backends.cuda.matmul.allow_tf32
        with critic_precision(PPOConfig(critic_tf32=True)):
            assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        with pytest.raises(RuntimeError):
            with critic_precision(PPOConfig(critic_tf32=True)):
                raise RuntimeError
        assert not torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def test_defaults_are_off():
    config = PPOConfig()
    assert not config.critic_tf32 and not config.critic_stream
    actor, critic, _, _ = fixed_batch()
    assert all(learner_speed.compiled_table(module) is None for module in table_modules(actor) + table_modules(critic))


@pytest.mark.skipif(not CUDA, reason='compiled tables are checked on CUDA')
def test_compiled_tables_match_eager_and_copies_stay_eager():
    actor, critic, _, _ = fixed_batch()
    actor, critic = actor.cuda(), critic.cuda()
    eager = [module.table() for module in table_modules(actor) + table_modules(critic)]
    compile_tables(actor, critic)
    try:
        modules = table_modules(actor) + table_modules(critic)
        compiled = [module.table() for module in modules]
        for left, right in zip(eager, compiled, strict=True):
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
        # Gradients flow into the same parameters.
        sum(table.sum() for table in compiled).backward()
        assert actor.cards.delta.weight.grad is not None
        # Export copies (deepcopy) are not registered: they run the eager build.
        copied = copy.deepcopy(actor)
        assert all(learner_speed.compiled_table(module) is None for module in table_modules(copied))
        with eager_tables():
            assert learner_speed.compiled_table(actor.cards) is None
        assert learner_speed.compiled_table(actor.cards) is not None
    finally:
        uncompile_tables(actor, critic)


@pytest.mark.skipif(not CUDA, reason='CUDA streams')
def test_critic_stream_gives_identical_gradients():
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        results = []
        for stream in (False, True):
            inputs = pass_inputs()
            inputs['actor'], inputs['critic'] = inputs['actor'].cuda(), inputs['critic'].cuda()
            config = PPOConfig(actor_tokens=35, critic_tokens=55, bf16=False, win_coef=0.0, critic_stream=stream)
            statistics, counts = run_pass(inputs, config, [np.array([0, 3, 5]), np.array([1, 2]), np.array([4, 6])])
            torch.cuda.synchronize()
            results.append(([p.detach().cpu() for model in ('actor', 'critic') for p in inputs[model].parameters()],
                            {name: [v.cpu().flatten().tolist() for v in values] for name, values in statistics.items()},
                            counts))
        (left, left_stats, left_counts), (right, right_stats, right_counts) = results
        assert left_counts == right_counts
        assert left_stats == right_stats
        assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))
    finally:
        torch.use_deterministic_algorithms(False)


def write_files(folder, sizes):
    folder.mkdir()
    paths = []
    for index, steps in enumerate(sizes):
        path = folder / f'part-{index:05d}.traj'
        path.write_text(str(index))
        path.with_name(path.name + '.meta.json').write_text(json.dumps({'steps': steps}))
        paths.append(path)
    return paths


def test_prefetch_offers_the_learner_order_and_identical_batches(tmp_path):
    paths = write_files(tmp_path / 'traj', [40, 30, 50, 20, 70, 10])
    done = []

    def files_of(folder):
        return sorted(path for path in folder.iterdir() if path.suffix == '.traj')

    def load(files):
        return [{'file': path.name, 'data': path.read_text()} for path in files]

    prefetch = Prefetch(tmp_path / 'traj', 60, files_of, load, lambda: bool(done))
    first = prefetch.get()
    second = prefetch.get()
    done.append(True)
    # The learner's rule: files in order until the steps reach the batch size; never a file twice.
    assert first['files'] == paths[:2] and second['files'] == paths[2:4]
    assert first['games'] == load(paths[:2]) and second['games'] == load(paths[2:4])


def test_prefetch_raises_a_decode_failure(tmp_path):
    write_files(tmp_path / 'traj', [100])

    def fail(files):
        raise ValueError('bad file')

    prefetch = Prefetch(tmp_path / 'traj', 50, lambda folder: sorted(folder.glob('*.traj')), fail, lambda: False)
    with pytest.raises(RuntimeError, match='prefetch thread failed'):
        prefetch.get()


def test_bench_minibatch_order_is_deterministic():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import learner_bench

    class Fake:
        lengths = np.random.default_rng(3).integers(10, 600, size=2000)
        recipe = learner_bench.ARMS
    first = learner_bench.Bench.batches(Fake, 11)
    second = learner_bench.Bench.batches(Fake, 11)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert learner_bench.same_games([{'items': [], 'critic_items': [], 'old_logp': np.zeros(0), 'reward': 1.0,
                                      'seqs': []}],
                                    [{'items': [], 'critic_items': [], 'old_logp': np.zeros(0), 'reward': 1.0,
                                      'seqs': []}])


def test_single_tables_match_blocks_and_exports_keep_blocks():
    actor, critic, _, _ = fixed_batch()
    modules = table_modules(actor) + table_modules(critic)
    blocked = [module.table().detach() for module in modules]
    learner_speed.single_tables(actor, critic)
    try:
        assert all(learner_speed.single_gemm(module) for module in modules)
        single = [module.table().detach() for module in modules]
        for left, right in zip(blocked, single, strict=True):
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
        copied = copy.deepcopy(actor)
        assert not any(learner_speed.single_gemm(module) for module in table_modules(copied))
        with eager_tables():
            assert not learner_speed.single_gemm(actor.cards)
    finally:
        learner_speed.single_tables(actor, critic, on=False)
    assert not any(learner_speed.single_gemm(module) for module in modules)


def test_fixed_blocks_single_pads_the_last_block():
    from swccg_policy.text_priors import BLOCK, fixed_blocks
    rows = torch.randn(BLOCK * 2 + 5, 7)
    seen = []

    def transform(block):
        seen.append(block.shape[0])
        return block * 2
    assert torch.equal(fixed_blocks(rows, transform, single=True), rows * 2)
    assert seen == [BLOCK * 3]
