"""Plan q-critic stage 2: the learner's option-value advantage (--advantage q). A_Q is centred under pi, the Q update
reaches only the chosen option's output, the default path runs nothing new, checkpoints carry the Q-net only when it
is active, and a learner update (and a restart from its checkpoint) runs on the test fixtures."""
import json
from pathlib import Path
import shutil
import threading

import numpy as np
import pytest
import torch

from swccg_policy import v4_rl
from swccg_policy.ppo_step import PPOConfig
from swccg_policy.qnet import from_actor, q_advantages, q_loss, save_qnet

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'runs/dojo/collect-test/trajectories'
CHECKPOINT = ROOT / 'runs/dojo/focus-s2/final.pt'
NEEDS_CHECKPOINT = pytest.mark.skipif(not CHECKPOINT.exists(), reason='needs the focus-s2 checkpoint')
NEEDS_DATA = pytest.mark.skipif(not DATA.is_dir() or not CHECKPOINT.exists(),
                                reason='needs runs/dojo/collect-test and the focus-s2 checkpoint')
NEEDS_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')


class Table(torch.nn.Module):
    """A toy network: row r of a parameter matrix is its output over the options of decision r (-inf where illegal)."""

    def __init__(self, values):
        super().__init__()
        self.values = torch.nn.Parameter(torch.as_tensor(values, dtype=torch.float32).clone())

    def forward(self, batch):
        return self.values[batch['row']].masked_fill(~batch['omask'], float('-inf'))

    def cache_card_table(self):
        pass

    def clear_tables(self):
        pass


def toy_stack(items, device):
    return {'row': torch.tensor([item['row'] for item in items], device=device),
            'omask': torch.as_tensor(np.stack([item['omask'] for item in items]), device=device)}


def toy_items(valid):
    return [{'tok': [0] * (3 + row % 5), 'row': row, 'omask': mask} for row, mask in enumerate(valid)]


def random_decisions(rows=12, options=6, seed=20261020):
    rng = np.random.default_rng(seed)
    valid = rng.random((rows, options)) < 0.7
    valid[:, 0] = True
    return rng, valid, rng.normal(size=(rows, options)).astype(np.float32), rng.normal(size=(rows, options))


def test_q_advantage_is_centred_under_pi():
    _, valid, values, logits = random_decisions()
    valid_t = torch.as_tensor(valid)
    pi = torch.softmax(torch.as_tensor(logits).float().masked_fill(~valid_t, float('-inf')), 1)
    q = torch.as_tensor(values).masked_fill(~valid_t, float('-inf'))
    per_option = torch.stack([q_advantages(q, valid_t, pi, torch.full((len(valid),), option))
                              for option in range(valid.shape[1])], 1)
    centred = (pi * per_option.masked_fill(~valid_t, 0.0)).sum(1)
    torch.testing.assert_close(centred, torch.zeros(len(valid)), atol=1e-6, rtol=0)


def test_learner_advantages_use_the_current_policy_over_the_sampled_options():
    rng, valid, values, logits = random_decisions()
    sequences = [[int(rng.choice(np.flatnonzero(mask)))] for mask in valid]
    # One decision with an excluded legal option: pi (as sampled) gives it no weight.
    excluded_row = next(row for row in range(len(valid)) if valid[row].sum() > 2)
    excluded = [[] for _ in valid]
    excluded[excluded_row] = [int(next(o for o in np.flatnonzero(valid[excluded_row]) if o != sequences[excluded_row][0]))]
    actor, qnet = Table(np.where(valid, logits, 0.0)), Table(np.where(valid, values, 0.0))
    advantage, chosen_values = v4_rl.option_advantages(actor, qnet, toy_items(valid), sequences, excluded,
                                                       PPOConfig(bf16=False, actor_tokens=16), stack=toy_stack)
    for row in range(len(valid)):
        options = [o for o in np.flatnonzero(valid[row]) if o not in excluded[row]]
        pi = np.exp(logits[row, options] - logits[row, options].max())
        pi /= pi.sum()
        expected = values[row, sequences[row][0]] - float((pi * values[row, options]).sum())
        assert advantage[row] == pytest.approx(expected, abs=1e-5)
        assert chosen_values[row] == pytest.approx(values[row, sequences[row][0]], abs=1e-6)


def test_q_loss_reaches_only_the_chosen_option():
    _, valid, values, _ = random_decisions()
    chosen = torch.as_tensor([int(np.flatnonzero(mask)[-1]) for mask in valid])
    targets = torch.linspace(-1, 1, len(valid))
    qnet = Table(np.where(valid, values, 0.0))
    loss, _ = q_loss(qnet, toy_stack(toy_items(valid), 'cpu'), chosen, targets)
    loss.backward()
    reached = qnet.values.grad != 0
    expected = torch.zeros_like(reached)
    expected[torch.arange(len(valid)), chosen] = True
    assert torch.equal(reached, expected)
    # The other options' outputs do not enter the loss.
    other = Table(np.where(valid, values, 0.0))
    with torch.no_grad():
        other.values.add_(torch.where(expected, 0.0, 5.0))
    assert float(q_loss(other, toy_stack(toy_items(valid), 'cpu'), chosen, targets)[0].detach()) == pytest.approx(float(loss.detach()))


def test_q_train_moves_only_the_chosen_outputs_towards_the_targets():
    _, valid, values, _ = random_decisions()
    sequences = [[int(np.flatnonzero(mask)[-1])] for mask in valid]
    sequences[3] = []  # no pick: not trained
    returns = np.linspace(-1, 1, len(valid)).astype(np.float32)
    qnet = Table(np.where(valid, values, 0.0))
    before = qnet.values.detach().clone()
    optimizer = torch.optim.AdamW(qnet.parameters(), lr=1e-2, weight_decay=0.0)
    loss, norm = v4_rl.q_train(qnet, optimizer, toy_items(valid), sequences, returns, 4,
                               PPOConfig(bf16=False, actor_tokens=16), 0.5, stack=toy_stack)
    assert loss > 0 and norm > 0
    changed = qnet.values.detach() != before
    for row, sequence in enumerate(sequences):
        if not sequence:
            assert not changed[row].any()
            continue
        assert changed[row].tolist() == [option == sequence[0] for option in range(valid.shape[1])]
        # One step towards the target.
        assert abs(qnet.values[row, sequence[0]].item() - returns[row]) < abs(before[row, sequence[0]].item() - returns[row])


@NEEDS_CHECKPOINT
def test_checkpoint_round_trip_with_and_without_the_qnet(tmp_path):
    from swccg_policy.v4_checkpoint import load, save
    actor, critic, tables, _ = load(CHECKPOINT)
    save(tmp_path / 'plain.pt', actor, critic, tables, {'iteration': 1})
    plain = torch.load(tmp_path / 'plain.pt', map_location='cpu', weights_only=False)
    assert set(plain) == {'cfg', 'critic_cfg', 'tables', 'vocab', 'table_sha', 'state', 'value_state', 'rl_meta'}
    qnet = from_actor(actor)
    with torch.no_grad():
        qnet.score[-1].bias.fill_(0.25)
    optimizer = torch.optim.AdamW(qnet.parameters(), lr=1e-4, weight_decay=0.0)
    save(tmp_path / 'q.pt', actor, critic, tables, {'iteration': 1}, extra=v4_rl.q_extra(qnet, optimizer))
    payload = torch.load(tmp_path / 'q.pt', map_location='cpu', weights_only=False)
    assert set(payload) - set(plain) == {'qnet_state', 'qnet_optimizer'}
    restored = from_actor(actor)
    restored.load_state_dict(payload['qnet_state'])
    for (name, a), (_, b) in zip(qnet.state_dict().items(), restored.state_dict().items()):
        assert torch.equal(a, b), name
    reloaded_actor = load(tmp_path / 'q.pt')[0]  # load ignores the Q-net's entries
    for (name, a), (_, b) in zip(actor.state_dict().items(), reloaded_actor.state_dict().items()):
        assert torch.equal(a, b), name
    with pytest.raises(ValueError, match='clash'):
        save(tmp_path / 'bad.pt', actor, critic, tables, {}, extra={'state': {}})


@pytest.fixture
def learner_globals():
    """learn() sets process-wide torch settings; restore them for the other tests."""
    saved = (torch.get_num_threads(), torch.backends.mha.get_fastpath_enabled(),
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    yield
    torch.set_num_threads(saved[0])
    torch.backends.mha.set_fastpath_enabled(saved[1])
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[2:]


def _run_dir(tmp_path, files):
    run = tmp_path / 'run'
    (run / 'traj').mkdir(parents=True, exist_ok=True)
    for path in files:
        shutil.copyfile(path, run / 'traj' / path.name)
        shutil.copyfile(path.with_name(path.name + '.meta.json'), run / 'traj' / (path.name + '.meta.json'))
    return run


def _learner(run, *extra):
    # Retention far above what one update writes: the learner deletes nothing here. A STOP file after 5 minutes ends
    # a learner that waits for trajectories (the assertions then fail instead of hanging).
    guard = threading.Timer(300, (run / 'STOP').touch)
    guard.start()
    try:
        v4_rl.main(['--run', str(run), '--initial', str(CHECKPOINT), '--max-iterations', '1', '--batch-steps', '150',
                    '--minibatch', '16', '--actor-tokens', '8192', '--critic-tokens', '8192', '--keep-consumed', '100',
                    '--keep-ckpt', '100', '--keep-snapshots', '100', *extra])
    finally:
        guard.cancel()
    return [json.loads(line) for line in (run / 'learner.jsonl').read_text().splitlines()]


def _fixture_files():
    from swccg_policy.trajectory import trajectory_files
    return sorted(trajectory_files(DATA))


@NEEDS_DATA
@NEEDS_GPU
def test_default_learner_runs_nothing_new(tmp_path, monkeypatch, learner_globals):
    def forbidden(*args, **kwargs):
        raise AssertionError('the default --advantage gae path called the Q-net code')
    for name in ('q_setup', 'option_advantages', 'q_train', 'q_extra'):
        monkeypatch.setattr(v4_rl, name, forbidden)
    run = _run_dir(tmp_path, _fixture_files()[:1])
    rows = _learner(run, '--actor-fp32', '--packed-attention', '--single-tables')
    assert len(rows) == 1 and not any(key.startswith(('q_', 'qnet')) for key in rows[0])
    assert set(rows[0]['phases']) == {'load', 'values_gae', 'ppo', 'checkpoint'}
    payload = torch.load(run / 'ckpt/latest.pt', map_location='cpu', weights_only=False)
    assert 'qnet_state' not in payload and 'qnet_optimizer' not in payload


@NEEDS_DATA
@NEEDS_GPU
def test_learner_q_update_and_restart(tmp_path, learner_globals):
    from swccg_policy.v4_checkpoint import load
    actor, _, tables, _ = load(CHECKPOINT)
    save_qnet(tmp_path / 'qnet-init.pt', from_actor(actor), tables.sha)
    files = _fixture_files()
    run = _run_dir(tmp_path, files[:1])
    rows = _learner(run, '--actor-fp32', '--packed-attention', '--single-tables', '--advantage', 'q',
                    '--qnet', str(tmp_path / 'qnet-init.pt'))
    first = rows[-1]
    # An untrained Q-net is 0 everywhere: A_Q is 0, the Q loss is the targets' mean square.
    assert first['qnet_source'] == str(tmp_path / 'qnet-init.pt')
    assert first['q_adv_std'] == 0 and first['corr_q_gae'] is None and first['q_mean'] == 0
    assert np.isfinite(first['q_loss']) and first['q_loss'] > 0 and first['q_seconds'] > 0
    assert {'q_advantage', 'q_train'} <= set(first['phases'])
    payload = torch.load(run / 'ckpt/latest.pt', map_location='cpu', weights_only=False)
    assert {'qnet_state', 'qnet_optimizer'} <= set(payload)
    assert payload['qnet_state']['score.3.bias'].abs().item() > 0  # trained away from the zeroed start
    # A restart continues the checkpoint's Q-net (no --qnet), with non-zero advantages now.
    _run_dir(tmp_path, files[1:3])
    rows = _learner(run, '--actor-fp32', '--packed-attention', '--single-tables', '--advantage', 'q')
    second = rows[-1]
    assert second['qnet_source'] == 'checkpoint'
    assert second['q_adv_std'] > 0 and second['corr_q_gae'] is not None and np.isfinite(second['q_loss'])



def test_stage_passes_the_q_flags_only_when_set():
    source = (ROOT / 'scripts/v31_stage.py').read_text()
    assert "*(['--advantage', 'q'] if getattr(args, 'advantage', 'gae') == 'q' else [])" in source
    assert "*(['--qnet', str(args.qnet)] if getattr(args, 'qnet', None) else [])" in source
    assert "*(['--qnet-lr', str(args.qnet_lr)] if getattr(args, 'qnet_lr', None) is not None else [])" in source
    for flag in ("'--advantage', choices=['gae', 'q'], default='gae'", "'--qnet', type=Path", "'--qnet-lr', type=float)"):
        assert flag in source
