"""Plan critic-v2: value lambda, win labels and loss, critic weight decay groups, accumulation, warm-up, the fresh
critic (reinit_critic.py) and the v2 critic's forward pass. Defaults must keep the previous behaviour."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from swccg_policy.ppo_step import PPOConfig, backward, win_loss
from swccg_policy.v4_rl import advantages, critic_groups, critic_warmup, rank_auc, win_labels

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
DATA = ROOT / 'runs/dojo/collect-test/trajectories'
CHECKPOINT = ROOT / 'runs/dojo/focus-s2/final.pt'
NEEDS_DATA = pytest.mark.skipif(not DATA.is_dir() or not CHECKPOINT.exists(),
                                reason='needs runs/dojo/collect-test and the focus-s2 checkpoint')
NEEDS_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')


def toy_games():
    rng = np.random.default_rng(912101)
    games = []
    for count, reward in ((5, 1.0), (3, -1.0), (4, 0.0)):
        games.append({'items': [None] * count, 'reward': reward,
                      'dojo_rewards': [[1, 0.25, 'x', 'y']] if count > 3 else []})
    values = rng.normal(size=12).astype(np.float32)
    return games, values


def test_value_lambda_default_is_identical():
    games, values = toy_games()
    base_adv, base_ret = advantages(games, values)
    same_adv, same_ret = advantages(games, values, value_lam=0.95)
    assert np.array_equal(base_adv, same_adv)
    np.testing.assert_allclose(same_ret, base_ret, rtol=0, atol=1e-6)


def test_value_lambda_one_is_monte_carlo():
    games, values = toy_games()
    adv, returns = advantages(games, values, value_lam=1.0)
    expected = []
    for game in games:
        rewards = np.zeros(len(game['items']))
        rewards[-1] = game['reward']
        for entry in game['dojo_rewards']:
            rewards[entry[0]] += entry[1]
        expected.extend(np.cumsum(rewards[::-1])[::-1])
    np.testing.assert_allclose(returns, expected, atol=1e-5)
    assert np.array_equal(adv, advantages(games, values)[0])  # the advantages keep lambda 0.95


def test_win_labels_and_masked_loss():
    games, _ = toy_games()
    labels = win_labels(games)
    assert labels[:5].tolist() == [1.0] * 5 and labels[5:8].tolist() == [0.0] * 3 and np.isnan(labels[8:]).all()
    logits = torch.tensor([2.0, -1.0, 5.0])
    targets = torch.tensor([1.0, 0.0, float('nan')])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[:2], targets[:2])
    torch.testing.assert_close(win_loss(logits, targets), expected)
    assert float(win_loss(logits, torch.full((3,), float('nan')))) == 0.0


def test_rank_auc_matches_sklearn():
    rng = np.random.default_rng(912102)
    scores = np.round(rng.normal(size=200), 1)  # with ties
    labels = rng.random(200) < 0.4
    assert rank_auc(scores, labels) == pytest.approx(roc_auc_score(labels, scores))
    assert rank_auc(scores, np.ones(200, bool)) is None


def test_critic_groups_skip_biases_and_norms():
    model = torch.nn.Sequential(torch.nn.Embedding(4, 3), torch.nn.Linear(3, 3), torch.nn.LayerNorm(3))
    assert list(critic_groups(model, 0.0)) == list(model.parameters())
    decay, plain = critic_groups(model, 0.01)
    assert decay['weight_decay'] == 0.01 and plain['weight_decay'] == 0.0
    assert {id(p) for p in decay['params']} == {id(model[0].weight), id(model[1].weight)}
    assert {id(p) for p in plain['params']} == {id(model[1].bias), id(model[2].weight), id(model[2].bias)}


def test_warmup_counts_from_the_run_start():
    assert critic_warmup({}, 'runs/a', 1862, 0) is None
    state = critic_warmup({}, 'runs/a', 1862, 20)
    assert state == {'run': 'runs/a', 'until': 1882}
    # a restart of the same run keeps the original end; a new run starts its own
    assert critic_warmup({'critic_warmup': state}, 'runs/a', 1870, 20) == state
    assert critic_warmup({'critic_warmup': state}, 'runs/b', 1900, 20) == {'run': 'runs/b', 'until': 1920}


def _load(device):
    import critic_diagnostics as cd
    from swccg_policy.v4_checkpoint import load
    from swccg_policy.v4_trajectory import TrajectoryReader
    from swccg_policy.trajectory import trajectory_files
    cd.setup()
    actor, critic, tables, checkpoint = load(CHECKPOINT, device)
    games = TrajectoryReader(tables, []).load(sorted(trajectory_files(DATA))[0])
    return actor, critic, tables, checkpoint, games


@pytest.fixture(scope='module')
def reinit_checkpoint(tmp_path_factory):
    """One fresh-critic checkpoint (425 MB) per test run, deleted afterwards: /tmp is a RAM disk and pytest keeps the
    temporary folders of its last three runs."""
    from reinit_critic import CRITIC_V2, reinit
    out = tmp_path_factory.mktemp('critic_v2') / 'reinit.pt'
    reinit(CHECKPOINT, out, 912001, CRITIC_V2)
    yield out
    out.unlink(missing_ok=True)


@NEEDS_DATA
def test_reinit_keeps_the_actor(reinit_checkpoint):
    from swccg_policy.v4_checkpoint import load
    out = reinit_checkpoint
    old_actor, _, _, old = load(CHECKPOINT, 'cpu')
    actor, critic, _, new = load(out, 'cpu')
    for (name, a), (_, b) in zip(old_actor.state_dict().items(), actor.state_dict().items(), strict=True):
        assert torch.equal(a, b), name
    assert critic.cfg.win_head and critic.cfg.input_scaling == 'v2' and critic.cfg.component_norm_scope == 'all'
    assert len(new['optimizer_state']) == 1 and new['optimizer_state'][0]['state'].keys() == old['optimizer_state'][0]['state'].keys()
    assert new['rl_meta']['iteration'] == old['rl_meta']['iteration'] and 'critic_reinit' in new['rl_meta']


@NEEDS_DATA
@NEEDS_GPU
def test_v2_critic_forward_and_accumulation(reinit_checkpoint):
    """The v2 critic runs on real items; two half-weighted minibatches accumulate to the gradient of their union."""
    from swccg_policy.v4_checkpoint import load
    from swccg_policy.v4_data import stack
    actor, critic, _, _ = load(reinit_checkpoint, 'cuda')
    games = _load('cpu')[4]
    items = [item for game in games[:2] for item in game['critic_items']][:16]
    with torch.no_grad():
        value, logits = critic.heads(stack(items, 'cuda'))
    assert value.shape == logits.shape == (len(items),) and torch.isfinite(value).all() and torch.isfinite(logits).all()
    torch.testing.assert_close(critic(stack(items, 'cuda')), value)

    actor_items = [item for game in games[:2] for item in game['items']][:16]
    returns = np.linspace(-1, 1, 16).astype(np.float32)
    wins = np.array([1.0, 0.0, np.nan, 1.0] * 4, np.float32)
    config = PPOConfig(bf16=False, win_coef=0.5)
    common = dict(sequences=[[0]] * 16, exclusions=[[]] * 16, old_logp=np.zeros(16, np.float32),
                  advantages=np.zeros(16, np.float32), returns=returns, stack=stack, config=config,
                  policy_on=False, wins=wins)
    whole = copy.deepcopy(critic)
    backward(actor, whole, actor_items, items, np.arange(16), **common)
    parts = copy.deepcopy(critic)
    for half in (np.arange(8), np.arange(8, 16)):
        backward(actor, parts, actor_items, items, half, critic_scale=0.5, **common)
    for (name, a), (_, b) in zip(whole.named_parameters(), parts.named_parameters()):
        if a.grad is not None:
            # the win loss is a masked mean per minibatch; equal halves with equal masks keep the identity
            torch.testing.assert_close(b.grad, a.grad, rtol=1e-3, atol=1e-6, msg=name)


@NEEDS_DATA
def test_legacy_checkpoint_loads_with_defaults():
    _, critic, _, _, _ = _load('cpu')
    assert critic.cfg.input_scaling == 'legacy' and not critic.cfg.win_head and not hasattr(critic, 'win')


def test_early_aucs_first_and_first_fifth():
    """Plan actor-v7-recipe: first decision and first fifth of each seat's decisions; undecided seats left out."""
    from swccg_policy.v4_rl import early_aucs
    games = [{'items': [0] * 10, 'reward': 1.0}, {'items': [0] * 10, 'reward': -1.0}, {'items': [0] * 5, 'reward': 0.0}]
    values = np.zeros(25, np.float32)
    values[0:2], values[10:12] = 0.9, 0.1      # early decisions rank the winner first
    values[2:10], values[12:20] = 0.1, 0.9     # later ones the loser (not part of either AUC)
    values[20:25] = 5.0                        # an undecided seat: ignored
    result = early_aucs(games, values)
    assert result == {'value_first_auc': 1.0, 'value_early_auc': 1.0}
    values[10] = 0.95
    assert early_aucs(games, values)['value_first_auc'] == 0.0
    assert early_aucs(games[:1], values[:10]) == {'value_first_auc': None, 'value_early_auc': None}
