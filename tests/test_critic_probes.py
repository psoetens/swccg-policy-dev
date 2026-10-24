"""Plan critic-probes: the component decomposition, owner pooling, split hygiene, card scaling, top-of-reserve targets."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
DATA = ROOT / 'runs/dojo/collect-test/trajectories'
CHECKPOINT = ROOT / 'runs/dojo/focus-s2/final.pt'
GPU_DATA = pytest.mark.skipif(not torch.cuda.is_available() or not DATA.is_dir() or not CHECKPOINT.exists(),
                              reason='needs a GPU, runs/dojo/collect-test and the focus-s2 checkpoint')


def _setup():
    import critic_probes as cp
    from swccg_policy.trajectory import trajectory_files
    from swccg_policy.v4_data import stack
    from swccg_policy.v4_trajectory import TrajectoryReader
    cp.cd.setup()
    actor, critic, tables, _ = cp.load_models('s2')
    game = TrajectoryReader(tables, []).load(sorted(trajectory_files(DATA))[0])[0]
    return cp, actor, critic, game, stack


@GPU_DATA
def test_components_sum_to_the_model_input():
    cp, actor, critic, game, stack = _setup()
    for model, items in ((critic.encoder, game['critic_items']), (actor, game['items'])):
        batch = stack(items[:6], 'cuda')
        with torch.no_grad():
            total = sum(cp.components(model, batch).values())
            states = cp.run(model, batch)
        torch.testing.assert_close(total, states['input'], rtol=0, atol=1e-4 * float(total.abs().max()))


@GPU_DATA
def test_card_scaling_changes_only_the_card_part():
    cp, _, critic, game, stack = _setup()
    batch = stack(game['critic_items'][:4], 'cuda')
    with torch.no_grad():
        before = cp.components(critic.encoder, batch)
        cp.scale_card(critic.encoder, 0.01)
        after = cp.components(critic.encoder, batch)
    for name in before:
        if name == 'card':
            torch.testing.assert_close(after[name], before[name] * 0.01)
        elif name != 'degree':  # degree depends only on edges; it is recomputed but identical
            assert torch.equal(after[name], before[name]), name
    torch.testing.assert_close(after['degree'], before['degree'])


def test_owner_means_by_hand():
    import critic_probes as cp
    tok = torch.zeros(1, 4, 8, dtype=torch.long)
    tok[0, :, 3] = torch.tensor([1, 1, 0, 1])     # owner: mine, mine, theirs, mine
    tok[0, 3, 0] = 2                              # the last token is knowledge, not a card
    batch = {'tok': tok, 'tmask': torch.tensor([[True, True, True, True]])}
    states = torch.tensor([[[1.0, 0], [3.0, 2], [5.0, 5], [100.0, 100]]])
    torch.testing.assert_close(cp.owner_means(states, batch), torch.tensor([[2.0, 1.0, 5.0, 5.0]]))


def test_top_of_reserve_extraction():
    import critic_probes as cp
    tok = np.zeros((5, 8), np.int32)
    tokf = np.zeros((5, 8), np.float32)
    # my reserve: blueprint 10 at index 2, blueprint 11 on top (zone 7, index 1); their reserve: blueprint 20 on top;
    # my hand: blueprint 12 (index 0, not a pile).
    for row, (bp, zone, owner, b) in enumerate([(10, 18, 1, 2), (11, 7, 1, 1), (20, 7, 0, 1), (12, 15, 1, 0),
                                                 (30, 0, 0, 0)]):
        tok[row, [1, 2, 3]] = bp, zone, owner
        tokf[row, 1] = b
    mine, theirs = cp.tops({'tok': tok, 'tokf': tokf})
    assert mine[0] == 11 and mine[1] == 1 and sorted(mine[2]) == [10, 11]
    assert theirs[0] == 20 and theirs[2] == [20]


@pytest.mark.skipif(not (ROOT / 'runs/dojo/collect-s2b/trajectories').is_dir(), reason='needs collect-s2b')
def test_train_split_excludes_evaluation_games():
    import critic_probes as cp
    train, held = cp.cd.split_files(ROOT / 'runs/dojo/collect-s2b/trajectories')
    held_games = cp.eval_games(held)
    train_games = {m['g'] for p in train for m in cp.cd.scan_meta(p)}
    overlap = train_games & held_games
    # the filter is needed (seats of one game can land in different files) and the samplers apply it
    kept = {m['g'] for p in train for m in cp.cd.scan_meta(p) if m['g'] not in held_games}
    assert not (kept & held_games)
    print('games in both splits before filtering:', len(overlap))
