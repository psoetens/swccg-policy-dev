"""Plan multiselect-count §1.2: the Done option template.

- Growth: the tables grow by the Done template through the existing path (text prior, prefix extension); a grown model
  computes exactly its parent's function on decisions without a Done row, and a Done row gets the new embedding row.
- Learner smoke: a trajectory with the steps of a step-by-step selection (a card step with nothing excluded, a step with
  the selected card excluded and Done picked) is written and read back (SWTRJ003) and goes through the learner's PPO
  backward; its log-probabilities are the sampler's (k = 1 over the options that are not excluded, Done a normal row).
"""
import copy

import numpy as np
import torch

from add_done_template import DONE, grow_done
from test_efficient_model import fixture
from swccg_policy.efficient import EfficientCritic, EfficientPolicy
from swccg_policy.ppo_step import PPOConfig, backward, seq_logprob, sequence_tensors
from swccg_policy.v4_data import parse, stack
from swccg_policy.v4_trajectory import TrajectoryReader, write_binary


def prior(texts):
    return np.full((len(texts), 768), 0.5, np.float16)


def card_selection(record, done):
    """The fixture decision as a card selection of two cards, optionally with the Done row last (StateEncoder)."""
    row = copy.deepcopy(record)
    row[4] = "CARD_SELECTION"
    row[9] = [["<card>", [], 0, None, "1", None], ["<card>", [], 1, None, "2", None]]
    if done:
        row[9].append([DONE, [], -1, None, "", None])
    return row


def test_done_growth_keeps_function_without_done_rows():
    tables, cfg, record = fixture()
    actor = EfficientPolicy(copy.deepcopy(cfg), tables).eval()
    critic = EfficientCritic(copy.deepcopy(cfg), tables).eval()
    grown = grow_done(tables, prior)
    tables.assert_prefix_of(grown)
    o = grown.vocabulary.payload["o"]
    assert o[DONE] == len(o) - 1 and DONE not in tables.vocabulary.payload["o"]
    batches = [stack([parse(record, tables)]), stack([parse(card_selection(record, False), tables)])]
    with torch.no_grad():
        expected = [(actor(b), critic(b)) for b in batches]
        actor.grow(tables, grown)
        critic.encoder.grow(tables, grown)
        for b, (logits, value) in zip(batches, expected):
            assert torch.equal(actor(b), logits)
            assert torch.equal(critic(b), value)
        with_done = parse(card_selection(record, True), grown)
        assert with_done["opt"][-1, 0] == o[DONE] and with_done["opt"][-1, 5] == 0  # own row, not a "none" option
        logits = actor(stack([with_done]))
        assert torch.isfinite(logits[0, :3]).all()
    # Before growth the Done template is an unknown key (a hashed OOV bucket), never an error.
    assert parse(card_selection(record, True), tables)["opt"][-1, 0] < 512


def test_learner_reads_and_trains_on_done_steps(tmp_path):
    tables, cfg, record = fixture()
    tables = grow_done(tables, prior)
    cfg.n_otmpl += 1
    actor = EfficientPolicy(copy.deepcopy(cfg), tables).eval()
    critic = EfficientCritic(copy.deepcopy(cfg), tables).eval()
    steps = [card_selection(record, False), card_selection(record, True)]
    items = [parse(row, tables) for row in steps]
    critic_items = []
    for row in steps:
        privileged = copy.deepcopy(row)
        privileged.extend([[token for token in row[8] if token[0] != 10], 7, row[11]])
        critic_items.append(parse(privileged, tables, critic=True))
    seqs, excluded = [[0], [2]], [None, [0]]  # first card, then Done with the first card excluded
    with torch.no_grad():
        logits = actor(stack(items)).float()
    old = []
    for index, item in enumerate(items):
        allowed = [i for i in range(len(item["opt"])) if i not in (excluded[index] or [])]
        log_probs = torch.log_softmax(logits[index, allowed], 0)
        old.append(float(log_probs[allowed.index(seqs[index][0])]))
    game = {"items": items, "critic_items": critic_items, "seqs": seqs, "excluded": excluded,
            "policy_versions": [3, 3], "old_logp": old, "knowledge_sequences": [1, 1], "repeats": [1, 1], "ks": [1, 1],
            "g": "game", "s": "ds", "o": "finished", "w": "ds", "reward": 1.0, "won": 1}
    path = tmp_path / "part-0.traj"
    write_binary(path, [game], tables)
    loaded = TrajectoryReader(tables).load(path)[0]
    assert loaded["seqs"] == seqs and loaded["excluded"] == excluded and loaded["ks"] == [1, 1]
    assert loaded["items"][1]["opt"][-1, 0] == tables.vocabulary.payload["o"][DONE]
    batch = stack(loaded["items"])
    picks, lengths, mask = sequence_tensors(np.arange(2), loaded["seqs"], loaded["excluded"], batch["omask"])
    with torch.no_grad():
        new = seq_logprob(actor(batch).float(), batch["omask"], batch["opt"][:, :, 5] > 0, picks, lengths, mask)
    np.testing.assert_allclose(new.numpy(), loaded["old_logp"], atol=1e-5)
    assert float(new[1]) > float(torch.log_softmax(logits[1], 0)[2])  # the excluded card leaves Done more mass
    statistics, counts = backward(actor.train(), critic.train(), loaded["items"], loaded["critic_items"], np.arange(2),
                                  loaded["seqs"], loaded["excluded"], loaded["old_logp"], np.array([0.5, 0.5], np.float32),
                                  np.array([1.0, 1.0], np.float32), stack, PPOConfig())
    assert counts["actor_microbatches"] >= 1
    assert all(torch.isfinite(torch.as_tensor(value)).all() for value in statistics.values())
    grad = actor.otmpl.residual.weight.grad
    assert grad is not None and grad[tables.vocabulary.payload["o"][DONE]].abs().sum() > 0  # the Done row learns


# ---------------------------------------------------------------- stage B: the "already selected" option flag

def selection_rows(record):
    """A card selection with the first card already selected (seventh option field 1), Done offered."""
    row = card_selection(record, True)
    row[9][0] = row[9][0] + [1]
    return row


def extended_models():
    from swccg_policy.efficient import EfficientConfig
    tables, cfg, record = fixture()
    tables = grow_done(tables, prior)
    cfg.n_otmpl += 1
    parent = EfficientPolicy(copy.deepcopy(cfg), tables).eval()
    extended = EfficientPolicy(EfficientConfig(**{**cfg.to_dict(), "option_selected": True}), tables).eval()
    missing, unexpected = extended.load_state_dict(parent.state_dict(), strict=False)
    assert not unexpected and missing == ["option_selected.weight"]
    assert list(dict(extended.named_parameters()))[-1] == "option_selected.weight"  # registered last
    return tables, cfg, record, parent, extended


def test_selected_flag_parses_and_validates():
    from swccg_policy.v4_contract import validate
    tables, _, record, _, _ = extended_models()
    item = parse(selection_rows(record), tables)
    assert item["opt"][:, 1].tolist() == [1, 0, 0]
    assert parse(card_selection(record, True), tables)["opt"][:, 1].tolist() == [0, 0, 0]  # six fields = not selected
    item["opt"][0, 1] = 2
    try:
        validate(item, tables.vocabulary, len(tables.cards.ids), actor=True)
        raise AssertionError("a flag outside {0, 1} must be refused")
    except ValueError:
        pass


def test_option_selected_extension_keeps_function_and_is_wired(tmp_path):
    from swccg_policy.v4_native import export_ts
    tables, _, record, parent, extended = extended_models()
    rows = [record, card_selection(record, False), card_selection(record, True), selection_rows(record)]
    with torch.no_grad():
        for row in rows:
            batch = stack([parse(row, tables)])
            assert torch.equal(extended(batch), parent(batch))
            assert torch.equal(parent(batch), parent(stack([parse(row, tables)])))
        flagged, plain = stack([parse(selection_rows(record), tables)]), stack([parse(card_selection(record, True), tables)])
        assert torch.equal(parent(flagged), parent(plain))  # the parent ignores the flag
        torch.nn.init.normal_(extended.option_selected.weight, std=0.5)
        changed = (extended(flagged) - extended(plain)).abs()
        assert changed[0, 0] > 1e-4  # the flagged option's logit moves
    traced = export_ts(extended, tmp_path / "actor.ts")
    reloaded = torch.jit.load(str(tmp_path / "actor.ts"))
    head = torch.cat([flagged[key][:, None] for key in ("dtype", "qtmpl", "qbp", "side")] + [flagged["glob"]], 1)
    arguments = [flagged[key] for key in ("tok", "tokf", "tmask", "opt", "omask")] + [head, flagged["kgrid"]]
    with torch.no_grad():
        torch.testing.assert_close(traced(*arguments), extended(flagged), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(reloaded(*arguments), extended(flagged), atol=1e-5, rtol=1e-5)


def test_extend_selection_script_keeps_function_and_optimizer_state(tmp_path):
    from extend_selection import extend
    from swccg_policy.v4_checkpoint import load, save
    tables, cfg, record = fixture()
    actor = EfficientPolicy(copy.deepcopy(cfg), tables).eval()
    critic = EfficientCritic(copy.deepcopy(cfg), tables).eval()
    optimizers = [torch.optim.AdamW(actor.parameters(), lr=1e-4), torch.optim.AdamW(critic.parameters(), lr=1e-4)]
    batch = stack([parse(record, tables)])
    for model, optimizer in zip((actor, critic), optimizers):  # one step, so every parameter has moments
        model.train()
        model(batch).float().sum().backward()
        optimizer.step()
        model.eval()
    source, out = tmp_path / "parent.pt", tmp_path / "extended.pt"
    save(source, actor, critic, tables, {"iteration": 3}, optimizers)
    report = extend(source, out, embed=prior)
    assert report["fresh_actor_parameters"] == 1 and report["optimizer_state"]
    extended, ext_critic, ext_tables, payload = load(out)
    assert extended.cfg.option_selected and DONE in ext_tables.vocabulary.payload["o"]
    with torch.no_grad():
        assert torch.equal(extended(batch), actor(batch))
        assert torch.equal(ext_critic(batch), critic(batch))
        assert torch.equal(extended(stack([parse(selection_rows(record), ext_tables)]))[0, :2],
                           extended(stack([parse(card_selection(record, True), ext_tables)]))[0, :2])
    state = payload["optimizer_state"][0]["state"]
    names = list(dict(extended.named_parameters()))
    residual = state[names.index("otmpl.residual.weight")]["exp_avg"]
    assert residual.shape[0] == len(ext_tables.vocabulary.payload["o"]) and not residual[-1].any()
    assert names.index("option_selected.weight") not in state


def test_learner_reads_the_selected_column(tmp_path):
    tables, cfg, record, _, actor = extended_models()
    critic = EfficientCritic(copy.deepcopy(cfg), tables).eval()
    rows = [card_selection(record, False), selection_rows(record)]
    items = [parse(row, tables) for row in rows]
    critic_items = []
    for row in rows:
        privileged = copy.deepcopy(row)
        privileged.extend([[token for token in row[8] if token[0] != 10], 7, row[11]])
        critic_items.append(parse(privileged, tables, critic=True))
    game = {"items": items, "critic_items": critic_items, "seqs": [[0], [2]], "excluded": [None, [0]],
            "policy_versions": [3, 3], "old_logp": [-0.7, -0.7], "knowledge_sequences": [1, 1], "repeats": [1, 1],
            "ks": [1, 1], "g": "game", "s": "ds", "o": "finished", "w": "ds", "reward": 1.0, "won": 1}
    path = tmp_path / "part-0.traj"
    write_binary(path, [game], tables)
    loaded = TrajectoryReader(tables).load(path)[0]
    assert loaded["items"][1]["opt"][:, 1].tolist() == [1, 0, 0] and not loaded["items"][0]["opt"][:, 1].any()
    statistics, _ = backward(actor.train(), critic.train(), loaded["items"], loaded["critic_items"], np.arange(2),
                             loaded["seqs"], loaded["excluded"], loaded["old_logp"], np.array([1.0, 1.0], np.float32),
                             np.array([1.0, 1.0], np.float32), stack, PPOConfig())
    assert all(torch.isfinite(torch.as_tensor(value)).all() for value in statistics.values())
    gradient = actor.option_selected.weight.grad
    assert gradient is not None and gradient.abs().sum() > 0  # the flag reaches the new projection
