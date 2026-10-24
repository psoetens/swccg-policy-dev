"""Real native behavior probabilities and FP32/BF16 independent PPO gradient checks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from swccg_policy.ppo_step import PPOConfig, backward, policy_loss, sequence_tensors
from swccg_policy.rl import seq_logprob
from swccg_policy.trajectory import trajectory_files
from swccg_policy.v4_checkpoint import load
from swccg_policy.v4_data import stack
from swccg_policy.v4_rl import advantages, predict_values
from swccg_policy.v4_trajectory import load_binary


def gradients(model):
    return torch.cat([parameter.grad.flatten() for parameter in model.parameters() if parameter.grad is not None])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--trajectories', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--logp-only', action='store_true', help='Audit a trained export without repeating gradient tests')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    actor, critic, tables, _ = load(args.checkpoint, 'cuda')
    games = [game for path in trajectory_files(args.trajectories) for game in load_binary(path, tables)]
    items = [item for game in games for item in game['items']]
    privileged = [item for game in games for item in game['critic_items']]
    sequences = [sequence for game in games for sequence in game['seqs']]
    exclusions = [excluded for game in games for excluded in game['excluded']]
    old = np.concatenate([game['old_logp'] for game in games])
    actor.cache_card_table()
    errors = []
    with torch.no_grad():
        for index, item in enumerate(items):
            batch = stack([item], 'cuda')
            picks, lengths, excluded = sequence_tensors([index], sequences, exclusions, batch['omask'])
            logits = actor(batch)
            probability = seq_logprob(logits, batch['omask'], batch['opt'][:, :, 5] > 0,
                                       picks, lengths, excluded)
            errors.append(abs(float(probability[0]) - float(old[index])))
    actor.clear_tables()
    report = {'games': len(games), 'decisions': len(items), 'table_sha': tables.sha,
              'behavior_logp': {'maximum_absolute': max(errors), 'p99_absolute': float(np.quantile(errors, .99)),
                                'over_1e6': sum(error > 1e-6 for error in errors)}, 'gradients': {},
              'scope': 'Behavior probabilities only' if args.logp_only else 'Behavior probabilities and gradients'}
    if not args.logp_only:
        values = predict_values(critic, privileged, PPOConfig(bf16=False), 64)
        advantage, returns = advantages(games, values)
        order = np.argsort([len(item['tok']) for item in items])
        selected = order[np.linspace(0, len(order) - 1, 12).astype(int)]
    # Use one fixed logical batch, independently packed into deliberately small budgets.
    for bf16 in (() if args.logp_only else (False, True)):
        config = PPOConfig(actor_tokens=1500, critic_tokens=2200, bf16=bf16)
        actor.zero_grad(set_to_none=True)
        critic.zero_grad(set_to_none=True)
        batch = stack([items[index] for index in selected], 'cuda')
        critic_batch = stack([privileged[index] for index in selected], 'cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            logits = actor(batch)
            prediction = critic(critic_batch)
        picks, lengths, excluded = sequence_tensors(selected, sequences, exclusions, batch['omask'])
        loss, _ = policy_loss(logits, batch, picks, lengths, excluded,
                              torch.as_tensor(old[selected], device='cuda'),
                              torch.as_tensor(advantage[selected], device='cuda'), config)
        value_loss = F.mse_loss(prediction.float(), torch.as_tensor(returns[selected], device='cuda'))
        (loss + config.value_coef * value_loss).backward()
        reference = [gradients(model).clone() for model in (actor, critic)]
        del logits, prediction, loss, value_loss, batch, critic_batch
        actor.zero_grad(set_to_none=True)
        critic.zero_grad(set_to_none=True)
        _, counts = backward(actor, critic, items, privileged, selected, sequences, exclusions,
                             old, advantage, returns, stack, config)
        precision = 'bf16' if bf16 else 'fp32'
        report['gradients'][precision] = {'packing': counts, 'models': {}}
        for name, expected, model in zip(('actor', 'critic'), reference, (actor, critic), strict=True):
            actual = gradients(model)
            difference = (actual - expected).abs()
            # BF16 has ~0.004 spacing at magnitude 0.5. Different GEMM shapes
            # and loss scaling can differ by a rounding unit near cancellation.
            atol, rtol = (5e-3, .03) if bf16 else (2e-5, 2e-4)
            report['gradients'][precision]['models'][name] = {
                'maximum_absolute': float(difference.max()),
                'relative_l2': float(difference.norm() / expected.norm().clamp_min(1e-8)),
                'atol': atol, 'rtol': rtol, 'passed': bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))}
        del reference
    report['passed'] = report['behavior_logp']['over_1e6'] == 0 and all(
        model['passed'] for result in report['gradients'].values() for model in result['models'].values())
    paths = [Path(__file__), Path('src/swccg_policy/ppo_step.py'), Path('src/swccg_policy/v4_rl.py')]
    report['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
