"""Plan critic-probes: where does the critic lose (or never get) the information to know who is favoured?

    critic_probes.py census      P-1 input-scale census (all checkpoints, actor and critic) + /60 site ranges
    critic_probes.py owner       P-2 owner / zone decodability through depth (fp32 and bf16)
    critic_probes.py matchup     P-3 matchup information per feature set (first decisions, oracle logit and outcome)
    critic_probes.py globals     P-4 global facts in G (input), G' and Q'
    critic_probes.py order       P-5 top-of-reserve verification and the destiny learnability probe (--variant A|B|C|D|E --seed)
    critic_probes.py bf16        P-6 critic-diagnostics D-2 cells with and without the card part scaled to the actor's norm

Read-only on checkpoints and model code: the input sum is reimplemented per component (tested against the model's own
input to the graph rounds), deeper states come from forward hooks, and the P-5 / P-6 card scaling patches the loaded
instance only. Data: runs/dojo/collect-s2b with critic-diagnostics' file split.
"""
import argparse
import json
import random
import zlib
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import critic_diagnostics as cd  # noqa: E402
from swccg_policy.efficient import grid_features, unpack_location_icons, unpack_location_state  # noqa: E402
from swccg_policy.ppo_step import microbatches  # noqa: E402
from swccg_policy.v4_checkpoint import load  # noqa: E402
from swccg_policy.v4_data import stack  # noqa: E402
from swccg_policy.v4_edges import degrees, sparse_edges  # noqa: E402
from swccg_policy.v4_trajectory import TrajectoryReader  # noqa: E402

CHECKPOINTS = {
    'amendment9-c6': ROOT / 'runs/knowledge-v3.1/pipeline-amendment9-v31-c6/final.pt',
    'a10c': ROOT / 'runs/knowledge-v3.1/v31-train8h-a10c/final.pt',
    'n2': ROOT / 'runs/knowledge-v3.1/v31-train8h-n2/final.pt',
    's2': ROOT / 'runs/dojo/focus-s2/final.pt',
    'cfp32-base': ROOT / 'runs/dojo/cfp32-base/final.pt',
    'cfp32-test': ROOT / 'runs/dojo/cfp32-test/final.pt',
}
RESERVE, HAND, FORCE, USED, LOST = 18, 15, 19, 20, 21  # ObservationV2.ZONES indices
TOP_RESERVE, TOP_FORCE, TOP_USED, TOP_LOST = 7, 8, 9, 10  # the top card of each pile has its own zone
ZONE_CLASS = {RESERVE: 'reserve', TOP_RESERVE: 'reserve', HAND: 'hand', FORCE: 'force', TOP_FORCE: 'force',
              USED: 'used', TOP_USED: 'used', LOST: 'lost', TOP_LOST: 'lost'}
GROUPS = {'card': (0,), 'location': (1,), 'knowledge': (2,), 'retained': (3, 5), 'context': (7,), 'edge': (8,),
          'reference': (9, 10)}
TOKENS = 32768


# ---------------------------------------------------------------- model access

ONLY = None  # plan critic-v2: --only restricts the checkpoints a command iterates over


def names(default):
    return [name for name in ONLY if name in CHECKPOINTS] if ONLY else list(default)


def load_models(name):
    actor, critic, tables, checkpoint = load(CHECKPOINTS[name], 'cuda')
    actor.clear_tables()
    critic.encoder.clear_tables()
    for model in (actor, critic):
        model.eval()
    actor.packed_attention = False
    critic.encoder.packed_attention = False
    return actor, critic, tables, checkpoint


def components(enc, batch):
    """Each input component's masked contribution [B, L, d]; their sum is the input to the first graph round."""
    table = enc.cards.table()
    tokens = batch['tok'].long()
    wide = batch['tokf'].float()
    floats = wide[..., :4]
    kind, blueprint, zone, owner, slot, _, count, flags = tokens.unbind(-1)
    context = kind == 7
    known = kind == 2
    retained = (kind == 3) | (kind == 5)
    reference = (kind == 9) | (kind == 10)
    numeric = torch.cat((count.unsqueeze(-1).float(), floats[..., 1:],
                         ((slot - 1) * (kind == 1)).unsqueeze(-1).float()), -1) / 60.0
    numeric = numeric * (~context & ~known).unsqueeze(-1)
    part = enc.part
    out = {'card': part('card', F.embedding(blueprint, table)) * (blueprint > 0).unsqueeze(-1),
           'kind': part('kind', enc.kind(kind)), 'zone': part('zone', enc.zone(zone)),
           'owner': part('owner', enc.owner(owner)),
           'numeric': part('numeric', enc.numeric(numeric)) * (~context & ~known).unsqueeze(-1)}
    if enc.cfg.location_icons:
        out['location_icons'] = enc.location_icons(unpack_location_icons(floats[..., 0], kind == 1))
    if enc.cfg.location_state:
        out['location_state'] = enc.location_state(unpack_location_state(wide[..., 4:8], kind == 1))
    out['entity_flags'] = part('entity_flags', enc.entity_flags(enc.bits(flags, 16))) * (kind <= 1).unsqueeze(-1)
    out['retained'] = part('retained', enc.scope(enc.bits(flags // 2, 3))
                           + enc.fact_flag(flags.remainder(2))) * retained.unsqueeze(-1)
    indices = torch.stack((floats[..., 0], floats[..., 2], floats[..., 3], count.float()), -1)
    mask = context.unsqueeze(-1)
    out['context_index'] = part('context_index', enc.context_index((indices + 1) / 60.0)) * mask
    out['context_value'] = part('context_value', enc.context_value(floats[..., 1:2] / 60.0)) * mask
    out['sentinel'] = part('sentinel', enc.sentinel(flags.remainder(8))) * mask
    out['context_role'] = part('context_role', enc.context_role(enc.bits(flags // 8, 4))) * mask
    out['context_template'] = part('context_template', enc.template('context', torch.where(context, slot, 0))) * mask
    bits = torch.stack((flags.remainder(2), (flags // 128).remainder(2)), -1).float()
    out['reference_flags'] = part('reference_flags', enc.reference_flags(bits)) * reference.unsqueeze(-1)
    out['position'] = part('position', enc.position((slot - 1).unsqueeze(-1).float() / 60.0)) * (
        reference | (kind == 5)).unsqueeze(-1)
    grid_index = torch.where(known, slot - 1, 0)
    raw_grid = batch['kgrid'].gather(1, grid_index.unsqueeze(-1).expand(-1, -1, 153)).float()
    _, projected = grid_features(raw_grid, enc.cfg.grid_indicators)
    out['grid'] = part('grid', enc.grid(projected)) * known.unsqueeze(-1)
    b, length, width = out['card'].shape
    source, target, _, _, valid = sparse_edges(batch)
    total = sum(out.values()).flatten(0, 1)
    degree = degrees(total, source, target, valid) / 60.0
    out['degree'] = part('degree', enc.degree(degree)).reshape(b, length, width)
    return out


class Depths:
    """Forward hooks: input to the graph rounds, the transformer input (after graph rounds and edge binding), the output
    of each transformer block, and the final normalised states."""

    def __init__(self, enc):
        self.enc, self.states, self.handles = enc, {}, []
        self.handles.append(enc.messages[0].register_forward_pre_hook(self._save('input', flat=True)))
        self.handles.append(enc.encoder.register_forward_pre_hook(self._save('graph')))
        for index, layer in enumerate(enc.encoder.layers):
            self.handles.append(layer.register_forward_hook(self._save_out(f'block{index + 1}')))
        self.handles.append(enc.norm.register_forward_hook(self._save_out('final')))

    def _save(self, name, flat=False):
        def hook(module, args, *rest):
            self.states[name] = args[0].detach()
        return hook

    def _save_out(self, name):
        def hook(module, args, output):
            self.states[name] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def run(model, batch, mode='fp32'):
    """Forward the encoder of an actor (EfficientPolicy) or critic; returns the Depths states (tokens at [:, 2:] for
    sequence states, input as [B*L, d])."""
    enc = model.encoder if hasattr(model, 'encoder') and hasattr(model, 'value') else model
    depths = Depths(enc)
    try:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=mode == 'bf16'):
            enc.encode(batch)
    finally:
        depths.close()
    states = depths.states
    b, length = batch['tok'].shape[:2]
    states['input'] = states['input'].reshape(b, length, -1)
    return states


def scale_card(enc, factor):
    """Measurement intervention (P-5 C, P-6): multiply the card-identity part by FACTOR on this loaded instance."""
    original = type(enc).part.__get__(enc)

    def part(name, value):
        out = original(name, value)
        return out * factor if name == 'card' else out
    enc.part = part


# ---------------------------------------------------------------- data

def eval_games(files):
    return {m['g'] for p in files for m in cd.scan_meta(p)}


def light_sample(reader, files, target, keep_actor=True, exclude=frozenset()):
    """About TARGET decisions spread over all games of FILES (games in EXCLUDE skipped), without values:
    (critic item, actor item, game meta, step)."""
    counts = [sum(m['n_steps'] for m in cd.scan_meta(p) if m['g'] not in exclude) for p in files]
    # Decode only an evenly spread subset of files holding ~4x TARGET decisions, then take every k-th decision.
    every = max(1, int(sum(counts) // (4 * target)))
    files = files[::every]
    total = sum(counts[::every])
    stride = max(1, total // target)
    out, counter = [], 0
    for path in files:
        for game in reader.load(path):
            if game['g'] in exclude:
                continue
            meta = {'g': game['g'], 's': game['s'], 'won': cd.outcome(game)}
            for index in range(len(game['items'])):
                if counter % stride == 0:
                    out.append((cd.detach(game['critic_items'][index]),
                                cd.detach(game['items'][index]) if keep_actor else None, meta, index))
                counter += 1
    return out


def first_items(reader, files, exclude=frozenset()):
    out = []
    for path in files:
        for game in reader.load(path):
            if game['g'] in exclude:
                continue
            out.append((cd.detach(game['critic_items'][0]), cd.detach(game['items'][0]),
                        {'g': game['g'], 's': game['s'], 'won': cd.outcome(game)}))
    return out


def batches(items, size=48):
    order = np.argsort([len(i['tok']) for i in items], kind='stable')
    for offset in range(0, len(order), size):
        for selected in microbatches(order[offset:offset + size], items, TOKENS):
            yield selected


# ---------------------------------------------------------------- P-1

def bf16(x):
    return x.to(torch.bfloat16).float()


def cmd_census(args):
    _, held = cd.split_files(args.data)
    result = {'checkpoints': {}, 'sites': None, 'card_table_weights': {}}
    sample = None
    for name in names(CHECKPOINTS):
        actor, critic, tables, checkpoint = load_models(name)
        if sample is None:
            sample = light_sample(TrajectoryReader(tables, []), held, args.n)
            result['sites'] = site_ranges(sample, critic.encoder)
        per_role = {}
        for role, model, index in (('actor', actor, 1), ('critic', critic.encoder, 0)):
            items = [s[index] for s in sample]
            acc = defaultdict(lambda: defaultdict(list))
            with torch.no_grad():
                for selected in batches(items):
                    batch = stack([items[i] for i in selected], 'cuda')
                    parts = components(model, batch)
                    total = sum(parts.values())
                    kind = batch['tok'][..., 0]
                    valid = batch['tmask']
                    total_norm = total.norm(dim=-1)
                    for group, kinds in GROUPS.items():
                        mask = valid & torch.isin(kind, torch.tensor(kinds, device=kind.device))
                        if not mask.any():
                            continue
                        s = total[mask]
                        sn = total_norm[mask].clamp_min(1e-12)
                        acc[group]['sum_norm'].append(sn.cpu())
                        rounded = bf16(s)
                        for comp, value in parts.items():
                            v = value[mask]
                            vn = v.norm(dim=-1)
                            if float(vn.max()) == 0:
                                continue
                            present = vn > 0
                            acc[group][f'{comp}:norm'].append(vn[present].cpu())
                            acc[group][f'{comp}:share'].append(((v * s).sum(-1).abs() / sn ** 2)[present].cpu())
                            survive = (rounded - bf16(s - v)).norm(dim=-1) / vn.clamp_min(1e-12)
                            acc[group][f'{comp}:bf16_survival'].append(survive[present].cpu())
            per_role[role] = {group: {key: float(torch.cat(values).float().mean()) for key, values in stats.items()}
                              for group, stats in acc.items()}
            per_role[role]['card_token_drowned'] = sorted(
                comp.split(':')[0] for comp in per_role[role].get('card', {})
                if comp.endswith(':share') and per_role[role]['card'][comp] < 0.02
                and per_role[role]['card'].get(comp.replace(':share', ':bf16_survival'), 1) < 0.5)
            enc = model
            result['card_table_weights'].setdefault(name, {})[role] = {
                n: float(p.norm()) for n, p in enc.cards.named_parameters()}
        result['checkpoints'][name] = {'iteration': checkpoint['rl_meta']['iteration'], **per_role}
        print(name, json.dumps({r: per_role[r].get('card', {}).get('card:norm') for r in per_role}), flush=True)
        del actor, critic
        torch.cuda.empty_cache()
    lineage = [n for n in ('amendment9-c6', 'a10c', 'n2', 's2') if n in result['checkpoints']]
    norms = [result['checkpoints'][n]['critic']['card']['card:norm'] for n in lineage]
    result['readout'] = {'critic_card_norm_lineage': norms,
                         'critic_card_growth_monotone': bool(all(b > a for a, b in zip(norms, norms[1:]))),
                         'drowned': {n: {r: result['checkpoints'][n][r]['card_token_drowned'] for r in ('actor', 'critic')}
                                     for n in result['checkpoints']}}
    return result


def site_ranges(sample, enc):
    """Raw value ranges at the /60 sites (before the division), over the sampled critic items (actor for option sites)."""
    values = defaultdict(list)
    for critic_item, actor_item, _, _ in sample:
        tok, f = critic_item['tok'], critic_item['tokf']
        kind, slot, count = tok[:, 0], tok[:, 4], tok[:, 6]
        card = kind == 0
        values['1 numeric: count'].append(count[card])
        piled = card & (f[:, 1] > 0)
        values['1 numeric: pile index b'].append(f[piled, 1])
        values['1 numeric: zone owner c (+-1)'].append(f[card, 2])
        values['1 numeric: inserted d (0/1)'].append(f[card, 3])
        values['1 numeric: location slot'].append((slot[kind == 1] - 1))
        ctx = kind == 7
        values['2 context_index: indices+1'].append(np.concatenate([f[ctx, 0], f[ctx, 2], f[ctx, 3], count[ctx]]) + 1)
        values['3 context_value'].append(f[ctx, 1])
        values['4 position: slot-1'].append(slot[np.isin(kind, (9, 10, 5))] - 1)
        packed = f[kind == 1, 0].round().astype(np.int64)
        values['9 location icon counts'].append(np.stack([(packed // 16 ** i) % 16 for i in range(4)], -1).ravel())
        if f.shape[1] >= 8:
            halves = f[kind == 1, 4:6].round().astype(np.int64)
            values['10 location power/ability us/them'].append(
                np.concatenate([(halves % 512).ravel() / 2, (halves // 512).ravel() / 2]))
            values['10 location drain us/them'].append(f[kind == 1, 6:8].ravel())
        known = kind == 2
        grid = critic_item['kgrid'][np.clip(slot[known] - 1, 0, None)] if known.any() else np.zeros((0, 153))
        values['11 grid lower bounds'].append(grid[:, 0::3].ravel())
        values['11 grid upper bounds'].append(grid[:, 1::3].ravel())
        values['12 ReferencePool n (question refs)'].append(np.array([(kind == 9).sum()]))
        head = critic_item['head'][4:22]
        for index, label in enumerate(GLOBAL_NAMES):
            values[f'6 glob: {label}'].append(np.array([head[index]]))
        if actor_item is not None and len(actor_item['opt']):
            values['8 option value (actor)'].append(actor_item['opt'][:, 4])
    out = {}
    for key, chunks in values.items():
        v = np.concatenate([np.asarray(c, np.float64).ravel() for c in chunks]) if chunks else np.zeros(0)
        if len(v):
            out[key] = {'min': float(v.min()), 'median': float(np.median(v)), 'p99': float(np.quantile(v, 0.99)),
                        'max': float(v.max()), 'p99_scaled': float(np.quantile(v, 0.99) / 60), 'n': int(len(v))}
    out['7 reference_match overlap (actor)'] = 'not measured (needs the option-pool internals)'
    return out


GLOBAL_NAMES = ('my turn number', 'opp turn number', 'phase', 'my turn flag', 'my life force', 'opp life force',
                'my reserve', 'opp reserve', 'my force pile', 'opp force pile', 'my used pile', 'opp used pile',
                'my lost pile', 'opp lost pile', 'my hand', 'opp hand', 'battle flag', 'drain flag')


# ---------------------------------------------------------------- P-2

def zone_class(zone):
    return ZONE_CLASS.get(int(zone), 'table/other')


def token_dataset(model, items, mode, per_item=5, seed=950001):
    """Card tokens (kind 0) at every depth: features per depth, labels mine / zone class."""
    rng = np.random.default_rng(seed)
    feats, mine, zone = defaultdict(list), [], []
    for selected in batches(items):
        batch = stack([items[i] for i in selected], 'cuda')
        states = run(model, batch, mode)
        for row, index in enumerate(selected):
            tok = items[index]['tok']
            cards = np.flatnonzero(tok[:, 0] == 0)
            if not len(cards):
                continue
            pick = rng.choice(cards, size=min(per_item, len(cards)), replace=False)
            mine.extend(tok[pick, 3].tolist())
            zone.extend(zone_class(z) for z in tok[pick, 2])
            for depth, state in states.items():
                offset = 0 if depth == 'input' else 2
                value = state[row, pick + offset].float()
                if depth == 'input' and mode == 'bf16':
                    value = bf16(value)
                feats[depth].append(value.cpu().numpy())
    return {k: np.concatenate(v) for k, v in feats.items()}, np.array(mine), np.array(zone)


def probe_accuracy(x_train, y_train, x_test, y_test, weight_decay=1e-4):
    """Linear (multinomial logistic) probe, full-batch L-BFGS on the GPU; held-out accuracy."""
    classes, y_tr = np.unique(y_train, return_inverse=True)
    lookup = {c: i for i, c in enumerate(classes)}
    y_te = np.array([lookup.get(c, -1) for c in y_test])
    mean, std = x_train.mean(0), x_train.std(0) + 1e-6
    xtr = torch.as_tensor((x_train - mean) / std, dtype=torch.float32, device='cuda')
    xte = torch.as_tensor((x_test - mean) / std, dtype=torch.float32, device='cuda')
    ytr = torch.as_tensor(y_tr, device='cuda')
    linear = nn.Linear(xtr.shape[1], len(classes)).cuda()
    optimizer = torch.optim.LBFGS(linear.parameters(), lr=1.0, max_iter=500, history_size=20,
                                  line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(linear(xtr), ytr) + weight_decay * linear.weight.pow(2).sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        predicted = linear(xte).argmax(1).cpu().numpy()
    return float((predicted == y_te).mean())


def cmd_owner(args):
    train_files, held = cd.split_files(args.data)
    result = {}
    for name in names(('s2', 'amendment9-c6')):
        actor, critic, tables, _ = load_models(name)
        reader = TrajectoryReader(tables, [])
        train = light_sample(reader, train_files, 400 if args.quick else 4000, exclude=eval_games(held))
        test = light_sample(reader, held, 100 if args.quick else 1000)
        for role, model, index in (('actor', actor, 1), ('critic', critic, 0)):
            for mode in ('fp32', 'bf16'):
                xtr, mtr, ztr = token_dataset(model, [s[index] for s in train], mode)
                xte, mte, zte = token_dataset(model, [s[index] for s in test], mode, seed=950002)
                xtr = {k: v[:20000] for k, v in xtr.items()}
                mtr, ztr = mtr[:20000], ztr[:20000]
                xte = {k: v[:5000] for k, v in xte.items()}
                mte, zte = mte[:5000], zte[:5000]
                cell = {}
                for depth in xtr:
                    cell[depth] = {'mine': probe_accuracy(xtr[depth], mtr, xte[depth], mte),
                                   'zone': probe_accuracy(xtr[depth], ztr, xte[depth], zte)}
                cell['n_train'], cell['n_test'] = int(len(mtr)), int(len(mte))
                cell['zone_majority'] = float(Counter(zte).most_common(1)[0][1] / len(zte))
                result[f'{name}:{role}:{mode}'] = cell
                print(name, role, mode, json.dumps({d: cell[d] for d in ('input', 'graph', 'final')}), flush=True)
        del actor, critic
        torch.cuda.empty_cache()
    readout = {}
    for key, cell in result.items():
        depths = [d for d in cell if isinstance(cell[d], dict)]
        readout[key] = {'mine_decodable_input': cell['input']['mine'] >= 0.99,
                        'mine_lost_in_depth': cell['input']['mine'] >= 0.99 and cell['final']['mine'] < 0.95,
                        'zone_lost_in_depth': cell['input']['zone'] >= 0.99 and cell['final']['zone'] < 0.95,
                        'min_mine': min(cell[d]['mine'] for d in depths)}
    for name in names(('s2', 'amendment9-c6')):
        for role in ('actor', 'critic'):
            a, b = result[f'{name}:{role}:fp32'], result[f'{name}:{role}:bf16']
            readout[f'{name}:{role}:bf16_fragile'] = any(
                a[d][t] - b[d][t] >= 0.02 for d in a if isinstance(a[d], dict) for t in ('mine', 'zone'))
    return {'cells': result, 'readout': readout}


# ---------------------------------------------------------------- P-3

def owner_means(states, batch):
    """Mean token state over my card tokens || over the opponent's card tokens (kind 0), [B, 2d]."""
    tok = batch['tok']
    card = (tok[..., 0] == 0) & batch['tmask']

    def mean(mask):
        return (states * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    return torch.cat((mean(card & (tok[..., 3] == 1)), mean(card & (tok[..., 3] == 0))), -1)


def first_features(model, items, value_model=None):
    """Feature sets 1-3 (+5 for a critic) at first decisions."""
    sets = defaultdict(list)
    for selected in batches(items):
        batch = stack([items[i] for i in selected], 'cuda')
        states = run(model, batch)
        final, inputs = states['final'].float(), states['input'].float()
        sets['1 Q\'|G\''].append(torch.cat((final[:, 0], final[:, 1]), -1).cpu().numpy())
        sets['2 final per owner'].append(owner_means(final[:, 2:], batch).cpu().numpy())
        sets['3 input per owner'].append(owner_means(inputs, batch).cpu().numpy())
        if value_model is not None:
            with torch.no_grad():
                if getattr(value_model.cfg, 'win_head', False):  # plan critic-v2: the win head as a 6th set
                    value, win = value_model.heads(batch)
                    sets['6 win head'].append(win.float().unsqueeze(-1).cpu().numpy())
                else:
                    value = value_model(batch)
                sets['5 critic value'].append(value.float().unsqueeze(-1).cpu().numpy())
        sets['_order'].append(np.asarray(selected))
    order = np.concatenate(sets.pop('_order'))
    inverse = np.argsort(order)
    return {k: np.concatenate(v)[inverse] for k, v in sets.items()}


def bag_of_cards(items, n_cards):
    from scipy.sparse import lil_matrix
    x = lil_matrix((len(items), 2 * n_cards), dtype=np.float32)
    for row, item in enumerate(items):
        tok = item['tok']
        for bp, owner in tok[tok[:, 0] == 0][:, [1, 3]]:
            x[row, int(bp) + (0 if owner == 1 else n_cards)] += 1
    return x.tocsr()


def fit_eval(x_train, y_logit, y_won, x_test, t_logit, t_won, views, sparse=False):
    if not sparse:
        scaler = StandardScaler().fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(x_train, y_logit)
    logistic = LogisticRegressionCV(Cs=np.logspace(-4, 2, 13), cv=5, max_iter=5000, scoring='roc_auc').fit(x_train, y_won)
    pred_logit, pred_won = ridge.predict(x_test), logistic.predict_proba(x_test)[:, 1]
    out = {}
    for view, mask in views.items():
        out[view] = {'oracle_r2': float(r2_score(t_logit[mask], pred_logit[mask])),
                     'outcome_auc': float(roc_auc_score(t_won[mask], pred_won[mask]))}
    return out


def cmd_matchup(args):
    import pickle
    train_files, held = cd.split_files(args.data)
    if args.quick:
        train_files, held = train_files[::10], held[::10]
    with open(args.work_diag / 'oracle.pkl', 'rb') as stream:
        oracle = pickle.load(stream)
    rows = cd.rollout_rows(args.rollouts)
    result = {}
    data = None
    for name in names(('s2', 'cfp32-base', 'cfp32-test')):
        actor, critic, tables, _ = load_models(name)
        if data is None:
            reader = TrajectoryReader(tables, [])
            data = {}
            held_games = eval_games(held)
            for split, files in (('train', train_files), ('test', held)):
                items = [x for x in first_items(reader, files, held_games if split == 'train' else frozenset()) if x[2]['won'] >= 0 and x[2]['g'] in rows]
                logit = np.array([cd.logit(cd.oracle_seat(oracle, m, rows[m['g']])) for _, _, m in items])
                won = np.array([m['won'] for _, _, m in items])
                pvp = np.array([rows[m['g']]['ls_ai' if m['s'] == 'ds' else 'ds_ai'].startswith('policy')
                                for _, _, m in items])
                data[split] = {'items': items, 'logit': logit, 'won': won, 'pvp': pvp}
            n_cards = len(tables.cards.ids)
            bags = {s: bag_of_cards([x[0] for x in data[s]['items']], n_cards) for s in data}
            views = {'policy_vs_policy': data['test']['pvp'], 'all': np.ones(len(data['test']['won']), bool)}
            tr, te = data['train'], data['test']
            result['4 bag of cards'] = fit_eval(bags['train'], tr['logit'], tr['won'], bags['test'], te['logit'],
                                                te['won'], views, sparse=True)
            result['oracle'] = {v: {'outcome_auc': float(roc_auc_score(te['won'][m], te['logit'][m]))}
                                for v, m in views.items()}
            result['n'] = {'train': int(len(tr['won'])), 'test': int(len(te['won'])), 'test_pvp': int(te['pvp'].sum())}
        tr, te = data['train'], data['test']
        for role, model, index in (('critic', critic.encoder, 0), ('actor', actor, 1)):
            if role == 'actor' and name != 's2':
                continue
            f_train = first_features(model, [x[index] for x in tr['items']], critic if role == 'critic' else None)
            f_test = first_features(model, [x[index] for x in te['items']], critic if role == 'critic' else None)
            result[f'{name}:{role}'] = {
                key: fit_eval(f_train[key], tr['logit'], tr['won'], f_test[key], te['logit'], te['won'], views)
                for key in f_train}
            print(name, role, json.dumps({k: v['policy_vs_policy'] for k, v in result[f'{name}:{role}'].items()}), flush=True)
        del actor, critic
        torch.cuda.empty_cache()
    readout = {}
    for name in names(('s2', 'cfp32-base', 'cfp32-test')):
        cell = result[f'{name}:critic']
        auc = {k: cell[k]['policy_vs_policy']['outcome_auc'] for k in cell}
        bag = result['4 bag of cards']['policy_vs_policy']['outcome_auc']
        readout[name] = {'readout_bottleneck': auc['2 final per owner'] >= auc["1 Q'|G'"] + 0.05,
                         'encoder_loses_it': auc['2 final per owner'] < auc['3 input per owner'] - 0.05,
                         'input_weak': auc['3 input per owner'] < bag - 0.05,
                         'head_does_not_use_it': auc["1 Q'|G'"] >= auc['5 critic value'] + 0.05, 'auc_pvp': auc,
                         'bag_auc_pvp': bag}
    return {'cells': result, 'readout': readout}


# ---------------------------------------------------------------- P-4

FACTS = {'life force difference': ('reg', lambda g: g[4] - g[5]), 'my reserve': ('reg', lambda g: g[6]),
         'my turn number': ('reg', lambda g: g[0]), 'phase': ('cls', lambda g: g[2]),
         'my turn flag': ('cls', lambda g: g[3]), 'battle flag': ('cls', lambda g: g[16]),
         'drain flag': ('cls', lambda g: g[17])}


def cmd_globals(args):
    _, held = cd.split_files(args.data)
    actor, critic, tables, _ = load_models('s2')
    sample = light_sample(TrajectoryReader(tables, []), held, args.n)
    games = np.array([zlib.crc32(m['g'].encode()) % 5 for _, _, m, _ in sample])
    train, test = games != 0, games == 0
    result = {}
    for role, model, index in (('actor', actor, 1), ('critic', critic.encoder, 0)):
        items = [s[index] for s in sample]
        feats = defaultdict(lambda: [None] * len(items))
        for selected in batches(items):
            batch = stack([items[i] for i in selected], 'cuda')
            states = run(model, batch)
            for row, i in enumerate(selected):
                feats['G (input)'][i] = states['graph'][row, 1].float().cpu().numpy()
                feats["G' (final)"][i] = states['final'][row, 1].float().cpu().numpy()
                feats["Q' (final)"][i] = states['final'][row, 0].float().cpu().numpy()
        heads = np.array([item['head'][4:22] for item in items], np.float64)
        result[role] = {}
        for fact, (kind, fn) in FACTS.items():
            y = np.array([fn(h) for h in heads])
            result[role][fact] = {}
            for depth, rows_ in feats.items():
                x = np.stack(rows_)
                scaler = StandardScaler().fit(x[train])
                xtr, xte = scaler.transform(x[train]), scaler.transform(x[test])
                if kind == 'reg':
                    model_ = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(xtr, y[train])
                    score = float(r2_score(y[test], model_.predict(xte)))
                else:
                    score = probe_accuracy(x[train], y[train].astype(int), x[test], y[test].astype(int))
                result[role][fact][depth] = score
            majority = None if kind == 'reg' else float(Counter(y[test].astype(int)).most_common(1)[0][1] / test.sum())
            result[role][fact]['majority'] = majority
        print(role, json.dumps(result[role]), flush=True)
    readout = {}
    for role in result:
        for fact, (kind, _) in FACTS.items():
            threshold = 0.9 if kind == 'reg' else 0.98
            cell = result[role][fact]
            readout[f'{role}:{fact}'] = {d: cell[d] >= threshold for d in ('G (input)', "G' (final)", "Q' (final)")}
            readout[f'{role}:{fact}']['lost'] = cell['G (input)'] >= threshold and cell["G' (final)"] < threshold \
                and cell["Q' (final)"] < threshold
    return {'scores': result, 'readout': readout, 'decisions': len(sample)}


# ---------------------------------------------------------------- P-5

def destiny_table(tables):
    cards = tables.cards.cards
    column = list(cards['feature_names']).index('num:destiny')
    return np.asarray(cards['features'])[:, column] * 5.0  # num:destiny is printed destiny / 5


def tops(item):
    """Per owner (mine, theirs): (top blueprint, its pile index, all reserve blueprints), or None if empty. The top
    card carries zone TOP_OF_RESERVE_DECK (7); the rest of the deck is RESERVE_DECK (18) with indices from 2."""
    tok, f = item['tok'], item['tokf']
    out = []
    for owner in (1, 0):
        reserve = (tok[:, 0] == 0) & np.isin(tok[:, 2], (RESERVE, TOP_RESERVE)) & (tok[:, 3] == owner)
        if not reserve.any():
            out.append(None)
            continue
        top = reserve & (tok[:, 2] == TOP_RESERVE)
        rows = np.flatnonzero(top) if top.any() else np.flatnonzero(reserve)[[np.argmin(f[reserve, 1])]]
        out.append((int(tok[rows[0], 1]), int(f[rows[0], 1]), tok[reserve, 1].tolist()))
    return out


def verify_top(reader, files):
    """Between consecutive decisions of a seat, when exactly one card left my reserve deck, was it the one at index 1?"""
    hits = total = 0
    chance = []
    for path in files:
        for game in reader.load(path):
            items = game['critic_items']
            for a, b in zip(items, items[1:]):
                ta, tb = tops(a)[0], tops(b)[0]
                if ta is None or tb is None:
                    continue
                before, after = Counter(ta[2]), Counter(tb[2])
                removed = before - after
                if sum(removed.values()) == 1 and not (after - before):
                    total += 1
                    card = next(iter(removed))
                    hits += int(card == ta[0])
                    chance.append(before[card] / len(ta[2]))
    return {'single_removals': total, 'top_match_rate': hits / max(total, 1),
            'chance_rate': float(np.mean(chance)) if chance else None, 'min_b_is_1': None}


class OrderProbe(nn.Module):
    def __init__(self, enc, width=256):
        super().__init__()
        self.enc = enc
        self.query = nn.Parameter(torch.randn(width) * 0.02)
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 4))  # mine/theirs: logit, value

    def forward(self, batch, frozen):
        with torch.set_grad_enabled(not frozen):
            hidden, _ = self.enc.encode(batch)
        hidden = hidden.float()
        mask = torch.cat((torch.ones_like(batch['tmask'][:, :2]), batch['tmask']), 1)
        scores = (hidden @ self.query).masked_fill(~mask, -1e9) / 16.0
        pooled = (torch.softmax(scores, 1).unsqueeze(-1) * hidden).sum(1)
        return self.head(pooled)


def order_dataset(items, destiny):
    keep, targets, baseline = [], [], []
    for item in items:
        t = tops(item)
        if t[0] is None or t[1] is None:
            continue
        keep.append(item)
        values = [destiny[t[0][0]], destiny[t[1][0]]]
        targets.append(values)
        baseline.append([np.mean(destiny[t[0][2]] >= 4), np.mean(destiny[t[1][2]] >= 4)])
    return keep, np.array(targets, np.float32), np.array(baseline)


def cmd_order(args):
    train_files, held = cd.split_files(args.data)
    actor, critic, tables, _ = load_models(args.target)
    reader = TrajectoryReader(tables, [])
    destiny = destiny_table(tables)
    result = {'variant': args.variant, 'seed': args.seed}
    if args.variant == 'verify':
        result['verify'] = verify_top(reader, held[:10] if args.quick else held[:60])
        sample = light_sample(reader, held, 500 if args.quick else 5000, keep_actor=False)
        bs = [t[1] for s in sample for t in tops(s[0]) if t is not None]
        result['verify']['min_b_is_1'] = float(np.mean(np.array(bs) == 1))
        result['destiny_distribution'] = {str(int(k)): int(v) for k, v in
                                          zip(*np.unique(destiny[2:], return_counts=True))}
        result['destiny_check'] = destiny_db_check(tables, destiny)
        return result
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    n_train, n_test = (2000, 600) if args.quick else (60000, 6000)
    train_items, train_y, _ = order_dataset([s[0] for s in light_sample(reader, train_files, n_train, False,
                                                                        exclude=eval_games(held))], destiny)
    test_items, test_y, test_base = order_dataset([s[0] for s in light_sample(reader, held, n_test, False)], destiny)
    test_items, test_y, test_base = test_items[:2000], test_y[:2000], test_base[:2000]
    enc = critic.encoder
    if args.variant == 'D':  # the critic architecture at random initialisation (seeded), no training history
        torch.manual_seed(args.seed)
        enc = type(critic)(enc.cfg, tables).cuda().encoder
        enc.packed_attention = False
    elif args.variant == 'E':  # the actor encoder (no knowledge mask in s2), fine-tuned: a differently trained network
        enc = actor
    if args.variant == 'C':
        census = json.loads((args.out / 'census.json').read_text())['checkpoints']['s2']
        factor = census['actor']['card']['card:norm'] / census['critic']['card']['card:norm']
        scale_card(enc, factor)
        result['card_scale'] = factor
    frozen = args.variant == 'A'
    probe = OrderProbe(enc).cuda()
    params = [probe.query, *probe.head.parameters()]
    if not frozen:
        params += list(enc.parameters())
        enc.train()
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.0)
    labels = torch.as_tensor(train_y)
    curve = []

    def evaluate():
        probe.eval()
        preds = []
        with torch.no_grad():
            for selected in batches(test_items, 40):
                out = probe(stack([test_items[i] for i in selected], 'cuda'), True)
                preds.append((np.asarray(selected), out.float().cpu().numpy()))
        probe.train()
        order = np.concatenate([p[0] for p in preds])
        out = np.concatenate([p[1] for p in preds])[np.argsort(order)]
        row = {}
        for k, side in enumerate(('mine', 'theirs')):
            y = test_y[:, k] >= 4
            row[side] = {'accuracy': float(accuracy_score(y, out[:, 2 * k] > 0)),
                         'auc': float(roc_auc_score(y, out[:, 2 * k])) if len(set(y)) == 2 else None,
                         'composition_baseline_accuracy': float(accuracy_score(y, test_base[:, k] > 0.5)),
                         'composition_baseline_auc': float(roc_auc_score(y, test_base[:, k])) if len(set(y)) == 2 else None,
                         'majority': float(max(y.mean(), 1 - y.mean())),
                         'value_r2': float(r2_score(test_y[:, k], out[:, 2 * k + 1] * 7))}
        return row
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        index = rng.choice(len(train_items), size=40, replace=False)
        batch = stack([train_items[i] for i in index], 'cuda')
        out = probe(batch, frozen).float()
        y = labels[index].cuda()
        loss = sum(F.binary_cross_entropy_with_logits(out[:, 2 * k], (y[:, k] >= 4).float())
                   + F.mse_loss(out[:, 2 * k + 1], y[:, k] / 7) for k in range(2))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step % 1000 == 0 or step == args.steps:
            curve.append({'step': step, 'seconds': time.monotonic() - started, 'loss': float(loss), **evaluate()})
            print(json.dumps(curve[-1]), flush=True)
    result.update({'curve': curve, 'train_examples': len(train_items), 'test_examples': len(test_items)})
    return result


def destiny_db_check(tables, destiny):
    """Compare the table's destiny with the card database for 20 blueprints with a destiny."""
    from swccg_policy.cards import DEFAULT_DB, load_db
    if not Path(DEFAULT_DB).exists():
        return {'checked': 0, 'note': f'{DEFAULT_DB} missing'}
    db = {str(c.get('cardId')): c for c in load_db(Path(DEFAULT_DB))}
    ids = list(tables.cards.ids)
    rng = np.random.default_rng(950003)
    candidates = [i for i in range(4, len(ids)) if ids[i] in db and db[ids[i]].get('destiny') not in (None, '')]
    checked = []
    for i in rng.choice(candidates, size=min(20, len(candidates)), replace=False):
        raw = db[ids[i]].get('destiny')
        try:
            expected = float(raw)
        except (TypeError, ValueError):
            expected = None
        checked.append({'id': ids[i], 'title': db[ids[i]].get('title'), 'db': raw, 'table': float(destiny[i]),
                        'match': expected is not None and abs(expected - destiny[i]) < 1e-6})
    return {'checked': len(checked), 'matches': sum(c['match'] for c in checked), 'rows': checked,
            'db_keys_found': len(candidates)}


# ---------------------------------------------------------------- P-6

def cmd_bf16(args):
    census = json.loads((args.out / 'census.json').read_text())['checkpoints']['s2']
    factor = census['actor']['card']['card:norm'] / census['critic']['card']['card:norm']
    out = {'card_scale': factor}
    for label in ('as-is', 'card-scaled'):
        actor, critic, tables, _ = load_models('s2')
        if label == 'card-scaled':
            scale_card(critic.encoder, factor)
        cd.setup()
        grads_args = argparse.Namespace(data=args.data, n=2000 if args.quick else 40000, sizes=[64, 1024],
                                        minibatches=3 if args.quick else 24)
        out[label] = cd.cmd_grads(grads_args, actor, critic, TrajectoryReader(tables, []))
        del actor, critic
        torch.cuda.empty_cache()
    a, b = out['as-is']['cells'], out['card-scaled']['cells']

    def med(cells, cell, key):
        return cells[cell].get(key, {}).get('median')
    out['readout'] = {
        'gae64_bf16_packed': [med(a, 'gae:64', 'bf16-packed:cos'), med(b, 'gae:64', 'bf16-packed:cos')],
        'gae64_tables': [med(a, 'gae:64', 'bf16-packed:cos:card+context tables'),
                         med(b, 'gae:64', 'bf16-packed:cos:card+context tables')],
        'mc1024_bf16_packed': [med(a, 'mc:1024', 'bf16-packed:cos'), med(b, 'mc:1024', 'bf16-packed:cos')]}
    r = out['readout']
    out['readout']['scale_explains_bf16'] = bool(r['gae64_bf16_packed'][1] - r['gae64_bf16_packed'][0] >= 0.15
                                                 and r['gae64_tables'][1] >= 0.8)
    return out


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['census', 'owner', 'matchup', 'globals', 'order', 'bf16'])
    parser.add_argument('--data', type=Path, default=ROOT / 'runs/dojo/collect-s2b/trajectories')
    parser.add_argument('--rollouts', type=Path, default=ROOT / 'runs/dojo/collect-s2b/rollouts')
    parser.add_argument('--work-diag', type=Path, default=ROOT / 'runs/dojo/critic-diag')
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/data/critic-probes')
    parser.add_argument('--n', type=int, default=6000, help='sampled decisions (census: 6000; globals: 40000)')
    parser.add_argument('--variant', choices=['verify', 'A', 'B', 'C', 'D', 'E'], default='verify')
    parser.add_argument('--seed', type=int, default=883001)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--name')
    parser.add_argument('--quick', action='store_true', help='smoke-test sizes')
    # Plan critic-v2: the probes as a regression suite on any checkpoint.
    parser.add_argument('--checkpoint', nargs='*', default=[], metavar='NAME=PATH',
                        help='register extra checkpoints under a name')
    parser.add_argument('--only', nargs='*', help='census / owner / matchup: these checkpoint names only')
    parser.add_argument('--target', default='s2', help='order: the checkpoint whose critic (or actor, E) is probed')
    args = parser.parse_args()
    global ONLY
    for spec in args.checkpoint:
        name, _, path = spec.partition('=')
        CHECKPOINTS[name] = Path(path)
    ONLY = args.only
    cd.setup()
    started = time.time()
    result = {'census': cmd_census, 'owner': cmd_owner, 'matchup': cmd_matchup, 'globals': cmd_globals,
              'order': cmd_order, 'bf16': cmd_bf16}[args.command](args)
    result.update({'command': args.command, 'seconds': time.time() - started,
                   'args': {k: str(v) for k, v in vars(args).items()}})
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.name or (f'order_{args.variant}_{args.seed}' if args.command == 'order' and args.variant != 'verify'
                         else 'order_verify' if args.command == 'order' else args.command)
    path = args.out / f'{name}.json'
    path.write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(path)


if __name__ == '__main__':
    main()
