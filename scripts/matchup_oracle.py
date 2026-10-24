"""Plan critic-diagnostics D-1: a matchup oracle, P(DS wins) from start-of-game facts only.

    matchup_oracle.py fit --corpus 'runs/dojo/cfp32-*/rollouts/*.csv' ... --exclude ids.txt --out runs/dojo/critic-diag/oracle

Features: one strength per DS deck and per LS deck (one-hot, decks keyed by file name) and the AI type of each seat
(policy:learner / policy:snap / advanced / random). Model: L2 logistic regression; C chosen by 5-fold log loss over games.
Games without a winner (cap, error) are dropped. Unknown decks at prediction time get strength 0 (the average deck).
"""
import argparse
import csv
import glob
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold

C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def read_rows(patterns):
    """Rollout CSV rows with a winner: {'g', 'ds_deck', 'ls_deck', 'ds_ai', 'ls_ai', 'ds_won'}."""
    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            for row in csv.DictReader(open(path)):
                if row['winner'] not in ('ds', 'ls'):
                    continue
                rows.append({'g': row['game'], 'ds_deck': Path(row['ds_deck']).name, 'ls_deck': Path(row['ls_deck']).name,
                             'ds_ai': row['ds_ai'], 'ls_ai': row['ls_ai'], 'ds_won': int(row['winner'] == 'ds')})
    return rows


def features(row):
    return {'ds:' + row['ds_deck']: 1.0, 'ls:' + row['ls_deck']: 1.0,
            'dsai:' + row['ds_ai']: 1.0, 'lsai:' + row['ls_ai']: 1.0}


class Oracle:
    def __init__(self, c=1.0):
        self.c = c
        self.vectorizer = DictVectorizer()
        self.model = LogisticRegression(C=c, max_iter=5000)

    def fit(self, rows):
        x = self.vectorizer.fit_transform([features(r) for r in rows])
        self.model.fit(x, np.array([r['ds_won'] for r in rows]))
        return self

    def p_ds(self, rows):
        return self.model.predict_proba(self.vectorizer.transform([features(r) for r in rows]))[:, 1]

    def strengths(self):
        names = self.vectorizer.get_feature_names_out()
        return {name: float(w) for name, w in zip(names, self.model.coef_[0])}


def choose_c(rows, seed=930001):
    y = np.array([r['ds_won'] for r in rows])
    scores = {}
    for c in C_GRID:
        losses = []
        for train, test in KFold(5, shuffle=True, random_state=seed).split(rows):
            oracle = Oracle(c).fit([rows[i] for i in train])
            losses.append(log_loss(y[test], oracle.p_ds([rows[i] for i in test]), labels=[0, 1]))
        scores[c] = float(np.mean(losses))
    return min(scores, key=scores.get), scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['fit'])
    parser.add_argument('--corpus', nargs='+', required=True, help='rollout CSV globs')
    parser.add_argument('--exclude', type=Path, help='file with game ids to leave out (the evaluation games)')
    parser.add_argument('--out', type=Path, required=True, help='output stem: <out>.pkl and <out>.json')
    args = parser.parse_args()
    rows = read_rows(args.corpus)
    excluded = set(args.exclude.read_text().split()) if args.exclude else set()
    rows = [r for r in rows if r['g'] not in excluded]
    c, scores = choose_c(rows)
    import matchup_oracle  # pickle the class under its module name, not __main__
    oracle = matchup_oracle.Oracle(c).fit(rows)
    y = np.array([r['ds_won'] for r in rows])
    p = oracle.p_ds(rows)
    strengths = oracle.strengths()
    decks = sorted(((k, v) for k, v in strengths.items() if k.startswith(('ds:', 'ls:'))), key=lambda kv: kv[1])
    summary = {'games': len(rows), 'excluded_ids': len(excluded), 'c': c, 'cv_log_loss': scores,
               'base_rate_ds': float(y.mean()), 'base_log_loss': float(log_loss(y, np.full(len(y), y.mean()))),
               'train_auc': float(roc_auc_score(y, p)),
               'ai_effects': {k: v for k, v in strengths.items() if k.startswith(('dsai:', 'lsai:'))},
               'deck_strength_sd': float(np.std([v for _, v in decks])),
               'weakest': decks[:10], 'strongest': decks[-10:]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out.with_suffix('.pkl'), 'wb') as stream:
        pickle.dump(oracle, stream)
    args.out.with_suffix('.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: summary[k] for k in ('games', 'c', 'base_log_loss', 'train_auc', 'deck_strength_sd')}))


if __name__ == '__main__':
    main()
