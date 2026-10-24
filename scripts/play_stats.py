import json, re, csv, glob, collections, sys, numpy as np, statistics as st
ex=np.load('runs/v-cards/cards-all.npz'); names=[str(n) for n in ex['feature_names']]
ci={n:i for i,n in enumerate(names) if n.startswith('cardCategory=')}
cat_of={str(bp):next((n.split('=')[1] for n,i in ci.items() if row[i]),None) for bp,row in zip(ex['ids'],ex['features'])}
norm=lambda s: re.sub(r'[•◇]|<>', '', s).strip()
by_name=collections.defaultdict(set)
for r in json.load(open('runs/v-cards/title-display-metadata.unmarked.json')):
    for nm in (r['display'], r['title']): by_name[norm(nm)].add(cat_of.get(r['blueprint']))
def category(nm):
    s={x for x in by_name.get(norm(nm), set()) if x}
    return min(s) if len(s)>=1 else None
def deck(path):
    return collections.Counter(cat_of.get(i) for i in re.findall(r'blueprintId="([^"]+)"', open(path).read()))
out={}
for tag in sys.argv[1:]:
    E=f'runs/knowledge-v3.1/evaluations/{tag}'
    rows={r['game'].rsplit('-',1)[1]:r for r in csv.DictReader(open(glob.glob(f'{E}/*.csv')[0]))}
    agg={'policy':collections.defaultdict(list),'advanced':collections.defaultdict(list)}
    top={'policy':collections.Counter(),'advanced':collections.Counter()}; unk=collections.Counter()
    for f in glob.glob(f'{E}/game-logs/*.log'):
        r=rows[f.rsplit('-',1)[1][:-4]]
        pside='ds' if 'agent.ts' in r['ds_ai'] else 'ls'
        lines=open(f).read().splitlines()
        for side,who in ((pside,'policy'),('ls' if pside=='ds' else 'ds','advanced')):
            d=deck(r['ds_deck'] if side=='ds' else r['ls_deck'])
            ints=effs=0
            for l in lines:
                m=re.match(side+r' (plays|deploys) (.+?)(?: as Starting Interrupt| from | to | on | at | as |$)', l)
                if not m: continue
                c=category(m.group(2))
                if c is None: unk[m.group(2)]+=1; continue
                if m.group(1)=='plays' and c=='INTERRUPT' and 'Starting Interrupt' not in l: ints+=1; top[who][norm(m.group(2))]+=1
                elif m.group(1)=='deploys' and c=='EFFECT': effs+=1
            a=agg[who]; a['i'].append(ints); a['e'].append(effs); a['t'].append(float(r['turns'])/2)
            a['di'].append(d['INTERRUPT']); a['de'].append(d['EFFECT'])
    print(f'== {tag}: {len(rows)} games, unmapped plays/deploys: {sum(unk.values())}')
    for who,a in agg.items():
        n=len(a['i'])
        print(f"  {who:8s}: interrupts/game {st.mean(a['i']):.1f} (median {st.median(a['i']):.0f}; 0 in {sum(1 for x in a['i'] if x==0)/n:.0%} of games), "
              f"per own turn {sum(a['i'])/sum(a['t']):.2f}, share of deck's interrupts {sum(a['i'])/sum(a['di']):.0%} | "
              f"effects/game {st.mean(a['e']):.1f}, share of deck's effects {sum(a['e'])/max(1,sum(a['de'])):.0%} | deck: {st.mean(a['di']):.1f} interrupts, {st.mean(a['de']):.1f} effects")
        print(f"            top interrupts: {top[who].most_common(5)}")
