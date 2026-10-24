"""Cheap summaries of existing TrainGen CSVs; no engine/stat queries.

turns counts turns started by DS plus turns started by LS. Two player-turns
are one round-equivalent. Unfinished games are censored and excluded from the
finished-game distribution, while their counts/rates remain visible.
"""
import argparse,collections,csv,json,statistics,time
from pathlib import Path


def distribution(values):
    values=sorted(values)
    if not values:return {'games':0,'mean':None,'median':None,'p90':None,'under_10_player_turns_rate':None,'under_10_round_equivalents_rate':None}
    position=.9*(len(values)-1);lo=int(position);hi=min(lo+1,len(values)-1)
    return {'games':len(values),'mean':statistics.mean(values),'median':statistics.median(values),
            'p90':values[lo]+(values[hi]-values[lo])*(position-lo),
            'under_10_player_turns_rate':sum(v<10 for v in values)/len(values),
            'under_10_round_equivalents_rate':sum(v<20 for v in values)/len(values)}


def summarize(rows,learner=None):
    finished=[r for r in rows if r.get('outcome')=='finished' and r.get('winner') in ('ds','ls')]
    valid=[r for r in finished if int(r.get('turns') or 0)>0]
    out={'games':len(rows),'outcomes':dict(collections.Counter(r.get('outcome') for r in rows)),
         'unfinished_rate':sum(r.get('outcome')!='finished' for r in rows)/len(rows) if rows else None,
         'missing_finished_turn_counts':len(finished)-len(valid),
         'finished_player_turns':distribution([int(r['turns']) for r in valid])}
    if learner:
        wins=[r for r in valid if r.get(r['winner']+'_ai')==learner]
        losses=[r for r in valid if r.get(r['winner']+'_ai')!=learner]
        out['winning_player_turns']=distribution([int(r['turns']) for r in wins])
        out['losing_player_turns']=distribution([int(r['turns']) for r in losses])
    return out


def opponent(row):
    ais=[row.get('ds_ai',''),row.get('ls_ai','')]
    if ais==['policy:learner','policy:learner']:return 'current_self_play'
    others=[a for a in ais if a!='policy:learner']
    return others[0] if len(others)==1 else ' / '.join(ais)


def read_rows(path):
    # An active JVM may have written an incomplete final CSV line.
    with path.open() as f:
        return [r for r in csv.DictReader(f) if r.get('error') is not None and r.get('turns','').isdigit()]


def atomic(path,text):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(text);tmp.replace(path)


def refresh(run):
    stages={};trend=[]
    for stage_dir in sorted((run/'stages').iterdir()):
        if not stage_dir.is_dir():continue
        stage=stage_dir.name;all_rows=[]
        files=sorted((run/'rollouts').glob(stage+'-*.csv'),key=lambda p:int(p.stem.rsplit('-',1)[1]))
        for path in files:
            rows=read_rows(path);all_rows.extend(rows)
            groups={'all':rows}
            for name in sorted(set(map(opponent,rows))):groups[name]=[r for r in rows if opponent(r)==name]
            for name,group in groups.items():
                s=summarize(group);trend.append({'stage':stage,'chunk':int(path.stem.rsplit('-',1)[1]),'opponent':name,
                    'observed_games':len(group),'unfinished_rate':s['unfinished_rate'],**s['finished_player_turns']})
        by_opp=collections.defaultdict(list);by_pair=collections.defaultdict(list)
        for r in all_rows:
            by_opp[opponent(r)].append(r)
            by_pair[(r['ds_deck'],r['ls_deck'],opponent(r))].append(r)
        stages[stage]={'all':summarize(all_rows),'latest_500_games':summarize(all_rows[-500:]),
                       'by_opponent':{k:summarize(v) for k,v in sorted(by_opp.items())},
                       'by_deck_pair':[{'ds_deck':k[0],'ls_deck':k[1],'opponent':k[2],**summarize(v)} for k,v in sorted(by_pair.items())]}
    evaluations={}
    for folder in sorted((run/'evaluations').glob('*')):
        if not folder.is_dir():continue
        agent='policy:ck:'+str((folder/'agent.ts').resolve())
        evaluations[folder.name]={}
        for path in sorted(folder.glob('*.csv')):
            rows=read_rows(path);s=summarize(rows,agent)
            pairs=collections.defaultdict(list)
            for r in rows:pairs[(r['ds_deck'],r['ls_deck'])].append(r)
            s['by_deck_pair']=[{'ds_deck':k[0],'ls_deck':k[1],**summarize(v,agent)} for k,v in sorted(pairs.items())]
            evaluations[folder.name][path.stem]=s
    result={'time':time.time(),'turn_definition':'DS turns started + LS turns started; two player-turns = one round-equivalent',
            'censoring':'Only finished games with a winner and positive turn count enter turn distributions. Unfinished outcomes are reported separately.',
            'ordering':'Training chunks in numeric order; completion order within each chunk. Deck-pair results retain opponent identity.',
            'stages':stages,'evaluations':evaluations}
    atomic(run/'game-lengths.json',json.dumps(result,indent=2)+'\n')
    if trend:
        import io
        out=io.StringIO();writer=csv.DictWriter(out,fieldnames=list(trend[0]));writer.writeheader();writer.writerows(trend)
        atomic(run/'game-length-trend.csv',out.getvalue())
    lines=['# Game length tracking','',result['turn_definition']+'.', '',result['censoring'],'',
           '## Training: finished games','', '| Stage / opponent | Games | Mean turns | Median | p90 | Under 10 turns |',
           '|---|---:|---:|---:|---:|---:|']
    def row(label,d):
        if not d['games']:return
        lines.append(f"| {label} | {d['games']} | {d['mean']:.1f} | {d['median']:g} | {d['p90']:.1f} | {d['under_10_player_turns_rate']:.1%} |")
    for stage,s in stages.items():
        for name,v in s['by_opponent'].items():row(stage+' / '+name,v['finished_player_turns'])
    lines += ['', '## Evaluation: winning games only','', '| Checkpoint / opponent | Wins | Mean turns | Median | p90 | Under 10 turns |',
              '|---|---:|---:|---:|---:|---:|']
    for tag,opps in evaluations.items():
        for name,v in opps.items():row(tag+' / '+name,v['winning_player_turns'])
    lines += ['', 'JSON also includes losses, unfinished rates, deck-pair breakdowns and under-10-round-equivalent rates.',
              'The CSV tracks each rollout chunk by opponent. Fixed held-out evaluations offer the cleaner learning comparison.',
              'Training mixtures and deck difficulty can change aggregate length; stronger defense can lengthen self-play.',
              '',f'Updated Unix time: {result["time"]:.0f}', '']
    atomic(run/'game-lengths.md','\n'.join(lines))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=Path('runs/knowledge-v3.1'));ap.add_argument('--watch',action='store_true');a=ap.parse_args()
    if not a.watch:refresh(a.run);return
    # Independent read-only observer of training; also collects final evaluation.
    while True:
        refresh(a.run)
        status_path=a.run/'campaign-status.json'
        try:
            if status_path.exists() and json.loads(status_path.read_text()).get('phase') in ('complete','stopped'):break
        except json.JSONDecodeError:
            pass  # controller may be in the middle of writing its status
        time.sleep(60)

if __name__=='__main__':main()
