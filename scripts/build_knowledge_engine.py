"""Build an isolated Gemp engine with the versioned knowledge instrumentation.

Baseline sources and bytecode remain untouched. All replacements assert their
anchors so an upstream change fails visibly. The generated diff is reviewable.
"""
from pathlib import Path
import re
import shutil
import subprocess
import os

ROOT=Path(__file__).resolve().parents[1]
# Plan engine-on-feat-brainiac: the base layer is a git archive of the Gemp branch that tracks every fix
# (SWCCG_GEMP_REPO, default .cache/gemp-swccg; feat/brainiac), plus the training harness kept in this repo (engine/harness); the knowledge
# instrumentation below and the per-run overlay go on top. The old base (the lab rig build/gemp, local-dev) is
# no longer read.
GEMP_REPO=Path(os.environ.get('SWCCG_GEMP_REPO', ROOT/'.cache/gemp-swccg'))  # cloned by scripts/setup.sh
GEMP_BRANCH=os.environ.get('SWCCG_GEMP_BRANCH', 'feat/brainiac')
HARNESS=ROOT/'engine/harness'
BASE=ROOT/'.cache/gemp-base-v4'
BUILD=Path(os.environ.get('SWCCG_BUILD_ROOT', ROOT/'.cache/gemp-knowledge-v5'))
LOGIC='gemp-swccg-logic/src/main/java/com/gempukku/swccgo/'

EDITED=[]

def edit(rel,fn):
    EDITED.append(rel)
    source=BASE/rel if (BASE/rel).exists() else HARNESS/rel.split('src/test/java/',1)[1]
    old=source.read_text()
    new=fn(old)
    (BUILD/rel).write_text(new)

def replace(s,a,b,count=1):
    if s.count(a)!=count: raise ValueError(f'expected {count} occurrences, found {s.count(a)}: {a[:100]}')
    return s.replace(a,b)

def state(s):
    s=replace(s,'public class GameState implements Snapshotable<GameState> {',
        'public class GameState implements Snapshotable<GameState> {\n'
        '    private com.gempukku.swccgo.game.knowledge.GameKnowledge _knowledge;\n'
        '    public com.gempukku.swccgo.game.knowledge.GameKnowledge getKnowledge() { return _knowledge; }')
    s=replace(s,'GameState snapshot = selfSnapshot;',
        'GameState snapshot = selfSnapshot;\n        snapshot._knowledge = _knowledge == null ? null : _knowledge.copy();')
    s=replace(s,'            addPlayerCards(playerId, decks, outsideOfDeckCards.get(playerId), library);\n        }',
        '            addPlayerCards(playerId, decks, outsideOfDeckCards.get(playerId), library);\n        }\n'
        '        if (Boolean.getBoolean("policy.knowledge")) {\n'
        '            _knowledge = new com.gempukku.swccgo.game.knowledge.GameKnowledge(_darkSidePlayer, _lightSidePlayer);\n'
        '            _knowledge.initialize(this);\n        }')
    s=replace(s,'            // Any "while active in play" modifiers are removed',
        '            if (_knowledge != null) _knowledge.remove(this, card);\n\n            // Any "while active in play" modifiers are removed')
    s=replace(s,'        card.setZone(cardZone);\n        card.setZoneOwner(zoneOwner);',
        '        card.setZone(cardZone);\n        card.setZoneOwner(zoneOwner);\n'
        '        if (_knowledge != null) _knowledge.add(this, card, top ? 0 : (indexOf != null ? indexOf : zoneCards.size()-1));')
    s=replace(s,'        zoneCards.add(card);\n\n        if (placement.getDirection()',
        '        zoneCards.add(card);\n        if (_knowledge != null) _knowledge.add(this, card, -1);\n\n        if (placement.getDirection()')
    s=replace(s,'    public void shufflePile(String player, Zone zone) {',
        '    public void shufflePile(String player, Zone zone) {\n        if (_knowledge != null) _knowledge.shuffle(player, zone);')
    s=replace(s,'            Collections.reverse(lostPile);',
        '            Collections.reverse(lostPile);\n            if (_knowledge != null) _knowledge.reverse(playerId, Zone.LOST_PILE);')
    s=replace(s,'            Collections.reverse(usedPile);',
        '            Collections.reverse(usedPile);\n            if (_knowledge != null) _knowledge.reverse(playerId, Zone.USED_PILE);')
    # Direct relocation paths bypass remove/add. Synchronize only on real moves,
    # not temporary modifier queries such as checkLocationConditions... .
    s=replace(s,'        for (GameStateListener listener : getAllGameStateListeners())\n            listener.cardMoved(card, this);',
        '        if (_knowledge != null) _knowledge.add(this, card, -1);\n'
        '        for (GameStateListener listener : getAllGameStateListeners())\n            listener.cardMoved(card, this);',count=s.count('        for (GameStateListener listener : getAllGameStateListeners())\n            listener.cardMoved(card, this);'))
    for signature in ['public void showCardOnScreen(PhysicalCard card) {',
                      'public void interruptPlayed(PhysicalCard card) {',
                      'public void epicEventPlayed(PhysicalCard card) {']:
        s=replace(s,signature,signature+'\n        if (_knowledge != null) _knowledge.reveal(this, card, true);')
    s=replace(s,'        if (card!=null && !affectedCards.isEmpty()) {',
        '        if (card!=null && !affectedCards.isEmpty()) {\n            if (_knowledge != null) _knowledge.reveal(this, card, true);')
    for signature in ['public void destinyDrawn(PhysicalCard card, String destinyText) {',
                      'public void activatedCard(String playerPerforming, PhysicalCard card) {']:
        s=replace(s,signature,signature+'\n        if (_knowledge != null) _knowledge.reveal(this, card, false);')
    s=replace(s,'            removeCardFromZone(usedPile.get(usedPile.size() - 1), usedPile.size() > 1, false);',
        '            if (_knowledge != null) _knowledge.expectRemoval(usedPile.get(usedPile.size()-1), usedPile.size()-1);\n'
        '            removeCardFromZone(usedPile.get(usedPile.size() - 1), usedPile.size() > 1, false);')
    s=replace(s,'            removeCardsFromZone(Collections.singleton(card), !lastCard, !lastCard);',
        '            if (_knowledge != null) _knowledge.expectRemoval(card, fromPileCards.size()-1);\n'
        '            removeCardsFromZone(Collections.singleton(card), !lastCard, !lastCard);')
    s=replace(s,'        getZoneCards(newCard.getZoneOwner(), newCard.getZone()).add(newCard);',
        '        getZoneCards(newCard.getZoneOwner(), newCard.getZone()).add(newCard);\n'
        '        if (_knowledge != null) _knowledge.add(this, newCard, -1);')
    s=replace(s,'            // Tell user interface to add cards to revealed sabacc hands',
        '            if (_knowledge != null) {\n'
        '                _knowledge.inspect(_darkSidePlayer, darkSabaccHand, true);\n'
        '                _knowledge.inspect(_lightSidePlayer, lightSabaccHand, true);\n            }\n'
        '            // Tell user interface to add cards to revealed sabacc hands')
    names=re.findall(r'private [^;\n]+ (_[A-Za-z]+State)\s*[;=]',s)
    # Ownership changes (cross-over, persona replacement, attached/stacked cards)
    # also transfer known inventory; original deck provenance is not pile ownership.
    s=re.sub(r'(\b(\w+)\.setOwner\([^;]+;)',r'\1\n            if (_knowledge != null) _knowledge.changeOwner(this, \2);',s)
    body='\n    public java.util.Map<String,Object> knowledgeStoredStates() {\n        java.util.Map<String,Object> result=new java.util.TreeMap<>();\n'
    for name in sorted(set(names)):
        body+=f'        result.put("{name}", {name});\n'
    body+='        return result;\n    }\n'
    s=s.rsplit('}',1)[0]+body+'}\n'
    return s

def decision(s):
    return replace(s,'    private Collection<PhysicalCard> _physicalCards;',
        '    private Collection<PhysicalCard> _physicalCards;\n'
        '    public Collection<PhysicalCard> getKnowledgeShownCards() { return Collections.unmodifiableCollection(_physicalCards); }')

def feedback(s):
    return replace(s,'        _awaitingDecisionMap.put(playerId, awaitingDecision);',
        '        if (_game.getGameState().getKnowledge() != null && awaitingDecision instanceof com.gempukku.swccgo.logic.decisions.ArbitraryCardsSelectionDecision) {\n'
        '            com.gempukku.swccgo.logic.decisions.ArbitraryCardsSelectionDecision d = (com.gempukku.swccgo.logic.decisions.ArbitraryCardsSelectionDecision) awaitingDecision;\n'
        '            _game.getGameState().getKnowledge().inspect(playerId, d.getKnowledgeShownCards(), false);\n        }\n'
        '        _awaitingDecisionMap.put(playerId, awaitingDecision);')

def verification(s):
    # Public verification is captured explicitly where the observation is granted.
    anchor='            // Emit effect result that card pile was verified by opponent'
    var='cardsInCardPiles' if 'cardsInCardPiles' in s else 'cardPile'
    return replace(s,anchor,
        f'            if (gameState.getKnowledge() != null) gameState.getKnowledge().inspect(_playerId, {var}, true);\n\n'+anchor)

def encoder(s):
    s=replace(s,'        public String type;',
        '        public String type;\n        public int version=VERSION;\n'
        '        public com.gempukku.swccgo.game.knowledge.ObservationV2 observation;\n'
        '        public String criticTokens;\n        public long knowledgeSequence;\n'
        '        public String trainingLine(String gameId,int index,List<Integer> chosen) {\n'
        '            String actor=line(gameId,index,chosen);\n'
        '            return criticTokens==null?actor:actor.substring(0,actor.length()-1)+","+criticTokens+","+knowledgeSequence+"]";\n        }')
    s=replace(s,"sb.append('[').append(VERSION)","sb.append('[').append(version)")
    s=replace(s,'        if (!QUERY_CACHE) return encodeImpl(game, player, d);',
        '        if (Boolean.getBoolean("policy.knowledge")) return KnowledgeStateEncoder.encode(game,player,d);\n'
        '        if (!QUERY_CACHE) return encodeImpl(game, player, d);')
    s=replace(s,'                int hi = Math.min(max, min + 30);',
        '                if (Boolean.getBoolean("policy.knowledge") && (long)max-min>4096) throw new IllegalArgumentException("INTEGER action range exceeds supported enumeration");\n'
        '                int hi = Boolean.getBoolean("policy.knowledge")?max:Math.min(max, min + 30);')
    s=replace(s,'    private static void encodeOptions(', '    static void encodeOptions(')
    s=replace(s,'    private static void appendStrings(', '    static void appendStrings(')
    s=replace(s,'        e.answers.add(answer == null ? "" : answer);',
        '        if(e.observation!=null) for(int i=0;i<textBps.size();i++) e.observation.reference(textBps.get(i),e.answers.size()+1,i);\n'
        '        e.answers.add(answer == null ? "" : answer);')
    return s

def policy_ai(s):
    s=replace(s,'String line = e.line(modelKey == null ? "play" : gameId, 0, List.of());',
        'String actorLine = e.line(modelKey == null ? "play" : gameId,0,List.of());\n'
        '            String line = modelKey == null ? actorLine : e.trainingLine(gameId,0,List.of());')
    return replace(s,'engine.query(modelKey, gameId, playerId, line, k, rep, undo,',
        'engine.query(modelKey, gameId, playerId, line, actorLine, k, rep, undo,')

def policy_engine(s):
    s=replace(s,'    private final Map<Path, Slot> snaps = new LinkedHashMap<>();',
        '    private final Map<Path, Slot> snaps = new LinkedHashMap<>();\n    private List<Path> eligibleSnaps=List.of();')
    s=replace(s,'        List<Path> keep = files;\n        snaps.keySet().removeIf(f -> !keep.contains(f)); // (their GPU memory is kept: at most a few per run)',
        '        eligibleSnaps=List.copyOf(files);\n'
        '        var retired=snaps.entrySet().iterator();\n'
        '        while(retired.hasNext()) {\n'
        '            var e=retired.next();\n'
        '            if(!eligibleSnaps.contains(e.getKey()) && !gameSnap.containsValue(e.getKey())) {\n'
        '                NativePolicy.free(e.getValue().handle);retired.remove();\n            }\n        }')
    s=replace(s,'                    if (snaps.isEmpty()) return bc;', '                    if (eligibleSnaps.isEmpty()) return bc;')
    s=replace(s,'                        List<Path> keys = new ArrayList<>(snaps.keySet());', '                        List<Path> keys = eligibleSnaps;')
    s=replace(s,'public List<Integer> query(String modelKey, String game, String side, String line, int k,',
        'public List<Integer> query(String modelKey, String game, String side, String line, String actorLine, int k,')
    s=replace(s,'        List<Integer> out;', '        List<Integer> out;\n        PolicySampler.Sample sample;')
    # Sampling stays on the batcher; compression runs on the requesting game thread.
    s=replace(s,'        requests.incrementAndGet();',
        '        if(r.sample!=null) writer.step(r.game,r.side,r.line,r.sample.seq,r.sample.logp,r.repeats,r.k,\n'
        '            r.excluded==null?null:Arrays.stream(r.excluded).boxed().collect(Collectors.toList()),r.policyLabel);\n'
        '        requests.incrementAndGet();')
    s=replace(s,'                writer.step(r.game, r.side, r.line, s.seq, s.logp, r.repeats, r.k,\n'
        '                        r.excluded == null ? null : Arrays.stream(r.excluded).boxed().collect(Collectors.toList()));',
        '                r.sample=s;')
    s=replace(s,'        r.tensors = PolicyTensors.fromRecord(line, maps);',
        '        long parseStart=com.gempukku.swccgo.game.knowledge.KnowledgeTiming.start();\n'
        '        r.tensors = PolicyTensors.fromRecord(actorLine, maps);\n'
        '        com.gempukku.swccgo.game.knowledge.KnowledgeTiming.end("tensor_parse",parseStart);')
    s=replace(s,'        long m = mtime(src);','''        String publishedLabel=null;
        Path manifest=run.resolve("ckpt/latest.ts.json");
        if(Files.exists(manifest)) try {
            var publication=com.google.gson.JsonParser.parseString(Files.readString(manifest)).getAsJsonObject();
            if(publication.has("file")) {
                src=run.resolve("ckpt").resolve(publication.get("file").getAsString());
                publishedLabel="learner:"+publication.get("iteration").getAsInt();
            }
        } catch(IOException e) {throw new IllegalStateException("cannot read policy publication",e);}
        long m = mtime(src);''')
    s=replace(s,'            learner = load("learner", src);',
        '            learner = load("learner", src);\n            if(publishedLabel!=null) learner.label=publishedLabel;')
    s=replace(s,'            learner.label = labelOf("learner", src);',
        '            learner.label = publishedLabel!=null?publishedLabel:labelOf("learner", src);')
    s=replace(s,'        PolicyTensors tensors;', '        PolicyTensors tensors;\n        String policyLabel;')
    s=replace(s,'                    if (NativePolicy.launch(g.getKey().handle, ins.get(i), sh[0], sh[1], sh[2]) != 0)\n                        throw new IllegalStateException("native launch: " + NativePolicy.lastError());',
        '                    synchronized(this) {\n'
        '                        if (NativePolicy.launch(g.getKey().handle, ins.get(i), sh[0], sh[1], sh[2]) != 0)\n'
        '                            throw new IllegalStateException("native launch: " + NativePolicy.lastError());\n'
        '                        for(Request request:grp) request.policyLabel=g.getKey().label;\n                    }')
    return s


def modifier_context(s):
    names=re.findall(r'private [^;\n]+ (_[A-Za-z]+(?:Modifiers|LimitCounters))\s*[;=]',s)
    body='\n    public java.util.Map<String,Object> knowledgeStoredContext() {\n        java.util.Map<String,Object> result=new java.util.TreeMap<>();\n'
    for name in sorted(set(names)):
        body+=f'        result.put("{name}", {name});\n'
    body+='        return result;\n    }\n'
    return s.rsplit('}',1)[0]+body+'}\n'


def binary_encoder(s):
    s=replace(s,'        public String criticTokens;', '        public String criticTokens;\n        public BinaryObservation binary;\n        public final List<Object[]> tensorOptions=new ArrayList<>();')
    return replace(s,'        e.answers.add(answer == null ? "" : answer);',
        '        e.tensorOptions.add(new Object[]{template,List.copyOf(textBps),src,bp,value});\n        e.answers.add(answer == null ? "" : answer);')

def binary_ai(s):
    s=replace(s,'String actorLine = e.line(modelKey == null ? "play" : gameId,0,List.of());',
        'String actorLine = engine!=null && e.binary!=null ? null : e.line(modelKey == null ? "play" : gameId,0,List.of());')
    s=replace(s,'String line = modelKey == null ? actorLine : e.trainingLine(gameId,0,List.of());',
        'String line = actorLine==null ? null : modelKey == null ? actorLine : e.trainingLine(gameId,0,List.of());')
    return replace(s,'line, actorLine, k, rep, undo,','line, actorLine, e, k, rep, undo,')

def binary_engine(s):
    s=replace(s,'String line, String actorLine, int k,','String line, String actorLine, StateEncoder.Encoded encoded, int k,')
    s=replace(s,'r.tensors = PolicyTensors.fromRecord(actorLine, maps);',
        'r.tensors = encoded.binary==null ? PolicyTensors.fromRecord(actorLine,maps) : encoded.binary.actor(encoded,maps);\n        if(encoded.binary!=null && Boolean.getBoolean("policy.checkBinaryParity")) BinaryTrajectoryCheck.check(encoded,maps,r.tensors);')
    s=replace(s,'if(r.sample!=null) writer.step(', 'if(r.sample!=null && encoded.binary==null) writer.step(')
    s=replace(s,'        requests.incrementAndGet();', '        if(r.sample!=null && encoded.binary!=null) writer.stepBinary(r.game,r.side,r.tensors,\n            encoded.binary.critic(encoded,maps),encoded.knowledgeSequence,r.sample.seq,r.sample.logp,\n            r.repeats,r.k,r.excluded,r.policyLabel,maps);\n        if(r.sample!=null && encoded.binary!=null && encoded.binary.legacy!=null) writer.stepLegacy(r.game,r.side,\n            encoded.binary.legacy.actor(encoded,maps),encoded.binary.legacy.critic(encoded,maps),encoded.knowledgeSequence,\n            r.sample.seq,r.sample.logp,r.repeats,r.k,r.excluded,r.policyLabel,maps);\n        requests.incrementAndGet();')
    return s


# Plan titles-and-learner-ablation A: getFullName marks card names (NameMarks, always on); every reader but a
# normalizer-v2 encoder gets today's text back. Used by prepare_efficient_overlay (training) and brainiac_overlay.
MARKS='com.gempukku.swccgo.game.knowledge.NameMarks'
NAME_MARK_EDITS={
    LOGIC+'logic/GameUtils.java':[
        ('        return card.getTitle() + getTitleSuffix(card.getBlueprint());',
         f'        return {MARKS}.mark(card.getBlueprintId(true), card.getTitle() + getTitleSuffix(card.getBlueprint()));',1),
        ('        return blueprint.getTitle() + getTitleSuffix(blueprint);',
         f'        return {MARKS}.mark(null, blueprint.getTitle() + getTitleSuffix(blueprint));',1)],
    LOGIC+'logic/decisions/AbstractAwaitingDecision.java':[
        ('    public String getText() {\n        return _text;\n    }',
         f'    public String getText() {{\n        return {MARKS}.read(_text);\n    }}',1),
        ('    public Map<String, String[]> getDecisionParameters() {\n        return _params;\n    }',
         f'    public Map<String, String[]> getDecisionParameters() {{\n        return {MARKS}.read(_params);\n    }}',1)],
    LOGIC+'game/state/actions/DefaultActionsEnvironment.java':[
        ('Objects.equal(existingAction.getText(), action.getText())',
         f'Objects.equal({MARKS}.strip(existingAction.getText()), {MARKS}.strip(action.getText()))',6)],
    LOGIC+'logic/timing/DefaultSwccgGame.java':[
        ('    public void takeSnapshot(String description) {\n',
         f'    public void takeSnapshot(String description) {{\n        description = {MARKS}.strip(description);\n',1)],
    LOGIC+'game/state/GameState.java':[
        ('    public void sendMessage(String message) {\n',
         f'    public void sendMessage(String message) {{\n        message = {MARKS}.strip(message);\n',1)],
}

def name_marks(relative,s):
    for before,after,count in NAME_MARK_EDITS[relative]:
        if s.count(before)!=count: raise ValueError(f'{relative}: expected {count} occurrences, found {s.count(before)}: {before[:80]}')
        s=s.replace(before,after)
    return s

def export_base():
    """BASE = git archive of GEMP_BRANCH (src/), plain; returns the exported commit. The harness enters BUILD only."""
    commit=subprocess.check_output(['git','-C',str(GEMP_REPO),'rev-parse',GEMP_BRANCH],text=True).strip()
    if BASE.exists(): shutil.rmtree(BASE)
    BASE.mkdir(parents=True)
    archive=subprocess.run(['git','-C',str(GEMP_REPO),'archive',commit,'src'],check=True,capture_output=True).stdout
    subprocess.run(['tar','-x','-m','--strip-components=1','-C',str(BASE)],input=archive,check=True)
    return commit

def compile_build():
    """Like the Brainiac overlay: Maven compiles the plain base (branch + harness; no install), then one javac compiles
    the instrumented files, the knowledge package and the bench (harness) sources into BUILD's server test-classes, which come
    first on every engine classpath (efficient_java_check.sh, v31_engine.sh) and shadow the plain classes. (The
    knowledge package needs gson and the cards module, so it cannot live in the logic module's Maven build.)"""
    # The plain branch with its own test sources (Gemp's test framework, used by our Java checks); the harness needs
    # the instrumented engine and is compiled by the javac below.
    # One reactor build (-am) for both goals: the sibling modules resolve from the reactor, not from ~/.m2, so a machine
    # that never ran `mvn install` on Gemp builds too (plan dev-distribution). The server module is built last, so its
    # classpath is the one left in cp.txt.
    subprocess.run(['mvn','-q','-DskipTests','test-compile','dependency:build-classpath','-pl','gemp-swccg-server','-am',
                    f'-Dmdep.outputFile={BUILD/"cp.txt"}'],cwd=BASE,check=True)
    # cp.txt is the third-party classpath every engine script reads: drop the reactor's own module folders
    entries=(BUILD/'cp.txt').read_text().strip().split(':')
    (BUILD/'cp.txt').write_text(':'.join(e for e in entries if not e.startswith(str(BASE)))+'\n')
    modules=('server','cards','logic','common')
    for module in modules:
        target=BASE/f'gemp-swccg-{module}/target'
        if (BUILD/f'gemp-swccg-{module}/target').exists(): shutil.rmtree(BUILD/f'gemp-swccg-{module}/target')
        shutil.copytree(target,BUILD/f'gemp-swccg-{module}/target')
    tests=BUILD/'gemp-swccg-server/target/test-classes'
    tests.mkdir(parents=True,exist_ok=True)
    third=[e for e in (BUILD/'cp.txt').read_text().strip().split(':') if '/com/gempukku/' not in e]
    classpath=[str(tests),*(str(BUILD/f'gemp-swccg-{m}/target/classes') for m in modules),*third]
    # The bench classes compile only together with the overlay every run compiles (scripts/efficient_java_check.sh,
    # whose sources prepare_efficient_overlay generates from this tree): compile that same set plus the harness classes
    # it does not list, in one javac, so the tree's test-classes hold every harness class.
    import prepare_efficient_overlay
    prepare_efficient_overlay.prepare()
    check=(ROOT/'scripts/efficient_java_check.sh').read_text()
    listed=[]
    for match in re.finditer(r'"\$(?:PWD|SWCCG_V31_SOURCE)"(/[^\s"\\]+\.java)',check):
        path=match.group(1)
        listed+=sorted(ROOT.glob(path[1:])) if '*' in path else [ROOT/path[1:]]
    names={p.name for p in listed}
    sources=[BUILD/rel for rel in EDITED if not rel.startswith('gemp-swccg-server/src/test/') and Path(rel).name not in names]
    sources+=listed
    # Every bench class of this repo (some, e.g. RolloutBackpressure, are not in the per-run list) and the whole production
    # Brainiac package, which shadows feat/brainiac's own server-module copy (as brainiac_overlay.py does).
    sources+=[p for p in sorted((ROOT/'engine/java/com/gempukku/swccgo/bench').glob('*.java')) if p.name not in names]
    sources+=[p for p in sorted((ROOT/'engine/production/com/gempukku/swccgo/ai/brainiac').glob('*.java'))
              if p.name not in names]
    names|={p.name for p in sources}
    harness={p.name for p in (HARNESS/'com/gempukku/swccgo/bench').glob('*.java')}
    sources+=[p for p in sorted((BUILD/'gemp-swccg-server/src/test/java/com/gempukku/swccgo/bench').glob('*.java'))
              if p.name in harness and p.name not in names]
    sources=list(dict.fromkeys(sources))
    subprocess.run(['javac','-proc:none','-nowarn','-cp',':'.join(classpath),'-d',str(tests),*map(str,sources)],check=True)

def snapshots_switch(s):
    """Training-only speed switch (formerly local-dev e5688e30): headless games take no revert snapshots."""
    s=replace(s,'    @Override\n    public void takeSnapshot(String description) {\n',
        '    // Snapshots only serve the revert feature; headless games (e.g. training) can turn them off.\n'
        '    private boolean _snapshotsEnabled = true;\n'
        '    public void setSnapshotsEnabled(boolean snapshotsEnabled) {\n        _snapshotsEnabled = snapshotsEnabled;\n    }\n\n'
        '    @Override\n    public void takeSnapshot(String description) {\n        if (!_snapshotsEnabled)\n            return;\n')
    return s

def main():
    commit=export_base()
    if BUILD.exists(): shutil.rmtree(BUILD)
    shutil.copytree(BASE,BUILD)
    bench=BUILD/'gemp-swccg-server/src/test/java/com/gempukku/swccgo/bench'
    bench.mkdir(parents=True,exist_ok=True)
    for p in (HARNESS/'com/gempukku/swccgo/bench').glob('*.java'): shutil.copy2(p,bench/p.name)
    (BUILD/'GEMP_SOURCE').write_text(f'{GEMP_REPO} {GEMP_BRANCH} {commit}\n')
    edit(LOGIC+'game/state/GameState.java',state)
    edit(LOGIC+'logic/modifiers/querying/ModifiersLogic.java',modifier_context)
    edit(LOGIC+'logic/decisions/ArbitraryCardsSelectionDecision.java',decision)
    edit(LOGIC+'logic/timing/DefaultUserFeedback.java',feedback)
    edit(LOGIC+'logic/timing/DefaultSwccgGame.java',snapshots_switch)
    for name in ['ChooseCardsFromPileEffect','ChooseCardsFromMultiplePilesEffect','ChooseCardFromHandOrCardPilesEffect']:
        edit(LOGIC+'logic/effects/choose/'+name+'.java',verification)
    dest=BUILD/LOGIC/'game/knowledge';dest.mkdir(exist_ok=True)
    for p in (ROOT/'engine/java/com/gempukku/swccgo/game/knowledge').glob('*.java'): shutil.copy2(p,dest/p.name)
    bench='gemp-swccg-server/src/test/java/com/gempukku/swccgo/bench/'
    edit(bench+'StateEncoder.java',lambda s: binary_encoder(encoder(s)))
    edit(bench+'PolicyAi.java',lambda s: binary_ai(policy_ai(s)))
    edit(bench+'PolicyEngine.java',lambda s: binary_engine(policy_engine(s)))

    edit(bench+'TrainGen.java',lambda s: replace(replace(s,'shard.write(enc.line(gameId, decisions, ch));','shard.write(enc.trainingLine(gameId, decisions, ch));'), '                        while ((g = next.getAndIncrement()) < games) {', '                        while ((g = next.getAndIncrement()) < games) {\n                            if(!RolloutBackpressure.await()) break;'))
    for p in (ROOT/'engine/java/com/gempukku/swccgo/bench').glob('*.java'): shutil.copy2(p,BUILD/bench/p.name)
    compile_build()
    print(BUILD, GEMP_BRANCH, commit)

if __name__=='__main__': main()
