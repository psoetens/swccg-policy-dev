package com.gempukku.swccgo.bench;

import com.gempukku.swccgo.game.knowledge.ObservationV2;
import java.util.*;

/** Direct observation -> tensor conversion (schema 4); no JSON parse and no hidden actor inputs. */
public final class BinaryObservation {
    private final ObservationV2 visible;
    public ObservationV2 privileged;
    /** Plan critic-noise stage 0: the same decision's actor view under the legacy knowledge rule (-Dpolicy.dualLegacy). */
    public BinaryObservation legacy;
    private final String question;
    private final List<String> questionCards;
    private final int[] globals;
    private final int side;
    public BinaryObservation(ObservationV2 visible,String question,List<String> cards,int[] globals,int side) {
        this.visible=visible;this.question=question;this.questionCards=List.copyOf(cards);this.globals=globals;this.side=side;
    }
    public PolicyTensors actor(StateEncoder.Encoded e,PolicyTensors.Maps maps) {return convert(e,maps,visible,true);}
    public PolicyTensors critic(StateEncoder.Encoded e,PolicyTensors.Maps maps) {
        if(privileged==null) throw new IllegalStateException("missing synchronized critic");
        return convert(e,maps,privileged,false);
    }
    /** Plan multiselect-count stage B: each tensor option with its "already selected" flag as a sixth element. */
    static List<Object[]> withSelected(StateEncoder.Encoded e) {
        if (e.tensorOptions.size() != e.selected.size()) throw new IllegalStateException("option flags out of step");
        List<Object[]> out = new ArrayList<>(e.tensorOptions.size());
        for (int i = 0; i < e.tensorOptions.size(); i++) {
            Object[] o = e.tensorOptions.get(i);
            out.add(new Object[]{o[0], o[1], o[2], o[3], o[4], e.selected.get(i) ? 1 : 0});
        }
        return out;
    }
    @SuppressWarnings("unchecked")
    private PolicyTensors convert(StateEncoder.Encoded e,PolicyTensors.Maps maps,ObservationV2 observation,boolean actor) {
        if (observation.schema == 4) {
            com.gempukku.swccgo.game.knowledge.KnowledgeGrid.validate(observation, !actor);
            return PolicyTensorsV4.convert(observation.tokens, observation.kgrid,
                actor ? withSelected(e) : List.of(), e.type, question, globals, side, maps, !actor);
        }
        throw new IllegalArgumentException("binary observations are schema 4 only (plan remove-old-generations)");
    }
}
