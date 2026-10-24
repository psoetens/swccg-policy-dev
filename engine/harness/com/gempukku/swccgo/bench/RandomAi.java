package com.gempukku.swccgo.bench;

import com.gempukku.swccgo.ai.SwccgAiController;
import com.gempukku.swccgo.game.state.GameState;
import com.gempukku.swccgo.logic.decisions.AwaitingDecision;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

/**
 * A patient random player (training-league opponent "random", weekend plan Addendum 7): the same rules as the test human of
 * swccg-policy/scripts/browser_bot_e2e.py. On "choose an action or pass" it passes 65 % of the time (never if passing is not
 * allowed), otherwise a random action; a random choice for other action choices, yes/no and multiple-choice questions; a random
 * number between min and min + 5; for card selections the minimum number of random selectable cards (step-by-step selections keep
 * the preselected cards). It shows that patience can win against players who act whenever they can.
 */
public class RandomAi implements SwccgAiController {
    static final double PASS = 0.65;

    @Override
    public String decide(String playerId, AwaitingDecision decision, GameState gameState) {
        ThreadLocalRandom rnd = ThreadLocalRandom.current();
        Map<String, String[]> p = decision.getDecisionParameters();
        switch (decision.getDecisionType().name()) {
            case "CARD_ACTION_CHOICE": {
                String[] ids = p.getOrDefault("actionId", new String[0]);
                boolean noPass = "true".equals(first(p, "noPass"));
                if (ids.length > 0 && (noPass || rnd.nextDouble() >= PASS)) return ids[rnd.nextInt(ids.length)];
                return "";
            }
            case "ACTION_CHOICE": {
                String[] ids = p.getOrDefault("actionId", new String[0]);
                return ids.length == 0 ? "" : ids[rnd.nextInt(ids.length)];
            }
            case "MULTIPLE_CHOICE": {
                String[] res = p.getOrDefault("results", new String[0]);
                return String.valueOf(res.length == 0 ? 0 : rnd.nextInt(res.length));
            }
            case "INTEGER": {
                int lo = parseInt(first(p, "min"), 0), hi = parseInt(first(p, "max"), lo);
                return String.valueOf(lo + rnd.nextInt(Math.max(0, Math.min(hi, lo + 5) - lo) + 1));
            }
            case "CARD_SELECTION":
            case "ARBITRARY_CARDS": {
                String[] ids = p.getOrDefault("cardId", new String[0]);
                String[] sel = p.get("selectable"), pre = p.get("preselected");
                List<String> keep = new ArrayList<>(), choose = new ArrayList<>();
                for (int i = 0; i < ids.length; i++) {
                    boolean isPre = pre != null && i < pre.length && "true".equals(pre[i]);
                    boolean selectable = sel == null || (i < sel.length && "true".equals(sel[i]));
                    if (isPre) keep.add(ids[i]);
                    else if (selectable) choose.add(ids[i]);
                }
                int min = parseInt(first(p, "min"), 1);
                Collections.shuffle(choose, rnd);
                List<String> answer = new ArrayList<>(keep);
                for (String c : choose) {
                    if (answer.size() >= min) break;
                    answer.add(c);
                }
                return String.join(",", answer);
            }
            default:
                return "";
        }
    }

    private static String first(Map<String, String[]> p, String k) {
        String[] v = p.get(k);
        return v == null || v.length == 0 ? null : v[0];
    }

    private static int parseInt(String s, int def) {
        try {
            return s == null ? def : Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            return def;
        }
    }
}
