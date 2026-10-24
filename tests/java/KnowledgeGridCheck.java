import com.gempukku.swccgo.game.knowledge.KnowledgeGrid;
import com.gempukku.swccgo.game.knowledge.PileKnowledge;
import java.util.List;

public final class KnowledgeGridCheck {
    private static void cell(int[] grid, int pile, int scope, int lower, int upper, int asserted) {
        int offset = KnowledgeGrid.offset(pile, scope);
        if (grid[offset] != lower || grid[offset + 1] != upper || grid[offset + 2] != asserted) {
            throw new AssertionError("Unexpected cell " + pile + ":" + scope);
        }
    }

    private static void rejects(Runnable operation, String message) {
        try {
            operation.run();
        } catch (IllegalStateException exception) {
            if (exception.getMessage().contains(message)) {
                return;
            }
            throw exception;
        }
        throw new AssertionError("Expected rejection: " + message);
    }

    public static void main(String[] args) {
        PileKnowledge ours = new PileKnowledge();
        ours.initialize(PileKnowledge.RESERVE, 4, List.of("1_1", "1_1"));
        ours.initialize(PileKnowledge.LOST, 1, List.of("1_2"));
        PileKnowledge disclosed = new PileKnowledge();
        disclosed.initialize(PileKnowledge.RESERVE, 4, List.of());
        int[] grid = KnowledgeGrid.row("1_1", new PileKnowledge[]{ours, disclosed});
        cell(grid, PileKnowledge.RESERVE, 0, 2, 4, 1);
        cell(grid, PileKnowledge.RESERVE, 1, 0, 4, 0);
        cell(grid, PileKnowledge.LOST, 0, 0, 0, 1);
        cell(grid, PileKnowledge.TABLE, 0, 0, 0, 0);
        for (int quantity = 0; quantity < 17; quantity++) {
            cell(grid, quantity, 2, 0, 0, 0);
        }
        PileKnowledge oversized = new PileKnowledge();
        oversized.initialize(PileKnowledge.RESERVE, 256, List.of());
        rejects(() -> KnowledgeGrid.row("1_1", new PileKnowledge[]{oversized}), "uint8");
        System.out.println("{\"passed\":true,\"defaults\":true,\"completeness\":true,\"overflow_rejected\":true}");
    }
}
