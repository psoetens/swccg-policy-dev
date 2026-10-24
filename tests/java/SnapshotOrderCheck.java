import com.gempukku.swccgo.bench.PolicyEngine;

import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/** Plan nvf-b-local: the newest snapshots are chosen by iteration number (snap-10000 after snap-9999). */
public final class SnapshotOrderCheck {
    public static void main(String[] args) {
        List<String> sorted = List.of("snap-10003.ts", "snap-9999.ts", "snap-0005.ts", "snap-10000.ts").stream()
                .map(Path::of).sorted(Comparator.comparingLong(PolicyEngine::iterationOf).thenComparing(Path::toString))
                .map(Path::toString).collect(Collectors.toList());
        if (!sorted.equals(List.of("snap-0005.ts", "snap-9999.ts", "snap-10000.ts", "snap-10003.ts")))
            throw new AssertionError(sorted.toString());
        System.out.println("SnapshotOrderCheck ok " + sorted);
    }
}
