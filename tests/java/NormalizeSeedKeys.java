import com.gempukku.swccgo.game.knowledge.TitleMatcher;
import com.google.gson.Gson;
import java.io.BufferedReader;
import java.io.InputStreamReader;

/** JSON-lines bridge to the exact Java runtime normalizer; no simulator or hidden state. */
public final class NormalizeSeedKeys {
    public static void main(String[] args) throws Exception {
        Gson json = new Gson();
        TitleMatcher matcher = TitleMatcher.shared();
        try (BufferedReader input = new BufferedReader(new InputStreamReader(System.in))) {
            for (String line; (line = input.readLine()) != null;) {
                String[] request = json.fromJson(line, String[].class);
                System.out.println(json.toJson(new String[]{request[0], request[1],
                    matcher.seedKey(request[0], request[1])}));
            }
        }
    }
}
