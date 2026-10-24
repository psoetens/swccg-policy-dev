package com.gempukku.swccgo.bench;

import com.gempukku.swccgo.game.SwccgCardBlueprintLibrary;
import com.gempukku.swccgo.logic.vo.SwccgDeck;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;
import java.util.List;

/**
 * Deck XML loading for the training harness (plan engine-on-feat-brainiac: moved out of GempBench, the engine
 * benchmark, which stays in gemp-swccg-agent with its EngineProfiler).
 */
final class DeckLoader {
    private DeckLoader() {
    }

    static SwccgDeck loadDeck(File file, SwccgCardBlueprintLibrary library) {
        try {
            var doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(file);
            SwccgDeck deck = new SwccgDeck(file.getName());
            // Mirrors DeckSerialization.buildDeckFromContents: unknown blueprints are dropped (with a warning here).
            for (String tag : List.of("card", "cardOutsideDeck")) {
                NodeList nodes = doc.getElementsByTagName(tag);
                for (int i = 0; i < nodes.getLength(); i++) {
                    String id = ((Element) nodes.item(i)).getAttribute("blueprintId");
                    var bp = library.getSwccgoCardBlueprint(id);
                    if (bp == null) {
                        System.err.println("WARN dropping unknown card " + id + " (" + tag + ") in " + file.getName());
                        continue;
                    }
                    // Preload the back side too: the library caches lazily in an unsynchronized HashMap.
                    if (bp.isFrontOfDoubleSidedCard())
                        library.getSwccgoCardBlueprint(library.stripBlueprintModifiers(id) + "_BACK");
                    if (tag.equals("card")) deck.addCard(id);
                    else deck.addCardOutsideDeck(id);
                }
            }
            return deck;
        } catch (Exception e) {
            throw new RuntimeException("cannot load deck " + file + ": " + e, e);
        }
    }
}
