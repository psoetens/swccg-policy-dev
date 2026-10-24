"""Freeze the existing card table's title lookup and report ambiguity deterministically."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, default=Path("data/cards/card_table_knowledge_v2.npz"))
    parser.add_argument("--out", type=Path, default=Path("runs/knowledge-v3.1/titles.tsv"))
    parser.add_argument("--report", type=Path, default=Path("reports/data/knowledge-v3.1/title-table.json"))
    parser.add_argument("--display-metadata", type=Path, help="Static Gemp display aliases for the card-table IDs")
    parser.add_argument("--virtual-display-only", action="store_true",
                        help="virtual cards whose display name differs from the title get only display-name aliases")
    args = parser.parse_args()
    cards = np.load(args.cards)
    rows = list(zip(map(str, cards["ids"][4:]), map(str, cards["titles"][4:]), strict=True))
    suffixes = set()
    metadata = None
    generic_aliases = 0
    if args.display_metadata:
        metadata = json.loads(args.display_metadata.read_text())
        if {row["blueprint"] for row in metadata} != {blueprint for blueprint, _ in rows}:
            raise ValueError("Display metadata blueprint set differs from the frozen card table")
        systems = sorted({row["system"] for row in metadata if row.get("system")})
        aliases = set(rows)
        bare_dropped = set()
        for row in metadata:
            blueprint = row["blueprint"]
            title = row["title"]
            display = row["display"]
            if not display.startswith(title):
                raise ValueError(f"Display title is not a title plus suffix: {blueprint}")
            suffix = display[len(title):]
            if suffix:
                suffixes.add(suffix)
            # Plan v-cards: Gemp names a virtual card with its "(V)" display name, so with --virtual-display-only a virtual
            # card whose display differs from its title gets no bare-title alias (the bare title stays the original card's).
            bare = not (args.virtual_display_only and row.get("virtual") and display != title)
            if not bare:
                bare_dropped.add((blueprint, title))
            if bare:
                aliases.add((blueprint, title))
            aliases.add((blueprint, display))
            if row["perSystem"]:
                for system in systems:
                    if bare:
                        aliases.add((blueprint, system + ": " + title))
                    aliases.add((blueprint, system + ": " + display))
                    generic_aliases += 1
        if args.virtual_display_only:
            # nor any alias that is exactly an original (non-virtual) card's name (V cards Gemp names like the original)
            # by set (>= 200): Gemp's "virtual" flag follows the (V) suffix, which some V-set cards lack
            virtual = {row["blueprint"] for row in metadata
                       if row.get("virtual") or int(row["blueprint"].split("_", 1)[0]) >= 200}
            original_names = {title for blueprint, title in aliases if blueprint not in virtual}
            bare_dropped |= {(blueprint, title) for blueprint, title in aliases
                             if blueprint in virtual and title in original_names}
        rows = sorted(aliases - bare_dropped)
    if any(not blueprint or not title or any(character in blueprint + title for character in "\t\r\n")
            for blueprint, title in rows):
        raise ValueError("Invalid blueprint/title TSV value")
    content = "".join(blueprint + "\t" + title + "\n" for blueprint, title in rows)
    if args.out.exists():
        if args.out.read_text() != content:
            raise ValueError("Existing frozen title table differs from card-table source")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content)
    titles = defaultdict(set)
    for blueprint, title in rows:
        titles[re.sub(r"^(?:•+|<>|◇)+", "", title.replace("&lt;&gt;", "<>"))].add(blueprint)
    ambiguous = {
        title: {"blueprints": sorted(blueprints), "canonical": min(blueprints)}
        for title, blueprints in sorted(titles.items()) if len(blueprints) > 1
    }
    report = {
        "blueprints": len({blueprint for blueprint, _ in rows}),
        "normalized_titles": len(titles), "ambiguous_titles": len(ambiguous),
        "normalization_revision": 2 if metadata else 1,
        "display_suffixes": sorted(suffixes), "generic_system_alias_pairs": generic_aliases,
        "unavailable_in_engine": [row["blueprint"] for row in metadata or [] if row["unavailableInEngine"]],
        "display_metadata_sha256": hashlib.sha256(args.display_metadata.read_bytes()).hexdigest()
            if args.display_metadata else None,
        "ambiguities": ambiguous, "card_table_sha256": hashlib.sha256(args.cards.read_bytes()).hexdigest(),
        "title_table_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "policy": "Use the sole permitted match if unique; otherwise canonical with explicit reference flag.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("blueprints", "normalized_titles", "ambiguous_titles")}))


if __name__ == "__main__":
    main()
