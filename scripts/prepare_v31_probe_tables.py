"""Append two offline scope-question priors to a disposable Decipher probe bundle."""

import hashlib
import json
from pathlib import Path

import numpy as np

from swccg_policy.cards import embed_texts
from swccg_policy.soft_vocabulary import PRIOR_MODEL, SoftVocabulary
from swccg_policy.table_bundle import TableBundle


SCOPE_QUESTIONS = {
    3: "Which blueprint is present in our private knowledge?",
    6: "Which blueprint is present in public knowledge?",
}


def main():
    parent = TableBundle.load("runs/knowledge-v3.1/tables-decipher")
    vocabulary = SoftVocabulary.build({"q": list(SCOPE_QUESTIONS.values())},
                                      lambda texts: embed_texts(texts, PRIOR_MODEL),
                                      "disposable-scope-probe-questions", parent.vocabulary)
    bundle = TableBundle(parent.cards, vocabulary, parent.sha, (*parent.ancestors, parent.sha))
    parent.assert_prefix_of(bundle)
    rows = {old: vocabulary.lookup("q", text) for old, text in SCOPE_QUESTIONS.items()}
    assert len(set(rows.values())) == 2
    assert not np.array_equal(vocabulary.priors["q"][rows[3]], vocabulary.priors["q"][rows[6]])
    output = Path("runs/knowledge-v3.1/tables-decipher-probes")
    bundle.save(output)
    report = {"parent_sha": parent.sha, "table_sha": bundle.sha, "scope_question_rows": rows,
              "scope_question_text": SCOPE_QUESTIONS, "distinct_priors": True,
              "prior_sha256": {old: hashlib.sha256(vocabulary.priors["q"][row].tobytes()).hexdigest()
                               for old, row in rows.items()},
              "purpose": "Disposable probe-only append batch; production and extension tables unchanged."}
    Path("reports/data/knowledge-v3.1/amendment5-probe-tables.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
