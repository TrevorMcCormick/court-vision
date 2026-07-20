"""grammar.json is the single vocabulary source: notation.py's
constants must come from it, every char must carry a label, and the
lint must accept the full fault vocabulary (the cv-18 'g' gap)."""

import json
from pathlib import Path

from courtvision import notation
from courtvision.notation import lint

GJ = Path(notation.__file__).with_name("grammar.json")


def test_grammar_file_shape():
    g = json.loads(GJ.read_text())
    assert g["version"] == 1
    assert set(g["vocab"]) == {
        "serve_digits", "shot_letters", "directions", "depths",
        "error_letters", "ending_marks", "fault_letters",
        "other_marks"}


def test_notation_constants_come_from_grammar():
    g = json.loads(GJ.read_text())
    assert notation.SHOT_LETTERS == set(g["vocab"]["shot_letters"])
    assert notation.FAULT_LETTERS == set(g["vocab"]["fault_letters"])
    assert notation.GRAMMAR["version"] == 1
    assert notation.GRAMMAR_PATH == GJ


def test_every_vocab_char_has_a_label():
    g = json.loads(GJ.read_text())
    chars = set("".join(g["vocab"].values())) | {"?"}
    missing = {c for c in chars if c not in g["labels"]}
    assert not missing, f"unlabeled: {missing}"


def test_foot_fault_is_known_vocabulary():
    # 1st = wide serve foot-faulted; 2nd = body serve, bh net UE
    assert lint("4g", "5b2n@") == []
