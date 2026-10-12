"""Synthetic templated notes: versioned source tags, coverage, no label leakage, field preservation."""
from __future__ import annotations

import numpy as np

from medical_tokenizer.data.notes import (IMAGE_CONTEXT, TEMPLATE_RULES, TEMPLATE_VERSION, VIDEO_CONTEXT, note_fields,
                                          stay_note, template_source)
from medical_tokenizer.data.physionet import TS_IDX, Stay
from medical_tokenizer.protocol.schema import Segment

from conftest import synthetic_stay


def _stay(obs, death=0, height=170.0):
    static = {"Age": 60.0, "Gender": 1.0, "Height": height, "ICUType": 3.0}
    return Stay("123", static, np.array(obs, np.float32).reshape(-1, 3), death)


def test_source_tag_is_versioned_and_synthetic():
    assert template_source("admission") == f"synthetic-template:admission@v{TEMPLATE_VERSION}"
    assert set(TEMPLATE_RULES) == {"admission", "request", "procedure"}


def test_negation_missing_values_and_times():
    empty = stay_note(_stay([], height=np.nan))
    assert "not recorded" in empty
    assert any(c in empty for c in ("No vital signs", "none charted"))
    assert any(c in empty for c in ("not assessed", "no GCS", "not documented"))
    assert "ventilat" in empty.lower() or "intubated" in empty.lower()
    full = stay_note(_stay([(7, TS_IDX["HR"], 80), (7, TS_IDX["MAP"], 90), (9, TS_IDX["Temp"], 37.0),
                            (12, TS_IDX["GCS"], 15), (30, TS_IDX["MechVent"], 1)]))
    assert "00:07" in full and "15" in full and "00:30" in full


def test_note_never_reads_outcome_or_later_hours():
    obs = [(5, TS_IDX["HR"], 80)]
    a, b = stay_note(_stay(obs, death=0)), stay_note(_stay(obs, death=1))
    assert a == b
    later = stay_note(_stay(obs + [(600, TS_IDX["GCS"], 3), (900, TS_IDX["MechVent"], 1)]))
    assert later == a


def test_image_and_video_notes_do_not_name_labels():
    from medical_tokenizer.data.video import CATARACT101_PHASES
    for t in VIDEO_CONTEXT:
        assert not any(p.lower() in t.lower() for p in CATARACT101_PHASES)
    for opts in IMAGE_CONTEXT.values():
        for t in opts:
            assert not any(w in t.lower() for w in ("pneumonia", "melanoma", "nevus", "neutrophil", "normal"))


def test_field_preservation_through_bpe(text_tok):
    rng = np.random.default_rng(0)
    for i in range(20):
        note = stay_note(synthetic_stay(i, rng))
        span = text_tok.encode(Segment("text", note, source=template_source("admission")))
        dec = text_tok.decode(span)
        assert dec == note and note_fields(dec) == note_fields(note)
