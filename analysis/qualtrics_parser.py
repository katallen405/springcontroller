"""
qualtrics_parser.py

Shared helpers for reading Qualtrics "choice text" CSV exports.

Qualtrics exports have three header rows:
  1. DataExportTag (e.g. "Q19") -- NOT reliable as a unique column key.
     A survey can (and CQHRI.qsf does) reuse the same export tag on more
     than one question -- e.g. "Q19" is both "highest level of education"
     and the Condition question in trial block 1. The raw CSV literally
     contains two columns both named "Q19".
  2. Question text (human-readable, also not unique -- e.g. all three
     Condition questions are just titled "Condition").
  3. A JSON blob like {"ImportId":"QID24"} -- this IS unique per question
     (and per sub-question, e.g. {"ImportId":"QID11_3"} for matrix item 3)
     and is what this module keys columns by.

Because of the tag collisions above, do NOT use pandas.read_csv() directly
on these exports -- it will auto-rename duplicate headers (Q19, Q19.1, ...)
which is fragile and easy to get backwards. Use load_qualtrics_csv() below
instead, and look columns up by ImportId via column_index().
"""

import csv
import json
import re


def load_qualtrics_csv(path):
    """
    Reads a Qualtrics CSV export.

    Returns (import_ids, data_rows) where:
      - import_ids: list[str], one per column, e.g. "QID24", "QID11_3",
        "QID23_TEXT". Parallel to each row in data_rows.
      - data_rows: list[list[str]], the actual response rows (header rows
        excluded).
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 3:
        raise ValueError(f"{path} does not look like a Qualtrics export "
                          f"(expected at least 3 header rows, got {len(rows)} total rows)")

    import_id_row = rows[2]
    import_ids = []
    for cell in import_id_row:
        try:
            import_ids.append(json.loads(cell)["ImportId"])
        except (json.JSONDecodeError, KeyError, TypeError):
            import_ids.append(cell)

    data_rows = rows[3:]
    return import_ids, data_rows


def column_index(import_ids, target_import_id):
    """Exact lookup of a single column index by ImportId. Raises if not
    found or if the ImportId is (unexpectedly) not unique."""
    matches = [i for i, v in enumerate(import_ids) if v == target_import_id]
    if not matches:
        raise KeyError(f"No column found with ImportId {target_import_id!r}")
    if len(matches) > 1:
        raise KeyError(f"ImportId {target_import_id!r} is not unique "
                        f"(found at columns {matches})")
    return matches[0]


# ---------------------------------------------------------------------------
# Canonical response scales, taken verbatim from the QSF question definitions
# (QID10 for fluency; QID11 Answers block for the 5-point agreement scale
# shared by all Perceived Social Intelligence Scale matrix items).
# Keyed case-insensitively (stripped + lowercased) since some historical
# responses have different capitalization than the current choice text
# (the choice wording was edited after some responses were already
# collected).
# ---------------------------------------------------------------------------

FLUENCY_SCALE = {
    "not at all, it was very difficult to get the robot to do something that worked for me": 1,
    "not very fluent, it worked but was surprising or stressful": 2,
    "acceptable but not good, the robot did what i wanted but not always the way i'd have preferred": 3,
    "good, the robot typically did what i needed it to do in a way that i felt comfortable with": 4,
    "as easy as i could imagine, like having a really great teammate who almost reads my mind to anticipate what i need": 5,
}

AGREEMENT_SCALE = {
    "strongly disagree": 1,
    "somewhat disagree": 2,
    "neither agree nor disagree": 3,
    "somewhat agree": 4,
    "strongly agree": 5,
}


def _normalize(text):
    return re.sub(r"\s+", " ", text or "").strip().lower()


def score_choice_text(text, scale_dict, *, context=""):
    """
    Case-insensitive lookup of `text` in `scale_dict`. Returns None (and
    prints a notice) for blank or unrecognized text instead of raising,
    so one bad/missing cell doesn't kill an entire analysis run.
    """
    key = _normalize(text)
    if not key:
        label = f" ({context})" if context else ""
        print(f"NOTICE: blank response text{label} -- treating as missing")
        return None
    score = scale_dict.get(key)
    if score is None:
        label = f" ({context})" if context else ""
        print(f"WARNING: unrecognized response text{label}: {text!r} -- treating as missing")
    return score
