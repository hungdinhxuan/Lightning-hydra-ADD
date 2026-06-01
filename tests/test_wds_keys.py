import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.wds_keys import make_wds_key


def test_make_wds_key_removes_dots_that_break_webdataset_grouping():
    key = make_wds_key("April/Synthesizers/Minimax/speech-2.8-turbo/sample.wav")

    assert "." not in key
    assert "/" not in key


def test_make_wds_key_uses_hash_to_avoid_normalization_collisions():
    first = make_wds_key("a/b.c.wav")
    second = make_wds_key("a/b_c.wav")

    assert first != second
