import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "fleurs_discrete_grid.py"
SPEC = importlib.util.spec_from_file_location("fleurs_discrete_grid", SCRIPT)
GRID = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRID)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [100 + (ord(char) % 50) for char in text]


def unit_rows(n=50):
    rows = []
    for item_id in range(1000, 1000 + n):
        for lang, extra in (("ga", 10), ("en", 0)):
            rows.append({
                "id": item_id,
                "lang": lang,
                "units": list(range((item_id - 990) + extra)),
            })
    return rows


def test_pair_grid_matches_six_mexa_headings_and_marks_speech_speech_na():
    assert set(GRID.PAIR_GRID) == {
        "speech_ga~text_ga",
        "speech_en~text_en",
        "text_ga~text_en",
        "speech_ga~speech_en",
        "speech_ga~text_en",
        "speech_en~text_ga",
    }
    assert GRID.PAIR_GRID["speech_ga~speech_en"] == ()
    assert GRID.PAIR_GRID["text_ga~text_en"] == (
        "text_ga2en", "text_en2ga"
    )


def test_subset_is_parallel_fixed_and_independent_of_shot_count():
    demos, scored, lengths = GRID.choose_ids(
        unit_rows(), n=34, demo_pool_size=3, max_source_units=100
    )
    assert len(demos) == 3
    assert len(scored) == 34
    assert not set(demos) & set(scored)
    assert scored == sorted(scored)
    assert all(set(lengths[item_id]) == {"ga", "en"} for item_id in demos + scored)


def test_discrete_asr_prompt_uses_sentinels_and_trained_demo_eos():
    tok = FakeTokenizer()
    spec = GRID.CONDITIONS["asr_ga"]
    row = {"ga_norm": "dia duit", "ga_raw": "Dia duit"}
    prompt = GRID.build_discrete_item(
        tok, spec, row, units=[1, 1, 2], include_answer=True
    )
    assert prompt[:2] == [GRID.BASE_VOCAB + 1000, GRID.BASE_VOCAB + 1]
    assert prompt[3:5] == [GRID.BASE_VOCAB + 2, GRID.BASE_VOCAB + 1001]
    assert prompt[5] == GRID.TRANSCRIPT_START
    assert prompt[-1] == GRID.IM_END


def test_text_demo_uses_document_eos_not_chat_eos():
    tok = FakeTokenizer()
    spec = GRID.CONDITIONS["text_ga2en"]
    row = {"ga_raw": "Dia duit", "en_raw": "Hello"}
    prompt = GRID.build_discrete_item(
        tok, spec, row, units=None, include_answer=True
    )
    assert prompt[-1] == GRID.END_OF_TEXT
