"""Base's raw prompt must be the ablations' prompt with only the unit block swapped.

The whole point of the base_raw arm is that nothing changes except the one
substitution base's vocabulary forces: it has no embedding for ids 152936-152938,
which sit past its 151936 rows, so the waveform occupies the slot the units occupy
and the sentinels go in as literal text. If the two builders ever drift apart the
"matched prompt" claim silently stops being true, which is exactly the kind of
confound this arm exists to remove -- hence this test.
"""
from ablations.eval.cluas_qa_units import build_prompt, build_tower_text
from ablations.eval.discrete_eval_common import (
    AUDIO_MARKERS,
    BASE_VOCAB,
    SPEECH_END,
    SPEECH_END_TEXT,
    SPEECH_START,
    SPEECH_START_TEXT,
    TRANSCRIPT_START,
    TRANSCRIPT_START_TEXT,
)
from ablations.eval.iwslt_st_units import (
    build_prompt as iwslt_build_prompt,
    build_tower_text as iwslt_build_tower_text,
)


class FakeTokenizer:
    """encode() returns the string itself, so prompt text survives into the id list."""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [text]


def render(prompt):
    """The ablation prompt as the text base would need to see for parity.

    Sentinels become their literal spellings and the whole unit run collapses to the
    tower's audio markers, emitted once at the speech-start boundary.
    """
    out = []
    for piece in prompt:
        if piece == SPEECH_START:
            out.append(SPEECH_START_TEXT + AUDIO_MARKERS)
        elif piece == SPEECH_END:
            out.append(SPEECH_END_TEXT)
        elif piece == TRANSCRIPT_START:
            out.append(TRANSCRIPT_START_TEXT)
        elif isinstance(piece, str):
            out.append(piece)
        else:
            # A unit id; absorbed into the markers emitted above.
            assert BASE_VOCAB <= piece < SPEECH_START, f"stray id {piece}"
    return "".join(out)


def test_just_audio_matches_apart_from_the_unit_block():
    ablation = build_prompt(
        FakeTokenizer(), "just_audio", "Cad é sin?", units=[1, 2, 3]
    )
    assert render(ablation) == build_tower_text("Cad é sin?", "just_audio")


def test_text_only_conditions_match_byte_for_byte():
    for condition in ("no_context", "just_transcript"):
        ablation = build_prompt(
            FakeTokenizer(), condition, "Cad é sin?", transcript="dia duit"
        )
        tower = build_tower_text("Cad é sin?", condition, transcript="dia duit")
        assert render(ablation) == tower
        assert AUDIO_MARKERS not in tower


def test_transcript_start_cue_is_carried_as_text():
    ablation = build_prompt(
        FakeTokenizer(), "just_audio", "Q", units=[1],
        answer_cue="transcript_start",
    )
    tower = build_tower_text("Q", "just_audio", answer_cue="transcript_start")
    assert render(ablation) == tower
    assert tower.endswith(TRANSCRIPT_START_TEXT)


def test_iwslt_prompts_match_for_every_task_cue():
    for cue in ("none", "english", "instructed"):
        ablation = iwslt_build_prompt(FakeTokenizer(), [1, 2, 3], cue)
        assert render(ablation) == iwslt_build_tower_text(cue)


def test_iwslt_instructed_keeps_the_transcript_boundary_before_the_cue():
    # The trained speech->text marker must survive into base's prompt too, or the
    # two arms are answering differently shaped questions.
    tower = iwslt_build_tower_text("instructed")
    assert tower.index(TRANSCRIPT_START_TEXT) > tower.index(SPEECH_END_TEXT)
    assert tower.endswith("\nEnglish:")


def test_fleurs_prompts_match_across_every_condition():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "fleurs_discrete_grid.py"
    spec_ = importlib.util.spec_from_file_location("fleurs_discrete_grid", script)
    grid = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(grid)

    row = {
        "ga_raw": "Dia duit", "en_raw": "Hello",
        "ga_norm": "dia duit", "en_norm": "hello",
    }
    for name, spec in grid.CONDITIONS.items():
        ablation = grid.build_discrete_item(
            FakeTokenizer(), spec, row, units=[1, 2, 3]
        )
        assert render(ablation) == grid.build_base_raw_item(spec, row), name


def test_tower_prompt_carries_the_transcript_it_is_conditioned_on():
    # Regression: build_tower_text took no transcript, so just_transcript silently
    # degraded to no_context and base was scored with the text withheld.
    tower = build_tower_text("Q", "just_transcript", transcript="dia duit")
    assert "dia duit" in tower
    assert tower != build_tower_text("Q", "no_context")
