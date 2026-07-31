from ablations.eval.cluas_qa_units import rubric_answer_candidates
from ablations.eval.discrete_eval_common import reference_token_budget


class _WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()


def test_reference_budget_is_reference_relative_with_floor_and_ceiling():
    tokenizer = _WhitespaceTokenizer()
    assert reference_token_budget(tokenizer, "one two three") == (3, 8)
    assert reference_token_budget(
        tokenizer, "one two three four five six seven eight",
        multiplier=1.25,
        minimum=2,
        hard_max=7,
    ) == (8, 7)


def test_cluas_inline_rubric_extracts_each_accepted_answer():
    rubric = (
        "Cén fáth? Chun cuidiú leo = 2 mharc "
        "Do dhaoine atá ag fulaingt = 1 mharc Don Afraic = 0"
    )
    assert rubric_answer_candidates(rubric, "Cén fáth?") == [
        "Chun cuidiú leo",
        "Do dhaoine atá ag fulaingt",
    ]


def test_cluas_numeric_answer_is_not_stripped():
    rubric = "Cé mhéad rannóg a bheidh ann? 6 = 2 mharc"
    assert rubric_answer_candidates(
        rubric, "Cé mhéad rannóg a bheidh ann?"
    ) == ["6"]
