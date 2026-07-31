#!/usr/bin/env python3
"""Run and validate the established Claude Sonnet CLUAS marking workflow."""
import argparse
import json
import subprocess
from pathlib import Path


CONDITIONS = ("just_audio", "no_context", "just_transcript")

PROMPT = """You are an experienced examiner marking An Chluastuiscint (the
listening-comprehension section of the Irish Leaving Certificate, Higher
Level). Mark strictly to the official marking scheme, exactly as the State
Examinations Commission does.

TASK: Read the input file, mark every candidate answer, and write a JSON
results file. Do not change the input file. Do not repair, normalise, or
rewrite any candidate answer.

INPUT FILE (read it): {input_path}
It is a JSON array of items: {{question_id, question (Irish), marking_scheme
(verbatim rubric -- THIS is the answer key), marks_available (int),
blanks_required (int), answers: {{just_audio, no_context, just_transcript}}}}.

OUTPUT FILE (write it, strict UTF-8): {output_path}
Shape: an object mapping each question_id to
{{"just_audio": {{"marks": <int>, "reason": "<short English>"}},
"no_context": {{...}}, "just_transcript": {{...}}}}.

HOW THE SCHEME WORKS:
- It lists accepted answers, each priced e.g. "Scléip = 2 mharc". Where
  several 2-mark answers are listed, ANY ONE earns full marks for that blank.
- A "= 1 mharc" tier is a partially correct answer (right thing, missing the
  detail that carries the 2nd mark). "= 0" marks an answer the examiners
  explicitly refuse.
- The rubric may contain stray PDF page numbers or line breaks. Ignore them;
  obey any examiner notes in parentheses.

MARKING RULES:
1. The candidate must answer in Irish. If the substantive answer is in
   English, award 0. Names, places, dates, numbers, currencies, URLs, and
   emails are language-neutral and valid where the scheme accepts them.
2. Mark MEANING expressed in Irish, not exact string overlap. Accept
   equivalent Irish phrasing, dialect forms, inflections, and minor spelling
   or grammar errors when the intended answer is unambiguous. Do not apply a
   separate language-quality deduction.
3. The candidate must supply the detail the scheme prices. Do not award 2
   marks for an answer the scheme prices at 1.
4. blanks_required is how many DISTINCT points the question demands. Two
   answers restating the same point earn one blank's marks, not two. Reward
   only distinct points, up to blanks_required.
5. Never award more than marks_available, and never award for a point the
   scheme does not list, however true.
6. Candidate answers were already bounded during decoding. Mark exactly the
   supplied candidate answer. A purely degenerate or empty answer scores 0.

Return marks as integers in [0, marks_available]. Write the complete output
file, then reply with only the three per-condition totals and any genuinely
ambiguous item IDs. Do not launch subagents.
"""


def validate(packet_path, verdict_path):
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    verdicts = json.loads(verdict_path.read_text(encoding="utf-8"))
    expected = {item["question_id"]: item for item in packet}
    if set(verdicts) != set(expected):
        missing = sorted(set(expected) - set(verdicts))
        extra = sorted(set(verdicts) - set(expected))
        raise ValueError(f"question ID mismatch: missing={missing}, extra={extra}")
    for qid, item in expected.items():
        result = verdicts[qid]
        if set(result) != set(CONDITIONS):
            raise ValueError(f"{qid}: condition keys are {sorted(result)}")
        maximum = item["marks_available"]
        for condition in CONDITIONS:
            decision = result[condition]
            marks = decision.get("marks")
            if isinstance(marks, bool) or not isinstance(marks, int):
                raise ValueError(f"{qid}/{condition}: non-integer marks {marks!r}")
            if not 0 <= marks <= maximum:
                raise ValueError(
                    f"{qid}/{condition}: marks {marks} outside [0, {maximum}]"
                )
            if not str(decision.get("reason", "")).strip():
                raise ValueError(f"{qid}/{condition}: missing reason")
    return verdicts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("packets", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="sonnet")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for packet_arg in args.packets:
        packet_path = Path(packet_arg).resolve()
        output_path = output_dir / packet_path.name
        prompt = PROMPT.format(
            input_path=packet_path,
            output_path=output_path,
        )
        completed = subprocess.run(
            [
                "claude",
                "--print",
                prompt,
                "--model",
                args.model,
                "--permission-mode",
                "acceptEdits",
                "--allowedTools",
                "Read,Write",
                "--disallowedTools",
                "Agent",
            ],
            check=False,
            text=True,
            capture_output=True,
            encoding="utf-8",
        )
        if completed.returncode:
            raise SystemExit(
                f"{packet_path.name}: Claude failed ({completed.returncode})\n"
                f"{completed.stderr.strip()}"
            )
        if not output_path.exists():
            raise SystemExit(
                f"{packet_path.name}: Claude did not write {output_path}\n"
                f"{completed.stdout.strip()}"
            )
        verdicts = validate(packet_path, output_path)
        totals = {
            condition: sum(v[condition]["marks"] for v in verdicts.values())
            for condition in CONDITIONS
        }
        print(f"{packet_path.stem}: valid; totals={totals}")


if __name__ == "__main__":
    main()
