# Discrete pre-ablation evaluation gate — 2026-07-30

## Contract

- Deterministic 10% subsets: CLUAS 33/334 questions, IWSLT 112/1120
  utterances, and FLEURS 34 parallel pairs.
- Greedy raw generation; raw IDs/text and the exact emitted terminal token are
  retained. There is no post-generation repair.
- Both trained EOS tokens are accepted. CLUAS also stops during decoding at a
  repeated `Ceist:` or `Freagra:` prompt boundary.
- Each row receives `ceil(reference_tokens * 1.25) + 1` generated tokens, with
  an 8-token floor and a hard safety ceiling. A cap hit is therefore a bounded
  model failure, not an invitation to clean or continue the output.
- Speech mismatch controls always use a different utterance.

## IWSLT results

All 112-row runs completed. The prompt is in English, preserves the trained
`<|transcript_start|>` boundary, and asks for English-only translation.

| arm | chrF++ | normalized chrF++ English | normalized chrF++ FOTHEIDIL | interpretation |
|---|---:|---:|---:|---|
| base, native audio tower | 12.59 | 13.53 | 11.61 | 79 translation-like / 33 transcription-like |
| AB1 text | 5.86 | 6.27 | 4.77 | weak text output |
| AB2 speech | 0.00 | 0.00 | 0.00 | emits unit IDs, not text |
| AB3 text + speech | 0.00 | 0.00 | 0.00 | emits unit IDs, not text |
| AB4 text + speech + paired ASR | 7.04 | 7.21 | 18.76 | 74 transcription-like / 24 translation-like / 14 ties |

The decisive result is that only paired ASR teaches the discrete-unit model to
return text, but it teaches Irish transcription rather than zero-shot English
speech translation. AB2/AB3 raw unit IDs are retained in the JSON artifacts.

## CLUAS generation gate

All five 33-question runs completed. Official marks still require the existing
LC-Aural-Bench judge.

- AB2 is the wrong-modality negative control: every decode is unit IDs, all
  text hypotheses are empty, and all 91 audio windows are identical under the
  mismatch comparison.
- AB3 often produces plausible Irish from the written question, but 46/91
  audio windows are identical under mismatch. On the inspected rugby item its
  selected real/mismatch answers are exactly identical and miss the rubric.
- AB4 produces selected text for all 33 audio questions and 90/91 audio windows
  differ under mismatch. On the same rugby item it names the two rubric facts
  (excellent play and friendly supporters), while mismatched speech changes the
  answer.
- A non-official maximum-rubric-chrF diagnostic is consistent with this:
  AB4 audio 16.86 versus blind 13.69; official judge marks are not inferred
  from this diagnostic.

Artifacts are under:

`/scratch/project_465002364/Qomhra-2/ablations/eval/output/partial10_pre_ablations`

## FLEURS results

The two-row raw/mismatch smoke passed in job `20446030` (different source IDs,
different outputs, dynamic caps recorded). All five 34-pair six-direction jobs
then completed without launch or parsing errors.

WER is reported for the first two columns (lower is better); chrF++ is reported
for the other four (higher is better).

| arm | ga speech→ga text | en speech→en text | ga text→en text | en text→ga text | ga speech→en text | en speech→ga text |
|---|---:|---:|---:|---:|---:|---:|
| base, native audio tower | 122.59 | **3.77** | 27.15 | **17.00** | **15.66** | **13.98** |
| AB1 text | 102.70 | 101.17 | 31.21 | 5.23 | 4.29 | 1.98 |
| AB2 speech | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AB3 text + speech | 100.00 | 100.00 | 32.57 | 12.06 | 0.02 | 1.08 |
| AB4 text + speech + paired ASR | **68.22** | 98.96 | **46.59** | 13.66 | 6.14 | 4.26 |

The base's 3.77 English WER validates the native-audio speech harness. AB4's
Irish WER improves from the base's 122.59 to 68.22, produces 34/34 non-empty
outputs with natural EOS, and its mismatched-audio output for one row becomes
the actual output for the row whose audio was substituted. This is direct
evidence that paired ASR learned Irish unit→text mapping. It did not teach
general speech translation: both AB4 speech-translation scores remain below
the native base, and its English ASR is poor as expected from Irish-only ASR.

Jobs: base `20446100`, AB2 `20446101`, AB4 `20446102`, AB1 `20446103`,
AB3 `20446104`. Output paths are
`ablations/eval/output/fleurs_grid_p10_<arm>_0shot_n34_all.json`.
