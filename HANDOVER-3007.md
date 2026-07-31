# HANDOVER-3007 — discrete-speech evaluation takeover

**Date:** 2026-07-30  
**Workspace:** `D:\VS-code-projects\Qomhra-2`  
**Purpose:** complete, inspect, and report the discrete-speech ablation evaluations. This file is the current operational handover; it records what is authoritative, what it supersedes, and what remains.

## 1. Current state in one paragraph

The four discrete ablations have completed the 10%-of-training pre-evaluation on FLEURS, IWSLT, and CLUAS, and MEXA has completed the current monolingual centred-MEXA@10 progression. Raw outputs are preserved in review CSVs; figures and tables are in `ablations/eval/output/review_visualisations/`. CLUAS was scored through Claude Code Sonnet under the local subscription workflow, not through an Anthropic API key. Several CLUAS and FLEURS generations hit `max_new_tokens`; this is recorded, not repaired, and the next generation pass should rerun capped examples with a larger adaptive budget.

## 2. LUMI access and execution

Login and account:

```bash
ssh lumi
# Slurm account:
project_465002364
```

Canonical remote repository and environment:

```text
/scratch/project_465002364/Qomhra-2/
/scratch/project_465002364/Qomhra/Qomhra_v2.sif
/scratch/project_465002364/Qomhra-2/full-train-est/train/qomhra-env.sqsh
/scratch/project_465002364/Qomhra-2/hf_cache/
```

Standard container setup (copy the existing scripts; do not invent a new environment):

```bash
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
singularity exec \
  -B /scratch/project_465002364:/scratch/project_465002364 \
  -B /scratch/project_465002364/Qomhra-2/full-train-est/train/qomhra-env.sqsh:/opt/qomhra-env:image-src=/ \
  --env PYTHONPATH=/opt/qomhra-env:/scratch/project_465002364/Qomhra-2/ablations/train:/scratch/project_465002364/Qomhra-2 \
  --env HF_HOME=/scratch/project_465002364/Qomhra-2/hf_cache \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  /scratch/project_465002364/Qomhra/Qomhra_v2.sif python <script>
```

Use `small-g`/`standard-g` for normal jobs and `dev-g` interactively for quick tests. Check before launching:

```bash
squeue -u "$USER"
```

Do not move thousands of small files through Lustre. Use the existing Parquet bundles and copy only a result file or a deliberate archive. The completed FLEURS trajectory jobs are no longer running; job IDs were 20448549–20448557.

## 3. Data and checkpoints

### Training data

```text
/scratch/project_465002364/audio/discrete_asr_9k/
/scratch/project_465002364/audio/discrete_ablation_9k/
/scratch/project_465002364/audio/unlabelled_subset_10h/
/scratch/project_465002364/Denorm/train/data/*.parquet
/scratch/project_465002364/Qomhra-2/full-train-est/tokens/
```

The paired/supervised ASR data used by the ablations is already recovered and wired into the training manifests. Read `handoff/1-PROGRESS.md`, `handoff/4-PROGRESS.md`, and `ablations/DATA_OVERVIEW.md` before rebuilding data.

### Evaluation bundles

```text
/scratch/project_465002364/Qomhra-2/ablations/eval/data/fleurs_parallel_test.parquet
/scratch/project_465002364/audio/mexa_fleurs_units/fleurs_mexa_units.parquet
/scratch/project_465002364/Qomhra-2/ablations/eval/data/cluas_all.parquet
/scratch/project_465002364/Qomhra-2/ablations/eval/data/iwslt2023_dev.parquet
```

Local copies are under `ablations/eval/data/`. FLEURS has the parallel Irish/English audio and text keyed by sentence ID; the current raw pre-evaluation uses `n=34`, while the MEXA set uses 100 paired sentences.

### Model mapping

The authoritative discrete-rerun checkpoints are:

```text
/scratch/project_465002364/Qomhra-2/ablations/train/output/discrete_rerun/ab1_text/checkpoints/
/scratch/project_465002364/Qomhra-2/ablations/train/output/discrete_rerun/ab2_speech/checkpoints/
/scratch/project_465002364/Qomhra-2/ablations/train/output/discrete_rerun/ab3_text_speech/checkpoints/
/scratch/project_465002364/Qomhra-2/ablations/train/output/discrete_rerun/ab4_text_speech_asr/checkpoints/
```

Mapping used everywhere in the review files:

| Label | Recipe |
|---|---|
| `base` | uncontinued starting model |
| `ab1` | text-only continued training |
| `ab2` | speech-only discrete NTP |
| `ab3` | mixed text + speech |
| `ab4` | text + speech + supervised ASR, aligned recipe |

The 10% checkpoints are approximately `step_000805` (AB3 is `step_000804` in the older manifest). Do not confuse these with old aligned-pilot paths from `HANDOVER-aligned-reeval.md` or old job IDs.

## 4. Authoritative results and supersession

The following current files supersede earlier smoke outputs, API-judge attempts, and hand-written summaries for the 10% pre-evaluation:

```text
ablations/eval/output/review_csv/fleurs_raw_outputs.csv
ablations/eval/output/review_csv/fleurs_numbers.csv
ablations/eval/output/review_csv/iwslt_raw_outputs.csv
ablations/eval/output/review_csv/iwslt_numbers.csv
ablations/eval/output/review_csv/cluas_raw_outputs.csv
ablations/eval/output/review_csv/cluas_numbers.csv
ablations/eval/output/review_csv/README.md
ablations/eval/output/review_visualisations/evaluation_tables.md
ablations/eval/output/review_visualisations/01_mexa_at10.png
ablations/eval/output/review_visualisations/02_fleurs_tasks.png
ablations/eval/output/review_visualisations/02b_fleurs_ab1_trajectory.png
ablations/eval/output/review_visualisations/03_iwslt_routes_and_scores.png
ablations/eval/output/review_visualisations/04_cluas_marks.png
```

Current row counts:

- FLEURS: 2,856 raw rows; 14 numeric rows (base, AB2–AB4, and AB1 at 10–90% plus final 100%).
- IWSLT: 560 raw rows and five numeric model rows.
- CLUAS: 495 raw rows and five numeric model rows; all 495 decisions are Sonnet-validated.
- MEXA: the current centred-MEXA@10 monolingual tables are derived from `Qomhra2-Paper/evals/discrete_mexa/monolingual/{pooled_units,asr_boundary}.csv`.

Older files remain for provenance and should not be deleted. In particular:

- `ablations/eval/output/five_min/` is smoke-only.
- `Qomhra2-Paper/evals/fleurs_*`, `cluas_*`, `IWSLT.csv`, and `mexa_*` are older paper/full-run artefacts unless explicitly regenerated from the current review CSVs.
- `ablations/eval/output/cluas_judge/judge_scored*` is the historical full-benchmark Sonnet result, not the current 33-question pre-eval.
- `ablations/eval/output/review_csv/cluas_sonnet_scored/*.json` is the current pre-eval Sonnet verdict source.

## 5. Evaluation definitions and scripts

### MEXA

MEXA reads internal representations, so it tests alignment without generation. `centered_mexa_at10` is the fraction of paired examples appearing in each other’s top ten after centring; chance is `0.0832868`. The current centred-MEXA run contains only the two monolingual conditions:

```text
speech_ga~text_ga
speech_en~text_en
```

The visualisation reports the best decoder layer at the final checkpoint and uses one continuous 0–1 scale with chance marked. The four requested cross-lingual columns (`text_ga~text_en`, `speech_ga~speech_en`, `speech_ga~text_en`, `speech_en~text_ga`) are **not yet available as centred-MEXA@10**; do not fill them with the older uncentred/top-1 values. Relevant scripts:

```text
ablations/eval/mexa_embed.py
ablations/eval/mexa_centered_pairs.py
ablations/eval/mexa_score.py
ablations/eval/mexa_monolingual_trajectory.py
ablations/eval/mexa_units.sh
```

### FLEURS

Conditions in the JSON/CSV are:

```text
asr_ga       speech_ga -> text_ga       WER/CER
asr_en       speech_en -> text_en       WER/CER
text_ga2en   text_ga   -> text_en       chrF++
text_en2ga   text_en   -> text_ga       chrF++
st_ga2en     speech_ga -> text_en       chrF++
st_en2ga     speech_en -> text_ga       chrF++
```

`ablations/eval/fleurs_prepare_review.py` flattens JSON results; `fleurs_discrete_grid.py` is the generation/evaluation entry point. The AB1 trajectory is in `02b_fleurs_ab1_trajectory.png`. Several conditions have high cap-hit rates, so the apparent score is partly a generation-budget result. Preserve raw `hypothesis`, `raw_output_text`, `stop_reason`, `generation_budget`, and `generated_tokens`; never repair the answer in place.

### IWSLT

`ablations/eval/iwslt_language_route.py` uses fastText `lid.176.ftz` to classify each raw response as English/translation, Irish/transcription, other, or empty, then routes the score to the correct reference. `iwslt_raw_outputs.csv` contains both references, language/confidence, route, both chrF++ values, routed score, stop reason, and cap budget. AB2/AB3 are empty/capped in this pre-eval; keep those as results, not missing data.

### CLUAS

Conditions are `just_audio`, `no_context` (blind), and `just_transcript`. Current marking is by Claude Code Sonnet using the established rubric prompt, not `ANTHROPIC_API_KEY` and not the old SDK cache. The reproducible local runner is:

```text
ablations/eval/cluas_sonnet_judge.py
ablations/eval/output/review_csv/cluas_sonnet_packets/{base,ab1,ab2,ab3,ab4}.json
ablations/eval/output/review_csv/cluas_sonnet_scored/{base,ab1,ab2,ab3,ab4}.json
```

The current totals are:

| Model | Audio /74 | Blind /74 | Transcript /74 | Audio − blind |
|---|---:|---:|---:|---:|
| Base | 0 | 0 | 15 | 0 |
| AB1 text | 2 | 3 | 45 | -1 |
| AB2 speech | 0 | 0 | 0 | 0 |
| AB3 mixed | 0 | 1 | 55 | -1 |
| AB4 aligned | 11 | 3 | 53 | +8 |

## 6. Immediate takeover order

1. Review `evaluation_tables.md` and the five PNGs with Joseph, starting with MEXA. Keep the limitation above visible: only two centred-MEXA@10 monolingual columns exist.
2. Quantify truncation from the raw CSVs. For FLEURS and CLUAS, list capped examples by model/condition; do not discard them.
3. Rerun only capped examples with a larger adaptive generation budget (reference token count plus margin; keep the original files immutable). Re-export files with an explicit `_uncapped`/`_rerun` suffix and compare raw outputs.
4. Complete the centred-MEXA@10 run for the four missing cross-lingual pairs if the embedding NPZs and exact sentence IDs are available. Reuse the existing 100-sentence Parquet and keep the same layer/checkpoint convention.
5. Regenerate paper-facing tables/figures only after Joseph agrees which corrected, uncapped scores should be primary. Never mix old full-run metrics with this 10% pre-ablation table without labelling the checkpoint and metric.

## 7. Useful commands

Local visualisation regeneration:

```powershell
python ablations/eval/visualise_preevals.py
python -m pytest ablations/eval/tests -q
```

Refresh current CSVs after adding result JSONs:

```powershell
python ablations/eval/fleurs_prepare_review.py `
  --inputs "ablations/eval/output/partial10_pre_ablations/fleurs_grid_p10_*_0shot_n34_all.json" `
  --numbers-out ablations/eval/output/review_csv/fleurs_numbers.csv `
  --raw-out ablations/eval/output/review_csv/fleurs_raw_outputs.csv
```

Inspect LUMI jobs/logs:

```bash
squeue -u "$USER"
tail -f /scratch/project_465002364/Qomhra-2/ablations/eval/output/<job-log>.log
```

The evaluator test suite currently passes 7 tests. Preserve this invariant after any changes to stopping, routing, marking, or CSV generation.
