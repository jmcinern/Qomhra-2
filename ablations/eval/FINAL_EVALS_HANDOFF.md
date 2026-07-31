# FINAL_EVALS_HANDOFF — for the write-up agent

**Date:** 2026-07-30
**Supersedes:** `HANDOVER-3007.md` for all four evals below. That file's checkpoint
mapping, LUMI access, and script descriptions are still accurate background reading;
this file only says what changed and which files are now current.

## What changed since HANDOVER-3007

1. **MEXA**: the four missing cross-lingual/cross-modal pairs are filled in (was 2 of
   6, now 6 of 6). `base`'s MEXA numbers were being measured through the wrong
   pathway and have been corrected — the fix is large, not cosmetic.
2. **FLEURS / IWSLT / CLUAS**: regenerated with the per-example generation budget
   doubled (`--reference-budget-multiplier 2.5`, was `1.25`), because too many
   generations were being cut off before the model finished. CLUAS re-marked fresh
   against the new generations.

## Files to use now (all in `ablations/eval/output/review_csv/`)

| Eval | Use this | Not this |
|---|---|---|
| FLEURS | `fleurs_numbers_final.csv`, `fleurs_raw_outputs_final.csv` | `fleurs_numbers.csv` / `fleurs_raw_outputs.csv` (old 1.25x budget; note these also carry the AB1 10-checkpoint trajectory, which the `_final` files do not — see below) |
| IWSLT | `iwslt_numbers_final.csv`, `iwslt_raw_outputs_final.csv` | `iwslt_numbers.csv` / `iwslt_raw_outputs.csv` |
| CLUAS | `cluas_numbers_final.csv`, `cluas_raw_outputs_final.csv` | `cluas_numbers.csv` / `cluas_raw_outputs.csv`; also `cluas_sonnet_scored/` (old marks) vs `cluas_sonnet_scored_final/` (new) |
| MEXA | `mexa_full6_pooled_units.csv`, `mexa_full6_asr_boundary.csv` **for ab1-ab4**, plus `mexa_base_native.csv` **for base** | do not use the `expanded_base` rows inside the two `mexa_full6_*.csv` files as base's value — see caveat 3 below |

**Scope note on FLEURS**: `fleurs_numbers_final.csv` / `fleurs_raw_outputs_final.csv`
cover only the 5 final-checkpoint model states (base, ab1, ab2, ab3, ab4) at
n=34/6 conditions — 1020 raw rows. The old `fleurs_raw_outputs.csv` additionally
carries the AB1 10%-step trajectory (2856 rows total), which was not rerun at the
doubled budget. If the write-up needs the AB1 trajectory figure, pull it from the old
file and label it as 1.25x-budget; don't mix it with the `_final` numbers in the same
table.

## Caveats to carry into the write-up, not resolve silently

1. **AB2 is dead, budget or not.** IWSLT and CLUAS: AB2 hits the generation cap on
   essentially every single item even at 2.5x budget (IWSLT: 112/112 empty; CLUAS:
   caps on every item across all three conditions). AB3 is also 100% empty on IWSLT.
   This is a real generation failure (never emits EOS), not something more tokens
   fixes. Report it as such — don't imply the doubled budget "didn't help enough."

2. **MEXA has no single canonical representation.** `pooled_units` (averaged hidden
   state over raw acoustic-unit positions) and `asr_boundary` (single causal-summary
   state at the position right before transcript generation) are both reported, on
   purpose. They diverge for AB2 specifically: `pooled_units` shows AB2's
   `speech_ga~speech_en` climbing to ~2x chance over training while `asr_boundary`
   stays flat near chance. AB2 never sees text or translation pairs, so this is a
   genuinely open, unexplained result — full numbers and reasoning are in
   `MEXA_PROGRESS.md` (2026-07-30 entries). Do not pick one representation as "the"
   MEXA score without flagging this; do not resolve the divergence by assumption.

3. **`base`'s MEXA number was wrong before today and is now fixed.** The discrete-rerun
   MEXA grid ran `base` through `--speech units` (raw mHuBERT unit ids fed straight in,
   audio tower untouched) via an "expanded_base" checkpoint, matching the pathway used
   for AB1-4 (which never train the audio tower). But `base`'s audio tower is real and
   working, and FLEURS/IWSLT/CLUAS all evaluate `base` through it. Testing `base`
   through raw unit ids it has no training on understated its actual cross-modal
   alignment by roughly an order of magnitude on some pairs (e.g. Irish speech↔text:
   0.09 → 0.92 once run through the native tower). `mexa_base_native.csv` is the
   corrected version; full comparison table in `MEXA_PROGRESS.md`.

4. **Generation budget context**: FLEURS/IWSLT/CLUAS use a per-example budget of
   `ceil(reference_length * multiplier) + 1`, not a flat token cap — the multiplier
   just went from 1.25 to 2.5. This was chosen because Irish tokenizes less
   efficiently than English, so a flat English-length-derived budget was too tight for
   the reasons the numbers actually needed. It is not a fixed `--max-new-tokens`
   change (that flag was already generous and was never the binding constraint —
   confirmed by checking that no capped row's budget ever equaled the flat ceiling).

## Where the reasoning trail lives

- `ablations/eval/MEXA_PROGRESS.md` — full MEXA history, including today's two entries
  (six-pair rescore + representation decision; base pathway correction).
- This file — the cross-eval summary for the doubled-budget rerun.
