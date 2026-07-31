# Qomhra-2 — Rerun: Discrete Speech

Master doc. Every agent reads this first. Single source of truth.

---

## 1. What this is

Rerun of the original ablations, fixing four things that broke the first attempt.

| # | Original mistake | Fix |
|---|---|---|
| 1 | Speech as Qwen audio-tower encodings; next-frame regression objective | Discrete mHuBERT units as ordinary tokens; next-token prediction |
| 2 | Few-shot prompts with mixed chat templates / EOS tokens, patched by post-processing | Zero-shot native contract. No post-processing. |
| 3 | Ablations trained for different step counts (8192 / 4396 / 8492) | Identical tokens and identical steps |
| 4 | No gate separating "not learning" from "learning, not generating" | Held-out loss per ablation, before any downstream eval |

### Superseded — do not use
- `configs/omni_speech.yaml`, `omni_both.yaml`, `omni_aligned.yaml` — continuous-audio regression. `speech`-final scored 0.0% on all CLUAS conditions.
- Checkpoints under `train/output/2026-07-1[78]_*`.
- `train/README.md` checkpoint-fraction claims (stale).

---

## 2. LUMI workflow

**Docs**
- `D:/VS-code-projects/LUMI-AI-Guide` — quick best practices (storage, env)
- `D:/VS-code-projects/lumi-userguide/docs` — comprehensive
- `full-train-est/` — throughput + tokenization reference

**Rules**
- Access: `ssh lumi`. Setanta: `ssh setanta`. Git with agent forwarding `-A`.
- **Container + squashfs env — never transfer lots of small files.**
- Container: `/scratch/project_465002364/Qomhra/Qomhra_v2.sif`
- Envs: `qomhra-env.sqsh` (torch/transformers), `mhubert-env.sqsh` (soundfile/faiss)
- Correct CPU binds + scratch bind from the example script. Use RCCL interconnect.
- Only the user edits READMEs.

**Traps**
- `/tmp` is per-login-node — it vanishes between sessions. Stage under `/scratch`.
- Never pass a partition list to `--partition` (association error).
- `small-g`: 200 running / 210 submitted. The 2-job limit is `dev-g` only.
- If `small`/`standard` is full, switch to the other and cancel the original.
- `dev-g` interactive node for quick debugging.

---

## 3. The four ablations

Data mixture is the **only** variable.

| # | Ablation | Stream |
|---|---|---|
| 1 | text | text tokens |
| 2 | speech | unit tokens |
| 3 | text + speech | 50/50, interleaved, never paired |
| 4 | text + speech + ASR | X% paired ASR, remainder 50/50 |

### Invariance contract
One parent config. Children differ in `data:` only.
- same seed, `seq_len`, `micro_batch_size`, `grad_acc`, lr, warmup, scheduler
- **same `freeze.mode`** — originally text=`llm_only` vs speech/both=`all`. Second variable.
- same init (expanded-vocab base), same total tokens, **same optimizer steps**
- no `gradient_checkpointing` (no audio tower now)
- **`seq_len` 1024 for all four.** Different context lengths between ablations would confound: it changes sequences-per-token and attention cost. Long context is not needed for these evals.

### 3 vs 4 is clean
ASR transcripts already sit in the text stream (`conversations_capr`); ASR audio already sits in the speech stream. So 3 and 4 see the **same content** — only 4 sees it **paired**. Isolates alignment, not new data.
- Caveat: within 4, that content appears twice. Accept and state; measure the overlap first.

---

## 4. Budget

**Constant = tokens.** Not GPU hours. Data held constant, compute allowed to vary.

- Budget = min(text, speech) = the speech corpus.
- Speech: 1,674,247,769 raw units → **~1.23 B after dedup** (estimate; A measures the real number when packing, and that becomes the budget).
- Text: 2.147 B available.
- Pre-ablation = 10% of budget.

### OPEN — decide before A packs

ASR tokens counted on the **text (loss-bearing) part**. But the supervised ASR corpus is small:

| | tokens |
|---|---:|
| supervised ASR, 413.4 h, total | ~65.6 M |
| of which loss-bearing (14.3%) | **~9.4 M** |

**9.4 M is 0.76% of a 1.23 B budget.** X% cannot be 25% — the entire corpus, one epoch, maxes out well below it.

Three options:
- **(a)** Use all supervised ASR, one epoch. X ≈ 0.8% by text tokens. Honest but a very weak dose.
- **(b)** Use pseudolabelled ASR from the 9K corpus — abundant, but transcripts are model-generated.
- **(c)** Repeat supervised ASR for several epochs — introduces a repetition confound the other ablations don't have.

Recommendation: **(b)**, with supervised ASR held out as the eval set. Needs your call.

---

## 5. Gates

**Primary gate — held-out loss.** Per ablation, on its own validation split. Train loss ↓ *and* held-out loss ↓ vs base. Runs before any downstream eval. Separates "not learning" from "learning, not generating" — the analogue of the retrieval test that settled the ASR question.

**Comparators — the four evals.** CLUAS, FLEURS, IWSLT, MEXA.

**Eval smoke rule.** Every eval must **reliably elicit a response on ~10 questions** before it runs in full. Looking for definitive evidence of learning and task adaptation, not fine-grained performance. No eval runs at scale until its smoke passes.

**MEXA.** Now an LLM-only probe — the audio tower is not trained. Covers all cross-modal / cross-lingual pairs in unit space. Prediction: ASR training raises `speech_ga ↔ text_ga`; ablations 3 and 4 build it, 1 and 2 do not.

---

## 6. Reporting contract

- **Plain English. No jargon.** Define any term you introduce.
- **Lead with status in one line, then the number.** Detail only if asked.
- **Do not volunteer interpretations.** Report measurements. Verify before mentioning.
- **No inner monologue.** Conclusions and numbers, not reasoning traces.
- Append to your `*_PROGRESS.md` at every step, blocker, or number — do not wait until the end.
- Smoke tests are your call. Full fan-outs need approval.
- Nothing uploaded externally without explicit approval.

---

## 7. Current state

**Ready**
- `audio/units_9k` — 21,885 files, 9,302.9 h, 1.674 B raw units. Verified 99.9996% vs CPU tokenizer, 0 bad dtype/range/empty.
- `audio/mhubert_codebook.npz`, `mhubert_quantizer.faissindex`
- `ablations/text/tokens_full/shards` — 2.147 B Omni-tokenized text tokens
- `audio/discrete_asr_250h` — 250 h paired ASR
- 250h ASR checkpoint: `train/output/2026-07-29_00-57-42_20368745/checkpoints/step_023863`

**Does not exist yet**
- packed unit `.bin` for the 9K corpus (A, first task)
- text+speech mixed loader (A)
- unit-space MEXA extraction (C)

**Known constraint**
- `data.py:832` rejects `micro_batch_size != 1` for `unit_asr_examples`.

**Do not change**
- Unit assignment uses faiss HNSW (approximate). An exact argmin is "more correct" but produces different unit IDs than every existing artefact. Changing it means re-tokenizing everything.
