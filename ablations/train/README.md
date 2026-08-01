# Qomhra whole-corpus ablations

Continual pretraining of the Qwen2.5-Omni-3B Thinker on Irish text, unlabelled
speech, and aligned ASR supervision. Real runs are finite **one-epoch** corpus
passes; their step counts are derived at startup rather than configured by hand.

## What a loader does

A dataset defines examples; a PyTorch `DataLoader` batches them and overlaps CPU
preprocessing with GPU training.

- **Text:** the Omni-tokenized corpus is shuffled and packed into 4,096-token
  windows.
- **Speech:** `manifest.tsv` durations define <=30 s chunks. WAV loading and mel
  extraction run in CPU workers while the GPU trains the previous batch.
- **Aligned ASR:** rows come from
  `/scratch/project_465002364/Denorm/train/data/supervised_ASR.parquet`; audio paths
  are already absolute. A stable 2.5% hash split is withheld from training.

Each dataset is shuffled deterministically, split across global ranks exactly once,
and padded only enough to give every rank complete optimizer steps. Loaders are not
passed through Accelerate's data preparation because they already shard themselves;
doing both would silently shard the corpus twice.

## The four runs

**The original speech runs are superseded.** `omni_speech`, `omni_both` and
`omni_aligned` trained speech as continuous audio-encoder frames with a next-frame
regression objective; it did not learn, and those three configs have moved to
`qomhra/configs/legacy/`. See `legacy/README.md` for what replaced each and why.
The rerun treats speech as discrete mHuBERT unit tokens — the `omni_unit_*`
configs. Its ablation set is defined in `QOMHRA-2-RERUN-DISCRETE-SPEECH.md` at the
repo root, not here.

```bash
cd /scratch/project_465002364/Qomhra-2/ablations/train

sbatch --nodes=2 train.sh --config-name omni_text
```

Text still consumes one complete selected corpus epoch.

## Smoke and scaling probes

`optim.max_steps` is null for real runs. Set it only for a deliberate probe:

```bash
sbatch --nodes=1 train.sh --config-name omni_unit_ntp_100h \
  optim.max_steps=40 checkpoint.save_final=false checkpoint.at_fractions=[]
sbatch --nodes=2 train.sh --config-name omni_unit_ntp_100h \
  optim.max_steps=40 checkpoint.save_final=false checkpoint.at_fractions=[]
```

Compare `train/seconds_per_step` and `train/units_per_second`. Two nodes should
approximately double global units/s without a large step-time regression.

## FSDP topology

The multi-node default is `HYBRID_SHARD`: parameters are sharded across the eight
GCDs within each node and that shard group is replicated across nodes. Gradients are
reduced in bf16, first within the node and then across replica groups over Slingshot.
At startup the rig logs and validates:

```text
[fsdp] topology: strategy=hybrid_shard, world=16, local=8,
                 shard_group=8, replica_group=2
```

Training aborts if a two-node run does not resolve to that topology.

## Checkpoints and evaluation

Full-state checkpoint gathers are expensive, so `base.yaml` defaults to saving at
50%, 90% and final, keeping the last three.

**The default is not what the runs actually did.** Every real run so far passed its
own `checkpoint.at_fractions` and `keep_last` on the sbatch line, and those
overrides win. The superseded `both` run
(`output/2026-07-18_10-05-47_19984354`) was launched with
`at_fractions=[0.1..0.9] keep_last=12` and holds **ten** checkpoints, one per 10%.
The discrete ASR runs use `[0.2, 0.4, 0.6, 0.8]`.

To know what a given run saved, read its own `.hydra/overrides.yaml` — not this
file and not `base.yaml`.

Evaluation scripts under `../eval/` load these full state dicts on one GCD for
before/after comparisons.
