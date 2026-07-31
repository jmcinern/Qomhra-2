# Ablations

## Summary

- We want to compare the use of different data modality mixtures for Irish Speech LLM pretraining.


## Base Model

Small enough for efficient GPU use and quick experimentation but enough for meaningful results.

- Omni (Text + Speech to Text + Speech): https://huggingface.co/Qwen/Qwen2.5-Omni-3B 

## Data

Speech: ~10K hours of audio from Irish language media: Destination: /scratch/project_465002364/audio/unlabelled_full/ (not sure if all ~1.1T transferred)
Text: ~2B Tokens of web scraped + curated Irish text (~2B mHubert tokens): project_465002364/Denorm/train/data/
ASR Transcripts: ASR transcripts of the speech data (200M text tokens). On LUMI: project_465002364/Denorm/train/data/conversations_ga.parquet

## Ablations

Need to consider what components of the Omni model we need to train: Audio Encoder -> Adapter -> LLM -> Decoder 

1. Just Text

Data: Text + ASR transcripts
Components: Freeze all but LLM

2. Just Speech

Data: Speech
Components: Train  all

3. Text + Speech 

Data: Text + ASR Transcripts + Speech 
Components: Train all

## Workflow

- Access LUMI HPC via "ssh lumi"
- Access Setanta server via "ssh setanta"
- Version control with git and agent forwarding "-A"
- For git be mindful of other agents on same project working in parallel
- File transfers with scp
- Only user edits README to have a mental model of how progress is going and to maintain control
- For config testing on LUMI can use an interactive node
- For LUMI:
-- Follow containr + squashfs env setup to prevent transferring lots of small files
-- Container: /appl/local/laifs/containers/lumi-multitorch-latest.sif
-- Ensure optimal CPU binds + correct scratch bind from example script
-- Ensure usage of rccl highspeed interconnect
-- Refer to D:\VS-code-projects\Qomhra-2\full-train-est\train\train.sh for the most recent Qwen8B (not multimodal) training config.
-- Refer to D:/VS-code-projects/LUMI-AI-Guide for quick best practices for AI training on LUMI: storage, env etc...
-- Refer to D:/VS-code-projects/lumi-userguide/docs for more comprehensive docs
-- Have a look at D:\VS-code-projects\Qomhra-2\full-train-est for efficinet training + tokenization info 

## Jobs

### Prepare ASR Data

1. Convert to clean text using capitalisation + punctuation restoration model
2. Tokenize using Qwen Omni
3. Count Tokens: X

### Prepare Text Data

1. Tokenize using Qwen Omni
2. Sample proportionally to get X tokens

## Prepare Speech Data

1. Use Audio Encoder + Adapter + LLM to get tokenize
2. Sample X tokens proportionally

