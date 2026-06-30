# Estimating the GPU hour time and total time for a full training run

## Summary 
Context: This is part of a project to train an Irish LLM and Speech LLM. We want to combine speech and text data for LLM and Speech LLM training.
Goal: We want to estimate the total time it would take on the LUMI HPC to train on the full text + audio corpus.
Task: (1) Transfer all data to LUMI (2) Pick representative LLM at sensible parameter size  (3) Tokenize representative subset of  data on LUMI (-C for heavy CPU, -G for heavy GPU) (4) Set up training on LUMI, multi GPU + multinode 
(5) Measure the tkn/sec (5) Quick profiling to see if any low-hanging fruit for optimisation.

## Model
Model selected for training time estimation purposes only. Will run eval on prospective candidates in the future. Can scale based on params + ave. words to token ratio.

- Qwen3-8B: https://huggingface.co/Qwen/Qwen3-8B 

## Tokenizer

Text:
-  Qwen3-8B


Speech: 
- https://huggingface.co/utter-project/mHuBERT-147
-- No refitting to Irish data, this is a quick worst case estimate
-- Don't dedup repeated tokens, this is for SLM training later
--  Fixed rate tokenization frequency of 50Hz so 180K tokens per hour
## Training Data
- Text: on LUMI at /scratch/project_465002364/Denorm/train/data in multiple parquet files by source
- Audio:  on LUMI at /scratch/project_465002364/audio/unlabelled_subset_10h/

### Extrapolate With Subsets
- Problem: Transferring the whole audio corpus will take a while and so will tokenization
- Solution:
-- Audio: Use a representative subset and calculate the hours/token ratio
-- Text: Use a representative subset and calculate the words/token ratio

Extrapolate: Total tokens = (total-hrs * tkn/hr) + (total-words * tkn/word)
-- Text: 890M words * 2.37 tkn/word = ~2.13B tokens (Qwen3-8B)
----- Text Tokenized: /scratch/project_465002364/Qomhra-2/full-train-est/tokens (eod sep.)
-------- Subset: /scratch/project_465002364/Qomhra-2/full-train-est/tokens/throughput_100k/sample.bin (114,207 words,  267,824 tokens)

---- Audio Tokenized: /scratch/project_465002364/Qomhra-2/full-train-est/mhubert/units/
## Workflow
- Access LUMI HPC via "ssh lumi"
- Version control with git and agent forwarding
- For git be mindful of other agents on same project working in parallel
- File transfers with scp
- User edits README so needs to have a mental model of how progress is going to maintain control
- For config testing on LUMI can use an interactive node
- For LUMI: 
-- Follow containr + squashfs env setup to prevent transferring lots of small files
-- Use /scratch/project_465002364/Qomhra/Qomhra_v2.sif as container
-- Ensure optimal CPU binds + correct scratch bind from example script
-- Ensure usage of rccl highspeed interconnect
-- Refer to D:/VS-code-projects/LUMI-AI-Guide for quick best practices
-- Refer to D:/VS-code-projects/lumi-userguide/docs for more comprehensive docs



# Agents
DONE: - Agent1 Transfer Audio: Transfer a subset of the audio: 10 hours from setanta -> LUMI, can reach directly with agent forwarding I think
DONE: - Agent2 Text tokenization set up: Take a subset of text corpus proportional to file sizes. Tokenize with Qwen tokenizer. Count words / token ratio.


DONE: - Agent3 Audio Tokenizer: set up the mHubert audio encoder, no refitting, no dedup
IN: PROG -  Agent4 - Training Setup: dependant on 2 and 3 and: set up simple LLM pretraining on audio + text tokens and time it. Also run pytorch profiler to save json file that I can look at in perfetto later. Train on one node across 8 GPUs. Example previous script of this for t5 training: D:\VS-code-projects\Denorm\train\nanoT5\train_denorm.sh
DONE  - Agent5 - Consider extrapolation for estimation: making sure that the formula makes sense, if so, sampling a small subset of text tokens (100K words) for testing throughput pruposes. 
DONE: - Agent 6: LUMI mirrors local — Qomhra and Qomhra-2 are separate repos. Qomhra-2 lives at /scratch/project_465002364/Qomhra-2 (this repo); the container/envs are still shared from /scratch/project_465002364/Qomhra.
