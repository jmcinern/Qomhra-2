# Pre-ablation evaluation tables

Raw model outputs are not repaired. `cap` columns report the percentage of examples that reached `max_new_tokens`; these cases require caution because some otherwise valid answers are visibly truncated.

## 1. MEXA

Centered mutual top-10 retrieval; chance is 0.0833. Values below are the best decoder-layer score at the final checkpoint, with that layer in parentheses. The current centered-MEXA@10 run contains the two monolingual speech–text pairs; the four cross-lingual pairs have not been scored with this metric yet.

### Pooled unit states

| Model | GA speech ↔ GA text | EN speech ↔ EN text |
|---|---:|---:|
| Base | 0.090 (L35) | 0.140 (L1) |
| AB1 · text | 0.100 (L35) | 0.120 (L17) |
| AB2 · speech | 0.100 (L13) | 0.140 (L17) |
| AB3 · mixed | 0.100 (L22) | 0.190 (L21) |
| AB4 · aligned | 0.260 (L22) | 0.190 (L2) |

### ASR-boundary state

| Model | GA speech ↔ GA text | EN speech ↔ EN text |
|---|---:|---:|
| Base | 0.100 (L0) | 0.100 (L0) |
| AB1 · text | 0.100 (L0) | 0.100 (L0) |
| AB2 · speech | 0.110 (L7) | 0.120 (L12) |
| AB3 · mixed | 0.140 (L10) | 0.190 (L17) |
| AB4 · aligned | 0.180 (L34) | 0.140 (L33) |

## 2. FLEURS

| Model | GA ASR WER↓ | EN ASR WER↓ | GA txt→EN chrF++↑ | EN txt→GA chrF++↑ | GA sp→EN chrF++↑ | EN sp→GA chrF++↑ |
|---|---:|---:|---:|---:|---:|---:|
| Base | 122.6 | 3.8 | 27.1 | 17.0 | 15.7 | 14.0 |
| AB1 · text | 102.7 | 101.2 | 31.2 | 5.2 | 4.3 | 2.0 |
| AB2 · speech | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| AB3 · mixed | 100.0 | 100.0 | 32.6 | 12.1 | 0.0 | 1.1 |
| AB4 · aligned | 68.2 | 99.0 | 46.6 | 13.7 | 6.1 | 4.3 |

### FLEURS AB1 checkpoint trajectory

| Training % | GA ASR WER↓ | EN ASR WER↓ | GA txt→EN chrF++↑ | EN txt→GA chrF++↑ | GA sp→EN chrF++↑ | EN sp→GA chrF++↑ |
|---|---:|---:|---:|---:|---:|---:|
| 10 | 103.7 | 101.2 | 44.2 | 22.5 | 2.4 | 1.2 |
| 20 | 100.9 | 101.3 | 47.5 | 17.8 | 2.0 | 1.3 |
| 30 | 106.4 | 101.8 | 39.6 | 2.8 | 3.3 | 2.5 |
| 40 | 115.6 | 101.4 | 28.7 | 3.1 | 5.9 | 1.9 |
| 50 | 104.5 | 101.4 | 27.8 | 8.4 | 2.5 | 2.0 |
| 60 | 109.0 | 102.9 | 25.9 | 2.9 | 4.6 | 2.4 |
| 70 | 112.9 | 101.0 | 36.6 | 3.9 | 5.3 | 2.5 |
| 80 | 105.8 | 100.9 | 30.5 | 4.3 | 3.5 | 1.4 |
| 90 | 117.0 | 100.5 | 30.6 | 6.5 | 3.8 | 2.2 |
| 100 | 102.7 | 101.2 | 31.2 | 5.2 | 4.3 | 2.0 |

### FLEURS max-token cap rate (%)

| Model | GA ASR | EN ASR | GA txt→EN | EN txt→GA | GA sp→EN | EN sp→GA |
|---|---:|---:|---:|---:|---:|---:|
| Base | 82.4 | 2.9 | 38.2 | 58.8 | 70.6 | 64.7 |
| AB1 · text | 94.1 | 97.1 | 47.1 | 0.0 | 79.4 | 58.8 |
| AB2 · speech | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| AB3 · mixed | 100.0 | 100.0 | 35.3 | 5.9 | 100.0 | 29.4 |
| AB4 · aligned | 0.0 | 100.0 | 35.3 | 14.7 | 97.1 | 82.4 |

## 3. IWSLT

| Model | Translation outputs % | Transcription outputs % | Other | Empty | Translation chrF++ | Routed chrF++ | Cap % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 98.2 | 0.9 | 1 | 0 | 12.6 | 11.5 | 82.1 |
| AB1 · text | 66.1 | 1.8 | 16 | 20 | 5.9 | 6.9 | 50.9 |
| AB2 · speech | 0.0 | 0.0 | 0 | 112 | 0.0 | — | 100.0 |
| AB3 · mixed | 0.0 | 0.0 | 0 | 112 | 0.0 | — | 100.0 |
| AB4 · aligned | 37.5 | 45.5 | 19 | 0 | 7.0 | 17.4 | 84.8 |

## 4. CLUAS

| Model | Transcript | Blind | Audio | Listening gain | Caps: transcript/blind/audio |
|---|---:|---:|---:|---:|---:|
| Base | 15/74 (20.3%) | 0/74 (0.0%) | 0/74 (0.0%) | +0 | 66.7/78.8/84.8 |
| AB1 · text | 45/74 (60.8%) | 3/74 (4.1%) | 2/74 (2.7%) | -1 | 57.6/51.5/72.7 |
| AB2 · speech | 0/74 (0.0%) | 0/74 (0.0%) | 0/74 (0.0%) | +0 | 100.0/100.0/100.0 |
| AB3 · mixed | 55/74 (74.3%) | 1/74 (1.4%) | 0/74 (0.0%) | -1 | 51.5/72.7/54.5 |
| AB4 · aligned | 53/74 (71.6%) | 3/74 (4.1%) | 11/74 (14.9%) | +8 | 51.5/72.7/93.9 |
