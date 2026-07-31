| model | asr_ga | asr_en | text_ga2en | text_en2ga | st_ga2en | st_en2ga |
|---|---:|---:|---:|---:|---:|---:|
| base · chat (own contract) | 219.03 | 3.77 | 27.41 | 15.61 | 16.21 | 12.63 |
| base_raw · matched prompt | 130.38 | 20.42 | 25.26 | 13.11 | 14.52 | 10.44 |
| AB1 · text | 131.68 | 111.05 | 31.50 | 5.23 | 4.78 | 1.89 |
| AB2 · speech | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AB3 · mixed | 100.00 | 100.00 | 33.32 | 12.15 | 0.02 | 1.08 |
| AB4 · aligned | 68.22 | 119.25 | 46.89 | 13.49 | 6.77 | 3.86 |

WER for asr_*, lower is better; chrF++ elsewhere, higher is better.

base_raw generation diagnostics (n, natural EOS, echoed sentinels):

| condition | n | natural EOS | sentinel echo |
|---|---:|---:|---:|
| asr_ga | 34 | 25 | 5 |
| asr_en | 34 | 30 | 12 |
| text_ga2en | 34 | 0 | 0 |
| text_en2ga | 34 | 0 | 0 |
| st_ga2en | 34 | 4 | 0 |
| st_en2ga | 34 | 5 | 0 |
