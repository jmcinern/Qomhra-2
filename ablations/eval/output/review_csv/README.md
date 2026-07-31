# Pre-evaluation review files

All CSV files use UTF-8 with a byte-order mark so that Irish text opens
correctly in spreadsheet software. Raw hypotheses are preserved; scoring and
diagnostic fields do not rewrite model output.

- `cluas_raw_outputs.csv`: one row per model, question, and condition. The
  official marking scheme, Sonnet marks/reasons, decoding stop, and raw token
  IDs are included.
- `cluas_numbers.csv`: compact CLUAS totals and generation diagnostics by model
  and condition. `listening_gain_marks` is `just_audio - no_context`.
- `iwslt_raw_outputs.csv`: one row per model output, with Irish transcription
  and English translation references, fastText language classification, and
  both chrF++ scores.
- `iwslt_numbers.csv`: compact IWSLT language-route counts and routed scores.
- `fleurs_raw_outputs.csv`: one row per model, direction, and example, with
  reference, raw hypothesis, stop information, and task metadata.
- `fleurs_numbers.csv`: compact FLEURS metrics and diagnostics by model. It is
  regenerated to add the AB1 checkpoint trajectory when the LUMI jobs finish.

Model names map as follows: `base` is the starting model; `ab1` is text-only,
`ab2` is speech-only, `ab3` is mixed speech and text, and `ab4` is aligned
speech-to-text.
