"""Convert the setanta supervised-ASR jsonl manifest to parquet.

Input rows: {"audio_filepath": str, "duration": float, "text": str}
"""
import pandas as pd

SRC = "/scratch/project_465002364/Denorm/train/data/staging/train_cleaned_without_tg4_remove_100dups_labelfiltered.jsonl"
DST = "/scratch/project_465002364/Denorm/train/data/supervised_ASR.parquet"

AUDIO_DIR = "/scratch/project_465002364/audio/supervised_asr"

df = pd.read_json(SRC, lines=True)
assert list(df.columns) == ["audio_filepath", "duration", "text"], df.columns
# source jsonl holds setanta paths; audio now lives flat in AUDIO_DIR on LUMI
df["audio_filepath"] = AUDIO_DIR + "/" + df["audio_filepath"].str.rsplit("/", n=1).str[-1]
df.to_parquet(DST, index=False)

print(f"rows={len(df)}")
print(df.dtypes)
print(df.head(2).to_string())
print(f"total_hours={df.duration.sum() / 3600:.1f}")
