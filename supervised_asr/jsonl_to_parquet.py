"""Convert the setanta supervised-ASR jsonl manifest to parquet.

Input rows: {"audio_filepath": str, "duration": float, "text": str}
"""
import pandas as pd

SRC = "/scratch/project_465002364/Denorm/train/data/staging/train_cleaned_without_tg4_remove_100dups_labelfiltered.jsonl"
DST = "/scratch/project_465002364/Denorm/train/data/supervised_ASR.parquet"

df = pd.read_json(SRC, lines=True)
assert list(df.columns) == ["audio_filepath", "duration", "text"], df.columns
df.to_parquet(DST, index=False)

print(f"rows={len(df)}")
print(df.dtypes)
print(df.head(2).to_string())
print(f"total_hours={df.duration.sum() / 3600:.1f}")
