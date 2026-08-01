"""Convert conversations_capr.jsonl (raw+clean triples) into a LUMI-format
parquet: text-only, one row per record, matching conversations_ga.parquet's
schema (text: string, source: string).

Usage: py -3 to_parquet.py <in.jsonl> <out.parquet>
"""
import sys
import json
import pyarrow as pa
import pyarrow.parquet as pq

def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    texts = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(" ".join(rec["conversation_capr"]))

    table = pa.table({
        "text": texts,
        "source": ["conversations_capr"] * len(texts),
    })
    pq.write_table(table, out_path)
    print(f"wrote {len(texts)} rows to {out_path}")

if __name__ == "__main__":
    main()
