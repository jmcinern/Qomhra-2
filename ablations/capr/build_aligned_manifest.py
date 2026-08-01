"""Build the conversations_aligned corpus manifest on LUMI.

Joins conversations_capr.jsonl records to audio that now lives on LUMI:
  1. exact mirror: /scratch/.../audio/conversations/<original setanta abspath>
  2. train_data records: remapped via traindata_to_setanta.tsv (the deleted
     /media/storage_ssd/liam/train_data staging dir -> unlabelled corpus path),
     found either in the conversations mirror or in unlabelled_full by basename.
Records whose audio can't be found are dropped and counted.

Output: aligned.jsonl (one line per record: audio relpath under --audio-root,
duration_s, clean text) + meta.json coverage report.

Usage (LUMI):
  python3 build_aligned_manifest.py \
    --jsonl /scratch/project_465002364/audio/conversations_capr.jsonl \
    --mapping /scratch/project_465002364/audio/traindata_to_setanta.tsv \
    --audio-root /scratch/project_465002364/audio \
    --out-dir /scratch/project_465002364/audio/conversations
"""
import argparse
import json
import os
import re
import unicodedata


def base_key(name):
    name = unicodedata.normalize("NFC", name)
    return re.sub(r"\.(wav|flac|mp3|opus|m4a)$", "", name, flags=re.I)


def index_tree(root):
    """basename-key -> relpath (first wins) for every audio file under root."""
    idx = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not re.search(r"\.(wav|flac|mp3|opus|m4a)$", f, flags=re.I):
                continue
            idx.setdefault(base_key(f), os.path.relpath(os.path.join(dirpath, f), root))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--audio-root", required=True,
                    help="dir containing conversations/ and unlabelled_full/")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    mirror = os.path.join(args.audio_root, "conversations")

    # train_data staging path -> setanta unlabelled path (extension may differ on disk)
    remap = {}
    with open(args.mapping, encoding="utf-8") as f:
        for line in f:
            old, new = line.rstrip("\n").split("\t")
            remap[old] = new

    print("indexing unlabelled_full ...", flush=True)
    unlab = index_tree(os.path.join(args.audio_root, "unlabelled_full"))
    print(f"  {len(unlab)} files", flush=True)
    print("indexing conversations mirror ...", flush=True)
    mir = index_tree(mirror)
    print(f"  {len(mir)} files", flush=True)

    n = kept = 0
    hours = words = 0.0
    dropped = {"no_mapping": 0, "not_found": 0, "empty_text": 0}
    out_path = os.path.join(args.out_dir, "aligned.jsonl")
    with open(args.jsonl, encoding="utf-8") as f, \
         open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            n += 1
            rec = json.loads(line)
            text = " ".join(t for t in rec["conversation_capr"] if t).strip()
            if not text:
                dropped["empty_text"] += 1
                continue
            src = rec["audio_filepath"]
            direct = src.lstrip("/")
            key = base_key(os.path.basename(remap.get(src, src)))
            if os.path.exists(os.path.join(mirror, direct)):
                audio = os.path.join("conversations", direct)
            elif key in mir:
                audio = os.path.join("conversations", mir[key])
            elif key in unlab:
                audio = os.path.join("unlabelled_full", unlab[key])
            else:
                dropped["not_found" if src in remap else "no_mapping"] += 1
                continue
            kept += 1
            hours += rec["duration"] / 3600.0
            words += len(text.split())
            out.write(json.dumps({
                "audio": audio,
                "duration_s": rec["duration"],
                "text": text,
            }, ensure_ascii=False) + "\n")

    meta = {
        "records_in": n,
        "records_kept": kept,
        "dropped": dropped,
        "hours": round(hours, 1),
        "words": int(words),
        "audio_root": args.audio_root,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
