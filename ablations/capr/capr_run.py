#!/usr/bin/env python3
"""Capitalisation & Punctuation Restoration (C&PR) over the ASR conversations corpus.

Streams `conversations.jsonl` (one record = one diarised audio span, with a
`conversation` list of speaker-diarised turns), sends each turn through a running
Marian C&PR `marian-server` websocket, and writes an output JSONL that preserves the
full (audio_filepath, offset, duration, raw turns, clean turns) tuple so the corpus
can later be used for ASR training as well as tokenisation.

The server is expected to already be running (see run_capr.sh) with the tuned config:
    marian-server -c <decoder.yml> -p 10002 --devices 0 --mini-batch 96 --maxi-batch 1000 --beam-size 1

Design notes:
  * Turns are batched ACROSS records (default 2000 turns/request) to keep the GPU fed;
    Marian batches them internally via maxi-batch. Empty turns are passed through
    without being sent (Marian would drop them and break the 1:1 line mapping).
  * `translate_exact` guarantees exactly one output line per input line: if a batch
    ever comes back with a mismatched line count, it recursively splits and retries
    down to single turns, so record/turn alignment can never silently drift.
  * Output is written one record per line, in input order; on restart we skip the
    records already present (resume), after dropping any trailing partial/corrupt line.
"""
import argparse
import json
import sys
import time

import websocket


def connect(url, timeout):
    return websocket.create_connection(url, timeout=timeout)


def translate_exact(ws_holder, url, timeout, texts):
    """Return C&PR output for `texts`, guaranteeing len(out) == len(texts).

    Sends the batch in one websocket round-trip; if the returned line count doesn't
    match (should only ever be a trailing empty line, which we strip), recursively
    splits the batch and retries so alignment is never lost. Reconnects on socket error.
    """
    if not texts:
        return []
    payload = "\n".join(texts)
    for attempt in range(3):
        try:
            ws = ws_holder[0]
            ws.send(payload)
            result = ws.recv()
            break
        except Exception:
            # reconnect and retry
            try:
                ws_holder[0].close()
            except Exception:
                pass
            time.sleep(2 * (attempt + 1))
            ws_holder[0] = connect(url, timeout)
    else:
        raise RuntimeError("marian-server unreachable after retries")

    out = result.split("\n")
    # Marian appends a trailing empty line (final newline); strip trailing blanks.
    while len(out) > len(texts) and out and out[-1] == "":
        out.pop()

    if len(out) == len(texts):
        return out

    # Mismatch (rare): split and recurse so we never misalign turns.
    if len(texts) == 1:
        # Single turn can't be split further; return best effort (out[0] or original).
        return [out[0] if out else texts[0]]
    mid = len(texts) // 2
    left = translate_exact(ws_holder, url, timeout, texts[:mid])
    right = translate_exact(ws_holder, url, timeout, texts[mid:])
    return left + right


def count_resumable(path):
    """Count complete, JSON-valid lines already written; rewrite file without a
    trailing partial line. Returns number of records to skip."""
    try:
        f = open(path, "r", encoding="utf-8")
    except FileNotFoundError:
        return 0
    good = []
    with f:
        for line in f:
            if not line.endswith("\n"):
                break  # partial final line
            s = line.strip()
            if not s:
                continue
            try:
                json.loads(s)
            except json.JSONDecodeError:
                break
            good.append(line)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(good)
    return len(good)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="conversations.jsonl")
    ap.add_argument("--output", required=True, help="output conversations_capr.jsonl")
    ap.add_argument("--url", default="ws://127.0.0.1:10002/translate")
    ap.add_argument("--batch-turns", type=int, default=2000)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--log-every", type=int, default=5000, help="log every N records")
    args = ap.parse_args()

    skip = count_resumable(args.output)
    if skip:
        print(f"[resume] {skip} records already done; skipping them.", flush=True)

    ws_holder = [connect(args.url, args.timeout)]
    out_f = open(args.output, "a", encoding="utf-8")

    # Pending buffers for cross-record batching with in-order output.
    pending = {}          # gidx -> record dict (awaiting clean turns)
    expect = {}           # gidx -> number of non-empty turns still to fill
    clean = {}            # gidx -> {turn_idx: clean_text}
    next_write = skip     # next gidx to flush to disk (resume point)
    batch = []            # list of (gidx, turn_idx, text)

    t0 = time.time()
    words_done = 0
    records_done = 0

    def flush_batch():
        nonlocal words_done
        if not batch:
            return
        texts = [t for (_, _, t) in batch]
        outs = translate_exact(ws_holder, args.url, args.timeout, texts)
        for (li, ti, src), dst in zip(batch, outs):
            clean[li][ti] = dst
            expect[li] -= 1
            words_done += len(src.split())
        batch.clear()

    def write_completed():
        nonlocal next_write, records_done
        while next_write in pending and expect.get(next_write, 1) == 0:
            rec = pending.pop(next_write)
            cmap = clean.pop(next_write)
            expect.pop(next_write, None)
            conv = rec.get("conversation", [])
            rec["conversation_capr"] = [cmap.get(i, "") for i in range(len(conv))]
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records_done += 1
            next_write += 1
        out_f.flush()

    def maybe_log():
        if records_done and records_done % args.log_every == 0:
            dt = time.time() - t0
            wpm = words_done / dt * 60 if dt else 0
            print(f"[{records_done:>7,} recs | {words_done:>12,} words | "
                  f"{dt/60:6.1f} min | {wpm:9,.0f} w/min]", flush=True)

    with open(args.input, "r", encoding="utf-8") as fin:
        for gidx, line in enumerate(fin):
            if gidx < skip:
                continue
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            li = gidx  # global line index = stable in-order key
            conv = rec.get("conversation", [])
            pending[li] = rec
            clean[li] = {}
            # queue only non-empty turns; empties are passed through as ""
            n_nonempty = 0
            for ti, turn in enumerate(conv):
                txt = (turn or "").strip().replace("\n", " ")
                if txt:
                    batch.append((li, ti, txt))
                    n_nonempty += 1
                else:
                    clean[li][ti] = ""
            expect[li] = n_nonempty
            if n_nonempty == 0:
                # nothing to translate; will be written by write_completed
                pass

            if len(batch) >= args.batch_turns:
                flush_batch()
                write_completed()
                maybe_log()

    # final drain
    flush_batch()
    write_completed()
    # write any trailing records whose turns were all empty / already complete
    for li in sorted(pending):
        rec = pending[li]
        cmap = clean.get(li, {})
        conv = rec.get("conversation", [])
        rec["conversation_capr"] = [cmap.get(i, "") for i in range(len(conv))]
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        records_done += 1
    out_f.flush()
    out_f.close()
    try:
        ws_holder[0].close()
    except Exception:
        pass

    dt = time.time() - t0
    print(f"[done] {records_done:,} records, {words_done:,} words in {dt/60:.1f} min "
          f"({words_done/dt*60:,.0f} w/min)", flush=True)


if __name__ == "__main__":
    main()
