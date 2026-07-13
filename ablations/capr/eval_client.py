#!/usr/bin/env python3
"""Run a single-file C&PR pass through a marian-server websocket, one hyp line per
input line. Distilled from capr_run.py's translate_exact — batches lines, strips the
trailing blank, and recursively splits on any line-count mismatch so the output stays
exactly 1:1 with the input (which the eval scorer assumes).

Usage:
    python3 eval_client.py --input clilstore.src --output hyp.txt --url ws://127.0.0.1:10002/translate
"""
import argparse
import time

import websocket


def connect(url, timeout):
    return websocket.create_connection(url, timeout=timeout)


def translate_exact(ws_holder, url, timeout, texts):
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
            try:
                ws_holder[0].close()
            except Exception:
                pass
            time.sleep(2 * (attempt + 1))
            ws_holder[0] = connect(url, timeout)
    else:
        raise RuntimeError("marian-server unreachable after retries")

    out = result.split("\n")
    while len(out) > len(texts) and out and out[-1] == "":
        out.pop()

    if len(out) == len(texts):
        return out

    if len(texts) == 1:
        return [out[0] if out else texts[0]]
    mid = len(texts) // 2
    left = translate_exact(ws_holder, url, timeout, texts[:mid])
    right = translate_exact(ws_holder, url, timeout, texts[mid:])
    return left + right


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--url", default="ws://127.0.0.1:10002/translate")
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        # keep the exact line structure; a marian line can't contain a newline,
        # so flatten any stray internal whitespace but keep one entry per input line
        src = [ln.rstrip("\n").replace("\n", " ") for ln in f]

    ws_holder = [connect(args.url, args.timeout)]
    hyp = [None] * len(src)

    # only send non-empty lines; empties pass through as "" (marian would drop them)
    idxs = [i for i, t in enumerate(src) if t.strip()]
    for i, t in enumerate(src):
        if not t.strip():
            hyp[i] = ""

    t0 = time.time()
    for b in range(0, len(idxs), args.batch):
        chunk = idxs[b:b + args.batch]
        outs = translate_exact(ws_holder, args.url, args.timeout, [src[i] for i in chunk])
        for i, o in zip(chunk, outs):
            hyp[i] = o
        done = min(b + args.batch, len(idxs))
        print(f"[{done}/{len(idxs)} lines | {time.time()-t0:.1f}s]", flush=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for line in hyp:
            f.write((line if line is not None else "") + "\n")

    try:
        ws_holder[0].close()
    except Exception:
        pass
    print(f"[done] {len(src)} lines -> {args.output} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
