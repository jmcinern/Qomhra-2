#!/usr/bin/env python3
"""Dose-response test for emitting Qwen's ``<|im_end|>`` chat-turn token.

Uses parallel Irish/English prompt banks, disjoint demonstrations and held-out
questions, configurable shot counts, and a JSON record of every generation.

  python -m eval.fewshot_endtoken --checkpoints <ckpt> --labels text --out results.json
"""
import argparse
import glob
import json
import math
import os
import re
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SNAPSHOT_GLOB = ("/scratch/project_465002364/Qomhra-2/hf_cache/hub/"
                 "models--Qwen--Qwen2.5-Omni-3B/snapshots/*")

# Demonstrations and tests are disjoint. Answers are inherently short, so the token
# budget should not force a well-formed response to stop mid-answer.
SHOTS = {
    "ga": [
        ("Cad é príomhchathair na Fraince?", "Is é Páras príomhchathair na Fraince."),
        ("Cén dath atá ar an spéir ar lá glan?", "Tá an spéir gorm ar lá glan."),
        ("Cé mhéad lá atá i seachtain?", "Tá seacht lá i seachtain."),
        ("Cad é a dó móide a trí?", "Is ionann a dó móide a trí agus a cúig."),
        ("Cén t-ainmhí a deir míá?", "Deir cat míá."),
        ("Cén séasúr a thagann i ndiaidh an earraigh?", "Tagann an samhradh i ndiaidh an earraigh."),
        ("Cé mhéad uair atá i lá?", "Tá ceithre huaire is fiche i lá."),
        ("Cén pláinéad ar a mairimid?", "Mairimid ar an Domhan."),
        ("Cad a reoiteann ag náid céim Celsius?", "Reonn uisce ag náid céim Celsius."),
        ("Cén dath atá ar shneachta úr?", "Tá sneachta úr bán."),
        ("Cad é a mhalairt de 'fuar'?", "Is é 'te' a mhalairt de 'fuar'."),
        ("Cé mhéad taobh atá ar thriantán?", "Tá trí thaobh ar thriantán."),
    ],
    "en": [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What colour is the sky on a clear day?", "The sky is blue on a clear day."),
        ("How many days are in a week?", "There are seven days in a week."),
        ("What is two plus three?", "Two plus three is five."),
        ("Which animal says meow?", "A cat says meow."),
        ("Which season comes after spring?", "Summer comes after spring."),
        ("How many hours are in a day?", "There are twenty-four hours in a day."),
        ("Which planet do we live on?", "We live on Earth."),
        ("What freezes at zero degrees Celsius?", "Water freezes at zero degrees Celsius."),
        ("What colour is fresh snow?", "Fresh snow is white."),
        ("What is the opposite of 'cold'?", "The opposite of 'cold' is 'hot'."),
        ("How many sides does a triangle have?", "A triangle has three sides."),
    ],
}
TESTS = {
    "ga": [
        "Cad é príomhchathair na hÉireann?", "Cén dath atá ar an bhféar?",
        "Cé mhéad mí atá i mbliain?", "Cad iad na teangacha oifigiúla in Éirinn?",
        "Cad é a ceathair móide a cúig?", "Cé mhéad cos atá ar dhamhán alla?",
        "Cad é an t-aigéan is mó ar domhan?",
        "Cén pláinéad ar a dtugtar an Pláinéad Dearg?",
        "Cad é a mhalairt de 'lá'?",
        "Cén teocht ag a bhfiuchann uisce i gcéimeanna Celsius?",
        "Cé mhéad lá atá i mbliain bhisigh?",
        "Cén uirlis cheoil a bhfuil eochracha dubha agus bána uirthi?",
        "Cad a thugtar ar mhadra óg?", "Cén treo ina n-éiríonn an ghrian?",
        "Cén cruth a bhfuil ceithre thaobh chothroma air?",
        "Cé a scríobh Romeo and Juliet?", "Cén mhór-roinn ina bhfuil an Éigipt?",
        "Cén gás a theastaíonn ó dhaoine chun análú?",
        "Cé mhéad nóiméad atá in uair an chloig?", "Cad é an chéad mhí den bhliain?",
        "Cén t-ainmhí a thugann olann dúinn?", "Cad é an fhoirmle cheimiceach d'uisce?",
        "Cén uimhir a thagann i ndiaidh nócha a naoi?",
        "Cén dath a bhíonn ar bhanana aibí de ghnáth?",
        "Cén pláinéad is gaire don Ghrian?",
        "Cén séasúr a thagann i ndiaidh an tsamhraidh?", "Cad a dhéanann beacha?",
        "Cén t-éan mór nach féidir leis eitilt agus a chónaíonn san Antartaice?",
        "Cad é airgeadra na Seapáine?", "Cad é an t-ainmhí is mó ar domhan?",
        "Cé mhéad taobh atá ar pheinteagán?", "Cad é príomhchathair na Spáinne?",
    ],
    "en": [
        "What is the capital city of Ireland?", "What colour is grass?",
        "How many months are in a year?", "What are the official languages of Ireland?",
        "What is four plus five?", "How many legs does a spider have?",
        "What is the largest ocean on Earth?", "Which planet is called the Red Planet?",
        "What is the opposite of 'day'?",
        "At what temperature does water boil in degrees Celsius?",
        "How many days are in a leap year?",
        "Which musical instrument has black and white keys?",
        "What is a young dog called?", "In which direction does the sun rise?",
        "Which shape has four equal sides?", "Who wrote Romeo and Juliet?",
        "Which continent is Egypt in?", "Which gas do humans need to breathe?",
        "How many minutes are in an hour?", "What is the first month of the year?",
        "Which animal gives us wool?", "What is the chemical formula for water?",
        "Which number comes after ninety-nine?", "What colour is a ripe banana usually?",
        "Which planet is closest to the Sun?", "Which season comes after summer?",
        "What do bees make?", "Which large flightless bird lives in Antarctica?",
        "What is the currency of Japan?", "What is the largest animal on Earth?",
        "How many sides does a pentagon have?", "What is the capital of Spain?",
    ],
}


def build(proc, lang, question, n_shots, demo_stop_token="<|im_end|>",
          prompt_format="chat"):
    if prompt_format == "raw":
        if demo_stop_token != "<|endoftext|>":
            raise ValueError("raw prompts require <|endoftext|> demonstration stops")
        parts = [
            f"Q: {q}\nA: {a}<|endoftext|>\n"
            for q, a in SHOTS[lang][:n_shots]
        ]
        parts.append(f"Q: {question}\nA:")
        text = "".join(parts)
        if text.count("<|endoftext|>") != n_shots:
            raise RuntimeError("raw prompt did not render one separator per demonstration")
        if "<|im_start|>" in text or "<|im_end|>" in text:
            raise RuntimeError("raw prompt unexpectedly contains Qwen chat markers")
        return text
    if prompt_format != "chat":
        raise ValueError(f"unsupported prompt format: {prompt_format}")
    conv = []
    for q, a in SHOTS[lang][:n_shots]:
        conv.append({"role": "user", "content": [{"type": "text", "text": q}]})
        conv.append({"role": "assistant", "content": [{"type": "text", "text": a}]})
    conv.append({"role": "user", "content": [{"type": "text", "text": question}]})
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    if demo_stop_token == "<|endoftext|>":
        # Change only completed assistant demonstrations. System, demonstration-user,
        # and held-out-user turns remain exactly as rendered by Qwen's official template.
        assistant_turn = re.compile(
            r"(<\|im_start\|>assistant\n.*?)(<\|im_end\|>)", re.DOTALL
        )
        text, replacements = assistant_turn.subn(
            lambda match: match.group(1) + demo_stop_token, text
        )
        if replacements != n_shots:
            raise RuntimeError(
                f"replaced {replacements} assistant demonstration stops; "
                f"expected {n_shots}"
            )
    elif demo_stop_token != "<|im_end|>":
        raise ValueError(f"unsupported demonstration stop token: {demo_stop_token}")
    # The official template must close every user/assistant demonstration turn and
    # leave exactly one open assistant header for generation.  Fail rather than run a
    # subtly malformed "few-shot" comparison.
    expected_closed_turns = (
        2 * n_shots + 2 if demo_stop_token == "<|im_end|>" else n_shots + 2
    )
    if text.count("<|im_end|>") != expected_closed_turns:
        raise RuntimeError(
            f"chat template rendered {text.count('<|im_end|>')} <|im_end|> tokens; "
            f"expected {expected_closed_turns} for {n_shots} shots"
        )
    if text.count("<|im_start|>assistant") != n_shots + 1:
        raise RuntimeError("chat template did not render the expected assistant turns")
    expected_endoftext = n_shots if demo_stop_token == "<|endoftext|>" else 0
    if text.count("<|endoftext|>") != expected_endoftext:
        raise RuntimeError(
            f"rendered {text.count('<|endoftext|>')} <|endoftext|> tokens; "
            f"expected {expected_endoftext}"
        )
    if not text.rstrip().endswith("<|im_start|>assistant"):
        raise RuntimeError("chat template did not leave an open assistant generation turn")
    return text


@torch.no_grad()
def run(thinker, proc, device, shot_counts, batch_size=1, max_new_tokens=60,
        demo_stop_token="<|im_end|>", prompt_format="chat"):
    tok = proc.tokenizer
    stop_ids = {
        token: tok.convert_tokens_to_ids(token)
        for token in ("<|im_end|>", "<|endoftext|>")
    }
    for token, token_id in stop_ids.items():
        if tok.convert_ids_to_tokens(token_id) != token:
            raise RuntimeError(f"tokenizer did not resolve {token} exactly (id={token_id})")
    if tok.pad_token_id is None:
        raise RuntimeError("tokenizer has no pad token for batched generation")
    old_padding_side = tok.padding_side
    tok.padding_side = "left"  # decoder-only batched generation must left-pad
    out = {}
    try:
        for lang in ("ga", "en"):
            out[lang] = {}
            for n_shots in shot_counts:
                rows = []
                questions = TESTS[lang]
                for start in range(0, len(questions), batch_size):
                    batch_q = questions[start:start + batch_size]
                    prompts = [
                        build(proc, lang, q, n_shots, demo_stop_token, prompt_format)
                        for q in batch_q
                    ]
                    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
                    prompt_width = enc.input_ids.shape[1]
                    gen = thinker.generate(
                        **enc, max_new_tokens=max_new_tokens, do_sample=False,
                        eos_token_id=list(stop_ids.values()), pad_token_id=tok.pad_token_id,
                    )
                    generated = gen[:, prompt_width:]
                    for offset, (q, prompt, new) in enumerate(zip(batch_q, prompts, generated)):
                        hits = [
                            (int(pos[0]), token)
                            for token, token_id in stop_ids.items()
                            if (pos := (new == token_id).nonzero(as_tuple=False).flatten()).numel()
                        ]
                        first_hit = min(hits) if hits else None
                        termination_token = first_hit[1] if first_hit else None
                        terminated = termination_token is not None
                        n_tokens = first_hit[0] + 1 if first_hit else int(new.shape[0])
                        kept = new[:n_tokens]
                        rows.append({
                            "id": start + offset,
                            "q": q,
                            "prompt": prompt,
                            "text": tok.decode(kept, skip_special_tokens=True),
                            "terminated": terminated,
                            "termination_token": termination_token,
                            "chat_terminated": termination_token == "<|im_end|>",
                            "n_tokens": n_tokens,
                            "token_ids": kept.tolist(),
                        })
                out[lang][str(n_shots)] = rows
                counts = {
                    token: sum(row["termination_token"] == token for row in rows)
                    for token in (*stop_ids, None)
                }
                print(f"[eval] lang={lang} shots={n_shots} "
                      f"im_end={counts['<|im_end|>']} endoftext={counts['<|endoftext|>']} "
                      f"none={counts[None]}", flush=True)
    finally:
        tok.padding_side = old_padding_side
    return out


def wilson_interval(k, n, z=1.959963984540054):
    """Two-sided Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def save_json(path, payload):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--shot-counts", nargs="+", type=int, default=[0, 1, 3, 6, 9])
    ap.add_argument("--test-limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="must remain 1: Qwen Thinker batching corrupts generation")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--out", default=None)
    ap.add_argument("--demo-stop-token", choices=["im_end", "endoftext"],
                    default="im_end")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--prompt-format", choices=["chat", "raw"], default="chat")
    args = ap.parse_args()
    if args.baseline is None:
        args.baseline = glob.glob(SNAPSHOT_GLOB)[0]
    if any(n < 0 or n > min(len(v) for v in SHOTS.values()) for n in args.shot_counts):
        ap.error(f"shot counts must be between 0 and {min(len(v) for v in SHOTS.values())}")
    if args.batch_size != 1:
        ap.error("--batch-size must be 1; batched Qwen Thinker generation produced "
                 "corrupted repeated-token outputs in the control run")
    if args.test_limit is not None:
        if not 1 <= args.test_limit <= min(len(v) for v in TESTS.values()):
            ap.error(f"--test-limit must be between 1 and {min(len(v) for v in TESTS.values())}")
        for lang in TESTS:
            TESTS[lang] = TESTS[lang][:args.test_limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    demo_stop_token = {
        "im_end": "<|im_end|>", "endoftext": "<|endoftext|>"
    }[args.demo_stop_token]
    from transformers import Qwen2_5OmniProcessor
    proc = Qwen2_5OmniProcessor.from_pretrained(args.baseline)
    rendered = build(
        proc, "ga", TESTS["ga"][0], 3, demo_stop_token, args.prompt_format
    )
    print(f"[template] format={args.prompt_format}: OK | "
          f"im_end={rendered.count('<|im_end|>')} | "
          f"endoftext={rendered.count('<|endoftext|>')} | "
          f"assistant_headers={rendered.count('<|im_start|>assistant')}", flush=True)

    from generate_compare import load_baseline
    payload = {
        "config": {
            "shot_counts": args.shot_counts,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "n_tests_per_language": {lang: len(rows) for lang, rows in TESTS.items()},
            "baseline": args.baseline,
            "checkpoints": args.checkpoints,
            "labels": args.labels,
            "demo_stop_token": demo_stop_token,
            "prompt_format": args.prompt_format,
            "demonstrations": SHOTS,
        },
        "results": {},
    }
    if not args.skip_baseline:
        print("[baseline] loading...", flush=True)
        thinker = load_baseline(args.baseline, device, dtype)  # text only -> sdpa is safe
        payload["results"]["baseline"] = run(
            thinker, proc, device, args.shot_counts, args.batch_size,
            args.max_new_tokens, demo_stop_token, args.prompt_format,
        )
        save_json(args.out, payload)
        del thinker
        torch.cuda.empty_cache()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
    from qomhra.checkpoint import load_for_eval
    for i, ckpt in enumerate(args.checkpoints):
        label = args.labels[i] if i < len(args.labels) else os.path.basename(ckpt)
        print(f"[{label}] loading...", flush=True)
        model, _ = load_for_eval(ckpt, device=device, dtype=dtype)
        payload["results"][label] = run(
            model.thinker, proc, device, args.shot_counts,
            args.batch_size, args.max_new_tokens, demo_stop_token, args.prompt_format,
        )
        save_json(args.out, payload)
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 88)
    print("TERMINATION RATE (either stop token; count/n, percent, Wilson 95% CI)")
    print("=" * 88)
    for label, result in payload["results"].items():
        print(f"\n{label}")
        for lang in ("ga", "en"):
            for n_shots in args.shot_counts:
                rows = result[lang][str(n_shots)]
                k, n = sum(x["terminated"] for x in rows), len(rows)
                lo, hi = wilson_interval(k, n)
                im_end = sum(x["termination_token"] == "<|im_end|>" for x in rows)
                endoftext = sum(x["termination_token"] == "<|endoftext|>" for x in rows)
                print(f"  {lang} shots={n_shots:2d}: {k:2d}/{n:<2d} "
                      f"{100*k/n:5.1f}%  CI [{100*lo:4.1f}, {100*hi:4.1f}] "
                      f"(im_end={im_end}, endoftext={endoftext}, none={n-k})")
    print("=" * 88, flush=True)
    save_json(args.out, payload)


if __name__ == "__main__":
    main()
