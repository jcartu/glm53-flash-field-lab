#!/usr/bin/env python3
"""Eviction proofing harness for the D-Rock LMCache validation.

Runs on the LMCache arm with a deliberately tiny L2 (L2_GB=2) so that a few
hundred K tokens of distinct prefixes overflow both L1 (GPU) and L2 (disk),
forcing store-eviction and restore-from-disk paths - exactly the code D-Rock's
commits touch (fail-closed writeback eviction, durable L2 stores, prefetch
eviction protection, writer-owned inode publishing).

Protocol:
  pass 1 (fill):     N distinct docs, record greedy answers + TTFT
  pass 2 (churn):    N filler docs (fresh keys) to evict pass-1 keys from L1/L2,
                     interleaved with re-asks of pass-1 docs in reverse order
  verdict:           pass-2 answers must match pass-1 token-for-token (temp 0);
                     any mismatch is cache corruption, TTFT deltas show hits

Usage: eviction_test.py <port> <out.json> [--docs N] [--words W]
"""
import argparse
import os
import json
import random
import time

import httpx

WORDS = (
    "atlas binder cobalt delta ember flint granite harbor ivory jasper kilo "
    "lunar mantle nectar opal pivot quartz raster sulfur tundra umber vector "
    "willow xenon yonder zephyr anchor bishop cipher dredger falcon gypsum"
).split()


def make_doc(seed: int, n_words: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def ask(port: int, seed: int, n_words: int, k: int, temp: float = 0.0):
    doc = make_doc(seed, n_words)
    words = doc.split()
    answer = words[k]
    prompt = f"Document:\n{doc}\n\nQuestion: what is word number {k + 1} of the document? Reply with that single word only."
    t0 = time.perf_counter()
    with httpx.Client(timeout=900.0) as c:
        r = c.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": os.environ.get("TEST_MODEL", "GLM-5.3"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": temp,
            },
        )
        r.raise_for_status()
        body = r.json()
    content = (body["choices"][0]["message"].get("content") or "").strip()
    return {
        "seed": seed,
        "k": k,
        "answer": answer,
        "reply": content,
        "contains": answer.lower() in content.lower(),
        "ttft_proxy_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "usage": body.get("usage", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("out")
    ap.add_argument("--docs", type=int, default=40)
    ap.add_argument("--words", type=int, default=12000)
    args = ap.parse_args()

    rng = random.Random(20260831)
    base_seed = 10_000

    print(f"[evict] pass 1: fill with {args.docs} docs "
          f"(~{args.docs * args.words} tokens of distinct prefixes)")
    pass1 = []
    for i in range(args.docs):
        k = rng.randrange(1, args.words - 1)
        rec = ask(args.port, base_seed + i, args.words, k)
        rec["idx"] = i
        pass1.append(rec)
        print(f"  doc {i:02d} ttft~{rec['ttft_proxy_ms']:.0f}ms ok={rec['contains']}")

    print("[evict] pass 2: churn with fresh filler docs + re-verify in reverse")
    pass2 = {}
    mismatches = []
    for i in reversed(range(args.docs)):
        # fresh filler to keep evicting old keys during the verification sweep
        ask(args.port, base_seed + 100_000 + i, args.words, 5)
        orig = pass1[i]
        rec = ask(args.port, base_seed + i, args.words, orig["k"])
        rec["idx"] = i
        pass2[i] = rec
        same = rec["reply"] == orig["reply"]
        if not same:
            mismatches.append({
                "idx": i, "seed": orig["seed"], "k": orig["k"],
                "expected_word": orig["answer"],
                "pass1_reply": orig["reply"][:200],
                "pass2_reply": rec["reply"][:200],
                "ground_truth_ok_p1": orig["contains"],
                "ground_truth_ok_p2": rec["contains"],
            })
        print(f"  doc {i:02d} identical={same} word_ok={rec['contains']} "
              f"ttft~{rec['ttft_proxy_ms']:.0f}ms (p1 {orig['ttft_proxy_ms']:.0f}ms)")

    ratios = [
        round(pass2[i]["ttft_proxy_ms"] / max(pass1[i]["ttft_proxy_ms"], 1), 3)
        for i in range(args.docs)
    ]
    report = {
        "docs": args.docs,
        "words_per_doc": args.words,
        "identical_reply_count": args.docs - len(mismatches),
        "mismatch_count": len(mismatches),
        "ground_truth_word_correct_p1": sum(r["contains"] for r in pass1),
        "ground_truth_word_correct_p2": sum(r["contains"] for r in pass2.values()),
        "ttft_ratio_p2_over_p1": {
            "median": sorted(ratios)[len(ratios) // 2],
            "min": min(ratios),
            "max": max(ratios),
        },
        "mismatches": mismatches[:20],
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "mismatches"}, indent=2))
    if mismatches:
        print(f"[evict] CORRUPTION: {len(mismatches)} mismatched replies")
    else:
        print("[evict] all pass-2 replies identical to pass-1")


if __name__ == "__main__":
    main()
