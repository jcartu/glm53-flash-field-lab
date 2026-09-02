#!/usr/bin/env python3
"""Prefix-cache persistence probe for the D-Rock LMCache validation.

Measures TTFT (time to first streamed token) for a large synthetic prompt:
  phase cold    - first-ever send (full prefill)
  phase warm    - immediate resend (L1/L2 hit if caching works)
  phase restart - resend after full container restart (L2 disk hit = LMCache win;
                  full cost again on the no-cache control arm)

Usage: prefix_test.py <port> <label> <out.json> [--tokens N]
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
    "willow xenon yonder zephyr anchor bishop cipher dredger"
).split()


def make_doc(seed: int, n_words: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def ttft_ms(port: int, prompt: str, max_tokens: int = 8) -> float:
    t0 = time.perf_counter()
    first = None
    with httpx.Client(timeout=600.0) as c:
        with c.stream(
            "POST",
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": os.environ.get("TEST_MODEL", "GLM-5.3"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": True,
            },
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line.startswith("data:") and '"content"' in line:
                    first = time.perf_counter()
                    break
    if first is None:
        raise RuntimeError("no streamed token observed")
    return (first - t0) * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("label")
    ap.add_argument("out")
    ap.add_argument("--tokens", type=int, default=24000)
    ap.add_argument("--skip-restart", action="store_true",
                    help="restart phase handled externally (container already bounced)")
    args = ap.parse_args()

    doc = make_doc(seed=42, n_words=args.tokens)
    prompt = f"Document:\n{doc}\n\nThe first word of this document is"

    res = {"label": args.label, "tokens_approx": args.tokens, "phases": {}}
    res["phases"]["cold"] = ttft_ms(args.port, prompt)
    res["phases"]["warm"] = ttft_ms(args.port, prompt)

    if not args.skip_restart:
        import subprocess
        subprocess.run(
            ["docker", "restart", os.environ.get("TEST_CONTAINER", "lmcache-glm53-5002")],
            check=True, capture_output=True, timeout=900,
        )
        for _ in range(90):
            try:
                if httpx.get(f"http://localhost:{args.port}/health", timeout=5).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(10)
        res["phases"]["restart"] = ttft_ms(args.port, prompt)

    res["ratios"] = {
        "warm_over_cold": round(res["phases"]["warm"] / res["phases"]["cold"], 3),
    }
    if "restart" in res["phases"]:
        res["ratios"]["restart_over_cold"] = round(res["phases"]["restart"] / res["phases"]["cold"], 3)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
