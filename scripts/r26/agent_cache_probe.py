#!/usr/bin/env python3
"""Measure deterministic multi-turn agent prefix reuse on the two R26 arms."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, importlib, json, math
import random, re, statistics, threading, time, traceback
from typing import Any
SCHEMA = "r26-agent-cache/v1"
ARMS = ("stock-r26", "drock-overlay")
SESSIONS, TURNS, MAX_TOKENS = 8, 12, 256
HIT_RATE_OBSERVATION = 0.95
SYSTEM_PROMPT = ("You are a careful software agent. Preserve all prior tool evidence byte-for-byte, "
                 "reason at low effort, and state the next concrete action after each tool result.")
TOOLS = [{"type": "function", "function": {"name": "inspect_workspace", "description": "Return deterministic workspace evidence for the requested step.", "parameters": {"type": "object", "properties": {"step": {"type": "integer"}}, "required": ["step"]}}}]
METRICS = {
    "local_prefix_queries": "vllm:prefix_cache_queries_total", "local_prefix_hits": "vllm:prefix_cache_hits_total",
    "external_prefix_queries": "vllm:external_prefix_cache_queries_total", "external_prefix_hits": "vllm:external_prefix_cache_hits_total",
    "prompt_tokens": "vllm:prompt_tokens_total", "prompt_tokens_cached": "vllm:prompt_tokens_cached_total"}
SOURCE_METRIC = "vllm:prompt_tokens_by_source_total"
SOURCE_KEYS = tuple(f"prompt_source_{name}" for name in ("local_compute", "local_cache_hit", "external_kv_transfer"))
METRIC_SAMPLE = "/home/josh/omp-workspace/drock-lmcache/r26-battery/cache-official-r26-fp8-80k-config-post-restart.vllm.metrics.txt"
LABEL_RE = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')
def load_runtime() -> Any:
    try:
        return importlib.import_module(".runtime", package=__package__)
    except (ImportError, TypeError):
        return importlib.import_module("runtime")
def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()
def dist(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None}
    values = sorted(values)
    def pct(q: float) -> float:
        at = (len(values) - 1) * q
        low, high = math.floor(at), math.ceil(at)
        return values[low] + (values[high] - values[low]) * (at - low)
    return {"count": len(values), "min": values[0], "p50": pct(.5), "p95": pct(.95), "max": values[-1], "mean": statistics.fmean(values)}
def make_spec(seed: int) -> dict[str, Any]:
    rng, sessions = random.Random(seed), []
    for session in range(SESSIONS):
        turns = [{"turn": turn, "target_tool_output_tokens": rng.randint(200, 600), "evidence_id": f"{rng.getrandbits(80):020x}"} for turn in range(TURNS)]
        sessions.append({"session": session, "nonce": f"{rng.getrandbits(96):024x}", "cache_salt": f"r26-agent-cache-{digest(f'{seed}:{session}')[:24]}", "turns": turns})
    basis = {"seed": seed, "system_prompt": SYSTEM_PROMPT, "tools": TOOLS, "sessions": sessions}
    return {**basis, "specification_sha256": digest(canonical(basis))}
def body_evidence(request: bytes, response: bytes) -> dict[str, str]:
    return {"raw_request_body": request.decode(), "raw_request_body_sha256": digest(request), "raw_response_body": response.decode(errors="replace"), "raw_response_body_sha256": digest(response)}
def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    import requests
    body, session, started = canonical(payload).encode(), requests.Session(), time.perf_counter()
    session.trust_env = False
    try:
        response = session.post(base_url.rstrip("/") + path, data=body, headers={"content-type": "application/json"}, timeout=(20, timeout))
        raw = response.content
        try: parsed = response.json()
        except ValueError: parsed = None
        return {"status": response.status_code, "response_headers": dict(response.headers),
                **body_evidence(body, raw), "parsed": parsed, "elapsed_seconds": time.perf_counter() - started,
                "error": None if response.ok else f"HTTP {response.status_code}"}
    except Exception as error:
        return {"status": None, "response_headers": {}, **body_evidence(body, b""), "parsed": None,
                "elapsed_seconds": time.perf_counter() - started, "error": f"{type(error).__name__}: {error}"}
    finally:
        session.close()
def ids_from(record: dict[str, Any]) -> list[int]:
    parsed = record.get("parsed")
    ids = parsed.get("tokens") if isinstance(parsed, dict) else None
    if record.get("status") != 200 or not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
        raise RuntimeError(f"tokenization failed: {record.get('error') or record.get('raw_response_body')}")
    return ids
def calibrate_tool(rt: Any, item: dict[str, Any], timeout: float) -> tuple[str, list[dict[str, Any]]]:
    filler, attempts = max(1, item["target_tool_output_tokens"] - 8), []
    for _ in range(8):
        content = f"EVIDENCE {item['evidence_id']}\n" + " datum" * filler
        http = post_json(rt.BASE_URL, "/tokenize", {"model": rt.MODEL_NAME, "prompt": content}, timeout)
        observed = len(ids_from(http))
        attempts.append({"filler_words": filler, "observed_tokens": observed, "http": http})
        if observed == item["target_tool_output_tokens"]:
            return content, attempts
        filler += item["target_tool_output_tokens"] - observed
        if filler < 1: break
    raise RuntimeError(f"could not calibrate {item['evidence_id']} to {item['target_tool_output_tokens']} tokens")
def prefix_len(left: list[int], right: list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b: return index
    return min(len(left), len(right))
def materialize(rt: Any, spec: dict[str, Any], timeout: float) -> dict[str, Any]:
    sessions = []
    for source in spec["sessions"]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Investigate the staged repository using the tool evidence. " f"Deterministic session nonce: {source['nonce']}."}]
        prior_messages: list[dict[str, Any]] = []
        prior_bytes, prior_ids, turn_rows = b"", [], []
        for item in source["turns"]:
            output, attempts = calibrate_tool(rt, item, timeout)
            call_id = f"call_{source['session']:02d}_{item['turn']:02d}_{item['evidence_id'][:8]}"
            messages += [
                {"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function",
                 "function": {"name": "inspect_workspace", "arguments": canonical({"step": item["turn"]})}}]},
                {"role": "tool", "tool_call_id": call_id, "content": output}]
            if prior_messages and messages[:len(prior_messages)] != prior_messages:
                raise RuntimeError("message transcript changed instead of appending")
            transcript = "".join(canonical(message) + "\n" for message in messages).encode()
            if prior_bytes and not transcript.startswith(prior_bytes):
                raise RuntimeError("canonical transcript bytes changed instead of appending")
            token_http = post_json(rt.BASE_URL, "/tokenize", {
                "model": rt.MODEL_NAME, "messages": messages, "tools": TOOLS, "add_generation_prompt": True,
                "chat_template_kwargs": {"reasoning_effort": "low"}}, timeout)
            token_ids = ids_from(token_http)
            reusable = prefix_len(prior_ids, token_ids) if prior_ids else 0
            turn_rows.append({**item, "tool_output": output, "tool_output_sha256": digest(output),
                "tool_output_tokens": attempts[-1]["observed_tokens"], "calibration_attempts": attempts,
                "message_count": len(messages), "transcript_bytes": len(transcript),
                "transcript_sha256": digest(transcript), "append_only": True,
                "prompt_tokens": len(token_ids), "prompt_token_ids_sha256": digest(canonical(token_ids)),
                "expected_reusable_tokens": reusable, "new_token_estimate": len(token_ids) - reusable,
                "tokenization": token_http})
            prior_messages, prior_bytes, prior_ids = list(messages), transcript, token_ids
        sessions.append({"session": source["session"], "nonce": source["nonce"],
            "cache_salt": source["cache_salt"], "cache_salt_sha256": digest(source["cache_salt"]),
            "messages": messages, "turns": turn_rows, "final_transcript_sha256": digest(prior_bytes)})
    basis = {"seed": spec["seed"], "system_prompt": SYSTEM_PROMPT, "tools": TOOLS,
             "sessions": [{"session": row["session"], "cache_salt": row["cache_salt"],
                           "messages": row["messages"]} for row in sessions]}
    return {"materialized": True, "seed": spec["seed"], "specification_sha256": spec["specification_sha256"],
            "trace_sha256": digest(canonical(basis)), "append_only": True, "sessions": sessions}
def labels(raw: str) -> dict[str, str]:
    return {m.group(1): bytes(m.group(2), "utf-8").decode("unicode_escape") for m in LABEL_RE.finditer(raw)}
def parse_metrics(raw: str) -> dict[str, Any]:
    wanted, values, cache_config = {metric: key for key, metric in METRICS.items()}, {}, {}
    for line in raw.splitlines():
        pieces = line.rsplit(None, 1)
        if not line or line.startswith("#") or len(pieces) != 2:
            continue
        metric, raw_value = pieces
        name = metric.split("{", 1)[0]
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if name in wanted:
            key = wanted[name]
        elif name == SOURCE_METRIC:
            source = labels(metric).get("source")
            key = f"prompt_source_{source}" if source else ""
        elif name == "vllm:cache_config_info":
            cache_config.update(labels(metric))
            continue
        else:
            continue
        if key:
            values[key] = values.get(key, 0.) + value
    return {"values": values, "cache_config": cache_config}
def metrics_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float | None]:
    result = {}
    for key in sorted(set(METRICS) | set(SOURCE_KEYS)):
        old, new = before["values"].get(key), after["values"].get(key)
        result[key] = new - old if old is not None and new is not None and new >= old else None
    return result
def stream_chat(base_url: str, payload: dict[str, Any], timeout: float,
                barrier: threading.Barrier) -> dict[str, Any]:
    import requests
    body, session = canonical(payload).encode(), requests.Session()
    session.trust_env = False
    raw_parts: list[bytes] = []
    usage, kv_params, parse_errors, finishes = {}, {}, [], []
    done, first_token, status, headers, started = False, None, None, {}, time.perf_counter()
    try:
        barrier.wait(timeout=30)
        started = time.perf_counter()
        with session.post(base_url.rstrip("/") + "/v1/chat/completions", data=body,
                          headers={"content-type": "application/json"}, timeout=(20, timeout),
                          stream=True) as response:
            status, headers, buffer = response.status_code, dict(response.headers), b""
            for chunk in response.iter_content(chunk_size=4096):
                if not chunk: continue
                arrived = time.perf_counter()
                raw_parts.append(chunk)
                buffer += chunk
                while b"\n" in buffer:
                    raw_line, buffer = buffer.split(b"\n", 1)
                    data = raw_line.rstrip(b"\r")
                    if not data.startswith(b"data:"): continue
                    data = data[5:].strip()
                    if data == b"[DONE]": done = True; continue
                    try:
                        event = json.loads(data)
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        parse_errors.append(f"{type(error).__name__}: {error}"); continue
                    usage = event["usage"] if isinstance(event.get("usage"), dict) else usage
                    kv_params = (event["kv_transfer_params"]
                                 if isinstance(event.get("kv_transfer_params"), dict) else kv_params)
                    for choice in event.get("choices") or []:
                        finish = choice.get("finish_reason")
                        finishes += [finish] if finish is not None else []
                        delta = choice.get("delta") or {}
                        if first_token is None and any(delta.get(key) for key in
                            ("content", "reasoning_content", "reasoning", "tool_calls", "function_call")):
                            first_token = arrived
        error = None if status == 200 else f"HTTP {status}"
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
    finally:
        session.close()
    finished, raw = time.perf_counter(), b"".join(raw_parts)
    first_token = finished if first_token is None and int(usage.get("completion_tokens") or 0) > 0 else first_token
    stats = kv_params.get("cached_token_stats") if isinstance(kv_params, dict) else None
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    return {"status": status, "response_headers": headers, **body_evidence(body, raw),
            "done_received": done, "parse_errors": parse_errors, "usage": usage,
            "cache_stats": stats if isinstance(stats, dict) else None, "usage_cached_tokens": cached,
            "ttft_seconds": first_token - started if first_token is not None else None,
            "elapsed_seconds": finished - started, "finish_reasons": finishes, "error": error,
            "ok": status == 200 and done and not parse_errors and bool(usage)}
def response_hits(response: dict[str, Any]) -> tuple[int | None, str | None]:
    stats = response.get("cache_stats")
    if isinstance(stats, dict):
        values = [stats.get(name) for name in
                  ("num_vllm_cached_tokens", "num_lmcache_extra_cached_tokens", "num_lmcache_cached_tokens")]
        if all(isinstance(value, int) and value >= 0 for value in values):
            return values[0] + values[1], "kv_transfer_params.cached_token_stats union"
    cached = response.get("usage_cached_tokens")
    return ((cached, "usage.prompt_tokens_details.cached_tokens")
            if isinstance(cached, int) and cached >= 0 else (None, None))
def request_payload(rt: Any, trace: dict[str, Any], session: dict[str, Any],
                    planned: dict[str, Any], turn: int) -> dict[str, Any]:
    return {"model": rt.MODEL_NAME, "messages": session["messages"][:planned["message_count"]],
            "tools": TOOLS, "tool_choice": "none", "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": MAX_TOKENS, "temperature": 0.,
            "seed": trace["seed"] + session["session"] * TURNS + turn,
            "chat_template_kwargs": {"reasoning_effort": "low"}, "cache_salt": session["cache_salt"],
            "kv_transfer_params": {"cached_token_stats": True}}
def execute(rt: Any, trace: dict[str, Any], cache: str,
            timeout: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    turns, rounds = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=SESSIONS) as pool:
        for turn_index in range(TURNS):
            before_raw = rt.metric_snapshot()
            before, barrier, futures, metadata = parse_metrics(before_raw), threading.Barrier(SESSIONS), [], []
            for session in trace["sessions"]:
                planned = session["turns"][turn_index]
                futures.append(pool.submit(stream_chat, rt.BASE_URL,
                    request_payload(rt, trace, session, planned, turn_index), timeout, barrier))
                metadata.append((session, planned))
            responses, after_raw = [future.result() for future in futures], rt.metric_snapshot()
            after = parse_metrics(after_raw)
            delta = metrics_delta(before, after)
            raw_block = after["cache_config"].get("block_size") or before["cache_config"].get("block_size")
            block_size = int(raw_block) if raw_block and raw_block.isdigit() else None
            round_id = f"{cache}-turn-{turn_index:02d}"
            rounds.append({"id": round_id, "turn": turn_index, "concurrent_requests": SESSIONS,
                           "metrics_before_raw": before_raw, "metrics_after_raw": after_raw,
                           "metrics_delta": delta,
                           "cache_config_labels": after["cache_config"] or before["cache_config"]})
            for (session, planned), response in zip(metadata, responses):
                prompt = (response.get("usage") or {}).get("prompt_tokens")
                observed, source = response_hits(response)
                expected = planned["expected_reusable_tokens"]
                turns.append({"series": cache, "session": session["session"], "turn": turn_index,
                    "cache_salt_sha256": session["cache_salt_sha256"], "trace_sha256": trace["trace_sha256"],
                    "transcript_sha256": planned["transcript_sha256"], "prompt_tokens": prompt,
                    "tokenized_prompt_tokens": planned["prompt_tokens"], "expected_reusable_tokens": expected,
                    "observed_hit_tokens": observed, "observed_hit_source": source,
                    "hit_rate": observed / expected if observed is not None and expected else None,
                    "per_request_new_token_estimate": prompt - expected if isinstance(prompt, int)
                    else planned["new_token_estimate"],
                    "tokens_prefilled_estimate": prompt - observed
                    if isinstance(prompt, int) and observed is not None else None,
                    "reusable_tokens_reprefilled": max(0, expected - observed) if observed is not None else None,
                    "vllm_block_size_tokens": block_size,
                    "vllm_block_alignment_loss_tokens": expected % block_size if block_size else None,
                    "lmcache_chunk_size_tokens": 4096 if cache == "lmcache" else None,
                    "lmcache_chunk_alignment_loss_tokens": expected % 4096 if cache == "lmcache" else None,
                    "server_metric_window": {"scope": "eight-request concurrent turn",
                                             "id": round_id, "delta": delta}, **response})
    return turns, rounds
def summarize(turns: list[dict[str, Any]], rounds: list[dict[str, Any]], cache: str) -> dict[str, Any]:
    reusable = [row for row in turns if row["expected_reusable_tokens"] > 0]
    expected = sum(row["expected_reusable_tokens"] for row in reusable)
    deltas = {key: sum(float(r["metrics_delta"].get(key) or 0) for r in rounds)
              for key in tuple(METRICS) + SOURCE_KEYS}
    cached, per_session, per_turn = deltas["prompt_tokens_cached"], [], []
    for session in range(SESSIONS):
        rows = [row for row in reusable if row["session"] == session]
        covered = [row for row in rows if row["observed_hit_tokens"] is not None]
        denominator = sum(row["expected_reusable_tokens"] for row in covered)
        observed = sum(row["observed_hit_tokens"] for row in covered)
        per_session.append({"session": session, "reusable_turns": len(rows),
            "per_request_hit_coverage": len(covered), "expected_reusable_tokens_covered": denominator,
            "observed_hit_tokens_covered": observed, "hit_rate": observed / denominator if denominator else None,
            "complete_per_request_observation": len(covered) == len(rows)})
    for round_row in rounds:
        rows = [row for row in turns if row["turn"] == round_row["turn"]]
        wanted, got = sum(row["expected_reusable_tokens"] for row in rows), \
                      round_row["metrics_delta"].get("prompt_tokens_cached")
        per_turn.append({"turn": round_row["turn"], "expected_reusable_tokens": wanted,
                         "observed_hit_tokens": got,
                         "hit_rate": got / wanted if wanted and got is not None else None,
                         "reusable_tokens_reprefilled": max(0., wanted - got) if got is not None else None})
    blocks = sorted({row["vllm_block_size_tokens"] for row in reusable if row["vllm_block_size_tokens"]})
    return {"requests_expected": SESSIONS * TURNS, "requests_completed": sum(bool(row["ok"]) for row in turns),
        "expected_reusable_tokens": expected, "observed_hit_tokens": cached,
        "hit_rate": cached / expected if expected else None,
        "reusable_tokens_reprefilled": max(0., expected - cached), "server_metric_deltas": deltas,
        "local_vllm_prefix_hit_tokens": deltas["local_prefix_hits"],
        "lmcache_external_prefix_hit_tokens": deltas["external_prefix_hits"],
        "prompt_source_local_cache_hit_tokens": deltas["prompt_source_local_cache_hit"],
        "prompt_source_external_kv_transfer_tokens": deltas["prompt_source_external_kv_transfer"],
        "alignment": {"observed_vllm_block_sizes": blocks,
            "vllm_block_floor_loss_tokens": sum(row["vllm_block_alignment_loss_tokens"] or 0 for row in reusable),
            "lmcache_chunk_size_tokens": 4096 if cache == "lmcache" else None,
            "lmcache_chunk_floor_loss_tokens": sum(row["lmcache_chunk_alignment_loss_tokens"] or 0
                                                    for row in reusable)},
        "ttft_seconds": dist([row["ttft_seconds"] for row in turns if row["ttft_seconds"] is not None]),
        "per_turn": per_turn, "per_session": per_session}
def integrity(cache: str, trace: dict[str, Any], turns: list[dict[str, Any]],
              rounds: list[dict[str, Any]]) -> dict[str, bool]:
    required = set(METRICS) | set(SOURCE_KEYS)
    return {"trace_append_only": trace.get("append_only") is True,
        "tool_outputs_200_to_600_tokens": all(200 <= item["tool_output_tokens"] <= 600
                                               for row in trace["sessions"] for item in row["turns"]),
        "all_turns_returned": len(turns) == SESSIONS * TURNS,
        "all_http_streams_complete": all(row.get("ok") is True for row in turns),
        "raw_bodies_saved": all(row.get("raw_request_body") and row.get("raw_response_body") for row in turns),
        "prompt_usage_matches_tokenizer": all(row.get("prompt_tokens") == row.get("tokenized_prompt_tokens")
                                                for row in turns),
        "ttft_observed": all(row.get("ttft_seconds") is not None for row in turns),
        "metric_windows_complete_and_monotonic": len(rounds) == TURNS and all(
            required <= {key for key, value in row["metrics_delta"].items() if value is not None} for row in rounds),
        "lmcache_response_stats_exposed": cache != "lmcache" or all(
            isinstance(row.get("cache_stats"), dict) for row in turns)}
def phase_plan(rt: Any, args: argparse.Namespace) -> dict[str, Any]:
    spec = make_spec(args.seed)
    selected = list(ARMS) if args.arm == "all" else [args.arm]
    return {"schema": "r26-agent-cache-plan/v1", "selected_arms": selected,
        "outputs": [str(rt.ROOT / f"agent-cache-{arm}.json") for arm in selected],
        "series_order": ["vram", "lmcache"],
        "runtime": {"tp": 4, "dcp": 4, "kv": "fp8_ds_mla", "spec": "mtp0"},
        "workload": {"sessions": SESSIONS, "turns": TURNS, "max_tokens": MAX_TOKENS,
                     "natural_eos": True, "reasoning_effort": "low", "seed": args.seed},
        "trace_specification_sha256": spec["specification_sha256"],
        "metric_names": {**METRICS, "prompt_tokens_by_source": SOURCE_METRIC},
        "metric_name_evidence": METRIC_SAMPLE, "observation_threshold": HIT_RATE_OBSERVATION}
def initial_report(rt: Any, arm: str, image: str, spec: dict[str, Any]) -> dict[str, Any]:
    return {"schema": SCHEMA, "arm": arm, "image": image,
        "config": {"model": rt.MODEL_NAME, "base_url": rt.BASE_URL, "tp": 4, "dcp": 4,
            "kv_cache_dtype": "fp8_ds_mla", "speculator": "mtp0", "series_order": ["vram", "lmcache"],
            "sessions": SESSIONS, "turns_per_session": TURNS, "concurrency": SESSIONS,
            "max_tokens": MAX_TOKENS, "natural_eos": True, "chat_template": "image default",
            "chat_template_kwargs": {"reasoning_effort": "low"},
            "l2_host_dir": str(rt.L2_HOST_ROOT / f"agent-cache-{arm}"),
            "metric_names": {**METRICS, "prompt_tokens_by_source": SOURCE_METRIC},
            "metric_name_evidence": METRIC_SAMPLE,
            "metric_uncertainty": ("Prometheus names were verified in the recorded R26 sample; per-request "
                                   "cached-token fields are reported only where exposed."),
            "scope": "No attribution to or claim about logprobz private fixes.", "series": []},
        "trace": {"materialized": False, **spec}, "turns": [], "summary": {}, "gates": []}
def run_arm(rt: Any, arm: str, args: argparse.Namespace) -> dict[str, Any]:
    image = rt.IMAGE if arm == "stock-r26" else rt.OVERLAY_IMAGE
    spec, output = make_spec(args.seed), f"agent-cache-{arm}.json"
    report, trace = initial_report(rt, arm, image, spec), None
    rt.save_json(output, report)
    for cache in ("vram", "lmcache"):
        label = f"agent-cache-{arm}-{cache}"
        series: dict[str, Any] = {"name": cache, "label": label, "booted": False, "rounds": []}
        turns, errors, stopped, captured = [], [], True, True
        env = {"FAIRNESS_ENGINE": "none"} if arm == "drock-overlay" else {}
        extra_args = ["--prefill-compute-share", "0.4"] if arm == "drock-overlay" else []
        if cache == "lmcache":
            env["LMCACHE_L2_HOST_DIR"] = str(rt.L2_HOST_ROOT / f"agent-cache-{arm}")
        try:
            series["booted"] = bool(rt.boot(
                label, image=image, tp=4, dcp=4, spec="mtp0", cache=cache, kv="fp8_ds_mla",
                extra_env=env or None, extra_args=extra_args or None))
            if series["booted"]:
                if trace is None:
                    trace = materialize(rt, spec, args.request_timeout)
                    report["trace"] = trace
                turns, series["rounds"] = execute(rt, trace, cache, args.request_timeout)
                report["turns"].extend(turns)
                series["summary"] = summarize(turns, series["rounds"], cache)
                try:
                    rt.capture(label + "-probe")
                except Exception as error:
                    captured = False
                    errors.append(f"capture failed: {type(error).__name__}: {error}")
            else:
                errors.append("runtime.boot returned false")
        except Exception as error:
            errors += [f"{type(error).__name__}: {error}", traceback.format_exc()]
        finally:
            try:
                rt.stop()
            except Exception as error:
                stopped = False
                errors.append(f"stop failed: {type(error).__name__}: {error}")
        checks = integrity(cache, trace, turns, series["rounds"]) if trace and turns else {
            "booted": bool(series["booted"]), "probe_completed": False}
        checks.update({"booted": bool(series["booted"]), "capture_completed": captured,
                       "container_stopped": stopped})
        passed = all(checks.values()) and not errors
        gate = {"name": f"runtime:agent-cache:{arm}:{cache}", "kind": "runtime_integrity",
                "passed": passed, "checks": checks, "errors": errors}
        report["gates"].append(gate)
        rt.record_gate(gate["name"], passed,
                       {"checks": checks, "errors": errors, "receipt": str(rt.ROOT / output)})
        summary = series.get("summary", {})
        report["gates"].append({
            "name": f"observation:agent-cache-hit-rate-at-least-{HIT_RATE_OBSERVATION:.2f}:{arm}:{cache}",
            "kind": "observation", "passed": None, "threshold": HIT_RATE_OBSERVATION,
            "observed": summary.get("hit_rate"),
            "threshold_met": summary.get("hit_rate") is not None and summary["hit_rate"] >= HIT_RATE_OBSERVATION,
            "affects_exit_status": False})
        series.update({"stopped": stopped, "captured": captured, "errors": errors})
        report["config"]["series"].append(series)
        report["summary"][cache] = summary
        rt.save_json(output, report)
    report["summary"]["runtime_integrity_passed"] = all(
        gate["passed"] is True for gate in report["gates"] if gate["kind"] == "runtime_integrity")
    rt.save_json(output, report)
    return report
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=(*ARMS, "all"), default="all")
    parser.add_argument("--seed", type=int, default=260905)
    parser.add_argument("--request-timeout", type=float, default=900.)
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.seed <= 2_000_000_000:
        parser.error("--seed must be between 0 and 2000000000")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    return args
def main(argv: list[str] | None = None) -> int:
    args, rt = parse_args(argv), load_runtime()
    if args.print_plan:
        print(json.dumps(phase_plan(rt, args), indent=2, ensure_ascii=False))
        return 0
    selected = ARMS if args.arm == "all" else (args.arm,)
    reports = [run_arm(rt, arm, args) for arm in selected]
    passed = all(report["summary"].get("runtime_integrity_passed") is True for report in reports)
    print(json.dumps({"schema": SCHEMA,
        "outputs": [str(rt.ROOT / f"agent-cache-{arm}.json") for arm in selected],
        "runtime_integrity_passed": passed, "observations_do_not_affect_exit_status": True}, indent=2))
    return 0 if passed else 1
if __name__ == "__main__":
    raise SystemExit(main())
