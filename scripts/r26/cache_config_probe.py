#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

GIB = 1024**3
SOURCE_LOCK_PATH = "/opt/glm53-flash/source.lock"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    os.replace(temporary, path)


def run_capture(args: list[str], timeout: int = 120) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            args,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        return {
            "args": args,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "wall_seconds": time.perf_counter() - started,
            "error": None,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "args": args,
            "returncode": 124,
            "stdout": error.stdout.decode(errors="replace")
            if isinstance(error.stdout, bytes)
            else (error.stdout or ""),
            "stderr": error.stderr.decode(errors="replace")
            if isinstance(error.stderr, bytes)
            else (error.stderr or ""),
            "wall_seconds": time.perf_counter() - started,
            "error": f"timeout after {timeout}s",
        }
    except OSError as error:
        return {
            "args": args,
            "returncode": 127,
            "stdout": "",
            "stderr": "",
            "wall_seconds": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        }


def http_get(url: str, timeout: int = 30) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            headers = dict(response.headers.items())
            status = int(response.status)
        return {
            "ok": 200 <= status < 300,
            "status": status,
            "headers": headers,
            "body": raw.decode(errors="replace"),
            "body_sha256": hashlib.sha256(raw).hexdigest(),
            "wall_seconds": time.perf_counter() - started,
            "error": None,
        }
    except urllib.error.HTTPError as error:
        raw = error.read()
        return {
            "ok": False,
            "status": int(error.code),
            "headers": dict(error.headers.items()),
            "body": raw.decode(errors="replace"),
            "body_sha256": hashlib.sha256(raw).hexdigest(),
            "wall_seconds": time.perf_counter() - started,
            "error": f"HTTPError: {error}",
        }
    except (OSError, TimeoutError) as error:
        return {
            "ok": False,
            "status": None,
            "headers": {},
            "body": "",
            "body_sha256": None,
            "wall_seconds": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        }


def parse_top(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines()[1:]:
        fields = line.split(None, 3)
        if not fields or not fields[0].isdigit():
            continue
        rows.append(
            {
                "pid": int(fields[0]),
                "ppid": int(fields[1]) if len(fields) > 1 and fields[1].isdigit() else None,
                "command": fields[2] if len(fields) > 2 else "",
                "args": fields[3] if len(fields) > 3 else "",
            }
        )
    return rows


def parse_nvidia_processes(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4 or not fields[1].isdigit():
            continue
        try:
            memory_mib: int | None = int(fields[3])
        except ValueError:
            memory_mib = None
        rows.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_gpu_memory_mib": memory_mib,
            }
        )
    return rows


def extract_json_argument(command: str, flag: str) -> object | None:
    match = re.search(rf"(?:^|\s){re.escape(flag)}(?:=|\s+)", command)
    if match is None:
        return None
    remainder = command[match.end() :].lstrip()
    if not remainder:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(remainder)
        return value
    except json.JSONDecodeError:
        return None


def extract_int_argument(command: str, flag: str) -> int | None:
    match = re.search(rf"(?:^|\s){re.escape(flag)}(?:=|\s+)(\d+)", command)
    return int(match.group(1)) if match else None


def parse_prometheus(text: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.rsplit(None, 1)
        if len(fields) != 2:
            continue
        name = fields[0].split("{", 1)[0]
        try:
            value = float(fields[1])
        except ValueError:
            continue
        totals[name] = totals.get(name, 0.0) + value
    return totals


def directory_inventory(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "files": 0, "bytes": 0, "error": None}
    files = 0
    byte_count = 0
    try:
        if path.exists():
            for root, _, names in os.walk(path):
                for name in names:
                    candidate = Path(root, name)
                    try:
                        byte_count += candidate.stat().st_size
                        files += 1
                    except OSError:
                        continue
        return {
            "path": str(path),
            "exists": path.exists(),
            "files": files,
            "bytes": byte_count,
            "error": None,
        }
    except OSError as error:
        return {
            "path": str(path),
            "exists": False,
            "files": files,
            "bytes": byte_count,
            "error": f"{type(error).__name__}: {error}",
        }


def host_path_for_container_path(
    inspect_data: dict[str, Any], container_path: str | None
) -> Path | None:
    if not container_path:
        return None
    selected: tuple[int, Path, str] | None = None
    for mount in inspect_data.get("Mounts", []):
        destination = str(mount.get("Destination") or "").rstrip("/")
        source = mount.get("Source")
        if not destination or not source:
            continue
        if container_path == destination or container_path.startswith(destination + "/"):
            candidate = (len(destination), Path(str(source)), destination)
            if selected is None or candidate[0] > selected[0]:
                selected = candidate
    if selected is None:
        return None
    _, source_path, destination = selected
    relative = container_path[len(destination) :].lstrip("/")
    return source_path / relative


def read_process_environment(pid: int) -> dict[str, str] | None:
    try:
        entries = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return None
    selected: dict[str, str] = {}
    for entry in entries:
        if b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        decoded_key = key.decode(errors="replace")
        if decoded_key in {
            "CUDA_VISIBLE_DEVICES",
            "LMCACHE_L2_ENABLED",
            "LMCACHE_L2_ROOT",
            "LMCACHE_INSTANCE_ID",
            "CACHE_MODE",
        }:
            selected[decoded_key] = value.decode(errors="replace")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture generated cache configuration and sidecar GPU isolation evidence."
    )
    parser.add_argument("--container", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vllm-port", type=int, required=True)
    parser.add_argument("--expected-image", required=True)
    parser.add_argument(
        "--expected-mode", choices=("l2-on", "l2-off", "native"), required=True
    )
    parser.add_argument("--expected-l2-host", type=Path)
    parser.add_argument("--max-l2-gb", type=float, default=160.0)
    parser.add_argument("--expected-l1-gb", type=int)
    parser.add_argument("--source-lock-path", default=SOURCE_LOCK_PATH)
    args = parser.parse_args()

    prefix = args.out.with_suffix("")
    top_result = run_capture(
        ["docker", "top", args.container, "-eo", "pid,ppid,comm,args"]
    )
    write_text(Path(f"{prefix}.top.txt"), top_result["stdout"] + top_result["stderr"])

    inspect_result = run_capture(["docker", "inspect", args.container])
    write_text(
        Path(f"{prefix}.inspect.raw.json"),
        inspect_result["stdout"] + inspect_result["stderr"],
    )
    try:
        inspect_rows = json.loads(inspect_result["stdout"])
        inspect_data = inspect_rows[0] if inspect_rows else {}
    except (json.JSONDecodeError, TypeError):
        inspect_data = {}

    gpu_result = run_capture(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    write_text(
        Path(f"{prefix}.nvidia-smi.csv"), gpu_result["stdout"] + gpu_result["stderr"]
    )

    source_lock_result = run_capture(
        ["docker", "exec", args.container, "cat", args.source_lock_path]
    )
    write_text(
        Path(f"{prefix}.source-lock.txt"),
        source_lock_result["stdout"] + source_lock_result["stderr"],
    )

    image_inspect_result = run_capture(
        ["docker", "image", "inspect", args.expected_image]
    )
    write_text(
        Path(f"{prefix}.image-inspect.raw.json"),
        image_inspect_result["stdout"] + image_inspect_result["stderr"],
    )
    try:
        image_rows = json.loads(image_inspect_result["stdout"])
        image_data = image_rows[0] if image_rows else {}
    except (json.JSONDecodeError, TypeError):
        image_data = {}

    processes = parse_top(top_result["stdout"])
    lmcache_processes = [
        row
        for row in processes
        if re.search(r"(?:^|/)lmcache\s+server(?:\s|$)", row["args"], re.IGNORECASE)
        or "lmcache server" in row["args"].lower()
    ]
    vllm_processes = [
        row for row in processes if re.search(r"(?:^|/)vllm\s+serve(?:\s|$)", row["args"])
    ]
    lmcache_command = lmcache_processes[0]["args"] if lmcache_processes else ""
    vllm_command = vllm_processes[0]["args"] if vllm_processes else ""

    adapter = extract_json_argument(lmcache_command, "--l2-adapter")
    if adapter is None:
        adapter = extract_json_argument(lmcache_command, "--l2-adapter-config")
    adapter_dict = adapter if isinstance(adapter, dict) else {}
    kv_events_config = extract_json_argument(vllm_command, "--kv-events-config")
    generated_l1_size_gb = extract_int_argument(lmcache_command, "--l1-size-gb")
    generated_l1_init_size_gb = extract_int_argument(
        lmcache_command, "--l1-init-size-gb"
    )
    l2_container_path = adapter_dict.get("base_path") or adapter_dict.get("path")
    if not l2_container_path:
        backend_params = adapter_dict.get("backend_params")
        if isinstance(backend_params, dict):
            l2_container_path = backend_params.get("file_path")
    host_l2_path = host_path_for_container_path(
        inspect_data, str(l2_container_path) if l2_container_path else None
    )

    status_port = extract_int_argument(lmcache_command, "--http-port")
    declared_metrics_port = extract_int_argument(lmcache_command, "--prometheus-port")
    http_base = f"http://127.0.0.1:{status_port}" if status_port is not None else None
    openapi = http_get(http_base + "/openapi.json") if http_base else {"ok": False, "body": "", "status": None}
    try:
        paths = json.loads(openapi["body"]).get("paths", {}) if openapi["ok"] else {}
    except (json.JSONDecodeError, AttributeError):
        paths = {}
    status_route = next((route for route in ("/status", "/api/status") if route in paths), None)
    status_url = http_base + status_route if http_base and status_route else None
    metrics_url = (
        http_base + "/metrics" if http_base and "/metrics" in paths
        else f"http://127.0.0.1:{declared_metrics_port}/metrics" if declared_metrics_port is not None
        else None
    )
    lmcache_metrics = (
        http_get(metrics_url)
        if metrics_url is not None
        else {
            "ok": False, "status": None, "headers": {}, "body": "",
            "body_sha256": None, "wall_seconds": 0.0,
            "error": "No LMCache metrics endpoint discovered from the live API or generated command",
        }
    )
    write_text(Path(f"{prefix}.lmcache.metrics.txt"), lmcache_metrics["body"])
    lmcache_status = (
        http_get(status_url)
        if status_url is not None
        else {
            "ok": False,
            "status": None,
            "headers": {},
            "body": "",
            "body_sha256": None,
            "wall_seconds": 0.0,
            "error": "No LMCache status route advertised by the live OpenAPI schema",
        }
    )
    write_text(Path(f"{prefix}.lmcache.status.txt"), lmcache_status["body"])
    try:
        status_json: object | None = json.loads(lmcache_status["body"])
    except json.JSONDecodeError:
        status_json = None

    vllm_metrics = http_get(f"http://127.0.0.1:{args.vllm_port}/metrics")
    write_text(Path(f"{prefix}.vllm.metrics.txt"), vllm_metrics["body"])

    gpu_processes = parse_nvidia_processes(gpu_result["stdout"])
    gpu_pids = {row["pid"] for row in gpu_processes}
    lmcache_rows = []
    for row in lmcache_processes:
        lmcache_rows.append(
            {
                **row,
                "selected_environment": read_process_environment(row["pid"]),
                "present_in_nvidia_smi_compute_apps": row["pid"] in gpu_pids,
            }
        )

    host_config = inspect_data.get("HostConfig") or {}
    shm_size = int(host_config.get("ShmSize") or 0)
    ipc_mode = str(host_config.get("IpcMode") or "private")
    configured_image = str((inspect_data.get("Config") or {}).get("Image") or "")
    repo_digests = image_data.get("RepoDigests") or []
    expected_digest = args.expected_image.split("@", 1)[1] if "@" in args.expected_image else None
    digest_observed = bool(
        configured_image == args.expected_image
        or args.expected_image in repo_digests
        or (
            expected_digest
            and any(str(item).endswith("@" + expected_digest) for item in repo_digests)
        )
    )

    status_num_l2: int | None = None
    if isinstance(status_json, dict):
        storage_manager = status_json.get("storage_manager", {})
        if isinstance(storage_manager, dict):
            controller = storage_manager.get("store_controller", {})
            value = controller.get("num_l2_adapters") if isinstance(controller, dict) else None
            if value is None:
                value = storage_manager.get("num_l2_adapters")
            if isinstance(value, int):
                status_num_l2 = value

    adapter_capacity = adapter_dict.get("max_capacity_gb")
    if adapter_capacity is None and isinstance(adapter_dict.get("backend_params"), dict):
        adapter_capacity = adapter_dict["backend_params"].get("max_capacity_gb")
    try:
        adapter_capacity_float = float(adapter_capacity)
    except (TypeError, ValueError):
        adapter_capacity_float = None
    adapter_eviction = (
        adapter_dict.get("eviction") if isinstance(adapter_dict.get("eviction"), dict) else {}
    )
    adapter_eviction_policy = str(
        adapter_eviction.get("eviction_policy")
        or adapter_eviction.get("policy")
        or ""
    ).upper()

    actual_l2_enabled = bool(adapter_dict)
    expected_host = args.expected_l2_host.resolve() if args.expected_l2_host else None
    observed_mount_sources = [
        Path(str(mount["Source"])).resolve()
        for mount in inspect_data.get("Mounts", [])
        if mount.get("Source") and mount.get("Destination") == "/lmcache-l2"
    ]
    expected_mount_observed = (
        True if expected_host is None else expected_host in observed_mount_sources
    )
    capacity_safe = (
        not actual_l2_enabled
        or (
            adapter_capacity_float is not None
            and 0 < adapter_capacity_float <= args.max_l2_gb
        )
    )
    l2_lru = not actual_l2_enabled or adapter_eviction_policy == "LRU"
    effective_mode_ok = False
    if args.expected_mode == "l2-on":
        effective_mode_ok = actual_l2_enabled and isinstance(status_num_l2, int)
        effective_mode_ok = effective_mode_ok and status_num_l2 > 0
    elif args.expected_mode == "l2-off":
        effective_mode_ok = (
            not actual_l2_enabled
            and isinstance(status_num_l2, int)
            and status_num_l2 == 0
        )
    elif args.expected_mode == "native":
        effective_mode_ok = bool(
            re.search(r"--kv-offloading-size(?:=|\s+)[1-9]", vllm_command)
            or re.search(
                r"['\"]kv_offloading_size['\"]\s*:\s*[1-9]",
                vllm_metrics["body"],
            )
        ) and not lmcache_processes
    l1_size_matches = (
        args.expected_l1_gb is None
        or (
            generated_l1_size_gb == args.expected_l1_gb
            and generated_l1_init_size_gb == args.expected_l1_gb
        )
    )
    observability_ok = bool(
        vllm_metrics["ok"]
        and (
            args.expected_mode == "native"
            or (lmcache_metrics["ok"] and lmcache_status["ok"])
        )
    )

    gpu_worker_pids = {
        row["pid"]
        for row in gpu_processes
        if row["pid"] not in {item["pid"] for item in lmcache_rows}
    }
    sidecar_cpu_only = bool(
        gpu_result["returncode"] == 0
        and len(gpu_worker_pids) >= 4
        and lmcache_rows
        and not any(row["present_in_nvidia_smi_compute_apps"] for row in lmcache_rows)
    )
    if args.expected_mode == "native":
        sidecar_cpu_only = True

    report = {
        "container": args.container,
        "expected": {
            "image": args.expected_image,
            "mode": args.expected_mode,
            "l2_host": str(args.expected_l2_host) if args.expected_l2_host else None,
            "max_l2_gb": args.max_l2_gb,
            "private_shm_bytes_at_least": 128 * GIB,
            "expected_l1_gb": args.expected_l1_gb,
        },
        "commands": {
            "docker_top": {k: v for k, v in top_result.items() if k not in {"stdout", "stderr"}},
            "docker_inspect": {
                k: v for k, v in inspect_result.items() if k not in {"stdout", "stderr"}
            },
            "nvidia_smi": {
                k: v for k, v in gpu_result.items() if k not in {"stdout", "stderr"}
            },
            "source_lock": {
                k: v
                for k, v in source_lock_result.items()
                if k not in {"stdout", "stderr"}
            },
            "image_inspect": {
                k: v
                for k, v in image_inspect_result.items()
                if k not in {"stdout", "stderr"}
            },
        },
        "image_provenance": {
            "configured_image": configured_image,
            "container_image_id": inspect_data.get("Image"),
            "repo_digests": repo_digests,
            "expected_digest_observed": digest_observed,
            "image_labels": (image_data.get("Config") or {}).get("Labels"),
            "source_lock_path": args.source_lock_path,
            "source_lock_read_ok": source_lock_result["returncode"] == 0,
            "source_lock_sha256": hashlib.sha256(
                source_lock_result["stdout"].encode()
            ).hexdigest()
            if source_lock_result["returncode"] == 0
            else None,
            "source_lock_scope_note": (
                "The embedded base source.lock is recorded as base-image evidence only. "
                "For a Python overlay it does not prove the overlaid package identity."
            ),
        },
        "container_config": {
            "shm_size_bytes": shm_size,
            "private_shm_128g_or_larger": shm_size >= 128 * GIB and ipc_mode != "host",
            "ipc_mode": ipc_mode,
            "mounts": inspect_data.get("Mounts", []),
        },
        "generated_processes": processes,
        "generated_lmcache_command": lmcache_command or None,
        "generated_vllm_command": vllm_command or None,
        "generated_kv_events_config": kv_events_config,
        "generated_l1_size_gb": generated_l1_size_gb,
        "generated_l1_init_size_gb": generated_l1_init_size_gb,
        "l2": {
            "actual_enabled": actual_l2_enabled,
            "adapter": adapter,
            "adapter_capacity_gb": adapter_capacity_float,
            "capacity_at_or_below_limit": capacity_safe,
            "eviction_policy": adapter_eviction_policy or None,
            "eviction_policy_is_lru": l2_lru,
            "status_num_l2_adapters": status_num_l2,
            "container_path": l2_container_path,
            "host_path": str(host_l2_path) if host_l2_path else None,
            "host_inventory": directory_inventory(host_l2_path),
            "expected_host_mount_observed": expected_mount_observed,
            "observed_l2_mount_sources": [str(path) for path in observed_mount_sources],
            "effective_mode_matches": effective_mode_ok,
        },
        "sidecar": {
            "lmcache_processes": lmcache_rows,
            "gpu_compute_processes": gpu_processes,
            "gpu_query_succeeded": gpu_result["returncode"] == 0,
            "non_sidecar_gpu_pid_count": len(gpu_worker_pids),
            "cpu_only": sidecar_cpu_only,
            "proof_basis": (
                "Host PIDs from docker top were compared with the actual PID list returned "
                "by nvidia-smi --query-compute-apps. CUDA_VISIBLE_DEVICES is retained only "
                "as secondary evidence."
            ),
        },
        "monitoring_endpoints": {
            "status": status_url,
            "metrics": metrics_url,
            "openapi_status": openapi.get("status"),
            "selection": "Live OpenAPI routes; declared dedicated metrics port only if HTTP API lacks metrics",
        },
        "lmcache_status": {
            **{k: v for k, v in lmcache_status.items() if k != "body"},
            "parsed": status_json,
        },
        "lmcache_metrics": {
            **{k: v for k, v in lmcache_metrics.items() if k != "body"},
            "totals": parse_prometheus(lmcache_metrics["body"]),
        },
        "vllm_metrics": {
            **{k: v for k, v in vllm_metrics.items() if k != "body"},
            "totals": parse_prometheus(vllm_metrics["body"]),
        },
        "checks": {
            "capture_complete": top_result["returncode"] == 0
            and inspect_result["returncode"] == 0,
            "image_digest_observed": digest_observed,
            "source_lock_read": source_lock_result["returncode"] == 0,
            "private_shm": shm_size >= 128 * GIB and ipc_mode != "host",
            "effective_mode": effective_mode_ok,
            "l2_capacity_safe": capacity_safe,
            "l2_eviction_policy_lru": l2_lru,
            "expected_l2_mount": expected_mount_observed,
            "sidecar_cpu_only": sidecar_cpu_only,
            "l1_size_matches": l1_size_matches,
            "runtime_metrics_and_status": observability_ok,
        },
    }
    report["passed"] = all(report["checks"].values())
    save_json(args.out, report)
    print(json.dumps({"out": str(args.out), "passed": report["passed"], "checks": report["checks"]}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
