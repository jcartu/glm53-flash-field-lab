#!/usr/bin/env python3
"""Bounded R29 external-cache lifecycle, durability, pressure, and cancellation probe.

The coordinator owns image selection and server launch.  This probe only talks to
loopback services, reads one explicitly mounted test L2 namespace, and restarts
the already-running, ownership-labelled test container once.  It never removes a
container or cache file.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA = "r29-external-cache-lifecycle-probe/v1"
OWNERSHIP_LABEL = "field-lab.battery"
OWNERSHIP_VALUE = "r26"
PRODUCTION_CONTAINER = "glm53-prod"
PRODUCTION_PORT = 5001
TEST_PORT = 5002
L2_ROOT = Path("/mnt/2king/lmcache-r26-battery")
PRODUCTION_L2 = Path("/mnt/2king/lmcache-l2")

CHAT_PATH = "/v1/chat/completions"
TOKENIZE_PATH = "/tokenize"
RESET_PATH = "/reset_prefix_cache"
SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 290064,
    "max_tokens": 1024,
}
CHAT_TEMPLATE_KWARGS = {"reasoning_effort": "high", "clear_thinking": False}
CANCELLATION_MAX_TOKENS = 4096
CACHE_CHUNK_TOKENS = 4096

MIN_CONTEXT_TOKENS = 8192
MAX_CONTEXT_TOKENS = 65536
MAX_GROWTH_TURNS = 12
MAX_PRESSURE_REQUESTS = 12
MAX_TOTAL_REQUESTS = 28
MAX_TOTAL_PROMPT_TOKENS = 2_000_000
MAX_SAFE_L2_CAPACITY_BYTES = 8 * 1024**3
MAX_OBSERVED_PAYLOAD_BYTES = 9 * 1024**3
MIN_FILESYSTEM_FREE_RESERVE_BYTES = 16 * 1024**3
MAX_INVENTORY_FILES = 20_000
MAX_MANIFEST_ROWS = 4096
MAX_MANIFEST_BLOB_BYTES = 64 * 1024**2
OVERALL_BUDGET_SECONDS = 4 * 60 * 60
REQUEST_TIMEOUT_SECONDS = 900
RESTART_TIMEOUT_SECONDS = 900
HEALTH_TIMEOUT_SECONDS = 900
SETTLE_TIMEOUT_SECONDS = 240
SETTLE_STABLE_POLLS = 3
SETTLE_POLL_SECONDS = 1.0

CHECKPOINT_PREFIXES = ("recurrent-checkpoint-v2-", "recurrent-checkpoint-v3-")
SELECTED_METRICS = (
    "vllm:request_success_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:external_prefix_cache_hits_created",
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_by_source_total",
)


class ProbeAbort(RuntimeError):
    """A fail-closed condition after which the probe must issue no more load."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(canonical_bytes(value))


def safe_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return (cleaned or "artifact")[:180]


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_bytes(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
        + b"\n",
    )


def first_choice(parsed: object) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {}
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {}
    return choices[0]


def common_prefix_tokens(left: list[int], right: list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def parse_metric_labels(raw: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for match in re.finditer(r'(\w+)="((?:\\.|[^"\\])*)"', raw):
        try:
            labels[match.group(1)] = bytes(match.group(2), "utf-8").decode(
                "unicode_escape"
            )
        except UnicodeDecodeError:
            labels[match.group(1)] = match.group(2)
    return labels


def parse_prometheus(raw: str) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    totals: dict[str, float] = {}
    cache_config: list[dict[str, str]] = []
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        pieces = line.rsplit(None, 1)
        if len(pieces) != 2:
            continue
        metric, raw_value = pieces
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        name = metric.split("{", 1)[0]
        labels = parse_metric_labels(metric)
        if name == "vllm:cache_config_info":
            cache_config.append(labels)
        if name in SELECTED_METRICS or name == "vllm:cache_config_info":
            samples.append({"name": name, "labels": labels, "value": value})
            totals[name] = totals.get(name, 0.0) + value
    return {"samples": samples, "totals": totals, "cache_config": cache_config}


def metric_total(parsed: dict[str, Any], name: str, **labels: str) -> float | None:
    values = [
        row["value"]
        for row in parsed.get("samples", [])
        if row.get("name") == name
        and all(row.get("labels", {}).get(key) == value for key, value in labels.items())
    ]
    return sum(values) if values else None


def monotonic_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or after < before:
        return None
    return after - before


def selected_metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name in SELECTED_METRICS:
        old = before.get("totals", {}).get(name)
        new = after.get("totals", {}).get(name)
        rows[name] = monotonic_delta(old, new)
    abort_before = metric_total(before, "vllm:request_success_total", finished_reason="abort")
    abort_after = metric_total(after, "vllm:request_success_total", finished_reason="abort")
    rows["vllm:request_success_total:abort"] = monotonic_delta(
        abort_before, abort_after
    )
    for source in ("local_compute", "local_cache_hit", "external_kv_transfer"):
        old = metric_total(before, "vllm:prompt_tokens_by_source_total", source=source)
        new = metric_total(after, "vllm:prompt_tokens_by_source_total", source=source)
        rows[f"vllm:prompt_tokens_by_source_total:{source}"] = monotonic_delta(
            old, new
        )
    return rows


def normalize_loopback_base_url(value: str) -> tuple[str, int]:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ValueError("--base-url must be a bare loopback HTTP origin")
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("--base-url must use 127.0.0.1, localhost, or ::1")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"invalid --base-url port: {error}") from error
    if port is None or not 1 <= port <= 65535:
        raise ValueError("--base-url requires an explicit valid port")
    if port == PRODUCTION_PORT:
        raise ValueError(f"refusing production port {PRODUCTION_PORT}")
    if port != TEST_PORT:
        raise ValueError(f"R29 test endpoint must use loopback port {TEST_PORT}")
    return f"http://127.0.0.1:{port}", port


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def ensure_no_symlink_components(path: Path, root: Path) -> list[str]:
    problems: list[str] = []
    cursor = root
    relative = path.relative_to(root)
    if root.is_symlink():
        problems.append(str(root))
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            problems.append(str(cursor))
    return problems


def env_map(inspected: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    config = inspected.get("Config") if isinstance(inspected.get("Config"), dict) else {}
    for entry in config.get("Env") or []:
        if isinstance(entry, str) and "=" in entry:
            key, value = entry.split("=", 1)
            result[key] = value
    return result


def _effective_cache_environment(container_id: str) -> dict[str, Any]:
    """Read only cache configuration inherited by the owned live services."""
    program = r'''
import json, os
from pathlib import Path
allowed = {"LMCACHE_ENABLED", "LMCACHE_TRANSFER_MODE", "LMCACHE_L2_ENABLED",
           "LMCACHE_L2_PATH", "LMCACHE_L2_CONFIG", "LMCACHE_L2_MAX_CAPACITY_GB",
           "LMCACHE_CHECKPOINT_IDENTITY", "LMCACHE_CHECKPOINT_INDEX_PATH",
           "LMCACHE_HTTP_PORT", "LMCACHE_MIN_SHM_GIB"}
candidates = []
for process in Path("/proc").iterdir():
    if not process.name.isdigit() or int(process.name) == os.getpid():
        continue
    try:
        command = (process / "cmdline").read_bytes().lower()
        if b"vllm" not in command and b"lmcache" not in command:
            continue
        entries = (process / "environ").read_bytes().split(b"\0")
    except OSError:
        continue
    values = {}
    for entry in entries:
        if b"=" in entry:
            key, value = entry.split(b"=", 1)
            key = key.decode(errors="replace")
            if key in allowed:
                values[key] = value.decode(errors="replace")
    if values.get("LMCACHE_CHECKPOINT_IDENTITY") and values.get("LMCACHE_L2_PATH"):
        candidates.append({"pid": int(process.name), "environment": values})
print(json.dumps(candidates))
'''
    completed = subprocess.run(
        ["docker", "exec", container_id, "/opt/venv/bin/python", "-c", program],
        capture_output=True, text=True, timeout=30, check=True,
    )
    candidates = json.loads(completed.stdout)
    if not candidates:
        raise ProbeAbort("No live service exposes resolved checkpoint/L2 configuration")
    merged: dict[str, str] = {}
    for row in candidates:
        for key, value in row["environment"].items():
            if key in merged and merged[key] != value:
                raise ProbeAbort(f"Live services disagree on resolved {key}")
            merged[key] = value
    return {"environment": merged, "processes": [row["pid"] for row in candidates]}


def summarize_container(inspected: dict[str, Any]) -> dict[str, Any]:
    config = inspected.get("Config") if isinstance(inspected.get("Config"), dict) else {}
    state = inspected.get("State") if isinstance(inspected.get("State"), dict) else {}
    host = inspected.get("HostConfig") if isinstance(inspected.get("HostConfig"), dict) else {}
    env = env_map(inspected)
    allowed_env = {
        key: env.get(key)
        for key in (
            "HOST",
            "PORT",
            "SERVED_MODEL_NAME",
            "CACHE_MODE",
            "LMCACHE_ENABLED",
            "LMCACHE_TRANSFER_MODE",
            "LMCACHE_L2_ENABLED",
            "LMCACHE_L2_PATH",
            "LMCACHE_L2_MAX_CAPACITY_GB",
            "LMCACHE_L2_CONFIG",
            "LMCACHE_CHECKPOINT_INDEX_PATH",
            "LMCACHE_CHECKPOINT_IDENTITY",
            "LMCACHE_HTTP_PORT",
            "LMCACHE_MP_HOST",
            "LMCACHE_MP_PORT",
            "LMCACHE_PROMETHEUS_PORT",
            "LMCACHE_SHM_NAME",
            "LMCACHE_MIN_SHM_GIB",
            "LMCACHE_INSTANCE_ID",
        )
        if key in env
    }
    return {
        "id": inspected.get("Id"),
        "name": inspected.get("Name"),
        "image_id": inspected.get("Image"),
        "configured_image": config.get("Image"),
        "path": inspected.get("Path"),
        "args": inspected.get("Args") or [],
        "labels": config.get("Labels") or {},
        "state": {
            "running": state.get("Running"),
            "status": state.get("Status"),
            "started_at": state.get("StartedAt"),
            "restart_count": inspected.get("RestartCount"),
            "oom_killed": state.get("OOMKilled"),
            "error": state.get("Error"),
        },
        "network_mode": host.get("NetworkMode"),
        "ipc_mode": host.get("IpcMode"),
        "shm_size": host.get("ShmSize"),
        "environment": allowed_env,
        "mounts": [
            {
                "type": row.get("Type"),
                "source": row.get("Source"),
                "destination": row.get("Destination"),
                "rw": row.get("RW"),
            }
            for row in inspected.get("Mounts") or []
            if isinstance(row, dict)
        ],
    }


def file_category(relative: str) -> str:
    name = Path(relative).name
    if name.endswith(".data"):
        if name.startswith(CHECKPOINT_PREFIXES):
            return "checkpoint_payload"
        return "other_payload"
    lowered = name.lower()
    if lowered.endswith(("-wal", "-shm", "-journal")):
        return "sqlite_sidecar"
    if lowered.endswith((".sqlite", ".sqlite3", ".db")):
        return "sqlite_database"
    if lowered.endswith((".tmp", ".partial", ".part")) or ".tmp." in lowered:
        return "temporary"
    return "metadata_other"


def parse_object_filename(filename: str) -> dict[str, Any] | None:
    if not filename.endswith(".data"):
        return None
    parts = filename[:-5].split("@")
    if len(parts) not in (4, 5):
        return None
    model, rank_text, group_text, chunk_hash = parts[:4]
    cache_salt = parts[4] if len(parts) == 5 else ""
    try:
        rank = int(rank_text, 16)
        group = int(group_text, 16)
        bytes.fromhex(chunk_hash)
    except (ValueError, TypeError):
        return None
    return {
        "model_name": model.replace("-SEP-", "/"),
        "kv_rank": rank,
        "object_group_id": group,
        "chunk_hash_hex": chunk_hash.lower(),
        "cache_salt": cache_salt,
    }


def object_identity(row: dict[str, Any]) -> str:
    return sha256_json(
        [
            row.get("model_name"),
            row.get("kv_rank"),
            row.get("object_group_id"),
            row.get("chunk_hash_hex"),
            row.get("cache_salt", ""),
        ]
    )


def expected_object_filename(row: dict[str, Any]) -> str:
    safe_model = str(row["model_name"]).replace("/", "-SEP-")
    base = (
        f"{safe_model}@{int(row['kv_rank']):#010x}@"
        f"{int(row['object_group_id']):x}@{row['chunk_hash_hex']}"
    )
    if row.get("cache_salt"):
        base += "@" + str(row["cache_salt"])
    return base + ".data"


def derive_manifest_references(row: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    payload = row.get("payload_parsed")
    if not isinstance(payload, dict):
        return [], "manifest payload is not a JSON object"
    version = payload.get("schema_version")
    if version not in (1, 2):
        return [], f"unsupported checkpoint payload schema {version!r}"
    groups = payload.get("page_groups")
    if not isinstance(groups, list) or not groups:
        return [], "manifest has no page_groups"
    namespace = row.get("namespace")
    generation = row.get("generation")
    world_size = row.get("world_size")
    if not isinstance(namespace, str) or not isinstance(generation, str):
        return [], "manifest identity columns are unavailable"
    if not isinstance(world_size, int) or not 1 <= world_size <= 1024:
        return [], "manifest world_size is invalid"
    references: list[dict[str, Any]] = []
    for group_id, group in enumerate(groups):
        if not isinstance(group, dict):
            return [], f"page group {group_id} is not an object"
        name = group.get("name")
        positions = group.get("positions")
        content_keys = group.get("content_keys", [])
        if not isinstance(name, str) or not isinstance(positions, list):
            return [], f"page group {group_id} is malformed"
        if version == 2 and (
            not isinstance(content_keys, list) or len(content_keys) != len(positions)
        ):
            return [], f"schema-2 page group {group_id} lacks content keys"
        semantic_kind = (
            "auxiliary"
            if "auxiliary" in name.lower()
            else "opaque_engine_group"
        )
        for rank in range(world_size):
            namespace_hash = hashlib.sha256(
                json.dumps([namespace, rank, group_id, name]).encode()
            ).hexdigest()
            for page_id, position in enumerate(positions):
                if type(position) is not int or position < 0:
                    return [], f"page group {group_id} has an invalid position"
                if version == 2:
                    content_key = content_keys[page_id]
                    if (
                        not isinstance(content_key, str)
                        or not re.fullmatch(r"[0-9a-f]{64}", content_key)
                    ):
                        return [], f"page group {group_id} has an invalid content key"
                    chunk_hash = content_key
                else:
                    chunk_hash = hashlib.sha256(
                        json.dumps([generation, name, position]).encode()
                    ).hexdigest()
                identity = {
                    "model_name": f"recurrent-checkpoint-v{3 if version == 2 else 2}-{namespace_hash}",
                    "kv_rank": rank,
                    "object_group_id": group_id,
                    "chunk_hash_hex": chunk_hash,
                    "cache_salt": "",
                }
                references.append(
                    {
                        **identity,
                        "identity": object_identity(identity),
                        "filename": expected_object_filename(identity),
                        "manifest_generation": generation,
                        "manifest_num_tokens": row.get("num_tokens"),
                        "schema_version": version,
                        "rank": rank,
                        "group_id": group_id,
                        "group_name": name,
                        "semantic_kind": semantic_kind,
                        "position": position,
                        "page_bytes": group.get("page_bytes"),
                    }
                )
    return references, None


def decode_json_blob(value: object) -> tuple[str | None, object | None, str | None]:
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    else:
        return None, None, f"unsupported blob type {type(value).__name__}"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        return None, None, f"UnicodeDecodeError: {error}"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        return text, None, f"JSONDecodeError: {error}"
    return text, parsed, None


def quote_sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def inspect_sqlite(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "schema_supported": False,
        "manifests": [],
        "references": [],
        "errors": [],
    }
    try:
        uri = "file:" + urllib.parse.quote(str(path), safe="/") + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    except sqlite3.Error as error:
        result["errors"].append(f"open: {type(error).__name__}: {error}")
        return result
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("BEGIN")
        result["pragmas"] = {
            "user_version": connection.execute("PRAGMA user_version").fetchone()[0],
            "journal_mode": connection.execute("PRAGMA journal_mode").fetchone()[0],
            "page_size": connection.execute("PRAGMA page_size").fetchone()[0],
            "page_count": connection.execute("PRAGMA page_count").fetchone()[0],
            "freelist_count": connection.execute("PRAGMA freelist_count").fetchone()[0],
            "data_version": connection.execute("PRAGMA data_version").fetchone()[0],
        }
        tables = connection.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        schemas = []
        for table_name, create_sql in tables:
            columns = [
                column[1]
                for column in connection.execute(
                    f"PRAGMA table_info({quote_sql_identifier(table_name)})"
                ).fetchall()
            ]
            count = connection.execute(
                f"SELECT COUNT(*) FROM {quote_sql_identifier(table_name)}"
            ).fetchone()[0]
            schemas.append(
                {
                    "name": table_name,
                    "columns": columns,
                    "row_count": count,
                    "create_sql": create_sql,
                }
            )
        result["tables"] = schemas
        checkpoint = next((row for row in schemas if row["name"] == "checkpoints"), None)
        expected = {
            "namespace",
            "start_tokens",
            "prefix_hash",
            "tail",
            "num_tokens",
            "generation",
            "world_size",
            "payload",
            "access_order",
        }
        if checkpoint is None or not expected.issubset(checkpoint["columns"]):
            result["errors"].append(
                "checkpoint table with the installed schema was not present"
            )
            return result
        if checkpoint["row_count"] > MAX_MANIFEST_ROWS:
            result["errors"].append(
                f"checkpoint row count {checkpoint['row_count']} exceeds probe bound {MAX_MANIFEST_ROWS}"
            )
            return result
        rows = connection.execute(
            "SELECT namespace,start_tokens,prefix_hash,tail,num_tokens,generation,"
            "world_size,payload,access_order FROM checkpoints "
            "ORDER BY access_order,generation"
        ).fetchall()
        blob_budget = 0
        for (
            namespace,
            start_tokens,
            prefix_hash,
            tail,
            num_tokens,
            generation,
            world_size,
            payload,
            access_order,
        ) in rows:
            prefix_bytes = bytes(prefix_hash) if isinstance(prefix_hash, (bytes, memoryview)) else b""
            tail_text, tail_parsed, tail_error = decode_json_blob(tail)
            payload_text, payload_parsed, payload_error = decode_json_blob(payload)
            blob_budget += len((tail_text or "").encode()) + len((payload_text or "").encode())
            if blob_budget > MAX_MANIFEST_BLOB_BYTES:
                result["errors"].append(
                    f"manifest blob bytes exceed probe bound {MAX_MANIFEST_BLOB_BYTES}"
                )
                break
            manifest = {
                "namespace": namespace,
                "start_tokens": start_tokens,
                "prefix_hash_hex": prefix_bytes.hex(),
                "tail_raw": tail_text,
                "tail_parsed": tail_parsed,
                "tail_error": tail_error,
                "tail_sha256": sha256_bytes((tail_text or "").encode()),
                "num_tokens": num_tokens,
                "generation": generation,
                "world_size": world_size,
                "payload_raw": payload_text,
                "payload_parsed": payload_parsed,
                "payload_error": payload_error,
                "payload_sha256": sha256_bytes((payload_text or "").encode()),
                "access_order": access_order,
            }
            manifest["manifest_key"] = sha256_json(
                [namespace, start_tokens, manifest["prefix_hash_hex"], tail_parsed]
            )
            references, reference_error = derive_manifest_references(manifest)
            manifest["reference_error"] = reference_error
            manifest["reference_count"] = len(references)
            manifest["reference_identities"] = [row["identity"] for row in references]
            result["manifests"].append(manifest)
            result["references"].extend(references)
        result["schema_supported"] = not result["errors"] and len(result["manifests"]) == len(rows)
    except sqlite3.Error as error:
        result["errors"].append(f"query: {type(error).__name__}: {error}")
    finally:
        try:
            connection.rollback()
        except sqlite3.Error:
            pass
        connection.close()
    return result


def scan_l2_once(path: Path) -> dict[str, Any]:
    """One bounded L2 walk. A listed entry that vanishes before stat is
    tolerated only when its name classifies as a temporary in-flight write;
    symlinks and every other OSError stay fail-closed."""
    files: list[dict[str, Any]] = []
    symlinks: list[str] = []
    errors: list[str] = []
    vanished_temporary: list[str] = []
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError as error:
            errors.append(f"{current}: {type(error).__name__}: {error}")
            continue
        for entry in sorted(entries, key=lambda item: item.name):
            entry_path = Path(entry.path)
            try:
                relative = str(entry_path.relative_to(path))
                if entry.is_symlink():
                    symlinks.append(relative)
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry_path)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                stat = entry.stat(follow_symlinks=False)
            except (OSError, ValueError) as error:
                if (isinstance(error, FileNotFoundError)
                        and file_category(entry_path.name) == "temporary"):
                    # LMCache atomic writes publish through
                    # .lmcache-write.tmp.<pid>.<n> files that are renamed away
                    # between listing and stat; that is an in-flight write, not
                    # an unreadable inventory. Symlinks and every other OSError
                    # stay fail-closed.
                    vanished_temporary.append(str(entry_path.relative_to(path)))
                    continue
                errors.append(f"{entry_path}: {type(error).__name__}: {error}")
                continue
            category = file_category(relative)
            parsed_key = parse_object_filename(entry.name)
            row = {
                "relative_path": relative,
                "category": category,
                "logical_bytes": stat.st_size,
                "allocated_bytes": getattr(stat, "st_blocks", 0) * 512,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
                "inode": stat.st_ino,
                "device": stat.st_dev,
                "nlink": stat.st_nlink,
                "object_key": parsed_key,
                "object_identity": object_identity(parsed_key) if parsed_key else None,
            }
            if category not in ("checkpoint_payload", "other_payload") and stat.st_size <= 64 * 1024**2:
                try:
                    row["content_sha256"] = sha256_bytes(entry_path.read_bytes())
                except OSError as error:
                    row["content_sha256_error"] = f"{type(error).__name__}: {error}"
            files.append(row)
            if len(files) > MAX_INVENTORY_FILES:
                raise ProbeAbort(
                    f"L2 namespace contains more than {MAX_INVENTORY_FILES} files"
                )
    return {
        "files": files,
        "symlinks": symlinks,
        "errors": errors,
        "vanished_temporary": vanished_temporary,
    }


def inventory_l2(
    path: Path, *, max_passes: int = 3, retry_sleep_seconds: float = 0.05
) -> dict[str, Any]:
    vanished_temporary: list[str] = []
    scan = scan_l2_once(path)
    passes = 1
    while scan["vanished_temporary"] and passes < max_passes:
        # A listed temporary file was renamed away mid-scan; re-scan bounded
        # until one pass observes no vanishing in-flight writes.
        vanished_temporary.extend(scan["vanished_temporary"])
        time.sleep(retry_sleep_seconds)
        passes += 1
        scan = scan_l2_once(path)
    if scan["vanished_temporary"]:
        vanished_temporary.extend(scan["vanished_temporary"])
    files = scan["files"]
    symlinks = scan["symlinks"]
    errors = scan["errors"]
    categories: dict[str, dict[str, int]] = {}
    for row in files:
        bucket = categories.setdefault(
            row["category"], {"count": 0, "logical_bytes": 0, "allocated_bytes": 0}
        )
        bucket["count"] += 1
        bucket["logical_bytes"] += int(row["logical_bytes"])
        bucket["allocated_bytes"] += int(row["allocated_bytes"])
    try:
        vfs = os.statvfs(path)
        filesystem = {
            "block_size": vfs.f_frsize,
            "capacity_bytes": vfs.f_blocks * vfs.f_frsize,
            "available_bytes": vfs.f_bavail * vfs.f_frsize,
        }
    except OSError as error:
        filesystem = {"error": f"{type(error).__name__}: {error}"}
    sqlite_rows = []
    for row in files:
        candidate = path / row["relative_path"]
        is_named = row["category"] == "sqlite_database"
        try:
            with candidate.open("rb") as handle:
                is_sqlite = handle.read(16) == b"SQLite format 3\x00"
        except OSError:
            is_sqlite = False
        if is_named or is_sqlite:
            sqlite_rows.append(inspect_sqlite(candidate))
    manifests = [
        manifest
        for database in sqlite_rows
        for manifest in database.get("manifests", [])
    ]
    references = [
        reference
        for database in sqlite_rows
        for reference in database.get("references", [])
    ]
    actual_checkpoint = {
        row["object_identity"]: row
        for row in files
        if row["category"] == "checkpoint_payload" and row.get("object_identity")
    }
    referenced = {row["identity"]: row for row in references}
    reference_supported = bool(sqlite_rows) and all(
        row.get("schema_supported") is True for row in sqlite_rows
    )
    unreferenced = [
        actual_checkpoint[key]
        for key in sorted(set(actual_checkpoint) - set(referenced))
    ]
    missing = [
        referenced[key]
        for key in sorted(set(referenced) - set(actual_checkpoint))
    ]
    fingerprint_basis = [
        [
            row["relative_path"],
            row["category"],
            row["logical_bytes"],
            row["allocated_bytes"],
            row["mtime_ns"],
            row["inode"],
        ]
        for row in files
        if row["category"] != "sqlite_sidecar" or not row["relative_path"].endswith("-shm")
    ]
    return {
        "path": str(path),
        "captured_at": utc_now(),
        "files": files,
        "file_count": len(files),
        "categories": categories,
        "logical_bytes": sum(int(row["logical_bytes"]) for row in files),
        "allocated_bytes": sum(int(row["allocated_bytes"]) for row in files),
        "filesystem": filesystem,
        "symlinks": symlinks,
        "errors": errors,
        "sqlite": sqlite_rows,
        "manifests": manifests,
        "manifest_count": len(manifests),
        "reference_analysis_supported": reference_supported,
        "referenced_checkpoint_payload_count": len(referenced),
        "actual_checkpoint_payload_count": len(actual_checkpoint),
        "unreferenced_checkpoint_payloads": unreferenced if reference_supported else [],
        "missing_referenced_checkpoint_payloads": missing if reference_supported else [],
        "vanished_temporary": sorted(set(vanished_temporary)),
        "inventory_passes": passes,
        "fingerprint": sha256_json(fingerprint_basis),
    }


def inventory_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    left = {row["relative_path"]: row for row in before.get("files", [])}
    right = {row["relative_path"]: row for row in after.get("files", [])}
    created = [right[path] for path in sorted(set(right) - set(left))]
    deleted = [left[path] for path in sorted(set(left) - set(right))]
    modified = []
    for path in sorted(set(left) & set(right)):
        fields = (
            "logical_bytes",
            "allocated_bytes",
            "mtime_ns",
            "ctime_ns",
            "inode",
            "content_sha256",
        )
        if any(left[path].get(field) != right[path].get(field) for field in fields):
            modified.append({"before": left[path], "after": right[path]})

    def payload(row: dict[str, Any]) -> bool:
        return row.get("category") in ("checkpoint_payload", "other_payload")

    payload_created = [row for row in created if payload(row)]
    payload_deleted = [row for row in deleted if payload(row)]
    payload_modified = [row for row in modified if payload(row["after"])]
    metadata_created = [row for row in created if not payload(row)]
    metadata_deleted = [row for row in deleted if not payload(row)]
    metadata_modified = [row for row in modified if not payload(row["after"])]
    before_payload = sum(
        row["logical_bytes"] for row in left.values() if payload(row)
    )
    after_payload = sum(
        row["logical_bytes"] for row in right.values() if payload(row)
    )
    before_metadata = before.get("logical_bytes", 0) - before_payload
    after_metadata = after.get("logical_bytes", 0) - after_payload
    return {
        "created": created,
        "deleted": deleted,
        "modified": modified,
        "payload_objects": {
            "created": payload_created,
            "deleted": payload_deleted,
            "modified": payload_modified,
            "created_count": len(payload_created),
            "deleted_count": len(payload_deleted),
            "modified_count": len(payload_modified),
            "created_logical_bytes": sum(row["logical_bytes"] for row in payload_created),
            "deleted_logical_bytes": sum(row["logical_bytes"] for row in payload_deleted),
            "net_logical_growth_bytes": after_payload - before_payload,
        },
        "metadata_files": {
            "created": metadata_created,
            "deleted": metadata_deleted,
            "modified": metadata_modified,
            "created_count": len(metadata_created),
            "deleted_count": len(metadata_deleted),
            "modified_count": len(metadata_modified),
            "created_logical_bytes": sum(row["logical_bytes"] for row in metadata_created),
            "deleted_logical_bytes": sum(row["logical_bytes"] for row in metadata_deleted),
            "net_logical_growth_bytes": after_metadata - before_metadata,
        },
        "filesystem_growth": {
            "net_logical_bytes": after.get("logical_bytes", 0) - before.get("logical_bytes", 0),
            "net_allocated_bytes": after.get("allocated_bytes", 0) - before.get("allocated_bytes", 0),
            "before_logical_bytes": before.get("logical_bytes", 0),
            "after_logical_bytes": after.get("logical_bytes", 0),
            "before_allocated_bytes": before.get("allocated_bytes", 0),
            "after_allocated_bytes": after.get("allocated_bytes", 0),
        },
        "write_accounting_scope": {
            "payload_write_evidence": "new or stat-changed .data objects after controller/lease drain",
            "metadata_write_evidence": "new or stat-changed non-payload files, including SQLite WAL/SHM",
            "file_growth_evidence": "net logical and st_blocks*512 allocated bytes",
            "physical_device_bytes_written": None,
            "physical_device_bytes_written_status": "unavailable: filesystem snapshots do not attribute block-device write traffic",
        },
    }


def manifest_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in snapshot.get("manifests", []) if isinstance(row, dict)]


def select_boundary_manifest(
    snapshot: dict[str, Any], emitted_tokens: int
) -> dict[str, Any]:
    # vLLM's final sampled token can be returned to the client before it has
    # been forwarded through model execution.  A request-boundary checkpoint
    # can therefore end at either the complete emitted sequence or exactly one
    # token before it.  Preserve the offset instead of pretending those
    # boundaries are identical.
    rows = manifest_rows(snapshot)
    selected_offset: int | None = None
    matches: list[dict[str, Any]] = []
    for offset in (0, -1):
        boundary = emitted_tokens + offset
        matches = [row for row in rows if row.get("num_tokens") == boundary]
        if matches:
            selected_offset = offset
            break
    if not matches or selected_offset is None:
        candidates = sorted(
            {
                row.get("num_tokens")
                for row in rows
                if isinstance(row.get("num_tokens"), int)
            }
        )
        return {
            "available": False,
            "emitted_sequence_tokens": emitted_tokens,
            "accepted_boundary_offsets": [0, -1],
            "candidate_num_tokens": candidates,
            "reason": (
                "no SQLite manifest matched the emitted prompt+completion "
                "boundary or the source-documented unforwarded final-token boundary"
            ),
        }
    selected = max(matches, key=lambda row: int(row.get("access_order") or 0))
    return {
        "available": True,
        "emitted_sequence_tokens": emitted_tokens,
        "selected_num_tokens": emitted_tokens + selected_offset,
        "boundary_offset_from_emitted": selected_offset,
        "selection_basis": (
            "exact emitted sequence"
            if selected_offset == 0
            else "final sampled token was not yet forwarded through model execution"
        ),
        "candidate_count": len(matches),
        "manifest": selected,
    }


def compare_manifest_references(
    before_selection: dict[str, Any], after_selection: dict[str, Any]
) -> dict[str, Any]:
    if not before_selection.get("available") or not after_selection.get("available"):
        return {
            "available": False,
            "before": before_selection,
            "after": after_selection,
        }
    left = before_selection["manifest"]
    right = after_selection["manifest"]
    left_ids = set(left.get("reference_identities") or [])
    right_ids = set(right.get("reference_identities") or [])
    return {
        "available": True,
        "same_manifest_key": left.get("manifest_key") == right.get("manifest_key"),
        "same_generation": left.get("generation") == right.get("generation"),
        "before_generation": left.get("generation"),
        "after_generation": right.get("generation"),
        "before_schema": (left.get("payload_parsed") or {}).get("schema_version"),
        "after_schema": (right.get("payload_parsed") or {}).get("schema_version"),
        "before_reference_count": len(left_ids),
        "after_reference_count": len(right_ids),
        "shared_reference_count": len(left_ids & right_ids),
        "added_reference_count": len(right_ids - left_ids),
        "removed_reference_count": len(left_ids - right_ids),
        "shared_reference_identities": sorted(left_ids & right_ids),
        "added_reference_identities": sorted(right_ids - left_ids),
        "removed_reference_identities": sorted(left_ids - right_ids),
    }


def checkpoint_payload_count(snapshot: dict[str, Any]) -> int:
    return int(snapshot.get("categories", {}).get("checkpoint_payload", {}).get("count", 0))


def checkpoint_payload_bytes(snapshot: dict[str, Any]) -> int:
    return int(
        snapshot.get("categories", {})
        .get("checkpoint_payload", {})
        .get("logical_bytes", 0)
    )


def unreferenced_ids(snapshot: dict[str, Any]) -> set[str]:
    if snapshot.get("reference_analysis_supported") is not True:
        return set()
    return {
        str(row.get("object_identity"))
        for row in snapshot.get("unreferenced_checkpoint_payloads", [])
        if row.get("object_identity")
    }


def service_idle(status: object) -> tuple[bool | None, dict[str, Any]]:
    if not isinstance(status, dict):
        return None, {"reason": "LMCache /status was not a JSON object"}
    storage = status.get("storage_manager")
    recurrent = status.get("recurrent_checkpoints")
    if not isinstance(storage, dict) or not isinstance(recurrent, dict):
        return None, {"reason": "LMCache status lacks storage/recurrent checkpoint state"}
    store = storage.get("store_controller")
    prefetch = storage.get("prefetch_controller")
    l1 = storage.get("l1_manager")
    if not all(isinstance(row, dict) for row in (store, prefetch, l1)):
        return None, {"reason": "LMCache status lacks controller state"}
    counters = {
        "store.pending_keys_count": store.get("pending_keys_count"),
        "store.in_flight_task_count": store.get("in_flight_task_count"),
        "prefetch.submission_queue_size": prefetch.get("submission_queue_size"),
        "prefetch.pending_queue_size": prefetch.get("pending_queue_size"),
        "prefetch.in_flight_request_count": prefetch.get("in_flight_request_count"),
        "checkpoint.pending_generations": recurrent.get("pending_generations"),
        "checkpoint.store_leases": recurrent.get("store_leases"),
        "checkpoint.retrieve_leases": recurrent.get("retrieve_leases"),
        "l1.write_locked_count": l1.get("write_locked_count"),
        "l1.read_locked_count": l1.get("read_locked_count"),
    }
    if not all(isinstance(value, int) and value >= 0 for value in counters.values()):
        return None, {"reason": "LMCache idle counters are missing", "counters": counters}
    healthy = bool(status.get("is_healthy")) and all(
        bool(row.get("is_healthy")) for row in (store, prefetch, l1)
    )
    return healthy and all(value == 0 for value in counters.values()), {
        "healthy": healthy,
        "counters": counters,
    }


def l2_capacity_status(status: object) -> dict[str, Any] | None:
    if not isinstance(status, dict):
        return None
    storage = status.get("storage_manager")
    if not isinstance(storage, dict):
        return None
    controller = storage.get("l2_eviction_controller")
    if not isinstance(controller, dict):
        return None
    adapters = controller.get("adapters")
    if not isinstance(adapters, list) or not adapters or not isinstance(adapters[0], dict):
        return None
    return dict(adapters[0])


def assistant_history_message(message: object) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise ProbeAbort("completion did not contain an assistant message")
    result: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content"),
    }
    reasoning = message.get("reasoning_content")
    if reasoning is None:
        reasoning = message.get("reasoning")
    if isinstance(reasoning, str):
        result["reasoning"] = reasoning
    if isinstance(message.get("tool_calls"), list):
        result["tool_calls"] = message["tool_calls"]
    return result


class LifecycleProbe:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.started_monotonic = time.monotonic()
        self.deadline = self.started_monotonic + OVERALL_BUDGET_SECONDS
        self.base_url, self.port = normalize_loopback_base_url(args.base_url)
        self.model = args.model
        self.container_name = args.container
        self.l2_dir = args.l2_dir
        self.output_dir = args.output_dir
        self.raw_dir = self.output_dir / "raw"
        self.inventory_dir = self.output_dir / "inventories"
        self.receipt_path = self.output_dir / "receipt.json"
        self.container_id: str | None = None
        self.cache_base_url: str | None = None
        self.capacity_bytes: int | None = None
        self.index_host_path: Path | None = None
        self.request_count = 0
        self.prompt_tokens_consumed = 0
        self.trigger_watermark = 0.8
        self.gates: list[dict[str, Any]] = []
        self.receipt: dict[str, Any] = {
            "schema": SCHEMA,
            "started_at": utc_now(),
            "complete": False,
            "passed": False,
            "invocation": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            "inputs": {
                "base_url_requested": args.base_url,
                "base_url_normalized": self.base_url,
                "model": self.model,
                "container": self.container_name,
                "l2_dir": str(self.l2_dir),
                "output_dir": str(self.output_dir),
                "context_tokens": args.context_tokens,
                "growth_turns": args.growth_turns,
            },
            "method": {
                "sequence": [
                    "preflight and fresh namespace",
                    "deterministic cold request",
                    "byte-identical request replay",
                    "owned-container restart (vLLM and LMCache launcher children)",
                    "post-restart replay",
                    "append-only conversation growth",
                    "bounded unique-namespace capacity pressure",
                    "stream cancellation while capacity remains pressured",
                ],
                "sampling": SAMPLING,
                "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
                "raw_http_evidence": "raw/*.request.json and raw/*.response.body",
                "filesystem_accounting": (
                    "settled file-stat snapshots separate .data payload objects, SQLite/metadata, "
                    "logical size and st_blocks*512 allocation"
                ),
                "unchanged_boundary_rule": (
                    "zero-payload replay is evaluated only when prompt and generated token IDs match; "
                    "growth and differing completions are labelled changed-boundary writes"
                ),
                "source_grounding": {
                    "pinned_r29_base": (
                        "localinferencelab/vllm@sha256:"
                        "e44e07e615287605f87bd4db916d683e39066e72a1ba94cf4149089c1ec21b49"
                    ),
                    "vllm_commit": "45361846d60622cb5211b902bc893963e5a9eaa6",
                    "installed_r29_lmcache_commit": (
                        "dcd6ec92b23c7da14a46e0b9bf23a078969ddd4d"
                    ),
                    "historical_probe_pattern": (
                        "/home/josh/omp-workspace/glm53-flash-field-lab/"
                        "scripts/r26/cache_probe.py"
                    ),
                    "guard_pattern": (
                        "/home/josh/omp-workspace/glm53-flash-field-lab/"
                        "scripts/r26/runtime.py"
                    ),
                    "checkpoint_index_contract": (
                        "CheckpointIndex SQLite checkpoints table; schema-1 "
                        "generation keys and schema-2 content keys"
                    ),
                    "filesystem_contract": (
                        "FSNativeL2Adapter reversible .data ObjectKey filenames"
                    ),
                },
                "safety_bounds": {
                    "overall_seconds": OVERALL_BUDGET_SECONDS,
                    "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
                    "max_context_tokens": MAX_CONTEXT_TOKENS,
                    "max_growth_turns": MAX_GROWTH_TURNS,
                    "max_pressure_requests": MAX_PRESSURE_REQUESTS,
                    "max_total_requests": MAX_TOTAL_REQUESTS,
                    "max_total_prompt_tokens": MAX_TOTAL_PROMPT_TOKENS,
                    "max_l2_capacity_bytes": MAX_SAFE_L2_CAPACITY_BYTES,
                    "max_observed_payload_bytes": MAX_OBSERVED_PAYLOAD_BYTES,
                    "minimum_filesystem_free_reserve_bytes": (
                        MIN_FILESYSTEM_FREE_RESERVE_BYTES
                    ),
                },
            },
            "gates": self.gates,
            "stages": {},
            "limitations": [],
        }

    def prepare_output(self) -> None:
        if not self.output_dir.is_absolute():
            raise ValueError("--output-dir must be absolute")
        if self.output_dir.exists():
            if not self.output_dir.is_dir():
                raise ValueError("--output-dir exists and is not a directory")
            if any(self.output_dir.iterdir()):
                raise ValueError("--output-dir must be absent or empty; refusing to overwrite evidence")
        else:
            self.output_dir.mkdir(parents=True)
        self.raw_dir.mkdir()
        self.inventory_dir.mkdir()
        self.persist()

    def persist(self) -> None:
        if self.output_dir.exists():
            atomic_write_json(self.receipt_path, self.receipt)

    def gate(
        self,
        name: str,
        status: str,
        detail: object,
        *,
        required: bool = True,
    ) -> str:
        if status not in ("pass", "fail", "unavailable", "not_applicable"):
            raise ValueError(f"invalid gate status {status!r}")
        self.gates.append(
            {
                "name": name,
                "status": status,
                "passed": True if status == "pass" else False if status == "fail" else None,
                "required": required,
                "detail": detail,
                "recorded_at": utc_now(),
            }
        )
        self.persist()
        return status

    def check_budget(
        self, additional_requests: int = 0, additional_prompt_tokens: int = 0
    ) -> None:
        if time.monotonic() >= self.deadline:
            raise ProbeAbort("overall probe wall-clock budget exhausted")
        if self.request_count + additional_requests > MAX_TOTAL_REQUESTS:
            raise ProbeAbort("total request-count budget would be exceeded")
        if (
            additional_prompt_tokens < 0
            or self.prompt_tokens_consumed + additional_prompt_tokens
            > MAX_TOTAL_PROMPT_TOKENS
        ):
            raise ProbeAbort("total prompt-token budget would be exceeded")

    def bounded_timeout(self, maximum_seconds: float) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 1.0:
            raise ProbeAbort("overall probe wall-clock budget exhausted")
        return min(maximum_seconds, remaining)

    def artifact(self, label: str, suffix: str) -> Path:
        return self.raw_dir / f"{safe_label(label)}.{suffix}"

    def http_exchange(
        self,
        label: str,
        method: str,
        base_url: str,
        path: str,
        *,
        payload: object | None = None,
        timeout: float = 30.0,
        capture: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not base_url.startswith("http://127.0.0.1:"):
            raise ProbeAbort(f"refusing non-loopback HTTP target {base_url!r}")
        raw_request = canonical_bytes(payload) if payload is not None else b""
        headers = {
            "Accept": "application/json",
            "X-Request-Id": safe_label(label),
            **(extra_headers or {}),
        }
        data: bytes | None = None
        if method == "POST":
            data = raw_request
            headers["Content-Type"] = "application/json"
        url = base_url.rstrip("/") + path
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        started = time.perf_counter()
        status: int | None = None
        response_headers: dict[str, str] = {}
        raw_response = b""
        error: str | None = None
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=timeout) as response:
                status = int(response.status)
                response_headers = dict(response.headers.items())
                raw_response = response.read()
        except urllib.error.HTTPError as caught:
            status = int(caught.code)
            response_headers = dict(caught.headers.items())
            raw_response = caught.read()
            error = f"HTTPError: {caught}"
        except (OSError, TimeoutError, urllib.error.URLError) as caught:
            error = f"{type(caught).__name__}: {caught}"
        elapsed = time.perf_counter() - started
        try:
            parsed: object | None = json.loads(raw_response)
        except (json.JSONDecodeError, UnicodeDecodeError):
            parsed = None
        record: dict[str, Any] = {
            "label": label,
            "method": method,
            "url": url,
            "status": status,
            "ok": status is not None and 200 <= status < 300,
            "response_headers": response_headers,
            "request_body_sha256": sha256_bytes(raw_request),
            "request_body_bytes": len(raw_request),
            "response_body_sha256": sha256_bytes(raw_response),
            "response_body_bytes": len(raw_response),
            "elapsed_seconds": elapsed,
            "parsed": parsed,
            "error": error,
        }
        if capture:
            request_path = self.artifact(label, "request.json")
            response_path = self.artifact(label, "response.body")
            atomic_write_bytes(request_path, raw_request)
            atomic_write_bytes(response_path, raw_response)
            record["request_body_path"] = str(request_path)
            record["response_body_path"] = str(response_path)
        return record

    def inspect_container(self, name: str) -> dict[str, Any] | None:
        try:
            completed = subprocess.run(
                ["docker", "inspect", name],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ProbeAbort(f"docker inspect failed: {type(error).__name__}: {error}") from error
        if completed.returncode:
            return None
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ProbeAbort(f"docker inspect returned invalid JSON: {error}") from error
        if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
            raise ProbeAbort("docker inspect returned an unexpected object count")
        return value[0]

    def container_top(self, label: str) -> dict[str, Any]:
        if self.container_id is None:
            raise ProbeAbort("container identity is not established")
        try:
            completed = subprocess.run(
                ["docker", "top", self.container_id, "-eo", "pid,ppid,comm,args"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            error = None
        except (OSError, subprocess.TimeoutExpired) as caught:
            completed = None
            error = f"{type(caught).__name__}: {caught}"
        stdout = completed.stdout if completed is not None else ""
        rows = []
        for line in stdout.splitlines()[1:]:
            fields = line.strip().split(None, 3)
            if len(fields) == 4 and fields[0].isdigit():
                rows.append(
                    {
                        "pid": int(fields[0]),
                        "ppid": int(fields[1]) if fields[1].isdigit() else None,
                        "comm": fields[2],
                        "args": fields[3],
                    }
                )
        lmcache = [
            row
            for row in rows
            if re.search(r"(?:^|[/ ])lmcache(?: |$).*server", row["args"], re.IGNORECASE)
        ]
        vllm = [
            row
            for row in rows
            if re.search(r"vllm|api_server|serve-glm53", row["args"], re.IGNORECASE)
            and row not in lmcache
        ]
        record = {
            "label": label,
            "command": ["docker", "top", self.container_id, "-eo", "pid,ppid,comm,args"],
            "returncode": completed.returncode if completed is not None else None,
            "stdout": stdout,
            "stderr": completed.stderr if completed is not None else "",
            "error": error,
            "rows": rows,
            "lmcache_processes": lmcache,
            "vllm_processes": vllm,
        }
        atomic_write_json(self.artifact(label, "docker-top.json"), record)
        return record

    def metrics(self, label: str, *, capture: bool = True) -> dict[str, Any]:
        exchange = self.http_exchange(
            label,
            "GET",
            self.base_url,
            "/metrics",
            timeout=30,
            capture=capture,
        )
        raw = b""
        response_path = exchange.get("response_body_path")
        if capture and isinstance(response_path, str):
            try:
                raw = Path(response_path).read_bytes()
            except OSError:
                raw = b""
        elif isinstance(exchange.get("parsed"), str):
            raw = str(exchange["parsed"]).encode()
        if not raw and exchange.get("status") is not None and not capture:
            # Non-captured metric calls are rare; fetch the body once with a plain opener.
            replay = self.http_exchange(
                label + "-body",
                "GET",
                self.base_url,
                "/metrics",
                timeout=30,
                capture=True,
            )
            response_path = replay.get("response_body_path")
            if isinstance(response_path, str):
                try:
                    raw = Path(response_path).read_bytes()
                except OSError:
                    raw = b""
            exchange = replay
        text = raw.decode("utf-8", errors="replace")
        return {"http": exchange, "parsed": parse_prometheus(text), "raw_sha256": sha256_bytes(raw)}

    def cache_status(self, label: str, *, capture: bool = True) -> dict[str, Any]:
        if self.cache_base_url is None:
            return {"http": {"ok": False}, "status": None}
        exchange = self.http_exchange(
            label,
            "GET",
            self.cache_base_url,
            "/status",
            timeout=30,
            capture=capture,
        )
        return {
            "http": exchange,
            "status": exchange.get("parsed") if exchange.get("ok") else None,
        }

    def health_snapshot(self, label: str, *, capture: bool = True) -> dict[str, Any]:
        vllm = self.http_exchange(
            label + "-vllm-health",
            "GET",
            self.base_url,
            "/health",
            timeout=15,
            capture=capture,
        )
        models = self.http_exchange(
            label + "-models",
            "GET",
            self.base_url,
            "/v1/models",
            timeout=30,
            capture=capture,
        )
        cache = (
            self.http_exchange(
                label + "-lmcache-health",
                "GET",
                self.cache_base_url,
                "/healthcheck",
                timeout=15,
                capture=capture,
            )
            if self.cache_base_url
            else {"ok": False, "error": "cache URL not established"}
        )
        model_rows = (
            models.get("parsed", {}).get("data", [])
            if isinstance(models.get("parsed"), dict)
            else []
        )
        model_present = any(
            isinstance(row, dict) and row.get("id") == self.model for row in model_rows
        )
        return {
            "healthy": bool(vllm.get("ok") and models.get("ok") and cache.get("ok") and model_present),
            "vllm": vllm,
            "models": models,
            "lmcache": cache,
            "model_present": model_present,
        }

    def require_health(self, label: str) -> dict[str, Any]:
        snapshot = self.health_snapshot(label)
        if not snapshot["healthy"]:
            raise ProbeAbort(f"service health failure at {label}")
        return snapshot

    def wait_health(self, label: str) -> dict[str, Any]:
        deadline = min(
            self.deadline, time.monotonic() + HEALTH_TIMEOUT_SECONDS
        )
        attempts: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            inspected = self.inspect_container(self.container_name)
            if inspected is None:
                attempts.append({"at": utc_now(), "container": "absent"})
                time.sleep(2)
                continue
            summary = summarize_container(inspected)
            if inspected.get("Id") != self.container_id:
                raise ProbeAbort("container identity changed while waiting for restart health")
            health = self.health_snapshot(label + "-attempt", capture=False)
            attempts.append(
                {
                    "at": utc_now(),
                    "running": summary["state"]["running"],
                    "started_at": summary["state"]["started_at"],
                    "healthy": health["healthy"],
                }
            )
            if summary["state"]["running"] is True and health["healthy"]:
                final = self.health_snapshot(label + "-ready", capture=True)
                return {"healthy": True, "attempts": attempts, "final": final, "container": summary}
            time.sleep(2)
        return {"healthy": False, "attempts": attempts, "final": None}

    def save_inventory(self, label: str, snapshot: dict[str, Any]) -> str:
        path = self.inventory_dir / f"{safe_label(label)}.json"
        atomic_write_json(path, snapshot)
        return str(path)

    def settle(self, label: str) -> dict[str, Any]:
        deadline = min(self.deadline, time.monotonic() + SETTLE_TIMEOUT_SECONDS)
        samples: list[dict[str, Any]] = []
        stable = 0
        prior_fingerprint: str | None = None
        final_snapshot: dict[str, Any] | None = None
        final_status: dict[str, Any] | None = None
        status_supported = False
        while time.monotonic() < deadline:
            status_record = self.cache_status(label + "-settle-status", capture=False)
            status = status_record.get("status")
            idle, idle_detail = service_idle(status)
            status_supported = status_supported or idle is not None
            snapshot = inventory_l2(self.l2_dir)
            if snapshot["symlinks"] or snapshot["errors"]:
                raise ProbeAbort(
                    f"unsafe or unreadable L2 inventory during {label}: "
                    f"symlinks={snapshot['symlinks']} errors={snapshot['errors']}"
                )
            fingerprint = snapshot["fingerprint"]
            if fingerprint == prior_fingerprint:
                stable += 1
            else:
                stable = 1
                prior_fingerprint = fingerprint
            samples.append(
                {
                    "at": utc_now(),
                    "idle": idle,
                    "idle_detail": idle_detail,
                    "filesystem_fingerprint": fingerprint,
                    "stable_polls": stable,
                    "checkpoint_payload_count": checkpoint_payload_count(snapshot),
                    "checkpoint_payload_bytes": checkpoint_payload_bytes(snapshot),
                }
            )
            final_snapshot = snapshot
            final_status = status if isinstance(status, dict) else None
            if idle is True and stable >= SETTLE_STABLE_POLLS:
                break
            if idle is None and stable >= max(8, SETTLE_STABLE_POLLS):
                break
            time.sleep(SETTLE_POLL_SECONDS)
        if final_snapshot is None:
            raise ProbeAbort(f"could not inventory L2 during {label}")
        final_idle, final_idle_detail = service_idle(final_status)
        settled = (
            stable >= SETTLE_STABLE_POLLS
            and (final_idle is True or (final_idle is None and stable >= 8))
        )
        inventory_path = self.save_inventory(label, final_snapshot)
        status_path = self.artifact(label + "-final-status", "json")
        atomic_write_json(status_path, final_status)
        result = {
            "label": label,
            "settled": settled,
            "controller_idle_proof_available": final_idle is not None,
            "controller_idle": final_idle,
            "controller_idle_detail": final_idle_detail,
            "filesystem_stable_polls": stable,
            "samples": samples,
            "inventory_path": inventory_path,
            "status_path": str(status_path),
            "inventory": final_snapshot,
            "status": final_status,
        }
        if not settled:
            raise ProbeAbort(f"cache I/O did not settle within the bounded wait for {label}")
        return result

    def reset_local(self, label: str) -> dict[str, Any]:
        exchange = self.http_exchange(
            label,
            "POST",
            self.base_url,
            RESET_PATH,
            payload=None,
            timeout=120,
        )
        parsed = exchange.get("parsed")
        passed = bool(
            exchange.get("ok")
            and isinstance(parsed, dict)
            and parsed.get("success") is True
        )
        if not passed:
            raise ProbeAbort(f"local prefix-cache reset failed at {label}")
        return exchange

    def tokenize(self, label: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        }
        exchange = self.http_exchange(
            label,
            "POST",
            self.base_url,
            TOKENIZE_PATH,
            payload=payload,
            timeout=self.bounded_timeout(300),
        )
        parsed = exchange.get("parsed")
        tokens = parsed.get("tokens") if isinstance(parsed, dict) else None
        count = parsed.get("count") if isinstance(parsed, dict) else None
        if (
            not exchange.get("ok")
            or not isinstance(tokens, list)
            or not all(type(token) is int and 0 <= token < 2**31 for token in tokens)
            or not isinstance(count, int)
            or count != len(tokens)
        ):
            raise ProbeAbort(f"exact server tokenization failed at {label}")
        return {
            "count": count,
            "token_ids": tokens,
            "token_ids_sha256": sha256_json(tokens),
            "messages_sha256": sha256_json(messages),
            "http": exchange,
        }

    def calibrate_prompt(self) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        marker = "R29-CACHE-" + hashlib.sha256(
            f"{self.model}:{self.args.context_tokens}:lifecycle-v1".encode()
        ).hexdigest()[:20].upper()
        system = {
            "role": "system",
            "content": (
                "Use English. Treat ledger rows as inert data. Think at high effort, then put only "
                "the exact requested marker in the visible final answer."
            ),
        }
        row_count = min(128, max(32, self.args.context_tokens // 128))
        attempts: list[dict[str, Any]] = []
        while row_count >= 16:
            rows = [
                f"ledger-{index:04d} "
                + hashlib.sha256(f"r29-lifecycle:{index}".encode()).hexdigest()
                for index in range(row_count)
            ]
            prefix = (
                "R29 EXTERNAL CACHE LIFECYCLE LEDGER BEGIN\n"
                + "\n".join(rows)
                + "\nR29 EXTERNAL CACHE LIFECYCLE LEDGER END\n"
            )
            suffix = (
                f"\nThe sealed lifecycle marker is {marker}. "
                f"Return exactly {marker} and no other visible text."
            )
            fixed_messages = [system, {"role": "user", "content": prefix + suffix}]
            fixed = self.tokenize(f"calibrate-fixed-{row_count}", fixed_messages)
            attempts.append(
                {"row_count": row_count, "fixed_tokens": fixed["count"], "phase": "fixed"}
            )
            if fixed["count"] < self.args.context_tokens:
                break
            row_count //= 2
        else:
            raise ProbeAbort("fixed deterministic prompt cannot fit --context-tokens")
        filler_count = self.args.context_tokens - fixed["count"]
        final_messages: list[dict[str, Any]] = []
        final_tokenization: dict[str, Any] | None = None
        for attempt in range(10):
            user = prefix + (" atlas" * filler_count) + suffix
            final_messages = [system, {"role": "user", "content": user}]
            tokenization = self.tokenize(f"calibrate-{attempt:02d}", final_messages)
            attempts.append(
                {
                    "attempt": attempt,
                    "row_count": row_count,
                    "filler_count": filler_count,
                    "observed_tokens": tokenization["count"],
                    "phase": "adjust",
                }
            )
            if tokenization["count"] == self.args.context_tokens:
                final_tokenization = tokenization
                break
            filler_count += self.args.context_tokens - tokenization["count"]
            if filler_count < 0:
                raise ProbeAbort("prompt calibration produced a negative filler count")
        if final_tokenization is None:
            raise ProbeAbort("prompt calibration did not reach the exact requested token count")
        metadata = {
            "marker": marker,
            "row_count": row_count,
            "filler_count": filler_count,
            "messages_sha256": sha256_json(final_messages),
            "canonical_messages_bytes": len(canonical_bytes(final_messages)),
            "tokenization": final_tokenization,
            "attempts": attempts,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        }
        path = self.output_dir / "prompt-calibration.json"
        atomic_write_json(path, metadata)
        metadata["receipt_path"] = str(path)
        return final_messages, marker, metadata

    def chat(
        self,
        label: str,
        messages: list[dict[str, Any]],
        marker: str,
        cache_salt: str,
        *,
        tokenization: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.check_budget(1)
        self.require_health(label + "-preflight")
        tokenized = tokenization or self.tokenize(label + "-tokenize", messages)
        self.check_budget(1, tokenized["count"])
        self.request_count += 1
        self.prompt_tokens_consumed += tokenized["count"]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": SAMPLING["max_tokens"],
            "temperature": SAMPLING["temperature"],
            "top_p": SAMPLING["top_p"],
            "seed": SAMPLING["seed"],
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            "cache_salt": cache_salt,
            "kv_transfer_params": {"cached_token_stats": True},
            "return_token_ids": True,
        }
        before_metrics = self.metrics(label + "-metrics-before")
        exchange = self.http_exchange(
            label,
            "POST",
            self.base_url,
            CHAT_PATH,
            payload=payload,
            timeout=self.bounded_timeout(REQUEST_TIMEOUT_SECONDS),
        )
        after_metrics = self.metrics(label + "-metrics-after")
        parsed = exchange.get("parsed")
        choice = first_choice(parsed)
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        usage = parsed.get("usage") if isinstance(parsed, dict) and isinstance(parsed.get("usage"), dict) else {}
        transfer = (
            parsed.get("kv_transfer_params")
            if isinstance(parsed, dict) and isinstance(parsed.get("kv_transfer_params"), dict)
            else {}
        )
        stats = transfer.get("cached_token_stats") if isinstance(transfer.get("cached_token_stats"), dict) else None
        prompt_ids = parsed.get("prompt_token_ids") if isinstance(parsed, dict) else None
        generated_ids = choice.get("token_ids")
        token_identity_available = bool(
            isinstance(prompt_ids, list)
            and all(type(token) is int for token in prompt_ids)
            and isinstance(generated_ids, list)
            and all(type(token) is int for token in generated_ids)
        )
        prompt_ids_match = token_identity_available and prompt_ids == tokenized["token_ids"]
        completion_tokens_match = token_identity_available and len(generated_ids) == usage.get("completion_tokens")
        emitted = prompt_ids + generated_ids if token_identity_available else []
        content = message.get("content") if isinstance(message.get("content"), str) else ""
        reasoning = message.get("reasoning_content")
        if reasoning is None:
            reasoning = message.get("reasoning")
        reasoning = reasoning if isinstance(reasoning, str) else ""
        finish_reason = choice.get("finish_reason")
        protocol_complete = bool(
            exchange.get("ok")
            and choice
            and isinstance(message, dict)
            and isinstance(finish_reason, str)
            and isinstance(usage.get("prompt_tokens"), int)
            and isinstance(usage.get("completion_tokens"), int)
        )
        row = {
            "label": label,
            "messages": messages,
            "messages_sha256": sha256_json(messages),
            "cache_salt": cache_salt,
            "cache_salt_sha256": sha256_bytes(cache_salt.encode()),
            "sampling": SAMPLING,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            "tokenization": tokenized,
            "http": exchange,
            "metrics": {
                "before": before_metrics,
                "after": after_metrics,
                "delta": selected_metric_delta(
                    before_metrics["parsed"], after_metrics["parsed"]
                ),
            },
            "summary": {
                "protocol_complete": protocol_complete,
                "finish_reason": finish_reason,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "visible_content": content,
                "visible_content_sha256": sha256_bytes(content.encode()),
                "reasoning_content": reasoning,
                "reasoning_content_sha256": sha256_bytes(reasoning.encode()),
                "assistant_message": message,
                "assistant_message_sha256": sha256_json(message),
                "visible_marker_exact": content.strip() == marker,
                "generation_budget_reached": finish_reason == "length",
                "cache_stats": stats,
            },
            "token_identity": {
                "available": token_identity_available,
                "returned_prompt_token_ids": prompt_ids,
                "returned_prompt_token_ids_sha256": sha256_json(prompt_ids) if isinstance(prompt_ids, list) else None,
                "generated_token_ids": generated_ids,
                "generated_token_ids_sha256": sha256_json(generated_ids) if isinstance(generated_ids, list) else None,
                "prompt_ids_match_tokenizer": prompt_ids_match,
                "completion_count_matches_usage": completion_tokens_match,
                "emitted_sequence_token_ids": emitted,
                "emitted_sequence_sha256": sha256_json(emitted) if emitted else None,
                "emitted_sequence_tokens": len(emitted),
            },
        }
        row["passed"] = bool(
            protocol_complete
            and finish_reason == "stop"
            and content.strip() == marker
            and usage.get("prompt_tokens") == tokenized["count"]
            and prompt_ids_match
            and completion_tokens_match
        )
        return row

    @staticmethod
    def request_cache_hits(request: dict[str, Any]) -> int | None:
        summary = request.get("summary", {})
        observation = {"source": "unavailable", "tokens": None}
        summary["cache_hit_observation"] = observation
        prompt_tokens = summary.get("prompt_tokens")
        if request.get("passed") is not True or type(prompt_tokens) is not int:
            return None
        stats = summary.get("cache_stats")
        if isinstance(stats, dict):
            value = stats.get("num_lmcache_cached_tokens")
            if type(value) is int and 0 <= value <= prompt_tokens:
                observation.update(source="per_request.num_lmcache_cached_tokens", tokens=value)
                return value
            return None

        # R29's recurrent connector returns no KVTransferParams. Its scheduler
        # instead records imported checkpoint tokens in these external counters.
        # Attribute a delta only to one completed, isolated request in one epoch.
        names = (
            "vllm:external_prefix_cache_hits_total",
            "vllm:external_prefix_cache_queries_total",
            "vllm:external_prefix_cache_hits_created",
            "vllm:request_success_total",
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:num_preemptions_total",
        )
        snapshots = [request.get("metrics", {}).get(side, {}) for side in ("before", "after")]
        if any(row.get("http", {}).get("ok") is not True for row in snapshots):
            return None
        values = [
            [row.get("parsed", {}).get("totals", {}).get(name) for name in names]
            for row in snapshots
        ]
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for row in values for value in row
        ):
            return None
        before, after = values
        if before[2] <= 0 or before[2] != after[2]:
            return None
        if any(row[index] != 0 for row in values for index in (4, 5)):
            return None
        if after[3] - before[3] != 1 or after[6] != before[6]:
            return None
        hits, queries = after[0] - before[0], after[1] - before[1]
        if not (0 <= hits <= queries <= prompt_tokens and queries > 0):
            return None
        if not float(hits).is_integer() or not float(queries).is_integer():
            return None
        observation.update(
            source="isolated_request.external_prefix_cache_hits_total",
            tokens=int(hits),
            external_queries=int(queries),
            completed_requests=1,
            counter_epoch=after[2],
        )
        return int(hits)

    def stage_request(
        self,
        label: str,
        messages: list[dict[str, Any]],
        marker: str,
        cache_salt: str,
        before_settle: dict[str, Any],
        *,
        tokenization: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        reset = self.reset_local(label + "-reset-local")
        request = self.chat(
            label,
            messages,
            marker,
            cache_salt,
            tokenization=tokenization,
        )
        after_settle = self.settle(label + "-settled")
        delta = inventory_delta(
            before_settle["inventory"], after_settle["inventory"]
        )
        boundary = select_boundary_manifest(
            after_settle["inventory"],
            request["token_identity"]["emitted_sequence_tokens"],
        )
        return {
            "label": label,
            "reset_local": reset,
            "request": request,
            "before_inventory_path": before_settle["inventory_path"],
            "after_settlement": after_settle,
            "inventory_delta": delta,
            "boundary_manifest": boundary,
        }

    def validate_paths_and_args(self) -> dict[str, Any]:
        if not self.l2_dir.is_absolute():
            raise ValueError("--l2-dir must be absolute")
        if not MIN_CONTEXT_TOKENS <= self.args.context_tokens <= MAX_CONTEXT_TOKENS:
            raise ValueError(
                f"--context-tokens must be in {MIN_CONTEXT_TOKENS}..{MAX_CONTEXT_TOKENS}"
            )
        if not 1 <= self.args.growth_turns <= MAX_GROWTH_TURNS:
            raise ValueError(f"--growth-turns must be in 1..{MAX_GROWTH_TURNS}")
        if self.container_name == PRODUCTION_CONTAINER:
            raise ValueError(f"refusing production container {PRODUCTION_CONTAINER!r}")
        if L2_ROOT.is_symlink():
            raise ValueError(f"test L2 root must not be a symlink: {L2_ROOT}")
        root = L2_ROOT.resolve(strict=True)
        if not self.l2_dir.exists() or not self.l2_dir.is_dir():
            raise ValueError(
                "--l2-dir must already exist as the mounted test namespace"
            )
        try:
            lexical_relative = self.l2_dir.relative_to(L2_ROOT)
        except ValueError as error:
            raise ValueError(
                f"--l2-dir must be lexically beneath {L2_ROOT}"
            ) from error
        if ".." in lexical_relative.parts:
            raise ValueError("--l2-dir must not contain parent traversal")
        symlinks = ensure_no_symlink_components(self.l2_dir, L2_ROOT)
        if symlinks:
            raise ValueError(f"--l2-dir contains symlink components: {symlinks}")
        resolved = self.l2_dir.resolve(strict=True)
        if resolved == root or not path_is_within(resolved, root):
            raise ValueError(f"--l2-dir must be a non-root descendant of {root}")
        production_cache = PRODUCTION_L2.resolve(strict=False)
        if resolved == production_cache or path_is_within(
            resolved, production_cache
        ):
            raise ValueError("refusing production L2 namespace")
        if (
            lexical_relative.parts
            and lexical_relative.parts[0].lower()
            in {"prod", "production", "shared"}
        ):
            raise ValueError(
                "--l2-dir must not use the root-level shared/production namespace"
            )
        output_resolved = self.output_dir.resolve(strict=False)
        production_output = PRODUCTION_L2.resolve(strict=False)
        if (
            output_resolved == production_output
            or path_is_within(output_resolved, production_output)
        ):
            raise ValueError("--output-dir must not modify the production L2 tree")
        if path_is_within(output_resolved, resolved) or path_is_within(resolved, output_resolved):
            raise ValueError("--output-dir and --l2-dir must not overlap")
        maximum_requests = 3 + self.args.growth_turns + MAX_PRESSURE_REQUESTS + 1
        maximum_tokens = maximum_requests * self.args.context_tokens
        if maximum_requests > MAX_TOTAL_REQUESTS or maximum_tokens > MAX_TOTAL_PROMPT_TOKENS:
            raise ValueError("requested lifecycle can exceed the probe request/token budget")
        return {
            "l2_root": str(root),
            "l2_resolved": str(resolved),
            "output_resolved": str(output_resolved),
            "symlink_components": symlinks,
            "maximum_planned_requests": maximum_requests,
            "maximum_planned_prompt_tokens": maximum_tokens,
        }

    def container_preflight(self) -> dict[str, Any]:
        inspected = self.inspect_container(self.container_name)
        if inspected is None:
            raise ProbeAbort(f"test container {self.container_name!r} does not exist")
        summary = summarize_container(inspected)
        labels = summary.get("labels") or {}
        production = self.inspect_container(PRODUCTION_CONTAINER)
        production_id = production.get("Id") if production else None
        ownership = (
            labels.get(OWNERSHIP_LABEL) == OWNERSHIP_VALUE
            and summary.get("name") != "/" + PRODUCTION_CONTAINER
            and summary.get("id") != production_id
            and summary.get("state", {}).get("running") is True
            and summary.get("state", {}).get("oom_killed") is not True
        )
        if not ownership:
            raise ProbeAbort("container ownership/non-production guard failed")
        self.container_id = str(summary["id"])
        env = env_map(inspected)
        effective = _effective_cache_environment(self.container_id)
        env.update(effective["environment"])
        host_ok = env.get("HOST") == "127.0.0.1"
        port_ok = env.get("PORT") == str(self.port)
        network_ok = summary.get("network_mode") == "host"
        cache_enabled = env.get("LMCACHE_ENABLED") == "1"
        transfer_mode = env.get("LMCACHE_TRANSFER_MODE", "engine_driven")
        l2_enabled = env.get("LMCACHE_L2_ENABLED", "1") == "1"
        checkpoint_identity_raw = env.get("LMCACHE_CHECKPOINT_IDENTITY", "")
        try:
            checkpoint_identity = json.loads(checkpoint_identity_raw)
        except json.JSONDecodeError:
            checkpoint_identity = None
        checkpoint_identity_ok = isinstance(checkpoint_identity, dict) and bool(
            checkpoint_identity
        )
        cache_port_text = env.get("LMCACHE_HTTP_PORT", "8085")
        if not cache_port_text.isdigit() or not 1 <= int(cache_port_text) <= 65535:
            raise ProbeAbort("container has an invalid LMCache HTTP port")
        cache_port = int(cache_port_text)
        if cache_port in (self.port, PRODUCTION_PORT):
            raise ProbeAbort("LMCache HTTP port collides with inference or production")
        self.cache_base_url = f"http://127.0.0.1:{cache_port}"
        l2_inside = env.get("LMCACHE_L2_PATH", "/lmcache-l2")
        if not l2_inside.startswith("/"):
            raise ProbeAbort("container L2 path is not absolute")
        matching_mounts = []
        requested_l2 = self.l2_dir.resolve()
        for mount in summary.get("mounts", []):
            if not mount.get("source") or not mount.get("destination") or mount.get("rw") is not True:
                continue
            try:
                relative = Path(l2_inside).relative_to(Path(str(mount["destination"])))
            except ValueError:
                continue
            mount_root = Path(str(mount["source"])).resolve(strict=True)
            effective_host = (mount_root / relative).resolve(strict=True)
            if requested_l2 not in (mount_root, effective_host):
                continue
            if not path_is_within(effective_host, L2_ROOT.resolve()) or effective_host == L2_ROOT.resolve():
                continue
            if ensure_no_symlink_components(mount_root / relative, L2_ROOT):
                continue
            matching_mounts.append({"mount": mount, "effective_host": effective_host})
        mount_ok = len(matching_mounts) == 1
        if mount_ok:
            self.l2_dir = matching_mounts[0]["effective_host"]
            self.receipt["inputs"]["effective_l2_dir"] = str(self.l2_dir)
        l2_config_raw = env.get("LMCACHE_L2_CONFIG")
        if l2_config_raw:
            try:
                l2_config = json.loads(l2_config_raw)
            except json.JSONDecodeError as error:
                raise ProbeAbort(f"LMCACHE_L2_CONFIG is invalid JSON: {error}") from error
        else:
            try:
                max_gb = float(env.get("LMCACHE_L2_MAX_CAPACITY_GB", "512"))
            except ValueError as error:
                raise ProbeAbort("LMCACHE_L2_MAX_CAPACITY_GB is invalid") from error
            l2_config = {
                "type": "fs_native",
                "base_path": l2_inside,
                "max_capacity_gb": max_gb,
                "eviction": {
                    "eviction_policy": "LRU",
                    "trigger_watermark": 0.8,
                    "eviction_ratio": 0.2,
                },
            }
        adapter_ok = (
            isinstance(l2_config, dict)
            and l2_config.get("type") == "fs_native"
            and l2_config.get("base_path") == l2_inside
        )
        try:
            capacity_gb = float(l2_config.get("max_capacity_gb", 0))
        except (TypeError, ValueError) as error:
            raise ProbeAbort("L2 max_capacity_gb is invalid") from error
        self.capacity_bytes = int(capacity_gb * 1024**3)
        eviction = l2_config.get("eviction") if isinstance(l2_config.get("eviction"), dict) else None
        eviction_ok = bool(
            eviction
            and eviction.get("eviction_policy") == "LRU"
            and isinstance(eviction.get("trigger_watermark"), (int, float))
            and 0 < float(eviction["trigger_watermark"]) < 1
            and isinstance(eviction.get("eviction_ratio"), (int, float))
            and 0 < float(eviction["eviction_ratio"]) <= 1
        )
        if eviction_ok:
            self.trigger_watermark = float(eviction["trigger_watermark"])
        capacity_ok = bool(
            0 < self.capacity_bytes <= MAX_SAFE_L2_CAPACITY_BYTES and eviction_ok
        )
        index_inside = env.get("LMCACHE_CHECKPOINT_INDEX_PATH")
        index_ok = False
        if index_inside:
            index_path = Path(index_inside)
            l2_path = Path(l2_inside)
            if index_path.is_absolute() and path_is_within(index_path, l2_path):
                self.index_host_path = self.l2_dir / index_path.relative_to(l2_path)
                index_ok = path_is_within(
                    self.index_host_path.resolve(strict=False), self.l2_dir.resolve()
                )
        min_shm_gib_text = env.get("LMCACHE_MIN_SHM_GIB", "96")
        min_shm_gib = int(min_shm_gib_text) if min_shm_gib_text.isdigit() else 96
        shm_ok = isinstance(summary.get("shm_size"), int) and summary["shm_size"] >= min_shm_gib * 1024**3
        service_contract = all(
            (
                host_ok,
                port_ok,
                network_ok,
                cache_enabled,
                transfer_mode == "engine_driven",
                l2_enabled,
                checkpoint_identity_ok,
                mount_ok,
                adapter_ok,
                capacity_ok,
                index_ok,
                shm_ok,
            )
        )
        if not service_contract:
            raise ProbeAbort("container loopback/external-checkpoint/L2 contract failed")
        return {
            "container": summary,
            "resolved_cache_environment": effective,
            "production_container_id": production_id,
            "checks": {
                "ownership_label": labels.get(OWNERSHIP_LABEL) == OWNERSHIP_VALUE,
                "not_production_name": summary.get("name") != "/" + PRODUCTION_CONTAINER,
                "not_production_id": summary.get("id") != production_id,
                "running": summary.get("state", {}).get("running") is True,
                "host_loopback": host_ok,
                "inference_port_matches": port_ok,
                "host_network_with_loopback_process_bind": network_ok,
                "lmcache_enabled": cache_enabled,
                "engine_driven_shm": transfer_mode == "engine_driven" and shm_ok,
                "l2_enabled": l2_enabled,
                "checkpoint_identity_present": checkpoint_identity_ok,
                "owned_l2_mounted_exactly_once_rw": mount_ok,
                "fs_native_adapter": adapter_ok,
                "bounded_l2_capacity_and_eviction": capacity_ok,
                "durable_index_inside_owned_l2": index_ok,
            },
            "checkpoint_identity": checkpoint_identity,
            "l2_config": l2_config,
            "capacity_bytes": self.capacity_bytes,
            "cache_base_url": self.cache_base_url,
            "checkpoint_index_host_path": str(self.index_host_path),
        }

    def preflight(self) -> dict[str, Any]:
        paths = self.validate_paths_and_args()
        container = self.container_preflight()
        health = self.require_health("preflight")
        top = self.container_top("preflight")
        settled = self.settle("initial")
        status = settled["status"]
        recurrent = status.get("recurrent_checkpoints") if isinstance(status, dict) else None
        status_contract = bool(
            isinstance(recurrent, dict)
            and recurrent.get("durable_index") is True
            and recurrent.get("store_leases") == 0
            and recurrent.get("retrieve_leases") == 0
            and recurrent.get("pending_generations") == 0
        )
        available_bytes = settled["inventory"].get("filesystem", {}).get(
            "available_bytes"
        )
        storage_headroom = bool(
            isinstance(available_bytes, int)
            and isinstance(self.capacity_bytes, int)
            and available_bytes - self.capacity_bytes
            >= MIN_FILESYSTEM_FREE_RESERVE_BYTES
        )
        payload_file_count = (
            checkpoint_payload_count(settled["inventory"])
            + int(
                settled["inventory"]
                .get("categories", {})
                .get("other_payload", {})
                .get("count", 0)
            )
        )
        fresh = (
            payload_file_count == 0
            and settled["inventory"].get("manifest_count") == 0
            and settled["inventory"].get("reference_analysis_supported") is True
            and not settled["inventory"].get("symlinks")
            and not settled["inventory"].get("errors")
        )
        capacity_status = l2_capacity_status(status)
        if capacity_status is not None and self.capacity_bytes is not None:
            status_capacity_matches = capacity_status.get("total_capacity_bytes") == self.capacity_bytes
        else:
            status_capacity_matches = None
        self.gate("preflight.loopback_only", "pass", {"base_url": self.base_url, "cache_base_url": self.cache_base_url})
        self.gate("preflight.test_container_owned", "pass", container["checks"])
        self.gate("preflight.isolated_l2_namespace", "pass", paths)
        self.gate(
            "preflight.bounded_capacity",
            (
                "pass"
                if (
                    self.capacity_bytes
                    and self.capacity_bytes <= MAX_SAFE_L2_CAPACITY_BYTES
                    and status_capacity_matches is not False
                )
                else "fail"
            ),
            {
                "declared_capacity_bytes": self.capacity_bytes,
                "live_status": capacity_status,
                "status_matches": status_capacity_matches,
            },
        )
        self.gate(
            "preflight.storage_headroom",
            "pass" if storage_headroom else "fail",
            {
                "filesystem_available_bytes": available_bytes,
                "declared_l2_capacity_bytes": self.capacity_bytes,
                "required_post-capacity_reserve_bytes": (
                    MIN_FILESYSTEM_FREE_RESERVE_BYTES
                ),
            },
        )
        self.gate(
            "preflight.external_checkpoint_services_healthy",
            "pass" if health["healthy"] and status_contract else "fail",
            {"health": health, "recurrent_checkpoints": recurrent},
        )
        self.gate(
            "preflight.fresh_namespace",
            "pass" if fresh else "fail",
            {
                "inventory_path": settled["inventory_path"],
                "checkpoint_payload_count": checkpoint_payload_count(
                    settled["inventory"]
                ),
                "all_payload_file_count": payload_file_count,
                "manifest_count": settled["inventory"].get("manifest_count"),
                "sqlite_reference_analysis_supported": settled["inventory"].get(
                    "reference_analysis_supported"
                ),
            },
        )
        if (
            not fresh
            or not status_contract
            or not storage_headroom
            or status_capacity_matches is False
        ):
            raise ProbeAbort(
                "fresh external-checkpoint namespace/safety preflight failed"
            )
        result = {
            "paths": paths,
            "container": container,
            "health": health,
            "processes": top,
            "initial_settlement": settled,
        }
        self.receipt["stages"]["preflight"] = result
        self.persist()
        return result

    def cold_and_replay(
        self,
        initial_settle: dict[str, Any],
        messages: list[dict[str, Any]],
        marker: str,
        calibration: dict[str, Any],
        salt: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        cold = self.stage_request(
            "cold",
            messages,
            marker,
            salt,
            initial_settle,
            tokenization=calibration["tokenization"],
        )
        cold_hits = self.request_cache_hits(cold["request"])
        cold_payload = cold["inventory_delta"]["payload_objects"]
        self.gate(
            "cold.request_correct",
            "pass" if cold["request"]["passed"] else "fail",
            cold["request"]["summary"],
        )
        self.gate(
            "cold.external_miss",
            "pass" if cold_hits == 0 else "fail" if cold_hits is not None else "unavailable",
            {"lmcache_cached_tokens": cold_hits},
        )
        self.gate(
            "cold.payload_published",
            "pass" if cold_payload["created_count"] > 0 and cold_payload["created_logical_bytes"] > 0 else "fail",
            cold_payload,
        )
        self.gate(
            "cold.manifest_observed",
            "pass" if cold["boundary_manifest"].get("available") else "unavailable",
            cold["boundary_manifest"],
        )
        if not cold["request"]["passed"] or cold_hits != 0 or cold_payload["created_count"] == 0:
            raise ProbeAbort("cold request did not establish a correct external-cache miss and payload store")

        replay = self.stage_request(
            "exact-replay",
            messages,
            marker,
            salt,
            cold["after_settlement"],
            tokenization=calibration["tokenization"],
        )
        assessment = self.assess_exact_replay(cold, replay, "replay")
        replay["assessment"] = assessment
        self.receipt["stages"]["cold"] = cold
        self.receipt["stages"]["exact_replay"] = replay
        self.persist()
        return cold, replay

    def assess_exact_replay(
        self, first: dict[str, Any], second: dict[str, Any], prefix: str
    ) -> dict[str, Any]:
        left = first["request"]
        right = second["request"]
        left_identity = left["token_identity"]
        right_identity = right["token_identity"]
        prompt_same = (
            left["http"].get("request_body_sha256") == right["http"].get("request_body_sha256")
            and left_identity.get("returned_prompt_token_ids") == right_identity.get("returned_prompt_token_ids")
        )
        completion_identity_available = bool(
            left_identity.get("available") and right_identity.get("available")
        )
        completion_same = bool(
            completion_identity_available
            and left_identity.get("generated_token_ids") == right_identity.get("generated_token_ids")
            and left["summary"].get("assistant_message_sha256")
            == right["summary"].get("assistant_message_sha256")
        )
        observable_identity_same = prompt_same and completion_same
        payload = second["inventory_delta"]["payload_objects"]
        settled = bool(
            second["after_settlement"].get("settled") is True
            and second["after_settlement"].get("controller_idle_proof_available")
            is True
            and second["after_settlement"].get("controller_idle") is True
        )
        no_payload_write = (
            payload.get("created_count") == 0
            and payload.get("modified_count") == 0
        )
        manifest_comparison = compare_manifest_references(
            first["boundary_manifest"], second["boundary_manifest"]
        )
        persisted_boundary_same = bool(
            manifest_comparison.get("available") is True
            and manifest_comparison.get("same_manifest_key") is True
            and first["boundary_manifest"].get("selected_num_tokens")
            == second["boundary_manifest"].get("selected_num_tokens")
        )
        unchanged_boundary = observable_identity_same and persisted_boundary_same
        hits = self.request_cache_hits(right)
        correct = bool(left.get("passed") and right.get("passed"))
        self.gate(
            f"{prefix}.correct_and_identical",
            "pass" if correct and observable_identity_same else "fail",
            {
                "both_correct": correct,
                "request_body_identical": prompt_same,
                "completion_identity_available": completion_identity_available,
                "completion_token_and_message_identity": completion_same,
                "persisted_checkpoint_boundary_same": persisted_boundary_same,
            },
        )
        prompt_tokens = right["summary"].get("prompt_tokens")
        minimum_hit_tokens = (
            max(1, prompt_tokens - CACHE_CHUNK_TOKENS)
            if isinstance(prompt_tokens, int) and prompt_tokens > 0
            else None
        )
        full_external_hit = bool(
            isinstance(hits, int)
            and isinstance(minimum_hit_tokens, int)
            and minimum_hit_tokens <= hits <= prompt_tokens
        )
        self.gate(
            f"{prefix}.external_hit",
            (
                "pass"
                if full_external_hit
                else "fail"
                if hits is not None and minimum_hit_tokens is not None
                else "unavailable"
            ),
            {
                "lmcache_cached_tokens": hits,
                "prompt_tokens": prompt_tokens,
                "minimum_expected_tokens": minimum_hit_tokens,
                "one_chunk_alignment_tolerance": CACHE_CHUNK_TOKENS,
            },
        )
        if unchanged_boundary and settled:
            dedup_status = "pass" if no_payload_write else "fail"
            dedup_reason = (
                "exact prompt, generated tokens, and persisted checkpoint "
                "boundary are unchanged"
            )
        elif not completion_identity_available:
            dedup_status = "unavailable"
            dedup_reason = "generated token identity is unavailable"
        elif not observable_identity_same:
            dedup_status = "unavailable"
            dedup_reason = (
                "completion history changed; new endpoint/recurrent payloads are "
                "legitimate changed-boundary writes and are not labelled duplicate "
                "historical-attention writes"
            )
        elif not persisted_boundary_same:
            dedup_status = "unavailable"
            dedup_reason = (
                "the SQLite manifest did not prove the same persisted checkpoint "
                "boundary, including the possible unforwarded final sampled token"
            )
        else:
            dedup_status = "unavailable"
            dedup_reason = "filesystem/controller settlement was not proven"
        self.gate(
            f"{prefix}.exact_payload_dedup",
            dedup_status,
            {
                "reason": dedup_reason,
                "payload_delta": payload,
                "metadata_delta": second["inventory_delta"]["metadata_files"],
                "filesystem_growth": second["inventory_delta"]["filesystem_growth"],
                "manifest_comparison": manifest_comparison,
            },
        )
        before_orphans = unreferenced_ids(first["after_settlement"]["inventory"])
        after_orphans = unreferenced_ids(second["after_settlement"]["inventory"])
        orphan_supported = (
            first["after_settlement"]["inventory"].get("reference_analysis_supported") is True
            and second["after_settlement"]["inventory"].get("reference_analysis_supported") is True
        )
        new_orphans = sorted(after_orphans - before_orphans)
        self.gate(
            f"{prefix}.no_new_unreferenced_payloads",
            ("pass" if not new_orphans else "fail") if orphan_supported else "unavailable",
            {
                "analysis_supported": orphan_supported,
                "before_count": len(before_orphans),
                "after_count": len(after_orphans),
                "new_unreferenced_identities": new_orphans,
            },
        )
        return {
            "prompt_same": prompt_same,
            "completion_identity_available": completion_identity_available,
            "completion_same": completion_same,
            "observable_identity_same": observable_identity_same,
            "persisted_boundary_same": persisted_boundary_same,
            "unchanged_boundary": unchanged_boundary,
            "no_payload_object_write": no_payload_write,
            "dedup_gate_status": dedup_status,
            "lmcache_cached_tokens": hits,
            "manifest_comparison": manifest_comparison,
            "new_unreferenced_payloads": new_orphans,
        }

    def ownership_guard(self) -> dict[str, Any]:
        inspected = self.inspect_container(self.container_name)
        if inspected is None:
            raise ProbeAbort("restart target disappeared")
        summary = summarize_container(inspected)
        production = self.inspect_container(PRODUCTION_CONTAINER)
        production_id = production.get("Id") if production else None
        passed = bool(
            summary.get("id") == self.container_id
            and summary.get("name") != "/" + PRODUCTION_CONTAINER
            and summary.get("id") != production_id
            and summary.get("labels", {}).get(OWNERSHIP_LABEL) == OWNERSHIP_VALUE
            and summary.get("state", {}).get("running") is True
        )
        if not passed:
            raise ProbeAbort("restart ownership guard failed immediately before docker restart")
        return {"passed": passed, "container": summary, "production_id": production_id}

    def restart_and_replay(
        self,
        replay: dict[str, Any],
        cold: dict[str, Any],
        messages: list[dict[str, Any]],
        marker: str,
        calibration: dict[str, Any],
        salt: str,
    ) -> dict[str, Any]:
        guard = self.ownership_guard()
        before_top = self.container_top("restart-before")
        before_container = guard["container"]
        command = ["docker", "restart", "--time", "120", str(self.container_id)]
        started = time.time()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.bounded_timeout(RESTART_TIMEOUT_SECONDS),
                check=False,
            )
            restart_record = {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as error:
            restart_record = {
                "command": command,
                "returncode": None,
                "stdout": (error.stdout or "") if isinstance(error.stdout, str) else "",
                "stderr": (error.stderr or "") if isinstance(error.stderr, str) else "",
                "timed_out": True,
                "error": f"TimeoutExpired: {error}",
            }
        restart_record["started_at_epoch"] = started
        restart_record["finished_at_epoch"] = time.time()
        atomic_write_json(self.artifact("restart", "docker-command.json"), restart_record)
        health = self.wait_health("restart")
        if not health.get("healthy"):
            raise ProbeAbort("owned test container did not restore both loopback services after restart")
        after_inspect = self.inspect_container(self.container_name)
        if after_inspect is None:
            raise ProbeAbort("owned test container disappeared after restart")
        after_container = summarize_container(after_inspect)
        after_top = self.container_top("restart-after")
        old_lmcache = {row["pid"] for row in before_top.get("lmcache_processes", [])}
        new_lmcache = {row["pid"] for row in after_top.get("lmcache_processes", [])}
        old_vllm = {row["pid"] for row in before_top.get("vllm_processes", [])}
        new_vllm = {row["pid"] for row in after_top.get("vllm_processes", [])}
        process_identity = {
            "lmcache_pre_pids": sorted(old_lmcache),
            "lmcache_post_pids": sorted(new_lmcache),
            "vllm_pre_pids": sorted(old_vllm),
            "vllm_post_pids": sorted(new_vllm),
            "lmcache_pid_set_replaced": bool(old_lmcache and new_lmcache and old_lmcache.isdisjoint(new_lmcache)),
            "vllm_pid_set_replaced": bool(old_vllm and new_vllm and old_vllm.isdisjoint(new_vllm)),
        }
        restart_passed = bool(
            restart_record.get("returncode") == 0
            and after_container.get("id") == self.container_id
            and after_container.get("state", {}).get("started_at")
            != before_container.get("state", {}).get("started_at")
            and health.get("healthy")
        )
        self.gate("restart.owned_container_guard", "pass", guard)
        self.gate(
            "restart.both_services_restarted",
            "pass" if restart_passed else "fail",
            {
                "docker_restart": restart_record,
                "before_started_at": before_container.get("state", {}).get("started_at"),
                "after_started_at": after_container.get("state", {}).get("started_at"),
                "two_loopback_health_checks": health,
                "process_identity": process_identity,
            },
        )
        if not restart_passed:
            raise ProbeAbort("owned-container restart failed")
        post_restart_idle = self.settle("post-restart-before-replay")
        post = self.stage_request(
            "post-restart-replay",
            messages,
            marker,
            salt,
            post_restart_idle,
            tokenization=calibration["tokenization"],
        )
        assessment = self.assess_exact_replay(cold, post, "restart")
        post["assessment"] = assessment
        manifest_survived = compare_manifest_references(
            replay["boundary_manifest"],
            select_boundary_manifest(
                post_restart_idle["inventory"],
                replay["request"]["token_identity"]["emitted_sequence_tokens"],
            ),
        )
        result = {
            "guard": guard,
            "docker_restart": restart_record,
            "before_container": before_container,
            "after_container": after_container,
            "before_processes": before_top,
            "after_processes": after_top,
            "process_identity": process_identity,
            "health": health,
            "post_restart_idle": post_restart_idle,
            "durable_manifest_before_request": manifest_survived,
            "replay": post,
        }
        self.receipt["stages"]["restart"] = result
        self.persist()
        return result
    def growth(
        self,
        parent_stage: dict[str, Any],
        before_settle: dict[str, Any],
        initial_messages: list[dict[str, Any]],
        salt: str,
    ) -> dict[str, Any]:
        history = list(initial_messages)
        parent_request = parent_stage["request"]
        history.append(
            assistant_history_message(
                parent_request["summary"]["assistant_message"]
            )
        )
        parent_boundary = parent_stage["boundary_manifest"]
        rows: list[dict[str, Any]] = []
        current_settle = before_settle
        for turn in range(1, self.args.growth_turns + 1):
            marker = "R29-GROWTH-" + hashlib.sha256(
                f"{turn}:{salt}".encode()
            ).hexdigest()[:20].upper()
            history.append(
                {
                    "role": "user",
                    "content": (
                        f"Conversation growth turn {turn}. Preserve the prior ledger and continuation. "
                        f"Think at high effort, then return exactly {marker} as visible text."
                    ),
                }
            )
            tokenized = self.tokenize(f"growth-{turn:02d}-tokenize", history)
            prior_emitted = parent_request["token_identity"].get("emitted_sequence_token_ids") or []
            rendered_common = common_prefix_tokens(prior_emitted, tokenized["token_ids"])
            messages_before = sha256_json(history)
            stage = self.stage_request(
                f"growth-{turn:02d}",
                history,
                marker,
                salt,
                current_settle,
                tokenization=tokenized,
            )
            request = stage["request"]
            hits = self.request_cache_hits(request)
            manifest_comparison = compare_manifest_references(
                parent_boundary, stage["boundary_manifest"]
            )
            created_identities = {
                row.get("object_identity")
                for row in stage["inventory_delta"]["payload_objects"]["created"]
                if row.get("category") == "checkpoint_payload"
                and row.get("object_identity")
            }
            previous_references = {
                reference["identity"]
                for database in current_settle["inventory"].get("sqlite", [])
                for reference in database.get("references", [])
            }
            current_references = {
                reference["identity"]
                for database in stage["after_settlement"]["inventory"].get("sqlite", [])
                for reference in database.get("references", [])
            }
            newly_referenced = current_references - previous_references
            stage["continuation"] = {
                "turn": turn,
                "parent_emitted_sequence_sha256": parent_request["token_identity"].get(
                    "emitted_sequence_sha256"
                ),
                "parent_emitted_sequence_tokens": len(prior_emitted),
                "current_prompt_token_ids_sha256": tokenized["token_ids_sha256"],
                "current_prompt_tokens": tokenized["count"],
                "rendered_common_prefix_tokens": rendered_common,
                "messages_before_sha256": messages_before,
                "append_only_message_history": True,
                "lmcache_cached_tokens": hits,
                "rendered_new_suffix_tokens": tokenized["count"] - rendered_common,
                "external_reuse_fraction_of_rendered_common": (
                    hits / rendered_common
                    if isinstance(hits, int) and rendered_common > 0
                    else None
                ),
                "created_payload_logical_bytes": stage["inventory_delta"][
                    "payload_objects"
                ]["created_logical_bytes"],
                "created_payload_bytes_per_new_rendered_token": (
                    stage["inventory_delta"]["payload_objects"][
                        "created_logical_bytes"
                    ]
                    / (tokenized["count"] - rendered_common)
                    if tokenized["count"] > rendered_common
                    else None
                ),
                "manifest_reference_comparison": manifest_comparison,
                "newly_referenced_payload_identity_count": len(newly_referenced),
                "created_newly_referenced_payload_identity_count": len(
                    created_identities & newly_referenced
                ),
                "created_previously_referenced_repopulation_count": len(
                    created_identities & previous_references
                ),
                "created_payload_identities_accounted_by_a_manifest": (
                    created_identities.issubset(current_references)
                    if (
                        current_settle["inventory"].get(
                            "reference_analysis_supported"
                        )
                        is True
                        and stage["after_settlement"]["inventory"].get(
                            "reference_analysis_supported"
                        )
                        is True
                    )
                    else None
                ),
                "write_classification": (
                    "changed-boundary payload additions; never treated as "
                    "exact-replay duplicate writes"
                ),
            }
            rows.append(stage)
            if not request["passed"]:
                raise ProbeAbort(f"conversation growth request {turn} failed correctness")
            history.append(assistant_history_message(request["summary"]["assistant_message"]))
            parent_request = request
            parent_boundary = stage["boundary_manifest"]
            current_settle = stage["after_settlement"]
            self.receipt["stages"]["growth"] = {"turns": rows, "complete": False}
            self.persist()
        all_correct = all(row["request"]["passed"] for row in rows)
        identities = all(
            row["request"]["token_identity"].get("available") is True
            and row["continuation"]["rendered_common_prefix_tokens"] > 0
            for row in rows
        )
        hits_available = all(isinstance(row["continuation"]["lmcache_cached_tokens"], int) for row in rows)
        external_reuse = hits_available and all(
            row["continuation"]["lmcache_cached_tokens"] > 0 for row in rows
        )
        accounting = all(
            row["after_settlement"].get("settled") is True
            and row["after_settlement"]["inventory"].get("reference_analysis_supported") is True
            for row in rows
        )
        schema2_rows = [
            row
            for row in rows
            if row["continuation"]["manifest_reference_comparison"].get("after_schema") == 2
        ]
        schema2_reuse = bool(schema2_rows) and all(
            row["continuation"]["manifest_reference_comparison"].get(
                "shared_reference_count", 0
            )
            > 0
            and row["continuation"][
                "created_payload_identities_accounted_by_a_manifest"
            ]
            is True
            for row in schema2_rows
        )
        self.gate("growth.all_requests_correct", "pass" if all_correct else "fail", {"turns": len(rows)})
        self.gate("growth.continuation_identity_complete", "pass" if identities else "fail", [row["continuation"] for row in rows])
        self.gate(
            "growth.external_reuse_observed",
            "pass" if external_reuse else "fail" if hits_available else "unavailable",
            [row["continuation"]["lmcache_cached_tokens"] for row in rows],
        )
        self.gate(
            "growth.payload_and_manifest_accounting",
            "pass" if accounting else "unavailable",
            {
                "settled_turns": sum(row["after_settlement"].get("settled") is True for row in rows),
                "reference_analysis_turns": sum(row["after_settlement"]["inventory"].get("reference_analysis_supported") is True for row in rows),
                "classification": "each turn intentionally changes the completion boundary",
            },
        )
        self.gate(
            "growth.schema2_shared_payload_identity",
            "pass" if schema2_reuse else "unavailable" if not schema2_rows else "fail",
            {
                "schema2_turns": len(schema2_rows),
                "comparisons": [row["continuation"]["manifest_reference_comparison"] for row in rows],
            },
            required=False,
        )
        result = {
            "complete": True,
            "turns": rows,
            "final_messages": history,
            "final_messages_sha256": sha256_json(history),
            "final_settlement": current_settle,
            "summary": {
                "all_correct": all_correct,
                "continuation_identity_complete": identities,
                "external_reuse_observed": external_reuse if hits_available else None,
                "accounting_complete": accounting,
                "schema2_shared_payload_identity": schema2_reuse if schema2_rows else None,
                "payload_created_bytes_per_turn": [
                    row["inventory_delta"]["payload_objects"]["created_logical_bytes"]
                    for row in rows
                ],
                "payload_created_count_per_turn": [
                    row["inventory_delta"]["payload_objects"]["created_count"]
                    for row in rows
                ],
                "prompt_tokens_per_turn": [row["request"]["summary"]["prompt_tokens"] for row in rows],
            },
        }
        self.receipt["stages"]["growth"] = result
        self.persist()
        return result

    def pressure(
        self,
        before_settle: dict[str, Any],
        messages: list[dict[str, Any]],
        marker: str,
        cold_created_bytes: int,
    ) -> dict[str, Any]:
        if self.capacity_bytes is None or self.capacity_bytes <= 0:
            raise ProbeAbort("bounded pressure requires a positive declared L2 capacity")
        if cold_created_bytes <= 0:
            raise ProbeAbort("bounded pressure cannot derive a per-request payload budget")
        current = before_settle
        available = current["inventory"].get("filesystem", {}).get(
            "available_bytes"
        )
        if (
            not isinstance(available, int)
            or available < MIN_FILESYSTEM_FREE_RESERVE_BYTES
        ):
            raise ProbeAbort(
                "filesystem free-space reserve is unavailable before pressure"
            )
        current_bytes = checkpoint_payload_bytes(current["inventory"])
        target_bytes = int(self.capacity_bytes * self.trigger_watermark)
        estimated = max(0, target_bytes - current_bytes)
        estimated_requests = max(3, math.ceil(estimated / cold_created_bytes) + 2)
        planned = min(MAX_PRESSURE_REQUESTS, estimated_requests)
        can_reach_estimated = estimated_requests <= MAX_PRESSURE_REQUESTS
        rows: list[dict[str, Any]] = []
        observed_eviction = False
        reached_watermark = False
        cumulative_created = 0
        latest_salt: str | None = None
        for index in range(planned):
            self.check_budget(1)
            latest_salt = "r29-pressure-" + hashlib.sha256(
                f"{self.model}:{self.args.context_tokens}:{index}".encode()
            ).hexdigest()[:24]
            stage = self.stage_request(
                f"pressure-{index:02d}",
                messages,
                marker,
                latest_salt,
                current,
            )
            if not stage["request"]["passed"]:
                raise ProbeAbort(f"pressure request {index} failed correctness")
            payload = stage["inventory_delta"]["payload_objects"]
            cumulative_created += int(payload["created_logical_bytes"])
            capacity = l2_capacity_status(stage["after_settlement"]["status"])
            usage = capacity.get("current_usage") if isinstance(capacity, dict) else None
            reached_watermark = reached_watermark or (
                isinstance(usage, (int, float))
                and float(usage) >= self.trigger_watermark
            )
            observed_eviction = observed_eviction or payload["deleted_count"] > 0
            stage["pressure"] = {
                "index": index,
                "unique_cache_salt": latest_salt,
                "lmcache_cached_tokens": self.request_cache_hits(stage["request"]),
                "capacity_status": capacity,
                "cumulative_created_payload_bytes": cumulative_created,
                "eviction_observed_from_deleted_payloads": payload["deleted_count"] > 0,
            }
            rows.append(stage)
            current = stage["after_settlement"]
            payload_now = checkpoint_payload_bytes(current["inventory"])
            if payload_now > MAX_OBSERVED_PAYLOAD_BYTES:
                raise ProbeAbort("observed payload namespace exceeded the hard disk budget")
            available = current["inventory"].get("filesystem", {}).get(
                "available_bytes"
            )
            if (
                not isinstance(available, int)
                or available < MIN_FILESYSTEM_FREE_RESERVE_BYTES
            ):
                raise ProbeAbort(
                    "filesystem free-space reserve was exhausted during pressure"
                )
            if observed_eviction or reached_watermark:
                break
            self.receipt["stages"]["pressure"] = {"complete": False, "requests": rows}
            self.persist()
        final_capacity = l2_capacity_status(current["status"])
        if isinstance(final_capacity, dict):
            final_used = final_capacity.get("total_bytes_used")
            final_declared = final_capacity.get("total_capacity_bytes")
            capacity_safe = bool(
                isinstance(final_used, int)
                and isinstance(final_declared, int)
                and 0 <= final_used <= final_declared == self.capacity_bytes
            )
        else:
            final_used = checkpoint_payload_bytes(current["inventory"])
            final_declared = self.capacity_bytes
            capacity_safe = final_used <= final_declared
        pressure_observed = observed_eviction or reached_watermark
        final_idle, final_idle_detail = service_idle(current["status"])
        self.gate(
            "pressure.budget_respected",
            "pass",
            {
                "planned": planned,
                "executed": len(rows),
                "estimated_requests_to_watermark": estimated_requests,
                "estimate_within_request_bound": can_reach_estimated,
                "cumulative_created_payload_bytes": cumulative_created,
                "hard_payload_budget": MAX_OBSERVED_PAYLOAD_BYTES,
            },
        )
        pressure_hits = [
            row["pressure"]["lmcache_cached_tokens"] for row in rows
        ]
        pressure_hits_available = all(
            isinstance(value, int) for value in pressure_hits
        )
        self.gate(
            "pressure.unique_namespaces_are_cold",
            (
                "pass"
                if pressure_hits_available
                and all(value == 0 for value in pressure_hits)
                else "fail"
                if pressure_hits_available
                else "unavailable"
            ),
            {
                "lmcache_cached_tokens_per_request": pressure_hits,
                "method": (
                    "each pressure request uses a distinct cache_salt and resets "
                    "the local prefix cache"
                ),
            },
        )
        self.gate(
            "pressure.capacity_or_eviction_observed",
            "pass" if pressure_observed else "fail" if can_reach_estimated else "unavailable",
            {
                "reached_watermark": reached_watermark,
                "payload_deletion_observed": observed_eviction,
                "final_capacity_status": final_capacity,
            },
        )
        self.gate(
            "pressure.capacity_accounting_safe",
            "pass" if capacity_safe else "fail",
            {
                "used_bytes": final_used,
                "capacity_bytes": final_declared,
                "filesystem_checkpoint_payload_bytes": checkpoint_payload_bytes(current["inventory"]),
            },
        )
        self.gate(
            "pressure.services_healthy_and_idle",
            (
                "pass"
                if final_idle is True
                else "fail"
                if final_idle is False
                else "unavailable"
            ),
            final_idle_detail,
        )
        result = {
            "complete": True,
            "declared_capacity_bytes": self.capacity_bytes,
            "target_watermark_bytes": target_bytes,
            "trigger_watermark": self.trigger_watermark,
            "starting_checkpoint_payload_bytes": current_bytes,
            "cold_payload_unit_bytes": cold_created_bytes,
            "estimated_requests_to_watermark": estimated_requests,
            "planned_requests": planned,
            "requests": rows,
            "observed_eviction": observed_eviction,
            "reached_watermark": reached_watermark,
            "final_capacity_status": final_capacity,
            "final_settlement": current,
            "latest_pressure_salt": latest_salt,
        }
        self.receipt["stages"]["pressure"] = result
        self.persist()
        return result

    def cancellation_exchange(
        self,
        label: str,
        messages: list[dict[str, Any]],
        cache_salt: str,
    ) -> dict[str, Any]:
        if not self.base_url.startswith("http://127.0.0.1:"):
            raise ProbeAbort("cancellation target is not loopback")
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": CANCELLATION_MAX_TOKENS,
            "temperature": SAMPLING["temperature"],
            "top_p": SAMPLING["top_p"],
            "seed": SAMPLING["seed"],
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            "cache_salt": cache_salt,
            "kv_transfer_params": {"cached_token_stats": True},
            "return_token_ids": True,
        }
        raw_request = canonical_bytes(payload)
        request_path = self.artifact(label, "request.json")
        response_path = self.artifact(label, "response.body")
        atomic_write_bytes(request_path, raw_request)
        parsed_url = urllib.parse.urlsplit(self.base_url)
        connection = http.client.HTTPConnection(
            parsed_url.hostname,
            parsed_url.port,
            timeout=self.bounded_timeout(REQUEST_TIMEOUT_SECONDS),
        )
        started = time.perf_counter()
        raw_parts: list[bytes] = []
        status: int | None = None
        headers: dict[str, str] = {}
        error: str | None = None
        event_count = 0
        data_event_observed = False
        done_observed = False
        try:
            connection.request(
                "POST",
                CHAT_PATH,
                body=raw_request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "Connection": "close",
                    "X-Request-Id": safe_label(label),
                },
            )
            response = connection.getresponse()
            status = int(response.status)
            headers = dict(response.getheaders())
            while sum(len(part) for part in raw_parts) < 256 * 1024 and event_count < 16:
                line = response.readline(64 * 1024)
                if not line:
                    break
                raw_parts.append(line)
                stripped = line.strip()
                if not stripped.startswith(b"data:"):
                    continue
                data = stripped[5:].strip()
                if data == b"[DONE]":
                    done_observed = True
                    break
                try:
                    event = json.loads(data)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                event_count += 1
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") if isinstance(choice, dict) else None
                    if isinstance(delta, dict) and any(
                        delta.get(key)
                        for key in ("content", "reasoning_content", "reasoning", "tool_calls")
                    ):
                        data_event_observed = True
                        break
                if data_event_observed:
                    break
        except (OSError, TimeoutError, http.client.HTTPException) as caught:
            error = f"{type(caught).__name__}: {caught}"
        finally:
            connection.close()
        raw_response = b"".join(raw_parts)
        atomic_write_bytes(response_path, raw_response)
        return {
            "label": label,
            "url": self.base_url + CHAT_PATH,
            "status": status,
            "response_headers": headers,
            "request_body_path": str(request_path),
            "request_body_sha256": sha256_bytes(raw_request),
            "request_body_bytes": len(raw_request),
            "response_body_path": str(response_path),
            "response_body_sha256": sha256_bytes(raw_response),
            "response_body_bytes": len(raw_response),
            "sse_events_read": event_count,
            "first_generated_data_event_observed": data_event_observed,
            "done_observed": done_observed,
            "client_closed_early": data_event_observed and not done_observed,
            "elapsed_seconds_before_close": time.perf_counter() - started,
            "error": error,
            "sampling": {
                **SAMPLING,
                "max_tokens": CANCELLATION_MAX_TOKENS,
                "ignore_eos": True,
            },
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            "cache_salt": cache_salt,
        }

    def cancellation(
        self,
        before_settle: dict[str, Any],
        messages: list[dict[str, Any]],
        cache_salt: str,
    ) -> dict[str, Any]:
        self.check_budget(1)
        self.require_health("cancellation-preflight")
        self.reset_local("cancellation-reset-local")
        tokenized = self.tokenize("cancellation-tokenize", messages)
        self.check_budget(1, tokenized["count"])
        self.request_count += 1
        self.prompt_tokens_consumed += tokenized["count"]
        before_metrics = self.metrics("cancellation-metrics-before")
        monitor_rows: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self.cancellation_exchange, "cancellation", messages, cache_salt
            )
            while not future.done():
                status = self.cache_status("cancellation-monitor", capture=False).get("status")
                recurrent = status.get("recurrent_checkpoints") if isinstance(status, dict) else None
                capacity = l2_capacity_status(status)
                monitor_rows.append(
                    {
                        "at": utc_now(),
                        "recurrent_checkpoints": recurrent,
                        "l2_capacity": capacity,
                    }
                )
                if len(monitor_rows) >= 2000:
                    raise ProbeAbort("cancellation status monitor exceeded its sample bound")
                time.sleep(0.05)
            exchange = future.result()
        if not exchange.get("client_closed_early"):
            raise ProbeAbort("stream cancellation did not close after a generated data event")
        abort_deadline = time.monotonic() + 60
        after_metrics = self.metrics("cancellation-metrics-after")
        delta = selected_metric_delta(before_metrics["parsed"], after_metrics["parsed"])
        while (
            delta.get("vllm:request_success_total:abort") in (None, 0.0)
            and time.monotonic() < abort_deadline
        ):
            time.sleep(1)
            after_metrics = self.metrics("cancellation-metrics-after-retry")
            delta = selected_metric_delta(before_metrics["parsed"], after_metrics["parsed"])
        after_settle = self.settle("cancellation-settled")
        file_delta = inventory_delta(before_settle["inventory"], after_settle["inventory"])
        before_orphans = unreferenced_ids(before_settle["inventory"])
        after_orphans = unreferenced_ids(after_settle["inventory"])
        orphan_supported = (
            before_settle["inventory"].get("reference_analysis_supported") is True
            and after_settle["inventory"].get("reference_analysis_supported") is True
        )
        new_orphans = sorted(after_orphans - before_orphans)
        final_idle, idle_detail = service_idle(after_settle["status"])
        abort_delta = delta.get("vllm:request_success_total:abort")
        abort_observed = isinstance(abort_delta, (int, float)) and abort_delta >= 1
        retrieve_lease_observed = any(
            isinstance(row.get("recurrent_checkpoints"), dict)
            and int(row["recurrent_checkpoints"].get("retrieve_leases") or 0) > 0
            for row in monitor_rows
        )
        health = self.require_health("cancellation-final")
        self.gate(
            "cancellation.server_abort_observed",
            "pass" if abort_observed else "unavailable",
            {"abort_counter_delta": abort_delta, "client_exchange": exchange},
        )
        self.gate(
            "cancellation.checkpoint_leases_drained",
            "pass" if final_idle is True else "fail" if final_idle is False else "unavailable",
            idle_detail,
        )
        self.gate(
            "cancellation.no_failed_generation_orphans",
            ("pass" if not new_orphans else "fail") if orphan_supported else "unavailable",
            {
                "analysis_supported": orphan_supported,
                "new_unreferenced_payload_identities": new_orphans,
                "before_unreferenced_count": len(before_orphans),
                "after_unreferenced_count": len(after_orphans),
                "payload_delta": file_delta["payload_objects"],
            },
        )
        self.gate(
            "cancellation.live_retrieve_observed_under_pressure",
            "pass" if retrieve_lease_observed else "unavailable",
            {
                "retrieve_lease_observed": retrieve_lease_observed,
                "monitor_samples": len(monitor_rows),
                "reason_if_unavailable": (
                    None
                    if retrieve_lease_observed
                    else "the HTTP status poll did not catch the short-lived SHM retrieve lease"
                ),
            },
            required=False,
        )
        self.gate("cancellation.services_remain_healthy", "pass" if health["healthy"] else "fail", health)
        result = {
            "tokenization": tokenized,
            "exchange": exchange,
            "metrics": {
                "before": before_metrics,
                "after": after_metrics,
                "delta": delta,
            },
            "status_monitor": monitor_rows,
            "after_settlement": after_settle,
            "inventory_delta": file_delta,
            "new_unreferenced_payload_identities": new_orphans,
            "retrieve_lease_observed": retrieve_lease_observed,
            "final_health": health,
        }
        self.receipt["stages"]["cancellation"] = result
        self.persist()
        return result

    def record_byte_evidence_limit(self, final_status: object) -> None:
        matches: list[dict[str, Any]] = []

        def walk(value: object, path: str) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    child = f"{path}.{key}" if path else str(key)
                    if re.search(r"checksum|checked_bytes|byte_equal|all_rank", str(key), re.IGNORECASE):
                        matches.append({"path": child, "value": item})
                    walk(item, child)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, f"{path}[{index}]")

        walk(final_status, "")
        detail = {
            "status": "unavailable",
            "http_status_fields_matching_byte_evidence_terms": matches,
            "reason": (
                "OpenAI responses expose visible output and token IDs, not checkpoint page bytes. "
                "LMCache /cache/checksums requires engine instance/block IDs and checks registered KV tensors; "
                "the serving API does not expose a safe request-to-physical-block map for recurrent bundles."
            ),
            "no_inference": "matching output/token IDs is not byte equality of all-rank KV/SHM payloads",
            "source_grounded_component_commands_for_main": [
                "python -m pytest -q tests/v1/multiprocess/test_checkpoint_storage.py::test_checkpoint_rpc_roundtrip_uses_metadata_and_shared_bytes",
                "python -m pytest -q tests/v1/multiprocess/test_checkpoint_storage.py::test_shm_roundtrip_all_ranks_groups_and_read_lease_lifetime",
                "python -m pytest -q tests/v1/multiprocess/test_checkpoint_storage.py::test_filesystem_restore_after_directory_and_storage_restart",
                "python -m pytest -q tests/v1/multiprocess/test_checkpoint_storage.py::test_payload_capacity_miss_preserves_a_pinned_checkpoint",
                "python -m pytest -q tests/v1/multiprocess/test_checkpoint_storage.py::test_cancel_pending_lookup_drains_locks_without_invalidating_payload",
                "python -m pytest -q tests/v1/multiprocess/test_checkpoint_storage.py::test_sustained_checkpoint_stores_reclaim_capacity_without_losing_a_read",
            ],
            "source_contract": {
                "checkpoint_rpc": "CheckpointModule CHECKPOINT_BEGIN/PREPARE_STORE/FINISH_STORE/FIND/BEGIN_RETRIEVE/POLL/FINISH",
                "shm_copy": "CheckpointPageCopier copies uint8 page_pool rows to/from CheckpointSlots and synchronizes its CUDA stream",
                "filesystem_restore": "real FS/FSNative adapter restart test compares every rank/group/page byte in SHM",
            },
        }
        self.receipt["limitations"].append(detail)
        self.gate(
            "evidence.all_rank_payload_byte_equality",
            "unavailable",
            detail,
            required=False,
        )

    def finalize(self) -> int:
        required = [row for row in self.gates if row.get("required")]
        passed = bool(required) and all(row.get("status") == "pass" for row in required)
        self.receipt["complete"] = True
        self.receipt["passed"] = passed
        self.receipt["finished_at"] = utc_now()
        self.receipt["elapsed_seconds"] = time.monotonic() - self.started_monotonic
        self.receipt["request_count"] = self.request_count
        self.receipt["prompt_tokens_consumed"] = self.prompt_tokens_consumed
        self.receipt["gate_summary"] = {
            status: sum(row.get("status") == status for row in self.gates)
            for status in ("pass", "fail", "unavailable", "not_applicable")
        }
        self.receipt["required_gate_count"] = len(required)
        self.persist()
        print(
            json.dumps(
                {
                    "receipt": str(self.receipt_path),
                    "complete": True,
                    "passed": passed,
                    "request_count": self.request_count,
                    "gate_summary": self.receipt["gate_summary"],
                },
                indent=2,
            )
        )
        return 0 if passed else 1

    def run(self) -> int:
        # Validate every path before creating the output directory. In
        # particular, an invalid output path under a cache tree must remain
        # completely untouched.
        self.validate_paths_and_args()
        self.prepare_output()
        try:
            preflight = self.preflight()
            messages, marker, calibration = self.calibrate_prompt()
            self.receipt["stages"]["prompt_calibration"] = calibration
            salt = "r29-lifecycle-" + hashlib.sha256(
                canonical_bytes(
                    {
                        "schema": SCHEMA,
                        "model": self.model,
                        "context_tokens": self.args.context_tokens,
                        "messages": messages,
                    }
                )
            ).hexdigest()[:24]
            cold, replay = self.cold_and_replay(
                preflight["initial_settlement"], messages, marker, calibration, salt
            )
            restart = self.restart_and_replay(
                replay, cold, messages, marker, calibration, salt
            )
            growth = self.growth(
                restart["replay"],
                restart["replay"]["after_settlement"],
                messages,
                salt,
            )
            pressure = self.pressure(
                growth["final_settlement"],
                messages,
                marker,
                cold["inventory_delta"]["payload_objects"]["created_logical_bytes"],
            )
            cancellation_salt = pressure.get("latest_pressure_salt") or salt
            cancellation = self.cancellation(
                pressure["final_settlement"], messages, cancellation_salt
            )
            exact_orphans = set(
                replay.get("assessment", {}).get("new_unreferenced_payloads", [])
            )
            final_orphans = unreferenced_ids(
                cancellation["after_settlement"]["inventory"]
            )
            final_inventory = cancellation["after_settlement"]["inventory"]
            final_reference_supported = (
                final_inventory.get("reference_analysis_supported") is True
            )
            self.gate(
                "orphan.final_unreferenced_payload_inventory",
                (
                    "pass"
                    if final_reference_supported and not final_orphans
                    else "fail"
                    if final_reference_supported
                    else "unavailable"
                ),
                {
                    "analysis_supported": final_reference_supported,
                    "unreferenced_checkpoint_payload_identities": sorted(
                        final_orphans
                    ),
                    "unreferenced_count": len(final_orphans),
                    "missing_referenced_payload_count": len(
                        final_inventory.get(
                            "missing_referenced_checkpoint_payloads", []
                        )
                    ),
                    "missing_reference_interpretation": (
                        "capacity-evicted payloads make their manifests safe "
                        "miss candidates; they are not unreferenced orphans"
                    ),
                },
            )
            if exact_orphans:
                reclaimed = exact_orphans - final_orphans
                self.gate(
                    "pressure.superseded_payload_reclamation",
                    "pass" if reclaimed == exact_orphans else "fail",
                    {
                        "superseded_after_exact_replay": sorted(exact_orphans),
                        "reclaimed_by_final_pressure_settlement": sorted(reclaimed),
                        "still_present": sorted(exact_orphans & final_orphans),
                    },
                    required=False,
                )
            else:
                self.gate(
                    "pressure.superseded_payload_reclamation",
                    "not_applicable",
                    {
                        "reason": "exact replay created no unreferenced payload generation",
                        "final_unreferenced_count": len(final_orphans),
                    },
                    required=False,
                )
            self.record_byte_evidence_limit(cancellation["after_settlement"]["status"])
        except (ProbeAbort, ValueError) as error:
            self.receipt["fatal_error"] = f"{type(error).__name__}: {error}"
            self.receipt["complete"] = False
            self.receipt["passed"] = False
            self.receipt["stopped_safely"] = True
            self.receipt["finished_at"] = utc_now()
            self.receipt["elapsed_seconds"] = time.monotonic() - self.started_monotonic
            self.persist()
            print(json.dumps({"receipt": str(self.receipt_path), "fatal_error": self.receipt["fatal_error"]}, indent=2), file=sys.stderr)
            return 2
        except KeyboardInterrupt as error:
            self.receipt["fatal_error"] = f"KeyboardInterrupt: {error}"
            self.receipt["complete"] = False
            self.receipt["passed"] = False
            self.receipt["stopped_safely"] = True
            self.receipt["finished_at"] = utc_now()
            self.receipt["elapsed_seconds"] = time.monotonic() - self.started_monotonic
            self.persist()
            print(json.dumps({"receipt": str(self.receipt_path), "fatal_error": self.receipt["fatal_error"]}, indent=2), file=sys.stderr)
            return 130
        except Exception as error:
            self.receipt["fatal_error"] = (
                f"unexpected probe harness exception: {type(error).__name__}: {error}"
            )
            self.receipt["complete"] = False
            self.receipt["passed"] = False
            self.receipt["stopped_safely"] = True
            self.receipt["unexpected_harness_exception"] = True
            self.receipt["finished_at"] = utc_now()
            self.receipt["elapsed_seconds"] = (
                time.monotonic() - self.started_monotonic
            )
            self.persist()
            print(
                json.dumps(
                    {
                        "receipt": str(self.receipt_path),
                        "fatal_error": self.receipt["fatal_error"],
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 2
        return self.finalize()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded R29 external-cache cold/replay/restart/growth/pressure/cancellation probe. "
            "The test container must already be running and labelled field-lab.battery=r26."
        )
    )
    parser.add_argument("--base-url", required=True, help="Loopback vLLM origin, for example http://127.0.0.1:5002")
    parser.add_argument("--model", required=True, help="Exact served model name")
    parser.add_argument("--container", required=True, help="Already-running owned test container name")
    parser.add_argument("--l2-dir", type=Path, required=True, help="Absolute isolated host L2 namespace under /mnt/2king/lmcache-r26-battery")
    parser.add_argument("--output-dir", type=Path, required=True, help="Absolute absent or empty evidence directory")
    parser.add_argument("--context-tokens", type=int, default=32768)
    parser.add_argument("--growth-turns", type=int, default=6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        probe = LifecycleProbe(args)
        code = probe.run()
    except (ValueError, OSError) as error:
        print(f"cache_lifecycle_probe: {type(error).__name__}: {error}", file=sys.stderr)
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()
