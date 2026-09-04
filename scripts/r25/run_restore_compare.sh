#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/josh/omp-workspace/drock-lmcache/r24-battery
CODE=SILVER-CEDAR-90024

docker rm -f r24-replay >/dev/null 2>&1 || true
"$ROOT/r24_dflash_replay_server.sh"
python3 "$ROOT/replay_one.py" "$CODE" "$ROOT/r24-restore-one.json"
docker logs r24-replay > "$ROOT/r24-restore-one.docker.log" 2>&1
docker rm -f r24-replay >/dev/null
sleep 5
"$ROOT/r25_replay_server.sh"
python3 "$ROOT/replay_one.py" "$CODE" "$ROOT/r25-restore-one.json"
docker logs r24-replay > "$ROOT/r25-restore-one.docker.log" 2>&1
docker rm -f r24-replay >/dev/null
echo RESTORE_COMPARE_DONE
