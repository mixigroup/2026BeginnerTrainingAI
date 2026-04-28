#!/bin/bash
set -euo pipefail

# メタデータからインスタンス名とゾーン取得
MD="http://metadata.google.internal/computeMetadata/v1"
H="Metadata-Flavor: Google"
INSTANCE_NAME=$(curl -fs -H "$H" "$MD/instance/name")
ZONE_FULL=$(curl -fs -H "$H" "$MD/instance/zone")
ZONE="${ZONE_FULL##*/}"

STATE_FILE="/tmp/idle_count"

IDLE_COUNT=48 # 10分ごとに実行され、48回連続=8時間アイドルなら停止

# 前回までのカウント
idle_count=$(cat "$STATE_FILE" 2>/dev/null || echo 0)

# GPUがあるか確認
if command -v nvidia-smi >/dev/null 2>&1; then
    # GPUプロセス数を確認
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l || echo 0)
    if [ "$procs" -eq 0 ]; then
        idle=1
    else
        idle=0
    fi
else
    # CPU平均使用率を確認 (直近1分の loadavg を CPU数で割る)
    cpu_load=$(awk '{print $1}' /proc/loadavg)
    cpu_cores=$(nproc)
    # awkで比較 (10%未満を idle とみなす)
    cmp=$(awk -v load="$cpu_load" -v cores="$cpu_cores" 'BEGIN { if ((load/cores) < 0.10) print 1; else print 0 }')
    idle=$cmp
fi

# idle判定
if [ "$idle" -eq 1 ]; then
    idle_count=$((idle_count + 1))
else
    idle_count=0
fi
echo "$idle_count" >"$STATE_FILE"

# shutdown
if [ "$idle_count" -ge $IDLE_COUNT ]; then
    rm -f "$STATE_FILE"
    gcloud workbench instances stop "$INSTANCE_NAME" --location="$ZONE" -q
fi
