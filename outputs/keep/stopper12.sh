#!/bin/bash
# Stops this pod once arm 12 has ended and its results were fetched, or 30 minutes after the end, or after 8 hours in all.
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
for i in $(seq 2880); do grep -qE "ARM12B_DONE|Traceback" /workspace/arm12.log 2>/dev/null && break; sleep 10; done
for i in $(seq 180); do [ -f /workspace/fetched ] && break; sleep 10; done
echo "stopping $RUNPOD_POD_ID at $(date -u +%T)"
runpodctl stop pod "$RUNPOD_POD_ID"
