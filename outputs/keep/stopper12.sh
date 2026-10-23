#!/bin/bash
# Stops this pod once arm 12 and its gallery have ended (or on a Traceback, or after 7.5 hours) and the laptop has
# fetched the results, or 45 minutes after that. The results are on the network volume, so a stop loses nothing.
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
for i in $(seq 2700); do grep -qE "GALLERY_READY|Traceback" /workspace/arm12.log 2>/dev/null && break; sleep 10; done
for i in $(seq 270); do [ -f /workspace/fetched ] && break; sleep 10; done
echo "stopping $RUNPOD_POD_ID at $(date -u +%T)"
runpodctl stop pod "$RUNPOD_POD_ID"
