#!/bin/bash
# Replaces stopper12.sh once the film is queued: stops the pod after the film (or a Traceback, or 11 hours) and the
# laptop's fetch, or 45 minutes after. Everything is on the network volume either way.
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
for i in $(seq 3960); do grep -qE "FILM_DONE|Traceback" /workspace/film9.log 2>/dev/null && break; grep -q Traceback /workspace/arm12.log 2>/dev/null && break; sleep 10; done
for i in $(seq 270); do [ -f /workspace/fetched_film ] && break; sleep 10; done
echo "stopping $RUNPOD_POD_ID at $(date -u +%T)"
runpodctl stop pod "$RUNPOD_POD_ID"
