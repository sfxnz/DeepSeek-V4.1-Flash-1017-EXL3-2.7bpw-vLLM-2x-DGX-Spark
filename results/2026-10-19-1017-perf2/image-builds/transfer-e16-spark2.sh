#!/usr/bin/env bash
# spark2 e16-rows is TRANSFERRED from spark1, not built: spark2 has no default route (its chain A build failed at
# `git clone https://github.com/vcruz305/vllm-exl3.git`: Could not resolve host: github.com). Main agent's option (a):
# docker save on spark1 | ssh spark2 (10.100.8.2) docker load; abort if spark2's free space drops under 40 GB.
LD=/home/sfxnz/projects/data/dsv41-1017/perf2/release/build-logs; LOG=$LD/transfer-e16-spark2.log
IMG=dsv41-flash-exl3-sm121:e16-rows
{ echo "start $(date -u +%FT%TZ) spark1 $IMG $(docker image inspect $IMG --format '{{.Id}}') size $(docker image inspect $IMG --format '{{.Size}}') B"
  echo "spark2 before: $IMG $(ssh -o BatchMode=yes spark2 docker image inspect $IMG --format '{{.Id}}'); df: $(ssh -o BatchMode=yes spark2 df -h / | tail -1)"; } >> $LOG
( docker save $IMG | ssh -o BatchMode=yes spark2 docker load ) >> $LOG 2>&1 &
P=$!
while kill -0 $P 2>/dev/null; do
  a=$(ssh -o BatchMode=yes spark2 "df --output=avail -B1G / | tail -1" | tr -d ' G')
  echo "$(date -u +%T) spark2 avail ${a}G" >> $LOG
  if [ "${a:-999}" -lt 40 ]; then echo "ABORT $(date -u +%T): spark2 free < 40 GB" >> $LOG; pkill -P $P; kill $P; break; fi
  sleep 3
done
wait $P; echo "pipeline rc=$? end $(date -u +%FT%TZ)" >> $LOG
echo "spark2 after: df: $(ssh -o BatchMode=yes spark2 df -h / | tail -1)" >> $LOG
