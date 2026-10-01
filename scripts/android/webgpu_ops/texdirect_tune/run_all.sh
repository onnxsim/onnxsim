#!/usr/bin/env bash
PY=/mnt/data/cache/claude-work/ortenv/bin/python
cd "$(dirname "$0")"
for job in "yolo26n 1 64" "yolo11n 1 256" "yolo11n 2 128" "yolo26n 2 128" "resnet50 1 512" "resnet50 2 256"; do
  set -- $job
  $PY tune_class.py $1 $2 $3 > log_class_$1_m$2_c$3.txt 2>&1
done
echo ALLDONE > alldone.txt
