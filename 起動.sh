#!/bin/bash
set -e
cd -- "$(dirname -- "$(readlink -f -- "$0")")"
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
printf '\n比較版: %s\n場所: %s\n\n' 'v18 行先指定後の発進待ち修正' "$PWD"
if [ "$#" -eq 0 ]; then set -- menu; fi
has_motion_mode=false
for arg in "$@"; do
  case "$arg" in --motion-mode|--motion-mode=*) has_motion_mode=true ;; esac
done
if [ "$has_motion_mode" = false ]; then set -- "$@" --motion-mode simultaneous; fi
exec /usr/bin/python3 "$PWD/run.py" "$@"
