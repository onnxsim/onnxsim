#!/bin/bash
# Run kernels.elf on hexagon-sim (timing model, v68 pipeline as the V65 proxy). Usage: ./run_sim.sh <op> <args...>
set -euo pipefail
T="${HEXAGON_TOOLS:-$HOME/.cache/hexagon-oa-19/Tools}"
cd "$(dirname "$0")"
LD_LIBRARY_PATH="${NCURSES_SHIM:-$HOME/.cache/ncshim}:${LD_LIBRARY_PATH:-}" \
  "$T/bin/hexagon-sim" -mv68 --timing --quiet kernels.elf -- "$@"
