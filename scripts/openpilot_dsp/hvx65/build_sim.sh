#!/bin/bash
# Build kernels.c for V65 (upstream clang: the Qualcomm 19.x toolchain dropped v65) and link it with the
# open-access toolchain's v68 standalone runtime so it runs on hexagon-sim (which models v68+ only).
# Usage: HEXAGON_TOOLS=<.../Tools> ./build_sim.sh [out.elf]
set -euo pipefail
T="${HEXAGON_TOOLS:-$HOME/.cache/hexagon-oa-19/Tools}"
CC="${CLANG:-clang-19}"
OUT="${1:-kernels.elf}"
cd "$(dirname "$0")"
"$CC" --target=hexagon -mcpu=hexagonv65 -mhvx=v65 -mhvx-length=128b -O2 -Wall -Werror \
  -I"$T/target/hexagon/include" -c kernels.c -o kernels_v65.o
# every HVX instruction in the object must exist on V65: the assembler enforced it; record the ISA flag
"$T/bin/hexagon-clang" -mv68 kernels_v65.o -o "$OUT" 2>&1 | grep -v "deprecated for input file" || true
echo "built $OUT"
