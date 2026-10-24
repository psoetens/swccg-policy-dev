#!/usr/bin/env bash
# Local build prerequisites for native/ (no system-wide install): native/cuda-home = a CUDA_HOME made of symlinks to the project
# venv's nvidia wheels (runtime, cuBLAS, cuSPARSE, ... headers and libcudart), Triton's cuda.h and the CCCL headers (pip wheel
# nvidia-cuda-cccl-cu12 into native/pydeps). Needed by native/build.sh (and by AOTInductor, see the plan). Then: native/build.sh
set -euo pipefail
cd "$(dirname "$0")/.."
V=.venv/lib/python3.12/site-packages
H=native/cuda-home
mkdir -p "$H/include" "$H/lib64/stubs"
[ -d native/pydeps/nvidia/cuda_cccl ] || uv pip install --target native/pydeps "nvidia-cuda-cccl-cu12==12.8.*"
for d in "$V"/nvidia/*/include "$V"/triton/backends/nvidia/include native/pydeps/nvidia/cuda_cccl/include; do
  for f in "$d"/*; do
    b=$(basename "$f"); [ "$b" = __init__.py ] && continue
    [ -e "$H/include/$b" ] || ln -s "$PWD/$f" "$H/include/$b"
  done
done
for f in "$V"/nvidia/cuda_runtime/lib/libcudart.so*; do ln -sfn "$PWD/$f" "$H/lib64/$(basename "$f")"; done
ln -sfn "$PWD/$V/nvidia/cuda_runtime/lib/libcudart.so.12" "$H/lib64/libcudart.so"
ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$H/lib64/libcuda.so"
ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$H/lib64/stubs/libcuda.so"
echo "CUDA_HOME=$PWD/$H"
