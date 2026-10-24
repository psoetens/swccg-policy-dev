#!/usr/bin/env bash
# Build native/build/libswccgpolicy.so (JNI, libtorch + CUDA graphs) against the libtorch of the project venv. No CUDA toolkit
# needed (no .cu sources): CUDA headers come from native/cuda-home (symlinks into the venv's nvidia wheels, see the plan).
set -euo pipefail
cd "$(dirname "$0")"
TORCH=$(cd .. && .venv/bin/python -c "import torch, os; print(os.path.dirname(torch.__file__))")
JDK=${JDK:-${JAVA_HOME:?JAVA_HOME or JDK must name a JDK 21 (source scripts/machine_env.sh)}}
NATIVE_OUT="${SWCCG_NATIVE_OUT:-build}"
mkdir -p "$NATIVE_OUT"
g++ -O2 -std=c++17 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=1 -DUSE_CUDA \
  -I"$JDK/include" -I"$JDK/include/linux" -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" -I"$PWD/cuda-home/include" \
  src/policy_jni.cpp -o "$NATIVE_OUT/libswccgpolicy.so" \
  -L"$TORCH/lib" -Wl,-rpath,"$TORCH/lib" -Wl,--no-as-needed -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda
echo "built $(ls -la "$NATIVE_OUT/libswccgpolicy.so")"
