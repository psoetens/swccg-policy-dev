#!/usr/bin/env bash
# One-time setup of a fresh clone (plan dev-distribution). Idempotent: rerun after a pull or a Gemp update.
#   scripts/setup.sh            # everything below
#   scripts/setup.sh --no-engine   # Python side only (no JDK / Gemp / engine build)
# Needs: git, git-lfs, curl, tar, rsync, uv (https://docs.astral.sh/uv/), an NVIDIA GPU + driver for training and
# evaluation (CUDA 12.8 wheels come with torch); docker only for the Brainiac server (scripts/brainiac_server.sh).
# Writes only inside the repo: .cache/tools (JDK 21, Maven), .cache/gemp-swccg (Gemp clone), .cache/gemp-knowledge-v5
# (the instrumented engine), .venv, models/D/policy.ts.
# Env (or .env): SWCCG_GEMP_URL (https://github.com/psoetens/gemp-swccg.git), SWCCG_GEMP_BRANCH (feat/brainiac),
#   SWCCG_GEMP_REPO (an existing Gemp clone to use instead of .cache/gemp-swccg).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
ENGINE=1; [ "${1:-}" = "--no-engine" ] && ENGINE=0
step() { echo; echo "== $*"; }
need() { command -v "$1" >/dev/null || { echo "missing: $1 ($2)"; exit 1; }; }
need git "your package manager"; need git-lfs "https://git-lfs.com"; need curl "package manager"; need tar "package manager"
need rsync "package manager"; need uv "curl -LsSf https://astral.sh/uv/install.sh | sh"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.cache/uv}" HF_HOME="${HF_HOME:-$ROOT/.cache/hf}"

step "Git LFS objects (models, tables, native library)"
git lfs install --local >/dev/null
git lfs pull

step "Python environment (uv sync: torch CUDA 12.8, onnxruntime, ...)"
uv sync --frozen
PY="$ROOT/.venv/bin/python"

if [ "$ENGINE" = 1 ]; then
    TOOLS="$ROOT/.cache/tools"; mkdir -p "$TOOLS"
    fetch() {  # url sha-tool expected-sum target-dir
        [ -d "$4" ] && return 0
        local archive="$TOOLS/$(basename "$1")"
        curl -fL --retry 3 -o "$archive" "$1"
        echo "$3  $archive" | "$2" -c - >/dev/null || { echo "checksum mismatch: $archive"; rm -f "$archive"; exit 1; }
        tar -xzf "$archive" -C "$TOOLS" && rm -f "$archive"
        [ -d "$4" ] || { echo "unexpected archive layout: $1"; exit 1; }
    }
    step "JDK 21 (Temurin 21.0.12.1+1) and Maven 3.9.9 into .cache/tools"
    fetch "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.12.1%2B1/OpenJDK21U-jdk_x64_linux_hotspot_21.0.12.1_1.tar.gz" \
        sha256sum ce79869e1307ed8ee1e2baa86a412b1eb5b75d10a01006d788a6f968bcfaee94 "$TOOLS/jdk-21.0.12.1+1"
    fetch "https://archive.apache.org/dist/maven/maven-3/3.9.9/binaries/apache-maven-3.9.9-bin.tar.gz" \
        sha512sum a555254d6b53d267965a3404ecb14e53c3827c09c3b94b5678835887ab404556bfaf78dcfe03ba76fa2508649dca8531c74bca4d5846513522404d48e8c4ac8b \
        "$TOOLS/apache-maven-3.9.9"
    source scripts/machine_env.sh
    java -version 2>&1 | head -1; mvn -v | head -1

    step "Gemp feat/brainiac"
    GEMP="${SWCCG_GEMP_REPO:-$ROOT/.cache/gemp-swccg}" BRANCH="${SWCCG_GEMP_BRANCH:-feat/brainiac}"
    if [ ! -d "$GEMP/.git" ]; then
        git clone --branch "$BRANCH" "${SWCCG_GEMP_URL:-https://github.com/psoetens/gemp-swccg.git}" "$GEMP"
    elif [ -z "${SWCCG_GEMP_REPO:-}" ]; then
        git -C "$GEMP" fetch origin "$BRANCH" && git -C "$GEMP" branch -f "$BRANCH" FETCH_HEAD 2>/dev/null \
            || git -C "$GEMP" merge --ff-only FETCH_HEAD
    fi
    git -C "$GEMP" log -1 --format='%h %s' "$BRANCH"

    step "Instrumented engine (.cache/gemp-knowledge-v5; Maven downloads its dependencies on the first build)"
    SWCCG_GEMP_REPO="$GEMP" SWCCG_GEMP_BRANCH="$BRANCH" PYTHONPATH=scripts:src "$PY" scripts/build_knowledge_engine.py
fi

step "TorchScript export of D (the fixed opponent in scripts/train_arm.sh)"
if "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    PYTHONPATH=src:scripts "$PY" scripts/export_torchscript.py models/D/policy.pt
else
    echo "no CUDA GPU visible: skipped (training and evaluation need one)"
fi

step "done. Next: QUICKSTART.md (uv run pytest -q tests; play, evaluate, train)"
