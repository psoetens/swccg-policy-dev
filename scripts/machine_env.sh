# Machine-local paths (plan host-migration M-2). Sourced by the engine scripts; a machine-local .env (not synced, not
# committed; see .env.example) overrides the defaults: the JDK and Maven scripts/setup.sh puts in .cache/tools, else
# the JAVA_HOME / mvn already on this machine.
# Sets JAVA_HOME, PATH and SWCCG_ENGINE_DEPS (a file holding the engine's dependency classpath).
_swccg_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$_swccg_root/.env" ]; then set -a; . "$_swccg_root/.env"; set +a; fi
_jdk="$(ls -d "$_swccg_root"/.cache/tools/jdk-21* 2>/dev/null | head -1)"
export JAVA_HOME="${SWCCG_JAVA_HOME:-${_jdk:-${JAVA_HOME:-}}}"
unset _jdk
export PATH="$JAVA_HOME/bin:$PATH"
# Plan engine-on-feat-brainiac: the engine tree is .cache/gemp-knowledge-v5 (since 2026-10-24, plan
# forfeit-reset-and-scan; v4 = tag engine-v4), built from feat/brainiac by build_knowledge_engine.py, which writes its
# own dependency classpath.
SWCCG_ENGINE_DEPS="${SWCCG_ENGINE_DEPS:-$_swccg_root/.cache/gemp-knowledge-v5/cp.txt}"
# Maven, for build_knowledge_engine.py (SWCCG_MAVEN_HOME overrides; default: the JDK's tools folder).
# Only builds need it; a machine without it (the server) must not fail here (callers run with set -euo pipefail).
if [ -z "${SWCCG_MAVEN_HOME:-}" ]; then
    for _maven in "$(dirname "$JAVA_HOME")"/apache-maven-*; do [ -d "$_maven" ] && SWCCG_MAVEN_HOME="$_maven"; done
    unset _maven
fi
if [ -n "${SWCCG_MAVEN_HOME:-}" ]; then export PATH="$SWCCG_MAVEN_HOME/bin:$PATH"; fi
# The project's own torch libraries, for the native library (its RUNPATH names the .venv of the machine that built it).
SWCCG_TORCH_LIB="$(ls -d "$_swccg_root"/.venv/lib/python3.*/site-packages/torch/lib 2>/dev/null | head -1)"
export LD_LIBRARY_PATH="$SWCCG_TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _swccg_root
