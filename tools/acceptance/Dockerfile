# Plan dev-distribution: a machine that has never seen the lab (no JDK, Maven, ~/.m2, Gemp rig or lab paths).
FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git git-lfs curl ca-certificates rsync docker.io procps time gcc g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/0.12.13/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
RUN userdel -r ubuntu 2>/dev/null; useradd -m -u 1000 -s /bin/bash dev
USER dev
WORKDIR /home/dev
