# Plan beta-server: Gemp feat/brainiac + the Brainiac overlay, compiled on the build machine (build.sh stages it).
FROM eclipse-temurin:21-jre-noble
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/* \
    && useradd --system --home /opt/gemp gemp && mkdir -p /data/unknown-keys && chown gemp /data/unknown-keys
COPY --chown=root:root . /opt/gemp/
USER gemp
WORKDIR /opt/gemp
EXPOSE 8080
ENTRYPOINT ["/bin/bash", "/opt/gemp/entrypoint.sh"]
