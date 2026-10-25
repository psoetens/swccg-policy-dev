#!/usr/bin/env bash
# Plan beta-server: start Gemp, then set the fresh hall operational with the ops admin (a fresh hall refuses games
# until an admin does; scripts/brainiac_server.sh start does the same with asdf). Env from compose.yml / .env.
set -u
cd /opt/gemp
CP=$(ls -d classes/* | sort | paste -sd:):$(ls lib/*.jar | paste -sd:)
DB_CONTAINER_NAME=db:3306 MYSQL_DATABASE=gemp-swccg MYSQL_USER=gemp MYSQL_PASSWORD="$DB_PASSWORD" \
INTERNAL_PORT=8080 WEB_PATH=/opt/gemp/web/ BRAINIAC_RELEASE_PATH=/brainiac/releases/$RELEASE_NAME \
HALL_DECISION_TIMEOUT_SECONDS="${HALL_DECISION_TIMEOUT_SECONDS:-300}" \
    java -Xmx"${GEMP_HEAP:-2560m}" --add-opens java.base/java.util=ALL-UNNAMED -Dlog4j.configurationFile=/opt/gemp/log4j.xml \
    -Dbeta.noDecisionTimerVsBot=true -Dpolicy.knowledge=true -Dpolicy.context=5 -Dpolicy.unknownKeysDir=/data/unknown-keys \
    -cp "$CP" com.gempukku.swccgo.async.SwccgoAsyncServer &
JAVA=$!
trap 'kill -TERM $JAVA; wait $JAVA; exit 0' TERM INT
URL=http://localhost:8080/gemp-swccg-server REF="Referer: http://localhost:8080/gemp-swccg/"
for _ in $(seq 1 120); do
    if curl -sf -o /dev/null http://localhost:8080/gemp-swccg/; then
        JAR=$(mktemp)
        if curl -sf -o /dev/null -c "$JAR" -H "$REF" --data-urlencode "login=$OPS_USER" --data-urlencode "password=$OPS_PASSWORD" "$URL/login" \
            && curl -sf -o /dev/null -b "$JAR" -H "$REF" --data "enabled=false" "$URL/admin/shutdown"; then
            echo "beta: hall operational"
        else
            echo "beta: hall NOT operational (ops admin login failed)"
        fi
        rm -f "$JAR"; break
    fi
    kill -0 $JAVA 2>/dev/null || break
    sleep 2
done
wait $JAVA
