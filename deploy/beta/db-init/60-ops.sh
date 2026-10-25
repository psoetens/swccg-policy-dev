#!/bin/bash
# Plan beta-server: the ops admin the gemp container logs in with to set the hall operational after each start.
# Its password is random, generated on the server into .env (deploy.sh init), never shown. Gemp stores SHA-256 hex.
mariadb -uroot -p"$MARIADB_ROOT_PASSWORD" "gemp-swccg" <<SQL
INSERT INTO player (name, password, type, last_login_reward, last_ip, create_ip)
VALUES ('$OPS_USER', SHA2('$OPS_PASSWORD', 256), 'au', '20170101', '127.0.0.1', '127.0.0.1');
SQL
