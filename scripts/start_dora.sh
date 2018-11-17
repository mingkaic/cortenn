#!/usr/bin/env bash

MYNAME="${0##*/}";

DEFAULT_PORT=10000;
DEFAULT_TIMEOUT=30;
DEFAULT_INTERVAL=1;

function usage {
    cat <<EOF
synopsis: start up dora server at specified port then wait at until server
started up. check start up at specified interval until timeout or server
reports SERVING

    -t timeout
        Number of seconds to wait for server startup.
        Positive integer, default value: $DEFAULT_TIMEOUT seconds.

    -i interval
        Interval between checks if the server is up.
        Positive integer, default value: $DEFAULT_INTERVAL seconds.

    -p port
        Port dora server is running on.
        Positive integer, default value: $DEFAULT_PORT seconds.

usage: $MYNAME [-t timeout] [-i interval] [-p port] cert_path
EOF
    exit 1;
}

declare -i timeout=$DEFAULT_TIMEOUT;
declare -i interval=$DEFAULT_INTERVAL;
declare -i port=$DEFAULT_PORT;

while getopts ":t:i:p:" option; do
    case "$option" in
        t) timeout=$OPTARG ;;
        i) interval=$OPTARG ;;
        p) port=$OPTARG ;;
        *) usage;;
    esac
done

if [ "$#" -lt 1 ];
then
    echo "Missing cert_path argument";
    usage;
fi

CERTPATH=$1;

export SERVER_NAME="localhost";
export SERVER_PORT=$port;

HEALTH_URL="https://$SERVER_NAME:$SERVER_PORT/v1/checkhealth";

if [ "$(curl -s -k $HEALTH_URL | jq -r '.status')" == "SERVING" ];
then
    echo "server already running @$SERVER_NAME:$SERVER_PORT... skipping";
    exit 0;
fi

docker run --rm -d -p "$SERVER_PORT:$SERVER_PORT" -v "$(realpath $CERTPATH)":/etc/ssl/dora\
    mkaichen/dora_server:latest -host 0.0.0.0 -servername "$SERVER_NAME" -port $SERVER_PORT\
    -key /etc/ssl/dora/server.key -cert /etc/ssl/dora/server.crt;

echo "waiting for dora to start serving";
((counter = timeout));
while [ "$(curl -s -k $HEALTH_URL | jq -r '.status')" != "SERVING" ] && ((counter > 0))
do
    printf '#';
    sleep $interval;
    ((counter -= $interval));
done
printf '\n';

if [ "$(curl -s -k $HEALTH_URL | jq -r '.status')" != "SERVING" ];
then
    echo "server failed to startup before timeout: $timeout seconds... giving up";
    exit 1;
fi
