#!/usr/bin/env bash

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";
openssl req -x509 -newkey rsa:4096 -keyout $THIS_DIR/server.key -out $THIS_DIR/server.crt -days 365 -nodes -subj '/CN=localhost';
