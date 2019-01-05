#!/usr/bin/env bash

MYNAME="${0##*/}";

function usage {
    cat <<EOF
synopsis: filter for trace files found by COVERAGE_OUTPUT_FILE=<file/path>
format in stdin, then stitch tracefiles together as outpath.

    outpath
        Path of the stitched output trace file

usage: stdin | $MYNAME outpath
EOF
    exit 1;
}

PIPE_IN="";
if [ -p /dev/stdin ];
then
    PIPE_IN=$(</dev/stdin);
else
    echo "Missing STDIN pipe";
    usage;
fi

if [ "$#" -lt 1 ];
then
    echo "Missing outpath argument";
    usage;
fi

OUTPATH="$1";

# extract coverage paths
PATHS_STR=$(echo "$PIPE_IN" | sed -rn 's/.*COVERAGE_OUTPUT_FILE=(.*)\/coverage\.dat.*/\1/p');

IFS=$'\n';
CPATHS=($PATHS_STR);

# make paths absolute
for ((i=0; i<${#CPATHS[@]}; i++))
do
    if [ -d "${CPATHS[i]}" ];
    then
        CPATHS[i]=$(realpath "${CPATHS[i]}");
    fi
done

# make all paths unique
IFS=$' ';
UCPATHS=($(printf "%s " "${CPATHS[@]}" | sort -u));

# start stitching tracefiles together
rm -f "$OUTPATH";
for CPATH in "${UCPATHS[@]}"
do
    if [ -d "$CPATH" ];
    then
        CFILE="$CPATH/coverage.dat";
        echo "Stitching file $CFILE";
        if [ -f "$OUTPATH" ];
        then
            lcov -a "$CFILE" -a "$OUTPATH" -o "$OUTPATH";
        else
            lcov -a "$CFILE" -o "$OUTPATH";
        fi
    fi
done
