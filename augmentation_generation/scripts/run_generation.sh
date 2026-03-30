#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 --type <all|clear|no_header|text|wrinkles>"
    exit 1
}

TYPE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)
            TYPE="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

[ -z "$TYPE" ] && usage

SUBSCRIPT="$SCRIPT_DIR/run_generation_${TYPE}.sh"

if [ ! -f "$SUBSCRIPT" ]; then
    echo "Unknown type: $TYPE"
    echo "Available types: all, clear, no_header, text, wrinkles"
    exit 1
fi

mkdir -p "$SCRIPT_DIR/../output/$TYPE"
bash "$SUBSCRIPT"
