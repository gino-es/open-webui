#!/usr/bin/env bash

# Script to run the chat migration CLI tool

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR" || exit

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if Python script exists
if [ ! -f "migrate_chats.py" ]; then
    echo "Error: migrate_chats.py not found in $(pwd)"
    exit 1
fi

# Run the migration script with all arguments passed through
python migrate_chats.py "$@" 