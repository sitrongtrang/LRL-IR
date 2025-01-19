#!/bin/sh

# Check if the first argument is a valid command
if [ "$1" = "monolingual_train" ] || [ "$1" = "monolingual_retrive" ]; then
    # Shift the first argument (the command) and pass the rest to the Python script
    shift
    exec python main.py "$@"
else
    # Print usage information if the command is not recognized
    echo "Usage: docker run <image> <command> [arguments...]"
    echo "Available commands: monolingual_train, monolingual_retrive"
    exit 1
fi