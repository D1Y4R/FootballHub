#!/bin/bash

# Set up proper library paths for C++ runtime
export LD_LIBRARY_PATH="/home/runner/.local/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Start the Flask application with uv's Python environment
exec uv run gunicorn --bind 0.0.0.0:2222 --reuse-port --reload main:app