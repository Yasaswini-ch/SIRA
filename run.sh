#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Optional: Ensure dependencies are installed (fallback if dev.nix missed it)
pip install -r requirements.txt

# Start the Flask app
export FLASK_APP=app.py
export FLASK_ENV=development
export FLASK_DEBUG=1

# Launch server dynamically on 0.0.0.0 and port 5000 so IDX can tunnel it
flask run --host=0.0.0.0 --port=5000
