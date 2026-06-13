#!/bin/bash
set -e
echo "Initializing Python virtual environment for Linux..."
python3 -m venv venv
source venv/bin/activate
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup complete. Run: source venv/bin/activate"
