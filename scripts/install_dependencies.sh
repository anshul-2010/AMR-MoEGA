#!/usr/bin/env bash
set -euo pipefail
# Use either conda or pip depending on your preference
echo "Installing dependencies (pip)..."
pip install -r requirements.txt
echo "Done"