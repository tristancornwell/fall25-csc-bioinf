#!/bin/bash
set -e

cd "$(dirname "$0")/code"

echo "Method            Language    Runtime"
echo "--------------------------------------"

# Run Python tests
python tests.py
# Run Codon tests
codon run tests.codon