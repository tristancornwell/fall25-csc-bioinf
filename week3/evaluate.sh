#!/bin/bash
set -e

echo "Language    Runtime"
echo "-------------------"

# Run Python tests and capture runtime
python_runtime=$(python tests/test_phylo.py | grep -o "[0-9.]*ms")
echo "python      $python_runtime"

# Run Codon tests and capture runtime
codon_runtime=$(codon run tests/test_phylo.codon | grep -o "[0-9.]*ms")
echo "codon       $codon_runtime"