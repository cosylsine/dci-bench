#!/usr/bin/env python3
"""Prepare or execute a frozen four-GPU KaLM candidate-pool batch."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dci_bench.data.pipeline.batch import main

if __name__ == '__main__':
    main()
