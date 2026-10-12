"""Configurable filesystem layout.

Override with environment variables when running outside the original node layout:
  PSKV_DATA_DIR   tokenizers, eval sets, packed corpora   (default /root/rescsa/data)
  PSKV_RUNS_DIR   training run outputs                    (default /root/rescsa/runs)
"""

import os

DATA = os.environ.get("PSKV_DATA_DIR", "/root/rescsa/data")
RUNS = os.environ.get("PSKV_RUNS_DIR", "/root/rescsa/runs")
