#!/usr/bin/env python3
"""Open the robot launcher without silently selecting the synthetic demo."""
import os
from pathlib import Path
import sys

launcher = Path(__file__).resolve().with_name('run.py')
os.execv(sys.executable, [sys.executable, str(launcher), *(sys.argv[1:] or ['menu'])])
