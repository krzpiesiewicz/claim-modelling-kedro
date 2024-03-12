#!/bin/bash
python -c '''
import sys
vstr = lambda tup: ".".join(map(str, tup))
required_python_version = (3, 8)
version = tuple(sys.version_info)[:3]
if version >= required_python_version:
  print(f"[OK] python3 version is {vstr(version)} (>= {vstr(required_python_version)})")
else:
  print(f"[Error] python3 version is {vstr(version)}. It has to be at least {vstr(required_python_version)}")
  exit(1)
'''

echo -e "> python3 -m venv venv"
python3 -m venv venv

echo -e "> source venv/bin/activate"
source venv/bin/activate

echo -e "> pip install -r solvency-models/src/requirements.txt"
pip install -r requirements.txt
