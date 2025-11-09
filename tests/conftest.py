# tests/conftest.py
import sys
from pathlib import Path

# Carpeta raíz del proyecto (la que contiene 'german_credit_ml')
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

