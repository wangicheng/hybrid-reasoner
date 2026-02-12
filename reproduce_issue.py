import sys
import os
import traceback

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.core.engine import HybridEngine

print("Attempting to initialize HybridEngine...")
try:
    engine = HybridEngine()
    print("Optimization successful!")
except Exception:
    traceback.print_exc()
