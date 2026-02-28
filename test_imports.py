#!/usr/bin/env python3
"""Quick test to verify all imports work correctly"""

import sys
print("Testing imports...")

try:
    import torch
    print("✓ torch imported successfully")
except ImportError as e:
    print(f"✗ torch import failed: {e}")
    sys.exit(1)

try:
    import cv2
    print("✓ cv2 imported successfully")
except ImportError as e:
    print(f"✗ cv2 import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✓ numpy imported successfully")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib imported successfully")
except ImportError as e:
    print(f"✗ matplotlib import failed: {e}")
    sys.exit(1)

try:
    from sklearn.metrics import roc_curve, auc
    print("✓ scikit-learn imported successfully")
except ImportError as e:
    print(f"✗ scikit-learn import failed: {e}")
    sys.exit(1)

try:
    from scipy.spatial.distance import cdist
    print("✓ scipy imported successfully")
except ImportError as e:
    print(f"✗ scipy import failed: {e}")
    sys.exit(1)

print("\n✓ All required imports are working!")
print("\nYou can now run:")
print("  python comprehensive_evaluation.py")
print("  python ablation_studies.py")
