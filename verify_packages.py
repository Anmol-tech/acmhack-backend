#!/usr/bin/env python3
"""
Verify all required packages are installed correctly.
"""

import sys

def check_import(package_name, import_name=None):
    """Try importing a package and report status."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name} - {e}")
        return False

print("=" * 70)
print("PACKAGE VERIFICATION")
print("=" * 70)

packages = [
    ("boto3", "boto3"),
    ("sagemaker", "sagemaker"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("requests", "requests"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("evaluate", "evaluate"),
    ("soundfile", "soundfile"),
    ("librosa", "librosa"),
]

print("\nChecking required packages:")
all_good = True
for display_name, import_name in packages:
    if not check_import(display_name, import_name):
        all_good = False

print("\n" + "=" * 70)
if all_good:
    print("✓ ALL PACKAGES INSTALLED SUCCESSFULLY!")
else:
    print("✗ Some packages are missing. Run: pip install -r requirements.txt")
print("=" * 70)

# Check Python version
print(f"\nPython version: {sys.version}")
print(f"Python executable: {sys.executable}")

sys.exit(0 if all_good else 1)
