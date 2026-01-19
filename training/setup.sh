#!/bin/bash
# Setup script for Veritas training environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "====================================================================="
echo "Veritas Training Environment Setup"
echo "====================================================================="
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found: Python $PYTHON_VERSION"

# Check pip
echo ""
echo "Checking pip..."
if ! command -v pip3 > /dev/null 2>&1; then
    echo "ERROR: pip3 not found. Please install pip first."
    exit 1
fi
echo "pip3 is available"

# Install dependencies
echo ""
echo "Installing Python dependencies from requirements.txt..."
pip3 install -r "$SCRIPT_DIR/requirements.txt"

# Verify installation
echo ""
echo "Verifying installation..."
python3 << 'EOF'
import sys
try:
    import sklearn
    import numpy
    import pandas
    import datasets
    import optuna
    print("✓ All required libraries installed successfully")
    print(f"  - scikit-learn: {sklearn.__version__}")
    print(f"  - numpy: {numpy.__version__}")
    print(f"  - pandas: {pandas.__version__}")
    sys.exit(0)
except ImportError as e:
    print(f"✗ Missing library: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "====================================================================="
    echo "Setup Complete!"
    echo "====================================================================="
    echo ""
    echo "You can now run training using:"
    echo "  ./run_training.sh start --quick"
    echo ""
    echo "For more information, see TRAINING_AUTOMATION.md"
else
    echo ""
    echo "Setup failed. Please check the error messages above."
    exit 1
fi
