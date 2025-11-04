#!/bin/bash

# English-Darija Translator Setup Script
# This script automates the initial project setup

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Welcome message
echo "=========================================="
echo "   English-Darija Translator Setup"
echo "=========================================="
echo ""

# Check prerequisites
print_info "Checking prerequisites..."

if ! command_exists python3; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
print_info "Found Python $PYTHON_VERSION"

if ! command_exists pip3; then
    print_error "pip3 is not installed. Please install pip3."
    exit 1
fi

print_info "All prerequisites met!"
echo ""

# Create directory structure
print_info "Creating directory structure..."
mkdir -p data/{raw,processed}
mkdir -p notebooks
mkdir -p src
mkdir -p models
mkdir -p tests

print_info "Directory structure created successfully!"
echo ""

# Create virtual environment
print_info "Creating virtual environment..."
python3 -m venv venv

print_info "Virtual environment created!"
echo ""

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_info "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_info "Dependencies installed successfully!"
else
    print_warn "requirements.txt not found. Skipping dependency installation."
fi
echo ""

# Create __init__.py
print_info "Creating package marker..."
cat > src/__init__.py << EOL
"""English-Darija Translation Package"""
__version__ = "1.0.0"
EOL

# Create .gitignore
print_info "Creating .gitignore..."
cat > .gitignore << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Model files (large)
models/*/pytorch_model.bin
models/*/model.safetensors
*.bin

# Data files (large)
data/raw/*.csv
data/processed/*.csv

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log

# Cache
cache/
.cache/
__pypackages__/

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch
*.pth
*.pt
checkpoints/

# Tensorboard
runs/
events.out.tfevents.*

# Environment variables
.env
.env.local
EOL

# Create .dockerignore
print_info "Creating .dockerignore..."
cat > .dockerignore << EOL
.git
.gitignore
*.md
notebooks/
tests/
venv/
env/
__pycache__/
*.pyc
.DS_Store
*.ipynb
.ipynb_checkpoints/
logs/
cache/
EOL

# Create run_pipeline.sh
print_info "Creating pipeline execution script..."
cat > run_pipeline.sh << EOL
#!/bin/bash

echo "=== English-Darija Translation Pipeline ==="
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Step 1: Preprocessing
echo "Step 1: Preprocessing data..."
python src/preprocessing.py
if [ \$? -ne 0 ]; then
    echo "❌ Preprocessing failed!"
    exit 1
fi
echo "✅ Preprocessing complete!"
echo ""

# Step 2: Training
echo "Step 2: Training model..."
python src/train.py --num_epochs 3 --batch_size 8
if [ \$? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi
echo "✅ Training complete!"
echo ""

# Step 3: Evaluation
echo "Step 3: Evaluating model..."
python src/evaluate.py --num_examples 5
if [ \$? -ne 0 ]; then
    echo "❌ Evaluation failed!"
    exit 1
fi
echo "✅ Evaluation complete!"
echo ""

echo "=== Pipeline Complete! ==="
echo "You can now run inference with:"
echo "  python src/inference.py --interactive"
EOL

chmod +x run_pipeline.sh

# Create quick start script
print_info "Creating quick start script..."
cat > quickstart.sh << EOL
#!/bin/bash

# Quick start script for inference

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run interactive inference
python src/inference.py --interactive
EOL

chmod +x quickstart.sh

# Create environment template
print_info "Creating environment template..."
cat > .env.example << EOL
# Model Configuration
MODEL_PATH=models/nllb-fine-tuned
MODEL_NAME=facebook/nllb-200-distilled-600M

# Data Configuration
RAW_DATA_PATH=data/raw/dataset.csv
PROCESSED_DATA_PATH=data/processed

# Training Configuration
NUM_EPOCHS=5
BATCH_SIZE=8
LEARNING_RATE=2e-5

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Device Configuration (cuda or cpu)
DEVICE=cuda
EOL

# Initialize git repository (optional)
if command_exists git; then
    read -p "Initialize git repository? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Initializing git repository..."
        git init
        git add .gitignore
        git commit -m "Initial commit: Add .gitignore"
        print_info "Git repository initialized!"
    fi
fi
echo ""

# Check for dataset
print_info "Checking for dataset..."
if [ -f "data/raw/dataset.csv" ]; then
    print_info "Dataset found in data/raw/dataset.csv"
else
    print_warn "Dataset not found!"
    print_warn "Please place your dataset.csv in data/raw/ directory"
    print_warn "The CSV should have two columns: 'english' and 'darija'"
fi
echo ""

# Print next steps
echo "=========================================="
echo "   Setup Complete! 🎉"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Place your dataset in data/raw/dataset.csv"
echo ""
echo "2. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Run the preprocessing:"
echo "   python src/preprocessing.py"
echo ""
echo "4. Train the model:"
echo "   python src/train.py --num_epochs 3"
echo ""
echo "5. Or run the complete pipeline:"
echo "   ./run_pipeline.sh"
echo ""
echo "6. For quick inference:"
echo "   ./quickstart.sh"
echo ""
echo "7. To start the API server:"
echo "   python src/api.py"
echo ""
echo "For more information, see:"
echo "  - README.md"
echo "  - SETUP_GUIDE.md"
echo "  - MIGRATION_GUIDE.md"
echo ""
echo "Happy translating! 🇲🇦"