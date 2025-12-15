#!/bin/bash
# Kascade Installation Script
# This script sets up all dependencies for Kascade
# Prerequisites: conda environment with Python 3.12 must be created and activated

set -e  # Exit on error

# Configuration
THIRD_PARTY_DIR="third_party"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${GREEN}==>${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

print_error() {
    echo -e "${RED}Error:${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    # Check if conda environment is activated
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        print_error "No conda environment detected. Please create and activate a conda environment first:"
        echo "  conda create -n kascade python=3.12"
        echo "  conda activate kascade"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "$PYTHON_VERSION" != "3.12" ]]; then
        print_warning "Python version is $PYTHON_VERSION, recommended is 3.12"
    fi
    
    # Check if CUDA is available
    if ! command -v nvcc &> /dev/null; then
        print_warning "nvcc not found. CUDA toolkit may not be installed."
    fi
    
    # Check if git is available
    if ! command -v git &> /dev/null; then
        print_error "git is not installed. Please install git first."
        exit 1
    fi
    
    echo "  Conda environment: $CONDA_DEFAULT_ENV"
    echo "  Python version: $PYTHON_VERSION"
}

# Detect GPU architecture
detect_gpu_arch() {
    print_step "Detecting GPU architecture..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "nvidia-smi not found. Cannot detect GPU architecture."
        echo "unknown"
        return
    fi

    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | cut -d'.' -f1)

    echo "$COMPUTE_CAP"
}

# Install PyTorch with CUDA 12.8
install_pytorch() {
    print_step "Installing PyTorch with CUDA 12.8..."
    pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
}

# Install CUDA via conda
install_cuda() {
    print_step "Installing CUDA 12.8.1 via conda..."
    conda install -y cuda -c nvidia/label/cuda-12.8.1

    echo ">> Configuring CUDA environment..."
    
    # Link the hidden headers to the standard location
    SOURCE_DIR="$CONDA_PREFIX/targets/x86_64-linux/include"
    DEST_DIR="$CONDA_PREFIX/include"

    # Loop through every file in the source
    for filepath in "$SOURCE_DIR"/*; do
        filename=$(basename "$filepath")
        
        # Only link if the destination does NOT exist yet
        if [ ! -e "$DEST_DIR/$filename" ]; then
            ln -s "$filepath" "$DEST_DIR/$filename"
        fi
    done
    }

# Install build tools
install_build_tools() {
    print_step "Installing build tools..."
    pip install packaging ninja
}

# Initialize and update git submodules
init_submodules() {
    print_step "Initializing git submodules..."
    git submodule update --init --recursive
}

# Install TileLang
install_tilelang() {
    print_step "Installing TileLang..."
    
    # Clean build artifacts if they exist
    if [ -d "$THIRD_PARTY_DIR/tilelang/build" ]; then
        print_warning "Cleaning TileLang build artifacts..."
        rm -rf "$THIRD_PARTY_DIR/tilelang/build"
        rm -rf "$THIRD_PARTY_DIR/tilelang/tilelang.egg-info"
        rm -rf "$THIRD_PARTY_DIR/tilelang/*.egg-info"
    fi
    
    # Install TileLang
    cd "$THIRD_PARTY_DIR/tilelang"
    pip install .
    cd ../..
}

# Install Flash Attention 3
install_flash_attention_3() {
    print_step "Installing Flash Attention 3..."
    
    # Build Flash Attention 3 from hopper directory
    cd "$THIRD_PARTY_DIR/flash-attention/hopper"
    python setup.py install
    cd ../../..
}

# Install Flash Attention 2
install_flash_attention_2() {
    print_step "Installing Flash Attention 2..."
    pip install flash-attn==2.8.0.post2 --no-build-isolation
}

# Install Kascade package
install_kascade() {
    print_step "Installing Kascade package..."
    pip install -e .
}

# Main installation flow
main() {
    echo "============================================"
    echo "       Kascade Installation Script"
    echo "============================================"
    echo ""
    
    check_prerequisites
    
    # Detect GPU architecture
    GPU_ARCH=$(detect_gpu_arch)
    echo "  Detected GPU architecture: $GPU_ARCH"
    
    # Get submodule commits for display
    TILELANG_COMMIT=$(cd "$THIRD_PARTY_DIR/tilelang" 2>/dev/null && git rev-parse HEAD 2>/dev/null || echo "not initialized")
    FA_COMMIT=$(cd "$THIRD_PARTY_DIR/flash-attention" 2>/dev/null && git rev-parse HEAD 2>/dev/null || echo "not initialized")
    
    echo ""
    echo "This script will install:"
    echo "  1. PyTorch with CUDA 12.8"
    echo "  2. CUDA 12.8.1 (via conda)"
    echo "  3. Build tools (packaging, ninja)"
    echo "  4. TileLang (submodule: ${TILELANG_COMMIT:0:7})"
    if [ "$GPU_ARCH" -ge 9 ] 2>/dev/null; then
        echo "  5. Flash Attention 3 (submodule: ${FA_COMMIT:0:7}) [H100/H800 detected]"
    else
        echo "  5. Flash Attention 2 only (v2.8.0.post2) [Non-Hopper GPU]"
    fi
    echo "  6. Kascade and its dependencies"
    echo ""
    
    read -p "Continue with installation? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
    
    install_pytorch
    install_cuda
    install_build_tools
    init_submodules
    install_tilelang
    
    if [ "$GPU_ARCH" -ge 9 ] 2>/dev/null; then
        print_step "Hopper GPU detected - Installing Flash Attention 3..."
        install_flash_attention_3
    else
        print_step "Non-Hopper GPU detected - Installing Flash Attention 2 only..."
        print_warning "Flash Attention 3 requires H100/H800 GPUs (Hopper architecture)"
        install_flash_attention_2
    fi

    install_kascade
    
    echo ""
    echo "============================================"
    echo -e "${GREEN}Installation complete!${NC}"
    echo "============================================"
    echo ""
    if [ "$GPU_ARCH" -lt 9 ] 2>/dev/null; then
        echo -e "${YELLOW}Note: Not using hopper architecture. Efficient Kascade and FA-3 will not work${NC}"
        echo ""
    fi
}

main "$@"
