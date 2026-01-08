#!/bin/bash
# Automated Setup Script for Calvin D -> D Benchmark
# This script automates the installation and setup process

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===============================================================================${NC}"
echo -e "${GREEN}CALVIN D -> D BENCHMARK SETUP${NC}"
echo -e "${GREEN}===============================================================================${NC}"
echo ""

# Default paths (can be overridden with environment variables)
VPP_DIR="${VPP_DIR:-/home/yangc/Lab/VPP/video-prediction-policy}"
MODELS_DIR="${MODELS_DIR:-/home/yangc/Lab/VPP/models}"
DATASET_DIR="${DATASET_DIR:-/home/yangc/Lab/VPP/calvin/task_D_D}"
CALVIN_ROOT="${CALVIN_ROOT:-/home/yangc/Lab/VPP/calvin}"

# Step 1: Check conda installation
echo -e "${YELLOW}[1/8] Checking conda installation...${NC}"
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found. Please install conda or miniconda.${NC}"
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo -e "${GREEN}✓ Conda found${NC}"
echo ""

# Step 2: Create conda environment
echo -e "${YELLOW}[2/8] Creating conda environment 'vpp'...${NC}"
if conda env list | grep -q "^vpp "; then
    echo -e "${GREEN}✓ Conda environment 'vpp' already exists${NC}"
else
    conda create -n vpp python==3.10 -y
    echo -e "${GREEN}✓ Conda environment 'vpp' created${NC}"
fi
echo ""

# Step 3: Activate conda environment
echo -e "${YELLOW}[3/8] Activating conda environment...${NC}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vpp
echo -e "${GREEN}✓ Conda environment activated${NC}"
echo ""

# Step 4: Install Calvin (optional)
echo -e "${YELLOW}[4/8] Installing Calvin environment (optional)...${NC}"
read -p "Do you want to install Calvin environment? (y/n) [n]: " install_calvin

if [ "$install_calvin" = "y" ] || [ "$install_calvin" = "Y" ]; then
    if [ ! -d "$CALVIN_ROOT" ]; then
        echo -e "${YELLOW}Cloning Calvin repository...${NC}"
        cd /home/yangc/Lab/VPP
        git clone --recurse-submodules https://github.com/mees/calvin.git
        echo -e "${GREEN}✓ Calvin repository cloned${NC}"
    else
        echo -e "${GREEN}✓ Calvin repository already exists${NC}"
    fi
    
    echo -e "${YELLOW}Installing Calvin...${NC}"
    cd $CALVIN_ROOT
    sh install.sh
    echo -e "${GREEN}✓ Calvin installed${NC}"
    
    # Add to .bashrc if not already there
    if ! grep -q "CALVIN_ROOT" ~/.bashrc; then
        echo "export CALVIN_ROOT=$CALVIN_ROOT" >> ~/.bashrc
        echo -e "${GREEN}✓ CALVIN_ROOT added to ~/.bashrc${NC}"
    fi
else
    echo -e "${YELLOW}Skipping Calvin installation${NC}"
fi
echo ""

# Step 5: Install VPP dependencies
echo -e "${YELLOW}[5/8] Installing VPP dependencies...${NC}"
cd $VPP_DIR
pip install -r requirements.txt
echo -e "${GREEN}✓ VPP dependencies installed${NC}"
echo ""

# Step 6: Install accelerate
echo -e "${YELLOW}[6/8] Installing accelerate...${NC}"
pip install accelerate
echo -e "${GREEN}✓ Accelerate installed${NC}"
echo ""

# Step 7: Install huggingface-hub
echo -e "${YELLOW}[7/8] Installing huggingface-hub...${NC}"
pip install huggingface-hub
echo -e "${GREEN}✓ huggingface-hub installed${NC}"
echo ""

# Step 8: Download models
echo -e "${YELLOW}[8/8] Downloading pre-trained models...${NC}"
mkdir -p $MODELS_DIR

echo -e "${YELLOW}  Downloading CLIP model (~600MB)...${NC}"
huggingface-cli download openai/clip-vit-base-patch32 \
    --local-dir $MODELS_DIR/clip-vit-base-patch32
echo -e "${GREEN}  ✓ CLIP model downloaded${NC}"

echo -e "${YELLOW}  Downloading SVD video model (~8GB)...${NC}"
huggingface-cli download yjguo/svd-robot-calvin-ft \
    --local-dir $MODELS_DIR/svd-robot-calvin
echo -e "${GREEN}  ✓ SVD video model downloaded${NC}"

echo -e "${YELLOW}  Downloading action model (~1GB)...${NC}"
huggingface-cli download yjguo/dp-calvin \
    --local-dir $MODELS_DIR/dp-calvin
echo -e "${GREEN}  ✓ Action model downloaded${NC}"
echo ""

# Step 9: Set environment variables
echo -e "${YELLOW}Setting environment variables...${NC}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CALVIN_ROOT=$CALVIN_ROOT
echo -e "${GREEN}✓ Environment variables set${NC}"
echo ""

# Step 10: Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
cd /home/yangc/Lab/VPP/scripts

if python test_d2d_setup.py; then
    echo -e "${GREEN}✓ Basic installation test passed${NC}"
else
    echo -e "${RED}✗ Installation test failed${NC}"
    exit 1
fi

# Step 11: Summary
echo ""
echo -e "${GREEN}===============================================================================${NC}"
echo -e "${GREEN}SETUP COMPLETE!${NC}"
echo -e "${GREEN}===============================================================================${NC}"
echo ""
echo "Next steps:"
echo "1. Download Calvin D -> D dataset (~500GB)"
echo "   Follow instructions at: https://github.com/mees/calvin"
echo "   Expected location: $DATASET_DIR"
echo ""
echo "2. Verify dataset structure:"
echo "   ls -la $DATASET_DIR/"
echo ""
echo "3. Run full verification test:"
echo "   python test_d2d_setup.py \\"
echo "       --video_model_path $MODELS_DIR/svd-robot-calvin \\"
echo "       --action_model_folder $MODELS_DIR/dp-calvin \\"
echo "       --clip_model_path $MODELS_DIR/clip-vit-base-patch32 \\"
echo "       --calvin_d2d_dir $DATASET_DIR"
echo ""
echo "4. Run benchmark:"
echo "   accelerate launch --num_processes=4 run_calvin_d2d_benchmark.py \\"
echo "       --config config/calvin_d2d_config.yaml \\"
echo "       --video_model_path $MODELS_DIR/svd-robot-calvin \\"
echo "       --action_model_folder $MODELS_DIR/dp-calvin \\"
echo "       --clip_model_path $MODELS_DIR/clip-vit-base-patch32 \\"
echo "       --calvin_d2d_dir $DATASET_DIR"
echo ""
echo "For detailed instructions, see: README.md and INSTALLATION_GUIDE.md"
echo ""
