# LayerGS: Decomposition and Inpainting of Layered 3D Human Avatars via 2D Gaussian Splatting

<!-- ### [Project Page](https://your-project-page.github.io) | [Paper](https://arxiv.org/abs/xxxx.xxxxx) | [Video](https://youtube.com) -->

> [Yinghan Xu](https://rockyxu66.github.io), [John Dingliana](https://www.scss.tcd.ie/john.dingliana/)
>
> Trinity College Dublin

**Abstract:** We propose a novel framework for decomposing arbitrarily posed humans into animatable multi-layered 3D human avatars, separating the body and garments. Conventional single-layer reconstruction methods lock clothing to one identity, while prior multi-layer approaches struggle with occluded regions. We overcome both limitations by encoding each layer as a set of 2D Gaussians for accurate geometry and photorealistic rendering, and inpainting hidden regions with a pretrained 2D diffusion model via score-distillation sampling (SDS). Our three-stage training strategy first reconstructs the coarse canonical garment via single-layer reconstruction, followed by multi-layer training to jointly recover the inner-layer body and outer-layer garment details. Experiments on two 3D human benchmark datasets (4D-Dress, Thuman2.0) show that our approach achieves better rendering quality and layer decomposition and recomposition than the previous state-of-the-art, enabling realistic virtual try-on under novel viewpoints and poses, and advancing practical creation of high-fidelity 3D human assets for immersive applications.

<!-- ![Teaser](assets/teaser.png) -->

## Installation

### Prerequisites

- Linux (tested on Ubuntu 22.04.4)
- NVIDIA GPU with CUDA support (tested on RTX 4090)
- Anaconda or Mamba

### Clone Repository

```bash
git clone --recursive https://github.com/RockyXu66/LayerGS.git
cd LayerGS

# If you already cloned without --recursive, run:
git submodule update --init --recursive

# Checkout specific commits for submodules
cd submodules/diff-gaussian-rasterization && git checkout 59f5f77 && cd ../..
cd submodules/simple-knn && git checkout 86710c2 && cd ../..
```

### Environment Setup

```bash
# 1. Create mamba environment
mamba create -n layergs python=3.10 -y
mamba activate layergs

# 2. Install CUDA toolkit (make sure versions are consistent)
mamba install -c nvidia cuda-nvcc=11.8 cuda-toolkit=11.8 -y

# 3. Fix CUDA headers for compilation
# The conda CUDA package places headers in a non-standard location,
# we need to create a symlink so compilers can find them
ln -sf $CONDA_PREFIX/targets/x86_64-linux/include/nv $CONDA_PREFIX/include/nv

# 4. Set CUDA environment variables (required for compiling CUDA extensions)
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# 5. Install PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 6. Install dependencies from requirements.txt
pip install -r requirements.txt

# 7. Install additional CUDA dependencies
mamba install -c nvidia cuda-cccl=11.8 -y

# 8. Install PyTorch3D & nvdiffrast
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git

# 9. Install custom CUDA extensions (submodules)
# Update the cuda kernel
cp scripts/diff_surfel_rasterization/forward.cu submodules/diff-surfel-rasterization/cuda_rasterizer
cp scripts/diff_surfel_rasterization/backward.cu submodules/diff-surfel-rasterization/cuda_rasterizer

pip install --no-build-isolation submodules/diff-surfel-rasterization
pip install --no-build-isolation submodules/simple-knn
pip install --no-build-isolation submodules/diff-gaussian-rasterization

# 10. Download files
cd <PROJECT_ROOT>
mkdir data
bash scripts/setup.sh

# 11. Install Meta's Sapiens Lite by following their instructions:
# https://github.com/facebookresearch/sapiens/blob/main/lite/README.md
# Set your SAPIENS_CHECKPOINT_ROOT at line 13 in tools/sapiens_seg.sh
# Set your SAPIENS_ENV python interpreter at line 76 in tools/sapiens_seg.sh
cp tools/sapiens_seg.sh <SAPIENS_PROJECT_ROOT>/lite/scripts/demo/torchscript
chmod 775 <SAPIENS_PROJECT_ROOT>/lite/scripts/demo/torchscript/sapiens_seg.sh
# Set <SAPIENS_PROJECT_ROOT> at line 38 in train-three-stages.sh

# 12. Install Blender
# Set Blender path at line 35 in train-three-stages.sh
```

### ImageReward Environment Setup

```bash
mamba deactivate

cd <PROJECT_ROOT>/tools
git clone https://github.com/THUDM/ImageReward.git
cd ImageReward

mamba create -n IR python=3.10 -y
mamba activate IR

pip install torch torchvision
pip install open_clip_torch==2.26.1

pip install -e .

pip install tabulate
pip install timm==0.6.13

mamba deactivate

# Update the IR_ENV path at line 491 in layergs/train/inner.py to point to your ImageReward Python interpreter.

cd ../..
mamba activate layergs
```

## Data Preparation

### 4D-Dress Dataset

1. Download the [4D-Dress dataset](https://eth-ait.github.io/4d-dress/)
2. Extract and place the data in <4D_DRESS_DATASET>
3. Set `DATASET_DIR` in `scripts/preprocess/dress_4d_utility.py`. Convert the data

```bash
# e.g. Example script for 00175_Inner_1 | Inner | Take4 | frame 110 | male
python scripts/preprocess/dress_4d_visualization.py \
    --tgt-folder data/4d-dress --subj 00175_Inner_1 \
    --outfit Inner --seq Take4 --frame 110 --gender male \
    --headless
```

3. Preprocess the mesh to normalize coordinates:

```bash
# Set subject name (e.g., 4d-dress-00175_Inner_1-Inner-Take4-f00110)
export SUBJECT_NAME=4d-dress-00175_Inner_1-Inner-Take4-f00110

python scripts/preprocess/normalize_mesh_for_gs.py \
    -f data/4d-dress/${SUBJECT_NAME} \
    --dataset_type 4d-dress \
    --norm_with_smplx
```

4. Render training images from multiple viewpoints:

```bash
cd data_preprocess
python render_mesh_pytorch3d.py \
    -f ../data/4d-dress/${SUBJECT_NAME}/${SUBJECT_NAME}_norm.obj \
    --bg_color 0.0,0.0,0.0

mkdir -p ../data/4d-dress/${SUBJECT_NAME}/images
mkdir -p ../data/4d-dress/${SUBJECT_NAME}/masks
cp ../data/4d-dress/${SUBJECT_NAME}/torch3d_imgs/* ../data/4d-dress/${SUBJECT_NAME}/images
cp ../data/4d-dress/${SUBJECT_NAME}/torch3d_masks/* ../data/4d-dress/${SUBJECT_NAME}/masks
```

5. Segment labels by Sapiens:

```bash
cd <SAPIENS_PROJECT_ROOT>/lite/scripts/demo/torchscript
./sapiens_seg.sh <LayerGS_PROJECT_ROOT>/data/4d-dress/${SUBJECT_NAME}/images <LayerGS_PROJECT_ROOT>/data/4d-dress/${SUBJECT_NAME}
```

## Project Structure

```
LayerGS/
├── config/                                             # Training configuration files
│   └── config-4d-dress/                                # 4D-Dress specific configs
├── data/                                               # Dataset directory
│   ├── 4d-dress/
│       ├── 4d-dress-00175_Inner_1-Inner-Take4-f00110/  # One subject data folder
│       ├── ...
├── data_preprocess/                                    # Data preprocessing scripts
├── deformer/                                           # SMPL-X deformation modules
├── gala_config/                                        # SDS guidance configuration
├── gala_utils/                                         # SDS utility functions
├── layergs/                                            # Main training code
│   ├── data/                                           # Data loading
│   ├── guidance/                                       # SDS guidance
│   ├── rendering/                                      # Gaussian rendering
│   ├── scene/                                          # Scene and Gaussian models
│   ├── train/                                          # Training scripts
│   └── utils/                                          # Utility functions
├── scripts/                                            # Training and preprocessing scripts
├── submodules/                                         # CUDA extensions
└── train-three-stages.sh                               # Main training script
```
## Training

### Quick Start

Run the three-stage training pipeline:

```bash
# Set GPU device (optional, defaults to GPU 0)

cd <PROJECT_ROOT>
export CUDA_DEVICE_ID=0

# Set Python environment
export PYTHON_ENV=/path/to/your/conda/envs/layergs/bin/python

# Run training (e.g. <SUBJECT_ID>=00175_Inner_1)
bash train-three-stages.sh <SUBJECT_ID>
```

### Training Stages

The pipeline consists of three main stages:

1. **Stage 1**: Single-layer reconstruction of the whole avatar
2. **Stage 1.1**: Extract coarse upper garment mesh
3. **Stage 2**: Train inner layer (body + lower garment) with frozen outer layer
4. **Stage 3**: Refine outer layer (upper garment)
5. **Stage 3.1**: Extract refined outer layer mesh

### Controlling Training Stages

You can control which stages to run via environment variables:

```bash
# Run only Stage 1 and 1.1
TRAIN_SINGLE_LAYER=true EXTRACT_COARSE_UPPER=true bash train-three-stages.sh 00175_Inner_1

# Run all stages
TRAIN_SINGLE_LAYER=true EXTRACT_COARSE_UPPER=true TRAIN_INNER_LAYER=true TRAIN_OUTER_LAYER=true bash train-three-stages.sh 00175_Inner_1

# Resume from a specific stage (set checkpoint names first in the script)
TRAIN_SINGLE_LAYER=false TRAIN_INNER_LAYER=true bash train-three-stages.sh 00175_Inner_1
```

### Output Structure

```
output/<SUBJECT_NAME>/
├── results/
│   ├── <STAGE1_NAME>/
│   │   └── iteration_10000/
│   │       ├── point_cloud_body_cano.ply
│   │       ├── point_cloud_comp_cano.ply
│   │       └── mesh/
|   |           ├── mesh-comp_cano.ply
|   |           └── segms_upper.obj
│   ├── <STAGE2_NAME>/
│   │   └── iteration_15000/
│   │       └── point_cloud_body_cano.ply
│   └── <STAGE3_NAME>/
│       └── iteration_10200/
│           ├── point_cloud_body_cano.ply
│           ├── point_cloud_garment_cano.ply
│           └── mesh/
|               ├── mesh-comp_cano.ply
|               └── segms_upper.obj
└── train/
```

## Mesh Extraction

To extract meshes from trained Gaussian representations:

```bash
# Set subject name and stage/iteration if not already set
export SUBJECT_NAME=4d-dress-00175_Inner_1-Inner-Take4-f00110
export STAGE_NAME=1225_1430_singlelayer  # Example stage name
export ITERATION=iteration_10000         # Example iteration

python -m scripts.extract_mesh \
    -s data/4d-dress/${SUBJECT_NAME} \
    -m output/${SUBJECT_NAME} \
    --ply_path output/${SUBJECT_NAME}/results/${STAGE_NAME}/${ITERATION}/point_cloud_comp_cano.ply \
    --mesh_res 1024
```

## License

This project is licensed under the terms specified in [LICENSE.md](LICENSE.md).

## Acknowledgements

We thank the authors of the following projects for their excellent work:
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)
- [GALA](https://github.com/snuvclab/gala)
- [4D-Dress Dataset](https://eth-ait.github.io/4d-dress/)
- [HumanGaussian](https://github.com/alvinliu0/HumanGaussian)
