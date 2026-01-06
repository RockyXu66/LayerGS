#!/bin/bash
#
# LayerGS Three-Stage Training Script
# ===================================
#
# This script trains a layered Gaussian splatting model in three stages:
#   Stage 1: Single-layer reconstruction of the whole avatar
#   Stage 1.1: Extract coarse upper garment mesh
#   Stage 2: Train inner layer (body + lower garment) with frozen outer layer
#   Stage 3: Refine outer layer (upper garment)
#   Stage 3.1: Extract refined upper garment mesh
#
# Usage:
#   bash ./train-three-stages.sh <SUBJECT_ID>
#   Example: bash ./train-three-stages.sh 00175_Inner_1
#
# Available subject IDs:
#   00122, 00127, 00127_Outer, 00152, 00156, 00160, 00174, 00175_Inner_1, 00190
#

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

DEVICE_ID=${CUDA_DEVICE_ID:-0}
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}
PORT=$((10000 + DEVICE_ID))
echo "Using GPU ${DEVICE_ID}, PORT ${PORT}"

PYTHON_ENV="${PYTHON_ENV:-python}"

# TODO: Set Blender path
BLENDER_ENV="/home/yixu/Tools/blender-4.2.0-linux-x64/blender";

# TODO: Set Sapiens project root path
SAPIENS_PROJECT_ROOT="/mnt/raid5/yixu/Projects/Tools/sapiens";

# Project paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${PROJECT_ROOT}/data/4d-dress"

# ============================================================================
# Subject Configuration
# ============================================================================

set_subject_config() {
    local subject_id=$1
    case "$subject_id" in
        00122)
            DATA_LAYER=Inner; TAKE_NAME=Take8; FRAME=11; GENDER=male; DATA_TYPE="pytorch3d"
            ;;
        00127)
            DATA_LAYER=Inner; TAKE_NAME=Take5; FRAME=45; GENDER=male; DATA_TYPE="pytorch3d"
            ;;
        00127_Outer)
            DATA_LAYER=Outer; TAKE_NAME=Take11; FRAME=11; GENDER=male; DATA_TYPE="pytorch3d"
            ;;
        00152)
            DATA_LAYER=Inner; TAKE_NAME=Take4; FRAME=55; GENDER=female; DATA_TYPE="pytorch3d"
            ;;
        00156)
            DATA_LAYER=Inner; TAKE_NAME=Take3; FRAME=23; GENDER=female; DATA_TYPE="pytorch3d"
            ;;
        00160)
            DATA_LAYER=Inner; TAKE_NAME=Take10; FRAME=148; GENDER=male; DATA_TYPE="pytorch3d"
            ;;
        00174)
            DATA_LAYER=Inner; TAKE_NAME=Take10; FRAME=11; GENDER=male; DATA_TYPE="pytorch3d"
            ;;
        00175_Inner_1)
            DATA_LAYER=Inner; TAKE_NAME=Take4; FRAME=110; GENDER=male; DATA_TYPE="pytorch3d"
            ;;
        00190)
            DATA_LAYER=Inner; TAKE_NAME=Take2; FRAME=11; GENDER=female; DATA_TYPE="pytorch3d"
            ;;
        *)
            echo "Error: Unknown SUBJECT_ID: $subject_id"
            echo "Available subject IDs: 00122, 00127, 00127_Outer, 00152, 00156, 00160, 00174, 00175_Inner_1, 00190"
            exit 1
            ;;
    esac
}

# ============================================================================
# Parse Arguments
# ============================================================================

if [ -z "$1" ]; then
    echo "Usage: $0 <SUBJECT_ID>"
    echo "Example: $0 00190"
    echo ""
    echo "Available subject IDs:"
    echo "  00122, 00127, 00127_Outer, 00152, 00156, 00160, 00174, 00175_Inner_1, 00190"
    exit 1
fi

SUBJECT_ID=$1
set_subject_config "$SUBJECT_ID"

FRAME_ID=$(printf "f%05d" "${FRAME}")
SUBJECT_NAME=4d-dress-${SUBJECT_ID}-${DATA_LAYER}-${TAKE_NAME}-${FRAME_ID}

echo "============================================"
echo "Training LayerGS for: ${SUBJECT_NAME}"
echo "Gender: ${GENDER}"
echo "Data type: ${DATA_TYPE}"
echo "============================================"

# ============================================================================
# Training Stage Flags (modify these to run specific stages)
# ============================================================================

TRAIN_SINGLE_LAYER=${TRAIN_SINGLE_LAYER:-true}         # Stage 1
EXTRACT_COARSE_UPPER=${EXTRACT_COARSE_UPPER:-true}     # Stage 1.1
TRAIN_INNER_LAYER=${TRAIN_INNER_LAYER:-true}           # Stage 2
TRAIN_OUTER_LAYER=${TRAIN_OUTER_LAYER:-true}            # Stage 3
EXTRACT_REFINED_UPPER=${EXTRACT_REFINED_UPPER:-true}    # Stage 3.1

# Stage iteration settings (will be set by earlier stages if running sequentially)
# Uncomment and modify these to resume from specific checkpoints
# STAGE1_NAME=0103_2045_singlelayer
# STAGE1_ITER=iteration_10000
# STAGE2_NAME=0103_2157_inner
# STAGE2_ITER=iteration_15000
# STAGE3_NAME=1204_1410_multilayer_2dgs_outer
# STAGE3_ITER=iteration_10200

# ============================================================================
# Stage 1: Single-layer for the whole avatar
# ============================================================================

if [ "$TRAIN_SINGLE_LAYER" = "true" ]; then
    echo ""
    echo ">>> Stage 1: Training single-layer model for whole avatar..."
    echo ""

    # Generate test camera views and ground-truth test images
    cd ${PROJECT_ROOT}/data_preprocess
    ${PYTHON_ENV} render_mesh_pytorch3d.py \
        -f ${DATASET_ROOT}/${SUBJECT_NAME}/${SUBJECT_NAME}_norm.obj \
        --test --bg_color 0.0,0.0,0.0

    STAGE1_VERSION="$(date "+%m%d_%H%M")"
    STAGE1_NAME="${STAGE1_VERSION}_singlelayer"
    ITER=10000

    cd ${PROJECT_ROOT}
    mkdir -p output/${SUBJECT_NAME}/results/${STAGE1_NAME}
    
    # Backup config files
    cp ${PROJECT_ROOT}/layergs/train/singlelayer.py output/${SUBJECT_NAME}/results/${STAGE1_NAME}/
    cp config/config-4d-dress/singlelayer-${GENDER}.yaml config/
    cp config/config-4d-dress/singlelayer-${GENDER}.yaml output/${SUBJECT_NAME}/results/${STAGE1_NAME}/

    # Run training
    ${PYTHON_ENV} -m scripts.train singlelayer \
        --data_type ${DATA_TYPE} \
        -s ${DATASET_ROOT}/${SUBJECT_NAME} \
        --scan_name ${SUBJECT_NAME}_norm \
        -m output/${SUBJECT_NAME} \
        --model_type multi-layer-2dgs \
        --sh_degree 0 \
        --apply_2dgs 1 \
        --save_iterations ${ITER} \
        --iterations ${ITER} \
        --config_file config/singlelayer-${GENDER}.yaml \
        --cur_version ${STAGE1_NAME} \
        --port ${PORT}

    STAGE1_ITER="iteration_${ITER}"
    
    # Copy body point cloud as composite
    cp output/${SUBJECT_NAME}/results/${STAGE1_NAME}/${STAGE1_ITER}/point_cloud_body_cano.ply \
       output/${SUBJECT_NAME}/results/${STAGE1_NAME}/${STAGE1_ITER}/point_cloud_comp_cano.ply

    echo ">>> Stage 1 completed!"
fi

# ============================================================================
# Stage 1.1: Extract coarse upper garment mesh
# ============================================================================

if [ "$EXTRACT_COARSE_UPPER" = "true" ]; then
    echo ""
    echo ">>> Stage 1.1: Extracting coarse upper garment mesh..."
    echo ""

    # Check if STAGE1_NAME and STAGE1_ITER are set
    if [ -z "$STAGE1_NAME" ] || [ -z "$STAGE1_ITER" ]; then
        echo "Error: STAGE1_NAME and STAGE1_ITER must be set"
        echo "Either run Stage 1 first, or set these variables manually"
        exit 1
    fi

    mkdir -p ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/${STAGE1_NAME}/${STAGE1_ITER}/mesh

    cd ${PROJECT_ROOT}
    ${PYTHON_ENV} -m scripts.generate_colmap_from_torch3d \
        --data_path ${DATASET_ROOT}/${SUBJECT_NAME} \
        --output_folder sparse_from_torch3d \
        --scale 1.95
    cd ${PROJECT_ROOT}
    ${PYTHON_ENV} -m scripts.extract_mesh \
        -s ${DATASET_ROOT}/${SUBJECT_NAME} \
        -m ${PROJECT_ROOT}/output/${SUBJECT_NAME} \
        --ply_path ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/${STAGE1_NAME}/${STAGE1_ITER}/point_cloud_comp_cano.ply \
        --iteration 10000 \
        --skip_test --skip_train --mesh_res 1024 \
        --data_type colmap --use_original_colmap --sparse_folder sparse_from_torch3d

    echo "Waiting for mesh extraction to complete..."
    sleep 20

    mv ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/${STAGE1_NAME}/${STAGE1_ITER}/mesh/fuse_post.ply \
       ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/${STAGE1_NAME}/${STAGE1_ITER}/mesh/mesh-comp_cano.ply

    echo "Mesh saved to: output/${SUBJECT_NAME}/results/${STAGE1_NAME}/${STAGE1_ITER}/mesh/mesh-comp_cano.ply"

    echo ""
    echo "Converting vertex color mesh to textured mesh by Blender..."
    ${BLENDER_ENV} -b -P ${PROJECT_ROOT}/tools/ply_to_obj.py -- ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/$STAGE1_NAME/$STAGE1_ITER/mesh/mesh-comp_cano.ply \
    ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/$STAGE1_NAME/$STAGE1_ITER/mesh/mesh-comp_cano.obj
    echo "Finished Blender conversion."

    echo ""
    echo "Rendering mesh-comp-cano.obj ..."
    ${PYTHON_ENV} ${PROJECT_ROOT}/tools/render_objs.py -f ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/$STAGE1_NAME/$STAGE1_ITER/mesh --mesh_name mesh-comp_cano.obj
    echo "Finished rendering mesh-comp-cano.obj"

    echo ""
    echo "Segmenting the rendering images by Sapiens..."
    cd ${SAPIENS_PROJECT_ROOT}/lite/scripts/demo/torchscript
    ./sapiens_seg.sh ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/$STAGE1_NAME/$STAGE1_ITER/mesh/segm_images ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/$STAGE1_NAME/$STAGE1_ITER/mesh
    echo "Finished segmentation."

    echo "Voting segmentation..."
    cd ${PROJECT_ROOT}
    ${PYTHON_ENV} tools/segment_3d_extract_gs_garment.py -f ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/$STAGE1_NAME/$STAGE1_ITER/mesh \
    --dataset_type dress-4d --mesh_name mesh-comp_cano.ply

    mv ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/$STAGE1_NAME/$STAGE1_ITER/mesh/segms_obj.obj ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/$STAGE1_NAME/$STAGE1_ITER/mesh/segms_upper.obj
    echo "Save the mesh to '${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/$STAGE1_NAME/$STAGE1_ITER/mesh/segms_upper.obj'"

    echo ">>> Stage 1.1 completed!"
fi

# ============================================================================
# Stage 2: Train inner layer (body + lower garment)
# ============================================================================

if [ "$TRAIN_INNER_LAYER" = "true" ]; then
    echo ""
    echo ">>> Stage 2: Training inner layer..."
    echo ""

    # Check prerequisites
    if [ -z "$STAGE1_NAME" ] || [ -z "$STAGE1_ITER" ]; then
        echo "Error: STAGE1_NAME and STAGE1_ITER must be set"
        exit 1
    fi

    STAGE2_VERSION="$(date "+%m%d_%H%M")"
    STAGE2_NAME="${STAGE2_VERSION}_inner"
    ITER=15000

    cd ${PROJECT_ROOT}
    mkdir -p output/${SUBJECT_NAME}/results/${STAGE2_NAME}
    
    # Backup config files
    cp ${PROJECT_ROOT}/layergs/train/inner.py output/${SUBJECT_NAME}/results/${STAGE2_NAME}/
    cp config/config-4d-dress/multilayer-inpaint-upper-${GENDER}.yaml config/
    cp config/config-4d-dress/multilayer-inpaint-upper-${GENDER}.yaml output/${SUBJECT_NAME}/results/${STAGE2_NAME}/

    # Run training
    ${PYTHON_ENV} -m scripts.train inner \
        --data_type ${DATA_TYPE} \
        -s ${DATASET_ROOT}/${SUBJECT_NAME} \
        --scan_name ${SUBJECT_NAME}_norm \
        -m output/${SUBJECT_NAME} \
        --model_type multi-layer-2dgs \
        --sh_degree 0 \
        --apply_2dgs 1 \
        --save_iterations ${ITER} \
        --iterations ${ITER} \
        --pretrained_name "results/${STAGE1_NAME}/${STAGE1_ITER}" \
        --mesh_name "segms_upper.obj" \
        --config_file config/multilayer-inpaint-upper-${GENDER}.yaml \
        --cur_version ${STAGE2_NAME} \
        --port ${PORT}

    STAGE2_ITER="iteration_${ITER}"
    echo ">>> Stage 2 completed!"
fi

# ============================================================================
# Stage 3: Refine outer layer (upper garment)
# ============================================================================

if [ "$TRAIN_OUTER_LAYER" = "true" ]; then
    echo ""
    echo ">>> Stage 3: Refining outer layer..."
    echo ""

    # Check prerequisites
    if [ -z "$STAGE1_NAME" ] || [ -z "$STAGE1_ITER" ] || \
       [ -z "$STAGE2_NAME" ] || [ -z "$STAGE2_ITER" ]; then
        echo "Error: Stage 1 and Stage 2 results must be available"
        exit 1
    fi

    STAGE3_VERSION="$(date "+%m%d_%H%M")"
    STAGE3_NAME="${STAGE3_VERSION}_outer"
    ITER=10200

    cd ${PROJECT_ROOT}
    mkdir -p output/${SUBJECT_NAME}/results/${STAGE3_NAME}
    
    # Backup config files
    cp ${PROJECT_ROOT}/layergs/train/outer.py output/${SUBJECT_NAME}/results/${STAGE3_NAME}/
    cp config/config-4d-dress/multilayer-inpaint-upper-refine-${GENDER}.yaml config/
    cp config/config-4d-dress/multilayer-inpaint-upper-refine-${GENDER}.yaml output/${SUBJECT_NAME}/results/${STAGE3_NAME}/

    # Run training
    ${PYTHON_ENV} -m scripts.train outer \
        --data_type ${DATA_TYPE} \
        -s ${DATASET_ROOT}/${SUBJECT_NAME} \
        --scan_name ${SUBJECT_NAME}_norm \
        -m output/${SUBJECT_NAME} \
        --model_type multi-layer-2dgs \
        --sh_degree 0 \
        --apply_2dgs 1 \
        --save_iterations 6600 ${ITER} \
        --iterations ${ITER} \
        --pretrained_name "results/${STAGE1_NAME}/${STAGE1_ITER}" \
        --mesh_name "segms_upper.obj" \
        --pretrained_inner_name "results/${STAGE2_NAME}/${STAGE2_ITER}" \
        --config_file config/multilayer-inpaint-upper-refine-${GENDER}.yaml \
        --cur_version ${STAGE3_NAME} \
        --port ${PORT}

    STAGE3_ITER="iteration_${ITER}"
    echo ">>> Stage 3 completed!"
fi

# ============================================================================
# Stage 3.1: Extract refined outer layer
# ============================================================================

if [ "$EXTRACT_REFINED_UPPER" = "true" ]; then
    echo ""
    echo ">>> Stage 3.1: Extract refined outer layer..."
    echo ""

    # Check prerequisites
    if [ -z "$STAGE1_NAME" ] || [ -z "$STAGE1_ITER" ] || \
       [ -z "$STAGE2_NAME" ] || [ -z "$STAGE2_ITER" ] || \
       [ -z "$STAGE3_NAME" ] || [ -z "$STAGE3_ITER" ]; then
        echo "Error: Stage 1, Stage 2 and Stage 3 results must be available"
        exit 1
    fi

    # run concatenate gaussians script
    cd ${PROJECT_ROOT}
    ${PYTHON_ENV} -m scripts.concatenate_gs --data_type pytorch3d --model_type multi-layer-2dgs --sh_degree 0 --apply_2dgs 1 --body_path output/${SUBJECT_NAME}/results/${STAGE3_NAME}/${STAGE3_ITER}/point_cloud_body_cano.ply --garment_path output/${SUBJECT_NAME}/results/${STAGE3_NAME}/${STAGE3_ITER}/point_cloud_garment_cano.ply --output_path output/${SUBJECT_NAME}/results/${STAGE3_NAME}/${STAGE3_ITER}/point_cloud_comp_cano.ply

    mkdir -p ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/$STAGE3_NAME/$STAGE3_ITER/mesh

    cd ${PROJECT_ROOT}
    ${PYTHON_ENV} -m scripts.extract_mesh \
        -s ${DATASET_ROOT}/${SUBJECT_NAME} \
        -m ${PROJECT_ROOT}/output/${SUBJECT_NAME} \
        --ply_path ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/${STAGE3_NAME}/${STAGE3_ITER}/point_cloud_comp_cano.ply \
        --iteration 10000 \
        --skip_test --skip_train --mesh_res 1024 \
        --data_type colmap --use_original_colmap --sparse_folder sparse_from_torch3d

    echo "Waiting for mesh extraction to complete..."
    sleep 20
    mv ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/${STAGE3_NAME}/${STAGE3_ITER}/mesh/fuse_post.ply \
       ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/${STAGE3_NAME}/${STAGE3_ITER}/mesh/mesh-comp_cano.ply
    echo "Mesh saved to: output/${SUBJECT_NAME}/results/${STAGE3_NAME}/${STAGE3_ITER}/mesh/mesh-comp_cano.ply"

    echo ""
    echo "Converting vertex color mesh to textured mesh by Blender..."
    ${BLENDER_ENV} -b -P ${PROJECT_ROOT}/tools/ply_to_obj.py -- ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/$STAGE1_NAME/$STAGE1_ITER/mesh/mesh-comp_cano.ply \
    ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/$STAGE3_NAME/$STAGE3_ITER/mesh/mesh-comp_cano.obj
    echo "Finished Blender conversion."

    echo ""
    echo "Rendering mesh-comp-cano.obj ..."
    ${PYTHON_ENV} ${PROJECT_ROOT}/tools/render_objs.py -f ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/$STAGE3_NAME/$STAGE3_ITER/mesh --mesh_name mesh-comp_cano.obj
    echo "Finished rendering mesh-comp-cano.obj"

    echo ""
    echo "Segmenting the rendering images by Sapiens..."
    cd ${SAPIENS_PROJECT_ROOT}/lite/scripts/demo/torchscript
    ./sapiens_seg.sh ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/$STAGE3_NAME/$STAGE3_ITER/mesh/segm_images ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/$STAGE3_NAME/$STAGE3_ITER/mesh
    echo "Finished segmentation."

    echo "Voting segmentation..."
    cd ${PROJECT_ROOT}
    ${PYTHON_ENV} tools/segment_3d_extract_gs_garment.py -f ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/$STAGE3_NAME/$STAGE3_ITER/mesh \
    --dataset_type dress-4d --mesh_name mesh-comp_cano.ply

    mv ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/$STAGE3_NAME/$STAGE3_ITER/mesh/segms_obj.obj ${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/$STAGE3_NAME/$STAGE3_ITER/mesh/segms_upper.obj
    echo "Save the mesh to '${PROJECT_ROOT}/output/${SUBJECT_NAME}/results/$STAGE3_NAME/$STAGE3_ITER/mesh/segms_upper.obj'"

    cp ${PROJECT_ROOT}/output/$SUBJECT_NAME/results/$STAGE3_NAME/$STAGE3_ITER/point_cloud_garment_cano.ply \
    ${PROJECT_ROOT}/output/$SUBJECT_NAME/results/$STAGE3_NAME/$STAGE3_ITER/point_cloud_upper_cano.ply
    echo ">>> Stage 3.1 completed!"
fi

echo ""
echo "============================================"
echo "Training pipeline completed!"
echo "============================================"
