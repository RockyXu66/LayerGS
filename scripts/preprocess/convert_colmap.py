#!/usr/bin/env python3
#
# COLMAP Converter for LayerGS
#
# Converts images to COLMAP format with camera calibration.
# Based on MipNeRF 360 converter.
#
# Usage:
#   python scripts/preprocess/convert_colmap.py -s /path/to/data
#

import os
import sys
import logging
import shutil
from argparse import ArgumentParser

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    parser = ArgumentParser(description="COLMAP converter for LayerGS")
    parser.add_argument("--no_gpu", action='store_true', help="Disable GPU for COLMAP")
    parser.add_argument("--skip_matching", action='store_true', help="Skip feature matching")
    parser.add_argument("--source_path", "-s", required=True, type=str, help="Path to input data")
    parser.add_argument("--camera", default="OPENCV", type=str, help="Camera model")
    parser.add_argument("--colmap_executable", default="", type=str, help="Path to COLMAP executable")
    parser.add_argument("--resize", action="store_true", help="Create resized image versions")
    parser.add_argument("--magick_executable", default="", type=str, help="Path to ImageMagick")
    args = parser.parse_args()
    
    colmap_command = f'"{args.colmap_executable}"' if args.colmap_executable else "colmap"
    magick_command = f'"{args.magick_executable}"' if args.magick_executable else "magick"
    use_gpu = 0 if args.no_gpu else 1

    if not args.skip_matching:
        os.makedirs(f"{args.source_path}/distorted/sparse", exist_ok=True)

        # Feature extraction
        feat_extraction_cmd = (
            f'{colmap_command} feature_extractor '
            f'--database_path {args.source_path}/distorted/database.db '
            f'--image_path {args.source_path}/input '
            f'--ImageReader.single_camera 1 '
            f'--ImageReader.camera_model {args.camera} '
            f'--SiftExtraction.use_gpu {use_gpu}'
        )
        exit_code = os.system(feat_extraction_cmd)
        if exit_code != 0:
            logging.error(f"Feature extraction failed with code {exit_code}")
            return exit_code

        # Feature matching
        feat_matching_cmd = (
            f'{colmap_command} exhaustive_matcher '
            f'--database_path {args.source_path}/distorted/database.db '
            f'--SiftMatching.use_gpu {use_gpu}'
        )
        exit_code = os.system(feat_matching_cmd)
        if exit_code != 0:
            logging.error(f"Feature matching failed with code {exit_code}")
            return exit_code

        # Bundle adjustment
        mapper_cmd = (
            f'{colmap_command} mapper '
            f'--database_path {args.source_path}/distorted/database.db '
            f'--image_path {args.source_path}/input '
            f'--output_path {args.source_path}/distorted/sparse '
            f'--Mapper.ba_global_function_tolerance=0.000001'
        )
        exit_code = os.system(mapper_cmd)
        if exit_code != 0:
            logging.error(f"Mapper failed with code {exit_code}")
            return exit_code

    # Image undistortion
    img_undist_cmd = (
        f'{colmap_command} image_undistorter '
        f'--image_path {args.source_path}/input '
        f'--input_path {args.source_path}/distorted/sparse/0 '
        f'--output_path {args.source_path} '
        f'--output_type COLMAP'
    )
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Undistortion failed with code {exit_code}")
        return exit_code

    # Move sparse files
    files = os.listdir(f"{args.source_path}/sparse")
    os.makedirs(f"{args.source_path}/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(args.source_path, "sparse", file)
        destination_file = os.path.join(args.source_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    # Optional: Create resized versions
    if args.resize:
        print("Creating resized image versions...")
        for scale, folder in [(2, "images_2"), (4, "images_4"), (8, "images_8")]:
            os.makedirs(f"{args.source_path}/{folder}", exist_ok=True)
            files = os.listdir(f"{args.source_path}/images")
            for file in files:
                source_file = os.path.join(args.source_path, "images", file)
                destination_file = os.path.join(args.source_path, folder, file)
                shutil.copy2(source_file, destination_file)
                resize_pct = 100 // scale
                exit_code = os.system(f'{magick_command} mogrify -resize {resize_pct}% {destination_file}')
                if exit_code != 0:
                    logging.error(f"Resize to {resize_pct}% failed")
                    return exit_code

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

