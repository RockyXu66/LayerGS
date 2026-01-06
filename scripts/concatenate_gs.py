from loguru import logger

import sys
from layergs.scene import GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from layergs.utils.train_utils import concat_gaussians

def training(dataset, args):
    gaussians_garment = GaussianModel(dataset.sh_degree, apply_2dgs=True)

    gaussians_body = GaussianModel(sh_degree=dataset.sh_degree, apply_2dgs=True)
    gaussians_body.load_ply(args.body_path)
    gaussians_body.init_obj(object_idx=6)
    gaussians_body.init_other()

    gaussians_garment = GaussianModel(sh_degree=dataset.sh_degree, apply_2dgs=True)
    gaussians_garment.load_ply(args.garment_path)
    gaussians_garment.init_obj(object_idx=3)
    gaussians_garment.init_other()

    gaussians_whole = concat_gaussians([gaussians_body, gaussians_garment])
    gaussians_whole.init_obj(object_idx=6)
    gaussians_whole.init_other()

    point_cloud_path = args.output_path
    gaussians_whole.save_ply(point_cloud_path)
    print(f'Save the new concatenated gs in {point_cloud_path}')
    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument( "--body_path", type=str)
    parser.add_argument( "--garment_path", type=str)
    parser.add_argument( "--output_path", type=str)
    args = parser.parse_args(sys.argv[1:])

    args = op.set_lr(args)      # set different learning rate based on the task

    # Read and parse the configuration file
    training(lp.extract(args), args)
