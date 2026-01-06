#
# LayerGS Utilities Module
#

from .loss_utils import l1_loss, ssim, loss_cls_3d
from .general_utils import safe_state, PILtoTorch, get_expon_lr_func, strip_symmetric, build_rotation
from .image_utils import psnr, mse
from .train_utils import (
    # Gaussian utilities
    save_multi_layer_gs,
    render_val_cam,
    save_img_sequence,
    concat_gaussians,
    # Training setup utilities
    set_sds,
    prepare_output_and_logger,
    log_system_info,
    setup_wandb,
    training_report_seg,
    setup_training_folders,
    setup_logging,
    get_smplx_path,
    create_folder,
    format_cmd_line,
    SAPIENS_SEG_LABELS,
    DRESS4D_SEG_LABELS,
)
from .camera_utils import (
    loadCam,
    cameraList_from_camInfos,
    camera_to_JSON,
)
from .graphics_utils import (
    getWorld2View2,
    getProjectionMatrix,
    focal2fov,
    fov2focal,
)
from .sh_utils import (
    eval_sh,
    RGB2SH,
    SH2RGB,
)
from .system_utils import (
    mkdir_p,
    searchForMaxIteration,
)
from .metrics import evaluate
from .gaussian_factory import (
    create_gaussians_from_mesh,
    create_gaussians_from_ply,
    load_pretrained_inner_gaussians,
)

__all__ = [
    # Loss functions
    "l1_loss",
    "ssim",
    "loss_cls_3d",
    # General utilities
    "safe_state",
    "PILtoTorch",
    "get_expon_lr_func",
    "strip_symmetric",
    "build_rotation",
    # Image utilities
    "psnr",
    "mse",
    # Training utilities
    "save_multi_layer_gs",
    "render_val_cam",
    "save_img_sequence",
    "concat_gaussians",
    "set_sds",
    "prepare_output_and_logger",
    "log_system_info",
    "setup_wandb",
    "training_report_seg",
    "setup_training_folders",
    "setup_logging",
    "get_smplx_path",
    "create_folder",
    "format_cmd_line",
    "SAPIENS_SEG_LABELS",
    "DRESS4D_SEG_LABELS",
    # Camera utilities
    "loadCam",
    "cameraList_from_camInfos",
    "camera_to_JSON",
    # Graphics utilities
    "getWorld2View2",
    "getProjectionMatrix",
    "focal2fov",
    "fov2focal",
    # SH utilities
    "eval_sh",
    "RGB2SH",
    "SH2RGB",
    # System utilities
    "mkdir_p",
    "searchForMaxIteration",
    # Metrics
    "evaluate",
    # Gaussian factory
    "create_gaussians_from_mesh",
    "create_gaussians_from_ply",
    "load_pretrained_inner_gaussians",
]
