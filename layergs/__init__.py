#
# LayerGS: Layered Gaussian Splatting for Human Avatars
#
# A multi-layer 2D Gaussian Splatting framework for reconstructing
# animatable human avatars with separable body and garment layers.
#

__version__ = "1.0.0"

from . import rendering
from . import train
from . import utils
from . import data
from . import scene
from . import guidance

__all__ = [
    "rendering",
    "train",
    "utils",
    "data",
    "scene",
    "guidance",
]

