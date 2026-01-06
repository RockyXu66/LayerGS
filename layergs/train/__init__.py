#
# LayerGS Training Module
#
# This module provides training functionality for the LayerGS pipeline:
# - Stage 1 (singlelayer): Train complete avatar as single layer
# - Stage 2 (inner): Train inner body layer while freezing outer garment
# - Stage 3 (outer): Train outer garment layer while freezing inner body
#

# Training modules
from . import singlelayer
from . import inner
from . import outer

__all__ = [
    # Training modules
    "singlelayer",
    "inner",
    "outer",
]

