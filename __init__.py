"""
3D Richards equation subsurface flow module for HiPIMS-GWF.

Quick start
-----------
from hipims.subsurface import Richards, RichardsInput

inp = RichardsInput()
inp.set_domain(Nx=1, Ny=1, Nz=50, dx=1.0, dy=1.0, dz=0.02)
inp.set_soil(soil_type='sandy_loam')
inp.set_initial_conditions(h0=-1.0)
inp.set_boundary_conditions(
    top={'type': 'rainfall', 'rate': 1e-5},
    bottom='free_drainage',
)
inp.set_time(t_end=3600.0, dt_init=60.0)
inp.set_output(output_dir='output/', output_interval=300.0)

solver = Richards(inp)
solver.run()
"""

from .Richards import Richards
from .RichardsInput import RichardsInput
from .soil_models import VanGenuchtenMualem, VanGenuchtenParams, SOIL_PRESETS
from .subsurface_preProcessing import build_grid
from .subsurface_postProcessing import (
    load_snapshots,
    reshape_3d,
    vertical_profile,
    horizontal_slice,
    water_balance_timeseries,
)

__all__ = [
    'Richards',
    'RichardsInput',
    'VanGenuchtenMualem',
    'VanGenuchtenParams',
    'SOIL_PRESETS',
    'build_grid',
    'load_snapshots',
    'reshape_3d',
    'vertical_profile',
    'horizontal_slice',
    'water_balance_timeseries',
]
