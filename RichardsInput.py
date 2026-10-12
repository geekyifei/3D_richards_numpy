"""
User-facing configuration class for the 3D Richards equation solver.
users instantiate RichardsInput, call set_*()
methods to configure the simulation, then pass the object to Richards.
"""

import numpy as np
from .soil_models import VanGenuchtenParams, SOIL_PRESETS


class RichardsInput:
    """
    Configuration container for a Richards equation simulation.

    Usage
    -----
    inp = RichardsInput()
    inp.set_domain(Nx=1, Ny=1, Nz=50, dx=1.0, dy=1.0, dz=0.02)
    inp.set_soil(soil_type='sandy_loam')
    inp.set_initial_conditions(h0=-1.0)
    inp.set_boundary_conditions(...)
    inp.set_time(t_end=3600.0, dt_init=60.0)
    inp.set_output(output_dir='output/', output_interval=300.0)
    solver = Richards(inp)
    solver.run()
    """

    def __init__(self):
        # Domain
        self.Nx = None
        self.Ny = None
        self.Nz = None
        self.dx = None
        self.dy = None
        self.dz = None

        # Soil
        self._soil_params = None  # VanGenuchtenParams

        # Initial conditions
        self._h0 = None           # ndarray shape (Nx*Ny*Nz,) or scalar

        # Boundary conditions 
        self.bc_top = None        # dict, see set_boundary_conditions
        self.bc_bottom = None     # dict
        self.bc_lateral = None    # 'no_flux' (only option for now)

        # Time control
        self.t_end = None
        self.dt_init = 60.0
        self.dt_max = 3600.0
        self.dt_min = 1e-3

        # Picard solver settings 
        self.picard_max_iter = 10
        self.picard_tol = 1e-3      # L1 norm tolerance [m]
        self.picard_dt_factor = 1.3 # multiply/divide dt by this factor
        self.picard_stable_thresh = 50  # steps before dt increase

        # Source / sink
        self._AET = 0.0             # actual evapotranspiration

        # Output
        self.output_dir = 'output/'
        self.output_interval = None  # seconds; None = save every step

    # ------------------------------------------------------------------
    # Domain
    # ------------------------------------------------------------------

    def set_domain(self, Nx, Ny, Nz, dx, dy, dz):
        """
        Define the 3D structured grid.

        Parameters
        ----------
        Nx, Ny, Nz : int  Number of cells in x, y, z
        dx, dy, dz : float  Cell size in x, y, z [m]

        Note: z increases upward.  Cell k=0 is the bottom, k=Nz-1 is the top.
        """
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.Nz = int(Nz)
        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)

    # ------------------------------------------------------------------
    # Soil parameters
    # ------------------------------------------------------------------

    def set_soil(self, soil_type=None, params=None):
        """
        Set soil hydraulic parameters.

        Parameters
        ----------
        soil_type : str, optional
            One of the preset names in soil_models.SOIL_PRESETS.
        params : VanGenuchtenParams, optional
            Custom parameter object (takes precedence over soil_type).
        """
        if params is not None:
            self._soil_params = params
        elif soil_type is not None:
            if soil_type not in SOIL_PRESETS:
                raise ValueError(
                    f"Unknown soil_type '{soil_type}'. "
                    f"Available: {list(SOIL_PRESETS.keys())}"
                )
            self._soil_params = SOIL_PRESETS[soil_type]
        else:
            raise ValueError("Provide either soil_type or params.")

    def set_soil_field(self, K_s, alpha, n, theta_s, theta_r, S_s=0.0):
        """
        Set spatially-varying soil parameters as arrays.
        All arrays must have shape (Nx*Ny*Nz,) or be broadcastable to it.
        """
        self._soil_params = VanGenuchtenParams(
            K_s=K_s, alpha=alpha, n=n,
            theta_s=theta_s, theta_r=theta_r, S_s=S_s,
        )

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------

    def set_initial_conditions(self, h0):
        """
        Set initial pressure head field.

        Parameters
        ----------
        h0 : float or ndarray
            Initial pressure head [m].
            Scalar → uniform over all cells.
            Array of shape (Nz,) → uniform in x-y, varying in z.
            Array of shape (Nx*Ny*Nz,) → full 3D field.
        """
        self._h0 = h0

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def set_boundary_conditions(self, top, bottom='free_drainage', lateral='no_flux'):
        """
        Set boundary conditions.

        Parameters
        ----------
        top : dict
            Top-face (k = Nz-1) BC.  One of:
              {'type': 'rainfall',    'rate': float_or_array}   [m/s]
              {'type': 'dirichlet',   'h': float_or_array}      [m]
              {'type': 'no_flux'}
            'rainfall' uses the mixed (Neumann/Dirichlet switch) logic
            from RichardsFoam3: Neumann when h < 0, Dirichlet (h=0)
            when the surface saturates.

        bottom : str or dict
            'free_drainage'    → ∂(h+z)/∂z = 1  i.e. gravity-only flux
            'no_flux'          → zero flux
            {'type': 'dirichlet', 'h': float}  → prescribed head

        lateral : str
            'no_flux'  (only option currently)
        """
        self.bc_top = top if isinstance(top, dict) else {'type': top}
        self.bc_bottom = bottom if isinstance(bottom, dict) else {'type': bottom}
        self.bc_lateral = lateral

    # ------------------------------------------------------------------
    # Time control
    # ------------------------------------------------------------------

    def set_time(self, t_end, dt_init=60.0, dt_max=3600.0, dt_min=1e-3):
        """
        Configure time-stepping parameters.

        Parameters
        ----------
        t_end    : float  Simulation end time [s]
        dt_init  : float  Initial time step [s]
        dt_max   : float  Maximum allowed time step [s]
        dt_min   : float  Minimum time step before aborting [s]
        """
        self.t_end = float(t_end)
        self.dt_init = float(dt_init)
        self.dt_max = float(dt_max)
        self.dt_min = float(dt_min)

    # ------------------------------------------------------------------
    # Solver settings
    # ------------------------------------------------------------------

    def set_solver_params(
        self,
        picard_max_iter=10,
        picard_tol=1e-3,
        picard_dt_factor=1.3,
        picard_stable_thresh=50,
    ):
        """
        Override default Picard iteration and dt-control parameters.

        Parameters
        ----------
        picard_max_iter      : int    Max Picard iterations per time step
        picard_tol           : float  L1 convergence tolerance [m]
        picard_dt_factor     : float  Factor to multiply/divide dt
        picard_stable_thresh : int    Consecutive easy steps before dt increase
        """
        self.picard_max_iter = int(picard_max_iter)
        self.picard_tol = float(picard_tol)
        self.picard_dt_factor = float(picard_dt_factor)
        self.picard_stable_thresh = int(picard_stable_thresh)

    # ------------------------------------------------------------------
    # Source / sink
    # ------------------------------------------------------------------

    def set_evapotranspiration(self, AET):
        """
        Set actual evapotranspiration sink [1/s].

        Parameters
        ----------
        AET : float or ndarray
            Volumetric sink term (positive = extraction).
            Shape (Nx*Ny*Nz,) or scalar.
        """
        self._AET = AET

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def set_output(self, output_dir, output_interval=None):
        """
        Configure output.

        Parameters
        ----------
        output_dir      : str    Directory for .npy output files
        output_interval : float  Save every this many seconds; None = every step
        """
        self.output_dir = output_dir
        self.output_interval = output_interval

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self):
        """Raise ValueError if required fields are not set."""
        required = {
            'Nx': self.Nx, 'Ny': self.Ny, 'Nz': self.Nz,
            'dx': self.dx, 'dy': self.dy, 'dz': self.dz,
            'soil params': self._soil_params,
            'initial conditions': self._h0,
            'BCs': self.bc_top,
            't_end': self.t_end,
        }
        for name, val in required.items():
            if val is None:
                raise ValueError(f"RichardsInput: '{name}' not set.")
