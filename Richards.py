"""
3D Richards equation solver (mixed form, FVM, Backward Euler, Picard).

Governing equation (following RichardsFoam3):

    C(h) ∂h/∂t = ∇·[K(h)(∇h + e_z)] - AET

where h is pressure head [m], z is elevation (positive upward),
C(h) is specific moisture capacity [1/m], K(h) is unsaturated
hydraulic conductivity [m/s], and AET is evapotranspiration [1/s].

Discretisation
--------------
* Cell-centred FVM on a 3D structured Cartesian grid (Nx × Ny × Nz).
* Cell ordering: p = i + j*Nx + k*Nx*Ny  (k=0 bottom, k=Nz-1 top).
* Backward Euler time integration.
* Picard (fixed-point) iteration for the nonlinearity.
* Inter-cell K computed by arithmetic mean at shared faces.

Linear algebra
--------------
**scipy.sparse is REQUIRED.**  At each Picard iteration a sparse
(7-band) linear system A h = b is assembled and solved with:

    scipy.sparse.linalg.spsolve(A, b)

The matrix A has shape (N, N) with up to 7 non-zeros per row.
For a 100×100×50 grid (N=500,000) a direct solver becomes memory-
intensive; profiling should determine when an iterative solver
(e.g., scipy.sparse.linalg.bicgstab with ILU preconditioning) is
preferable.  The solver choice can be swapped by changing
_solve_linear_system().

Dependencies: numpy, scipy
"""

import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .RichardsInput import RichardsInput
from .soil_models import VanGenuchtenMualem, VanGenuchtenParams
from .subsurface_preProcessing import (
    build_grid,
    prepare_initial_conditions,
    prepare_soil_arrays,
    prepare_source_term,
)


class Richards:
    """
    3D Richards equation solver.

    Parameters
    ----------
    inp : RichardsInput
        Fully configured input object.

    Example
    -------
    solver = Richards(inp)
    solver.run()
    """

    def __init__(self, inp: RichardsInput):
        inp.validate()
        self._inp = inp

        # Build grid
        self._grid = build_grid(inp)
        Nx, Ny, Nz = self._grid['Nx'], self._grid['Ny'], self._grid['Nz']
        self._Nx = Nx
        self._Ny = Ny
        self._Nz = Nz
        self._N = self._grid['N']
        self._dx = self._grid['dx']
        self._dy = self._grid['dy']
        self._dz = self._grid['dz']
        self._zflat = self._grid['zflat']  # elevation of each cell centre [m]

        # Soil model
        soil = prepare_soil_arrays(inp, self._grid)
        self._soil = soil
        _params = VanGenuchtenParams(
            K_s=soil['K_s'], alpha=soil['alpha'], n=soil['n'],
            theta_s=soil['theta_s'], theta_r=soil['theta_r'], S_s=soil['S_s'],
        )
        self._vgm = VanGenuchtenMualem(_params)

        # State arrays
        self._h = prepare_initial_conditions(inp, self._grid).copy()  # pressure head [m]
        self._theta = self._vgm.theta(self._h)                         # water content [-]

        # Source term
        self._AET = prepare_source_term(inp, self._grid)

        # Time
        self.t = 0.0
        self._dt = inp.dt_init
        self._dt_max = inp.dt_max
        self._dt_min = inp.dt_min

        # Picard controls (mirror RichardsFoam3 defaults)
        self._picard_max_iter = inp.picard_max_iter
        self._picard_tol = inp.picard_tol
        self._picard_dt_factor = inp.picard_dt_factor
        self._picard_stable_thresh = inp.picard_stable_thresh
        self._stable_count = 0   # consecutive steps with < 3 Picard iters

        # Pre-build neighbour index arrays (static for structured grid)
        self._build_neighbour_indices()

        # Output
        self._output_dir = inp.output_dir
        self._output_interval = inp.output_interval
        self._next_output_t = 0.0
        self._h_history = []   # list of (t, h_snapshot)

        # Sparse linear solver (swap this to change backend)
        self._linear_solver = 'spsolve'  # or 'bicgstab'

    # ------------------------------------------------------------------
    # Grid connectivity
    # ------------------------------------------------------------------

    def _build_neighbour_indices(self):
        """
        Pre-compute flat indices of the 6 neighbours for every cell.

        Shape of each array: (N,)
        Value -1 indicates a boundary (no neighbour in that direction).
        """
        N = self._N
        Nx, Ny, Nz = self._Nx, self._Ny, self._Nz

        p = np.arange(N, dtype=np.int64)

        # Decompose flat index
        k = p // (Nx * Ny)
        rem = p % (Nx * Ny)
        j = rem // Nx
        i = rem % Nx

        def nb(di, dj, dk):
            ni = i + di
            nj = j + dj
            nk = k + dk
            valid = (
                (ni >= 0) & (ni < Nx) &
                (nj >= 0) & (nj < Ny) &
                (nk >= 0) & (nk < Nz)
            )
            idx = np.where(valid, ni + nj * Nx + nk * Nx * Ny, -1)
            return idx.astype(np.int64)

        self._nb_xp = nb(+1,  0,  0)   # x+
        self._nb_xm = nb(-1,  0,  0)   # x-
        self._nb_yp = nb( 0, +1,  0)   # y+
        self._nb_ym = nb( 0, -1,  0)   # y-
        self._nb_zp = nb( 0,  0, +1)   # z+ (upward)
        self._nb_zm = nb( 0,  0, -1)   # z- (downward)

        # Boolean masks for boundary faces
        self._is_top    = (k == Nz - 1)   # top face cells
        self._is_bottom = (k == 0)         # bottom face cells
        self._is_xp     = (i == Nx - 1)
        self._is_xm     = (i == 0)
        self._is_yp     = (j == Ny - 1)
        self._is_ym     = (j == 0)

    # ------------------------------------------------------------------
    # Conductivity at faces (arithmetic mean)
    # ------------------------------------------------------------------

    def _face_K(self, K_cell):
        """
        Compute K at the 6 faces of each cell using arithmetic mean.

        Returns
        -------
        dict with keys 'xp', 'xm', 'yp', 'ym', 'zp', 'zm'
        Each array has shape (N,).  At boundary faces the own-cell K is used.
        """
        def arith(nb_idx):
            K_nb = np.where(nb_idx >= 0, K_cell[nb_idx], K_cell)
            return 0.5 * (K_cell + K_nb)

        return {
            'xp': arith(self._nb_xp),
            'xm': arith(self._nb_xm),
            'yp': arith(self._nb_yp),
            'ym': arith(self._nb_ym),
            'zp': arith(self._nb_zp),
            'zm': arith(self._nb_zm),
        }

    # ------------------------------------------------------------------
    # Sparse matrix assembly
    # ------------------------------------------------------------------

    def _assemble(self, h, h_old, dt):
        """
        Assemble the sparse linear system  A * h_new = b  for one
        Picard iteration, with K and C frozen at the current h.

        The discretised mixed-form Richards equation for cell p:

          C_p * (h_p - h_old_p) / dt
              = sum_faces { K_f * [(h_nb - h_p)/dl_f + g_f] / dl_f }
              - AET_p

        where g_f = +1 for the z+ face, -1 for z-, 0 for x/y faces
        (gravity contribution), and dl_f is the face spacing (dx/dy/dz).

        Rearranging into  A * h_new = b:

          A[p,p]  = C_p/dt + sum_faces(K_f/dl_f^2)
          A[p,nb] = -K_f/dl_f^2  (for each interior neighbour nb)
          b[p]    = C_p/dt * h_old_p + sum_faces(K_f * g_f / dl_f) - AET_p

        Boundary conditions modify specific rows (see _apply_bcs).

        Parameters
        ----------
        h     : ndarray (N,)  current Picard iterate (h^ν)
        h_old : ndarray (N,)  pressure head at start of time step
        dt    : float         current time step [s]

        Returns
        -------
        A : scipy.sparse.csr_matrix  shape (N, N)
        b : ndarray (N,)
        """
        N = self._N
        dx, dy, dz = self._dx, self._dy, self._dz

        K_cell = self._vgm.K(h)
        C_cell = self._vgm.C(h, h_old)

        Kf = self._face_K(K_cell)

        # Diagonal accumulator and RHS
        diag = C_cell / dt
        rhs = C_cell / dt * h_old - self._AET

        # Off-diagonal entries (row, col, value) collected in lists
        rows, cols, vals = [], [], []

        # Helper: add contribution from one face direction
        def _add_face(nb_idx, K_f, dl, gravity_sign):
            # gravity_sign: +1 for z+ face, -1 for z-, 0 for x/y
            nonlocal diag, rhs

            alpha_f = K_f / (dl * dl)
            diag += alpha_f

            # Interior faces: off-diagonal entry + gravity source
            interior = nb_idx >= 0
            p_in = np.where(interior)[0]
            nb_in = nb_idx[p_in]

            rows.append(p_in)
            cols.append(nb_in)
            vals.append(-alpha_f[p_in])

            # Gravity source term (z-direction only)
            if gravity_sign != 0:
                rhs += K_f / dz * gravity_sign
                # At boundary faces (no neighbour), the boundary treatment
                # is handled in _apply_bcs; nothing extra here for interior.

        _add_face(self._nb_xp, Kf['xp'], dx, 0)
        _add_face(self._nb_xm, Kf['xm'], dx, 0)
        _add_face(self._nb_yp, Kf['yp'], dy, 0)
        _add_face(self._nb_ym, Kf['ym'], dy, 0)
        _add_face(self._nb_zp, Kf['zp'], dz, +1)
        _add_face(self._nb_zm, Kf['zm'], dz, -1)

        # Assemble diagonal
        p_all = np.arange(N, dtype=np.int64)
        rows.append(p_all)
        cols.append(p_all)
        vals.append(diag)

        rows_arr = np.concatenate(rows).astype(np.int64)
        cols_arr = np.concatenate(cols).astype(np.int64)
        vals_arr = np.concatenate(vals)

        A = sp.csr_matrix(
            (vals_arr, (rows_arr, cols_arr)),
            shape=(N, N),
            dtype=float,
        )

        return A, rhs

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _apply_bcs(self, A, b, h, dt):
        """
        Modify A and b in-place to enforce boundary conditions.

        Supported BC types
        ------------------
        Top face (k = Nz-1):
          'rainfall'   : mixed Neumann/Dirichlet (RichardsFoam3 style)
          'dirichlet'  : prescribed head
          'no_flux'    : zero normal flux (already handled by ghost-cell absence)

        Bottom face (k = 0):
          'free_drainage' : ∂(h+z)/∂z = 1  →  flux = K_s (gravity drainage)
          'no_flux'       : zero flux
          'dirichlet'     : prescribed head

        Lateral faces:
          'no_flux'       : default (no extra modification needed because
                            boundary faces have no neighbour in the assembly)

        Note: the 'no_flux' lateral condition is implicitly satisfied
        because _assemble() does not add off-diagonal entries for boundary
        faces (nb_idx = -1 is excluded).  Only the gravity source term at
        the top and bottom z-faces needs explicit treatment.

        Parameters
        ----------
        A : lil_matrix (will be converted to csr after this call)
        b : ndarray (N,)
        h : ndarray (N,) current iterate
        dt: float
        """
        A = A.tolil()
        bc_top = self._inp.bc_top
        bc_bot = self._inp.bc_bottom
        dz = self._dz
        N = self._N
        p_top = np.where(self._is_top)[0]
        p_bot = np.where(self._is_bottom)[0]

        # ---- TOP FACE ----
        top_type = bc_top.get('type', 'no_flux')

        if top_type == 'dirichlet':
            h_bc = float(bc_top['h'])
            for p in p_top:
                A[p, :] = 0.0
                A[p, p] = 1.0
                b[p] = h_bc

        elif top_type == 'rainfall':
            # Mixed BC (RichardsFoam3 rainFluxMixedFvPatchScalarField):
            #   If h < 0 at top cell: Neumann  ∂h/∂n = rainfall/K - 1
            #   If h >= 0            : Dirichlet h = 0 (surface ponding head)
            rate = float(bc_top.get('rate', 0.0))  # [m/s], positive downward
            K_top = self._vgm.K(h[p_top])
            h_top_cell = h[p_top]
            for idx, p in enumerate(p_top):
                if h_top_cell[idx] < 0.0:
                    # Neumann: prescribed gradient at the top face
                    # flux_top = K_face * (∂h/∂z + 1) = rate
                    # → ∂h/∂z = rate/K - 1
                    # The z+ face of the top cell is a boundary; in the
                    # assembled matrix the z+ alpha_f contribution still
                    # appears on the diagonal (no off-diagonal for boundary
                    # face). We correct the RHS to impose the prescribed flux.
                    K_f = K_top[idx]
                    # Remove the spurious diagonal contribution from the
                    # absent z+ neighbour and replace with the prescribed flux.
                    A[p, p] -= K_f / (dz * dz)   # undo alpha_f added in _assemble
                    # Add prescribed Neumann flux to RHS:
                    # Divergence contribution = F_top / dz
                    # F_top = -rate (positive rate = inflow, sign: flux in = positive divergence)
                    b[p] += rate / dz
                    # Remove gravity source from top face that was added in _assemble
                    b[p] -= K_f / dz
                else:
                    # Dirichlet: surface saturated, set h = 0
                    A[p, :] = 0.0
                    A[p, p] = 1.0
                    b[p] = 0.0

        # no_flux: nothing to do (absence of z+ neighbour already gives zero flux)

        # ---- BOTTOM FACE ----
        bot_type = bc_bot if isinstance(bc_bot, str) else bc_bot.get('type', 'free_drainage')

        if bot_type == 'free_drainage':
            # Free drainage: ∂(h+z)/∂z = 1 at bottom face
            # → flux = K(h)  (gravity drainage, no capillary gradient)
            # The z- face of the bottom cell is a boundary.
            # In _assemble, the z- alpha_f is on the diagonal but no off-diagonal.
            # We impose: K_f * (0 + 1) / dz = K_f / dz as the divergence contribution.
            # This is equivalent to a ghost cell at h = h_bottom_cell
            # (so ∂h/∂z = 0) with gravity = 1.
            # → The gravity source already handles this: b[p] already has
            #   -K_zm/dz from the z- face in _assemble.
            # No modification needed beyond what _assemble already produced.
            pass

        elif bot_type == 'no_flux':
            # Zero flux at bottom: remove gravity source added by z- face
            K_bot = self._vgm.K(h[p_bot])
            for idx, p in enumerate(p_bot):
                # Undo the gravity contribution from z- face (added with sign -1)
                b[p] += K_bot[idx] / dz
                # Undo the alpha_f from z- face on diagonal (no Darcy flux)
                A[p, p] -= K_bot[idx] / (dz * dz)

        elif bot_type == 'dirichlet':
            h_bc = float(bc_bot['h']) if isinstance(bc_bot, dict) else 0.0
            for p in p_bot:
                A[p, :] = 0.0
                A[p, p] = 1.0
                b[p] = h_bc

        return A.tocsr(), b

    # ------------------------------------------------------------------
    # Linear system solver
    # ------------------------------------------------------------------

    def _solve_linear_system(self, A, b):
        """
        Solve  A * h = b  and return h.

        Current backend: scipy.sparse.linalg.spsolve (direct LU).
        For large grids (N > ~200,000) switch to bicgstab with ILU
        by setting self._linear_solver = 'bicgstab'.

        ALGEBRAIC SOLVER NOTE
        ---------------------
        spsolve uses SuperLU (bundled with scipy) — no extra install needed.
        For N > ~10^5 cells the fill-in of the LU factors grows; memory
        usage can be 10-50× the matrix storage.  At that point, switch to:

            solver = 'bicgstab'

        which uses scipy.sparse.linalg.bicgstab with an ILU(0)
        preconditioner (scipy.sparse.linalg.spilu).  This requires no
        extra dependencies.

        If cuSPARSE (GPU) is desired in a later phase, replace this method.
        """
        if self._linear_solver == 'spsolve':
            return spla.spsolve(A, b)

        elif self._linear_solver == 'bicgstab':
            # ILU(0) preconditioner
            try:
                ilu = spla.spilu(A.tocsc(), drop_tol=1e-4, fill_factor=10)
                M = spla.LinearOperator(A.shape, ilu.solve)
            except Exception:
                M = None  # fall back to unpreconditioned
            h_new, info = spla.bicgstab(A, b, M=M, tol=1e-8, maxiter=1000)
            if info != 0:
                print(f"    [Richards] BiCGSTAB did not converge (info={info}); "
                      "using result anyway.")
            return h_new

        else:
            raise ValueError(f"Unknown linear solver: '{self._linear_solver}'")

    # ------------------------------------------------------------------
    # Single time step (Picard loop)
    # ------------------------------------------------------------------

    def _step(self, dt):
        """
        Advance the solution by one time step dt using Picard iteration.

        Returns
        -------
        converged     : bool
        n_picard_iters: int
        residual_norm : float   (L1 norm at convergence / N)
        """
        h_old = self._h.copy()
        h = h_old.copy()   # Picard iterate starts at h^n

        converged = False
        n_iter = 0
        residual = np.inf

        for nu in range(self._picard_max_iter):
            A, b = self._assemble(h, h_old, dt)
            A, b = self._apply_bcs(A, b, h, dt)

            h_new = self._solve_linear_system(A, b)

            residual = np.sum(np.abs(h_new - h)) / self._N
            h = h_new
            n_iter = nu + 1

            if residual < self._picard_tol:
                converged = True
                break

        # Accept or reject the time step
        if converged:
            self._h = h
            self._theta = self._vgm.theta(self._h)
            self.t += dt
        # If not converged, _step returns False and the caller reduces dt.

        return converged, n_iter, residual

    # ------------------------------------------------------------------
    # Adaptive time stepping (mirrors RichardsFoam3 setDeltaT logic)
    # ------------------------------------------------------------------

    def _update_dt(self, n_iter, converged, dt):
        """
        Adjust dt based on Picard convergence count.

        Rules (from RichardsFoam3 setDeltaT.H):
        - If last step needed >= picard_max_iter - 2 iters (or did not
          converge): dt /= factor  (immediate reduction)
        - If last step needed < 3 iters for picard_stable_thresh
          consecutive steps: dt *= factor  (cautious increase)
        - Always clamp to [dt_min, dt_max].
        """
        factor = self._picard_dt_factor

        if not converged or n_iter >= self._picard_max_iter - 2:
            self._stable_count = 0
            dt = dt / factor
        elif n_iter < 3:
            self._stable_count += 1
            if self._stable_count >= self._picard_stable_thresh:
                dt = dt * factor
                self._stable_count = 0
        else:
            self._stable_count = 0

        return float(np.clip(dt, self._dt_min, self._dt_max))

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def run(self):
        """
        Run the full simulation from t=0 to t_end.

        Saves snapshots according to output_interval.
        Prints time step progress (mirrors HiPIMS flood.py style).
        """
        os.makedirs(self._output_dir, exist_ok=True)

        t_end = self._inp.t_end
        dt = self._dt
        self._next_output_t = 0.0

        self._save_output()   # save initial condition

        while self.t < t_end - 1e-10:
            # Clamp dt to not overshoot t_end or next output time
            dt = min(dt, t_end - self.t)
            if self._output_interval is not None:
                dt = min(dt, self._next_output_t - self.t + 1e-12)
            dt = max(dt, self._dt_min)

            converged, n_iter, residual = self._step(dt)

            if not converged:
                dt = self._update_dt(n_iter, False, dt)
                if dt < self._dt_min:
                    raise RuntimeError(
                        f"Richards: time step collapsed below dt_min={self._dt_min} "
                        f"at t={self.t:.4f} s.  Check soil parameters and BCs."
                    )
                # Retry with reduced dt (the step was not accepted)
                continue

            dt = self._update_dt(n_iter, True, dt)
            print(
                f"  t={self.t:10.3f} s  dt={dt:.3e} s  "
                f"Picard={n_iter:2d}  res={residual:.2e}"
            )

            if self._should_output():
                self._save_output()

        print(f"Richards: simulation complete at t={self.t:.3f} s")

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _should_output(self):
        if self._output_interval is None:
            return True
        if self.t >= self._next_output_t - 1e-10:
            return True
        return False

    def _save_output(self):
        """Save h and theta snapshots as .npy files."""
        tag = f"{self.t:.3f}".replace('.', '_')
        np.save(os.path.join(self._output_dir, f"h_{tag}.npy"), self._h)
        np.save(os.path.join(self._output_dir, f"theta_{tag}.npy"), self._theta)

        if self._output_interval is not None:
            self._next_output_t += self._output_interval

        self._h_history.append((self.t, self._h.copy()))

    # ------------------------------------------------------------------
    # Post-step diagnostics
    # ------------------------------------------------------------------

    def water_balance(self, h_initial=None):
        """
        Compute total water volume in the domain [m³].

        Parameters
        ----------
        h_initial : ndarray (N,), optional
            Initial h for computing ΔS.  If None, just returns current volume.

        Returns
        -------
        V_current : float  current total water volume [m³]
        delta_S   : float  change in storage since h_initial (if provided)
        """
        dV = self._dx * self._dy * self._dz
        theta_now = self._vgm.theta(self._h)
        V_now = float(np.sum(theta_now) * dV)
        if h_initial is None:
            return V_now, None
        theta_init = self._vgm.theta(h_initial)
        delta_S = (np.sum(theta_now) - np.sum(theta_init)) * dV
        return V_now, float(delta_S)

    def darcy_flux(self):
        """
        Compute cell-centre Darcy flux vector [m/s].

        Returns
        -------
        Ux, Uy, Uz : ndarray (N,) each
            Components of U = -K(h)(∇h + e_z).
        """
        h = self._h
        K_cell = self._vgm.K(h)
        dx, dy, dz = self._dx, self._dy, self._dz

        def gradient_component(nb_idx, dl, is_bnd_lo, is_bnd_hi, gravity=0.0):
            h_nb = np.where(nb_idx >= 0, h[nb_idx], h)
            grad = (h_nb - h) / dl + gravity
            # Zero flux at domain boundaries
            grad = np.where(is_bnd_hi & (nb_idx < 0), 0.0, grad)
            return grad

        dh_dx = 0.5 * (
            gradient_component(self._nb_xp, dx, self._is_xm, self._is_xp) -
            gradient_component(self._nb_xm, dx, self._is_xm, self._is_xp)
        )
        dh_dy = 0.5 * (
            gradient_component(self._nb_yp, dy, self._is_ym, self._is_yp) -
            gradient_component(self._nb_ym, dy, self._is_ym, self._is_yp)
        )
        dhdz_plus_gz = 0.5 * (
            gradient_component(self._nb_zp, dz, self._is_bottom, self._is_top, gravity=1.0) -
            gradient_component(self._nb_zm, dz, self._is_bottom, self._is_top, gravity=-1.0)
        )

        Ux = -K_cell * dh_dx
        Uy = -K_cell * dh_dy
        Uz = -K_cell * dhdz_plus_gz

        return Ux, Uy, Uz
