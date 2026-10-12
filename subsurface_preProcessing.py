"""
Pre-processing utilities for the 3D Richards equation solver.

Builds grid coordinate arrays, flattens initial condition fields,
and validates boundary condition specifications.

Linear algebra requirement: none.
"""

import numpy as np
from .RichardsInput import RichardsInput


def build_grid(inp: RichardsInput):
    """
    Build 3D structured grid arrays from a RichardsInput object.

    Cell ordering (C-order, row-major):
        global index p = i + j * Nx + k * Nx * Ny

    where:
        i ∈ [0, Nx)  →  x-direction
        j ∈ [0, Ny)  →  y-direction
        k ∈ [0, Nz)  →  z-direction (k=0 = bottom, k=Nz-1 = top)

    Parameters
    ----------
    inp : RichardsInput

    Returns
    -------
    grid : dict with keys:
        'Nx', 'Ny', 'Nz', 'N'       grid dimensions and total cell count
        'dx', 'dy', 'dz'             cell sizes [m]
        'x', 'y', 'z'               1D coordinate arrays of cell centres [m]
        'xc', 'yc', 'zc'            3D arrays of cell-centre coordinates, shape (Nx,Ny,Nz)
        'zflat'                     flattened z-coordinates [m], shape (N,)
    """
    Nx, Ny, Nz = inp.Nx, inp.Ny, inp.Nz
    dx, dy, dz = inp.dx, inp.dy, inp.dz
    N = Nx * Ny * Nz

    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    z = (np.arange(Nz) + 0.5) * dz  # z=0 at bottom of domain

    # 3D coordinate arrays
    xc, yc, zc = np.meshgrid(x, y, z, indexing='ij')  # shape (Nx, Ny, Nz)

    return {
        'Nx': Nx, 'Ny': Ny, 'Nz': Nz, 'N': N,
        'dx': dx, 'dy': dy, 'dz': dz,
        'x': x, 'y': y, 'z': z,
        'xc': xc, 'yc': yc, 'zc': zc,
        'zflat': zc.ravel(),
    }


def prepare_initial_conditions(inp: RichardsInput, grid: dict):
    """
    Expand initial pressure head to a flat array of length N.

    Parameters
    ----------
    inp  : RichardsInput
    grid : dict from build_grid()

    Returns
    -------
    h0 : ndarray, shape (N,), dtype float64
    """
    N = grid['N']
    Nx, Ny, Nz = grid['Nx'], grid['Ny'], grid['Nz']
    h0_raw = inp._h0

    h0 = np.empty(N, dtype=float)

    if np.isscalar(h0_raw):
        h0[:] = float(h0_raw)

    elif np.asarray(h0_raw).shape == (Nz,):
        # Uniform in x-y, varying in z
        # Reshape to (Nx, Ny, Nz) then flatten
        h0_3d = np.broadcast_to(
            np.asarray(h0_raw, dtype=float)[np.newaxis, np.newaxis, :],
            (Nx, Ny, Nz)
        ).copy()
        h0[:] = h0_3d.ravel()

    elif np.asarray(h0_raw).shape == (N,):
        h0[:] = np.asarray(h0_raw, dtype=float)

    else:
        raise ValueError(
            f"h0 shape {np.asarray(h0_raw).shape} not compatible with grid "
            f"(Nz={Nz}, N={N})."
        )

    return h0


def prepare_soil_arrays(inp: RichardsInput, grid: dict):
    """
    Expand soil parameter scalars/arrays to flat arrays of length N.

    Returns
    -------
    soil : dict with keys 'K_s', 'alpha', 'n', 'theta_s', 'theta_r', 'S_s'
           each an ndarray of shape (N,)
    """
    N = grid['N']
    p = inp._soil_params

    def _expand(val):
        arr = np.asarray(val, dtype=float)
        if arr.shape == ():
            return np.full(N, arr)
        if arr.shape == (N,):
            return arr.copy()
        raise ValueError(f"Soil parameter shape {arr.shape} incompatible with N={N}.")

    return {
        'K_s':    _expand(p.K_s),
        'alpha':  _expand(p.alpha),
        'n':      _expand(p.n),
        'theta_s': _expand(p.theta_s),
        'theta_r': _expand(p.theta_r),
        'S_s':    _expand(p.S_s),
    }


def prepare_source_term(inp: RichardsInput, grid: dict):
    """
    Expand AET source/sink to flat array of shape (N,) [1/s].
    """
    N = grid['N']
    aet = np.asarray(inp._AET, dtype=float)
    if aet.shape == ():
        return np.full(N, float(aet))
    if aet.shape == (N,):
        return aet.copy()
    raise ValueError(f"AET shape {aet.shape} incompatible with N={N}.")


def cell_index(i, j, k, Nx, Ny):
    """Return the flat cell index for cell (i, j, k)."""
    return int(i) + int(j) * Nx + int(k) * Nx * Ny


def ijk_from_index(p, Nx, Ny):
    """Return (i, j, k) from flat cell index p."""
    k = p // (Nx * Ny)
    rem = p % (Nx * Ny)
    j = rem // Nx
    i = rem % Nx
    return i, j, k
