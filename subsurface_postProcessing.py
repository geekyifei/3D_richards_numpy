"""
Post-processing utilities for the Richards equation solver.

Loads .npy snapshots and provides:
  - reshaping to 3D arrays (Nx, Ny, Nz)
  - horizontal-slice extraction (for GeoTIFF export)
  - vertical profile extraction
  - water balance time series

Linear algebra requirement: none.
"""

import os
import glob
import numpy as np


def load_snapshots(output_dir, field='h'):
    """
    Load all saved snapshots for a given field.

    Parameters
    ----------
    output_dir : str   Directory written by Richards._save_output()
    field      : str   'h' or 'theta'

    Returns
    -------
    times  : ndarray (T,)      simulation times [s]
    data   : ndarray (T, N)    field values at each time
    """
    pattern = os.path.join(output_dir, f"{field}_*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No '{field}' snapshots in {output_dir}")

    times, arrays = [], []
    for fpath in files:
        # Filename format: h_<t_string>.npy  where t_string = "0_000" for t=0.0
        basename = os.path.splitext(os.path.basename(fpath))[0]
        t_str = basename[len(field) + 1:].replace('_', '.')
        times.append(float(t_str))
        arrays.append(np.load(fpath))

    order = np.argsort(times)
    return np.array(times)[order], np.array(arrays)[order]


def reshape_3d(flat_array, Nx, Ny, Nz):
    """
    Reshape a flat (N,) field array to (Nx, Ny, Nz).

    Cell ordering: p = i + j*Nx + k*Nx*Ny
    """
    return flat_array.reshape(Nx, Ny, Nz)


def horizontal_slice(flat_array, Nx, Ny, Nz, k):
    """
    Extract a horizontal (x-y) slice at depth layer k.

    Parameters
    ----------
    k : int   layer index (0 = bottom, Nz-1 = top)

    Returns
    -------
    slice_2d : ndarray (Nx, Ny)
    """
    arr3d = reshape_3d(flat_array, Nx, Ny, Nz)
    return arr3d[:, :, k]


def vertical_profile(flat_array, Nx, Ny, Nz, i=0, j=0):
    """
    Extract a vertical (z) profile at cell (i, j).

    Returns
    -------
    profile : ndarray (Nz,)
    """
    arr3d = reshape_3d(flat_array, Nx, Ny, Nz)
    return arr3d[i, j, :]


def water_balance_timeseries(output_dir, Nx, Ny, Nz, dx, dy, dz):
    """
    Compute total water volume over time from saved theta snapshots.

    Parameters
    ----------
    Nx, Ny, Nz : int    grid dimensions
    dx, dy, dz : float  cell sizes [m]

    Returns
    -------
    times   : ndarray (T,)   simulation times [s]
    volumes : ndarray (T,)   total water volume in domain [m³]
    """
    times, data = load_snapshots(output_dir, field='theta')
    dV = dx * dy * dz
    volumes = np.sum(data, axis=1) * dV
    return times, volumes


def export_vertical_profiles(output_dir, Nx, Ny, Nz, dz, field='h'):
    """
    Export vertical profiles at cell (0,0) for all time steps.

    Returns
    -------
    times    : ndarray (T,)
    z_coords : ndarray (Nz,)   cell-centre z coordinates [m]
    profiles : ndarray (T, Nz)
    """
    times, data = load_snapshots(output_dir, field=field)
    z_coords = (np.arange(Nz) + 0.5) * dz

    profiles = np.array([
        vertical_profile(data[t_idx], Nx, Ny, Nz, i=0, j=0)
        for t_idx in range(len(times))
    ])
    return times, z_coords, profiles
