"""
Soil constitutive models for the Richards equation.
Implements the van Genuchten–Mualem model following the same formulation
as RichardsFoam3 (Orgogozo et al., 2014):

    Se(h)  = [1 + (alpha*|h|)^n]^(-m)         for h < 0
    theta(h) = theta_r + (theta_s - theta_r) * Se
    K(h)   = K_s * Se^l * [1 - (1 - Se^(1/m))^m]^2
    C(h)   = S_s + (theta_s - theta_r) * dSe/dh

where m = 1 - 1/n, l = 0.5 (Mualem pore-connectivity).

Note on C(h): the specific moisture capacity is
evaluated by finite-differencing Se between the current and previous
pressure head to avoid singularity as h -> 0.
"""

import numpy as np

class VanGenuchtenParams:

    def __init__(
        self,
        K_s,
        alpha,
        n,
        theta_s,
        theta_r,
        S_s=0.0,
        l=0.5,
    ):
        """
        Parameters
        ----------
        K_s     : float or ndarray  Saturated hydraulic conductivity [m/s]
        alpha   : float or ndarray  Inverse capillary length [1/m]
        n       : float or ndarray  Shape parameter [-]
        theta_s : float or ndarray  Saturated volumetric water content [-]
        theta_r : float or ndarray  Residual volumetric water content [-]
        S_s     : float or ndarray  Specific storage [1/m], default 0
        l       : float             Mualem pore-connectivity, default 0.5
        """
        self.K_s = np.asarray(K_s, dtype=float)
        self.alpha = np.asarray(alpha, dtype=float)
        self.n = np.asarray(n, dtype=float)
        self.theta_s = np.asarray(theta_s, dtype=float)
        self.theta_r = np.asarray(theta_r, dtype=float)
        self.S_s = np.asarray(S_s, dtype=float)
        self.l = float(l)
        self.m = 1.0 - 1.0 / self.n  # Mualem shape parameter


class VanGenuchtenMualem:
    """
    Van Genuchten–Mualem constitutive model.
    All methods accept and return numpy arrays and operate element-wise,
    so they work for any grid shape (1D, 3D flattened, etc.).
    """

    def __init__(self, params: VanGenuchtenParams):
        self._p = params

    # ------------------------------------------------------------------
    # Effective saturation
    # ------------------------------------------------------------------

    def Se(self, h):
        """
        Effective (normalised) saturation [-].

        Se = 1                            for h >= 0
        Se = [1 + (alpha*|h|)^n]^(-m)    for h < 0
        """
        h = np.asarray(h, dtype=float)
        p = self._p
        Se_unsat = np.power(
            1.0 + np.power(p.alpha * np.abs(np.minimum(h, 0.0)), p.n),
            -p.m,
        )
        return np.where(h >= 0.0, 1.0, Se_unsat)

    # ------------------------------------------------------------------
    # Volumetric water content
    # ------------------------------------------------------------------

    def theta(self, h):
        """
        Volumetric water content [-].

        theta = theta_r + (theta_s - theta_r) * Se(h)
        """
        p = self._p
        return p.theta_r + (p.theta_s - p.theta_r) * self.Se(h)

    # ------------------------------------------------------------------
    # Unsaturated hydraulic conductivity
    # ------------------------------------------------------------------

    def K(self, h):
        """
        Unsaturated hydraulic conductivity [m/s].

        K(h) = K_s * Se^l * [1 - (1 - Se^(1/m))^m]^2    for h < 0
        K(h) = K_s                                        for h >= 0
        """
        h = np.asarray(h, dtype=float)
        p = self._p
        Se_val = self.Se(h)

        # Clip Se to avoid numerical issues at Se ~ 0 or 1
        Se_clipped = np.clip(Se_val, 1e-10, 1.0 - 1e-10)

        K_unsat = (
            p.K_s
            * np.power(Se_clipped, p.l)
            * np.power(
                1.0 - np.power(1.0 - np.power(Se_clipped, 1.0 / p.m), p.m),
                2.0,
            )
        )
        return np.where(h >= 0.0, p.K_s, K_unsat)

    # ------------------------------------------------------------------
    # Specific moisture capacity (finite-difference form, per RichardsFoam3)
    # ------------------------------------------------------------------

    def C(self, h, h_prev, eps=1e-10):
        """
        Specific moisture capacity [1/m] using finite-difference of Se.

        Following RichardsFoam3, C is evaluated as:

            C(h) = S_s + (theta_s - theta_r) * (Se(h) - Se(h_prev))
                         / (h - h_prev + eps * sign)

        This avoids the singularity of the analytic dSe/dh at h -> 0.

        Parameters
        ----------
        h      : current pressure head [m]
        h_prev : pressure head at previous Picard iterate [m]
        eps    : regularisation to prevent division by zero

        Returns
        -------
        C      : specific moisture capacity [1/m]
        """
        h = np.asarray(h, dtype=float)
        h_prev = np.asarray(h_prev, dtype=float)
        p = self._p

        dh = h - h_prev
        # Regularise denominator (same sign as dh, minimum magnitude eps)
        dh_reg = np.where(np.abs(dh) < eps, eps * np.sign(dh + eps), dh)

        dSe = self.Se(h) - self.Se(h_prev)
        return p.S_s + (p.theta_s - p.theta_r) * dSe / dh_reg

    # ------------------------------------------------------------------
    # Analytic C (for unit tests / comparison only)
    # ------------------------------------------------------------------

    def C_analytic(self, h):
        """
        Analytic specific moisture capacity [1/m].

        Derived directly from dθ/dh = (θ_s - θ_r) * dSe/dh:

            dSe/dh = m*n*α*(α|h|)^(n-1) * (1 + (α|h|)^n)^(-(m+1))  for h < 0

        so:
            C(h) = S_s + (θ_s - θ_r) * m*n*α*(α|h|)^(n-1)
                         * (1 + (α|h|)^n)^(-(m+1))               for h < 0
            C(h) = S_s                                            for h >= 0
        """
        h = np.asarray(h, dtype=float)
        p = self._p
        abs_h = np.maximum(np.abs(h), 1e-12)
        alpha_h_n = np.power(p.alpha * abs_h, p.n)  # (α|h|)^n

        dSe_dh = (
            p.m * p.n * p.alpha
            * np.power(p.alpha * abs_h, p.n - 1.0)
            * np.power(1.0 + alpha_h_n, -(p.m + 1.0))
        )
        return p.S_s + np.where(h >= 0.0, 0.0, (p.theta_s - p.theta_r) * dSe_dh)


# ---------------------------------------------------------------------------
# Preset soil parameter sets (Carsel & Parrish, 1988)
# ---------------------------------------------------------------------------

SOIL_PRESETS = {
    "sand": VanGenuchtenParams(
        K_s=8.25e-5, alpha=14.5, n=2.68, theta_s=0.43, theta_r=0.045
    ),
    "loamy_sand": VanGenuchtenParams(
        K_s=4.05e-5, alpha=12.4, n=2.28, theta_s=0.41, theta_r=0.057
    ),
    "sandy_loam": VanGenuchtenParams(
        K_s=1.23e-5, alpha=7.5, n=1.89, theta_s=0.41, theta_r=0.065
    ),
    "loam": VanGenuchtenParams(
        K_s=2.89e-6, alpha=3.6, n=1.56, theta_s=0.43, theta_r=0.078
    ),
    "silt": VanGenuchtenParams(
        K_s=6.94e-7, alpha=1.6, n=1.37, theta_s=0.46, theta_r=0.034
    ),
    "silt_loam": VanGenuchtenParams(
        K_s=1.25e-6, alpha=2.0, n=1.41, theta_s=0.45, theta_r=0.067
    ),
    "sandy_clay_loam": VanGenuchtenParams(
        K_s=3.64e-6, alpha=5.9, n=1.48, theta_s=0.39, theta_r=0.100
    ),
    "clay_loam": VanGenuchtenParams(
        K_s=7.22e-7, alpha=1.9, n=1.31, theta_s=0.41, theta_r=0.095
    ),
    "silty_clay_loam": VanGenuchtenParams(
        K_s=1.94e-7, alpha=1.0, n=1.23, theta_s=0.43, theta_r=0.089
    ),
    "sandy_clay": VanGenuchtenParams(
        K_s=3.33e-6, alpha=2.7, n=1.23, theta_s=0.38, theta_r=0.100
    ),
    "silty_clay": VanGenuchtenParams(
        K_s=5.56e-8, alpha=0.5, n=1.09, theta_s=0.36, theta_r=0.070
    ),
    "clay": VanGenuchtenParams(
        K_s=5.56e-7, alpha=0.8, n=1.09, theta_s=0.38, theta_r=0.068
    ),
    # Haverkamp et al. (1977) Yolo light clay (used for benchmark)
    "yolo_light_clay": VanGenuchtenParams(
        K_s=1.23e-7, alpha=1.478e-2, n=1.259, theta_s=0.495, theta_r=0.124
    ),
}
