# -*- coding: utf-8 -*-
"""
Thermodynamic functions for VLE calculations.

Antoine equation:  log10(P / bar) = A - B / (T/K + C)
DIPPR equation:    ln(P / Pa)     = A + B/T + C*ln(T) + D*T^E
"""

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Benzene-Toluene Antoine constants  (P in bar, T in Kelvin)
# Derived from Perry's / NIST mmHg form, converted to bar and K.
# Verified: benzene bp 80.1 °C → 1.013 bar; toluene bp 110.6 °C → 1.012 bar.
# ---------------------------------------------------------------------------
BENZENE_ANTOINE = [4.03055, 1211.033, -52.36]
TOLUENE_ANTOINE = [4.07954, 1344.800, -53.668]


# ---------------------------------------------------------------------------
# Vapour-pressure correlations
# ---------------------------------------------------------------------------

def antoine(T, Par):
    """Vapour pressure in bar.  T in Kelvin."""
    return np.power(10, Par[0] - Par[1] / (np.asarray(T) + Par[2]))


def T_boil_antoine(P, Par):
    """Boiling temperature (K) at pressure P (bar) using Antoine constants."""
    return Par[1] / (Par[0] - np.log10(P)) - Par[2]


def Psat_DIPPR(T, Par):
    """Vapour pressure in Pa.  T in Kelvin.  5-parameter DIPPR correlation."""
    return np.exp(
        Par[0]
        + Par[1] / np.asarray(T)
        + Par[2] * np.log(np.asarray(T))
        + Par[3] * np.power(np.asarray(T), Par[4])
    )


def DIPPR_error(T, P, Par):
    return np.abs(P - Psat_DIPPR(T, Par))


def T_boil_DIPPR(Tguess, P, Par):
    res = minimize(DIPPR_error, Tguess, args=(P, Par), method="nelder-mead", tol=1e-3)
    return res.x


# ---------------------------------------------------------------------------
# Activity coefficient (NRTL-Aspen)
# ---------------------------------------------------------------------------

def NRTL_Aspen(T, Sys, Txy, x=None):
    if x is None:
        x = []
    tau12 = Sys.NRTL.A12 + Sys.NRTL.B12 / T
    tau21 = Sys.NRTL.A21 + Sys.NRTL.B21 / T
    alpha  = Sys.NRTL.C12 + Sys.NRTL.D12 / T

    G12 = np.exp(-alpha * tau12)
    G21 = np.exp(-alpha * tau21)

    S21  = G21 / (x[0] + x[1] * G21)
    S12  = tau12 * G12 / (np.power(x[1] + x[0] * G12, 2))
    Gam1 = np.power(x[1], 2) * (tau21 * np.power(S21, 2) + S12)

    S12  = G12 / (x[1] + x[0] * G21)
    S21  = tau21 * G21 / (np.power(x[0] + x[1] * G21, 2))
    Gam2 = np.power(x[0], 2) * (tau12 * np.power(S12, 2) + S21)

    return np.exp(Gam1), np.exp(Gam2)


# ---------------------------------------------------------------------------
# Helper for T-xy / P-xy minimisation
# ---------------------------------------------------------------------------

def delta_Txy(T, Sys, Txy, x=None):
    if x is None:
        x = []
    Gam1 = Gam2 = 1
    if Sys.NRTL_mode:
        Gam1, Gam2 = NRTL_Aspen(T, Sys, Txy, x)
    if len(Sys.ParC1) > 3:
        P1 = Psat_DIPPR(T, Sys.ParC1)
        P2 = Psat_DIPPR(T, Sys.ParC2)
    else:
        P1 = antoine(T, Sys.ParC1)
        P2 = antoine(T, Sys.ParC2)
    return np.abs(Sys.P_sys - x[0] * P1 * Gam1 - x[1] * P2 * Gam2)


# ---------------------------------------------------------------------------
# Data-holder classes (kept for legacy / NRTL use)
# ---------------------------------------------------------------------------

class Txy:
    def __init__(self, size=50):
        self.size   = size
        self.x      = []
        self.y      = []
        self.T      = []
        self.gamma1 = []
        self.gamma2 = []


class Pxy:
    def __init__(self, size=50):
        self.size   = size
        self.x      = []
        self.y      = []
        self.P      = []
        self.gamma1 = []
        self.gamma2 = []


# ---------------------------------------------------------------------------
# Legacy diagram builders (mutate Txy/Pxy objects — kept for compatibility)
# ---------------------------------------------------------------------------

def calc_Txy_simple(Sys, Txy_obj):
    if len(Txy_obj.x) == 0:
        for i in range(0, Txy_obj.size + 1):
            Txy_obj.x.append(i / Txy_obj.size)
            if i == 0:
                Txy_obj.T.append(Sys.BP2)
                Txy_obj.y.append(0)
                Txy_obj.gamma1.append(1)
                Txy_obj.gamma2.append(1)
            elif i == Txy_obj.size:
                Txy_obj.T.append(Sys.BP1)
                Txy_obj.y.append(1)
                Txy_obj.gamma1.append(1)
                Txy_obj.gamma2.append(1)
            else:
                T0  = Txy_obj.T[i - 1]
                res = minimize(
                    delta_Txy, T0,
                    args=(Sys, Txy_obj, [Txy_obj.x[i], 1 - Txy_obj.x[i]]),
                    method="nelder-mead", tol=1e-6,
                )
                Txy_obj.T.append(res.x)
                Gam1 = Gam2 = 1
                if Sys.NRTL_mode:
                    Gam1, Gam2 = NRTL_Aspen(
                        Txy_obj.T[i], Sys, Txy_obj,
                        [Txy_obj.x[i], 1 - Txy_obj.x[i]],
                    )
                Txy_obj.gamma1.append(Gam1)
                Txy_obj.gamma2.append(Gam2)
                if len(Sys.ParC1) > 3:
                    Txy_obj.y.append(
                        Txy_obj.x[i] * Psat_DIPPR(Txy_obj.T[i], Sys.ParC1) * Gam1 / Sys.P_sys
                    )
                else:
                    Txy_obj.y.append(
                        Txy_obj.x[i] * antoine(Txy_obj.T[i], Sys.ParC1) * Gam1 / Sys.P_sys
                    )
    else:
        Txy_obj.T[0] = Sys.BP2
        Txy_obj.T[Txy_obj.size] = Sys.BP1
        for i in range(1, Txy_obj.size):
            T0  = Txy_obj.T[i - 1]
            res = minimize(
                delta_Txy, T0,
                args=(Sys, Txy_obj, [Txy_obj.x[i], 1 - Txy_obj.x[i]]),
                method="nelder-mead", tol=1e-6,
            )
            Txy_obj.T[i] = res.x
            Gam1 = Gam2 = 1
            if Sys.NRTL_mode:
                Gam1, Gam2 = NRTL_Aspen(
                    Txy_obj.T[i], Sys, Txy_obj,
                    [Txy_obj.x[i], 1 - Txy_obj.x[i]],
                )
            Txy_obj.gamma1[i] = Gam1
            Txy_obj.gamma2[i] = Gam2
            if len(Sys.ParC1) > 3:
                Txy_obj.y[i] = (
                    Txy_obj.x[i] * Psat_DIPPR(Txy_obj.T[i], Sys.ParC1) * Gam1 / Sys.P_sys
                )
            else:
                Txy_obj.y[i] = (
                    Txy_obj.x[i] * antoine(Txy_obj.T[i], Sys.ParC1) * Gam1 / Sys.P_sys
                )
    return Txy_obj


def calc_Pxy_simple(Sys, Pxy_obj):
    if len(Pxy_obj.x) == 0:
        for i in range(0, Pxy_obj.size + 1):
            Pxy_obj.x.append(i / Pxy_obj.size)
            if i == 0:
                Pxy_obj.P.append(Sys.BP2)
                Pxy_obj.y.append(0)
                Pxy_obj.gamma1.append(1)
                Pxy_obj.gamma2.append(1)
            elif i == Pxy_obj.size:
                Pxy_obj.P.append(Sys.BP1)
                Pxy_obj.y.append(1)
                Pxy_obj.gamma1.append(1)
                Pxy_obj.gamma2.append(1)
            else:
                Gam1 = Gam2 = 1
                if Sys.NRTL_mode:
                    Gam1, Gam2 = NRTL_Aspen(
                        Pxy_obj.T[i], Sys, Pxy_obj,
                        [Pxy_obj.x[i], 1 - Pxy_obj.x[i]],
                    )
                Pxy_obj.gamma1.append(Gam1)
                Pxy_obj.gamma2.append(Gam2)
                if len(Sys.ParC1) > 3:
                    P1 = Pxy_obj.x[i] * Psat_DIPPR(Sys.T_sys, Sys.ParC1) * Gam1
                    P2 = (1 - Pxy_obj.x[i]) * Psat_DIPPR(Sys.T_sys, Sys.ParC2) * Gam2
                else:
                    P1 = Pxy_obj.x[i] * antoine(Sys.T_sys, Sys.ParC1) * Gam1
                    P2 = (1 - Pxy_obj.x[i]) * antoine(Sys.T_sys, Sys.ParC2) * Gam2
                Pxy_obj.y.append(P1 / (P1 + P2))
                Pxy_obj.P.append(P1 + P2)
    else:
        Pxy_obj.P[0] = Sys.BP2
        Pxy_obj.P[Pxy_obj.size] = Sys.BP1
        for i in range(1, Pxy_obj.size):
            Gam1 = Gam2 = 1
            if Sys.NRTL_mode:
                Gam1, Gam2 = NRTL_Aspen(
                    Pxy_obj.T[i], Sys, Pxy_obj,
                    [Pxy_obj.x[i], 1 - Pxy_obj.x[i]],
                )
            Pxy_obj.gamma1[i] = Gam1
            Pxy_obj.gamma2[i] = Gam2
            if len(Sys.ParC1) > 3:
                P1 = Pxy_obj.x[i] * Psat_DIPPR(Sys.T_sys, Sys.ParC1) * Gam1
                P2 = (1 - Pxy_obj.x[i]) * Psat_DIPPR(Sys.T_sys, Sys.ParC2) * Gam2
            else:
                P1 = Pxy_obj.x[i] * antoine(Sys.T_sys, Sys.ParC1) * Gam1
                P2 = (1 - Pxy_obj.x[i]) * antoine(Sys.T_sys, Sys.ParC2) * Gam2
            Pxy_obj.y[i] = P1 / (P1 + P2)
            Pxy_obj.P[i] = P1 + P2
    return Pxy_obj


# ---------------------------------------------------------------------------
# Clean functional interface — used by app.py
# ---------------------------------------------------------------------------

def compute_pxy(T_sys_K, par_c1, par_c2, n_points=50):
    """
    Ideal Raoult's law Pxy diagram at constant temperature.

    Parameters
    ----------
    T_sys_K : float
        System temperature in Kelvin.
    par_c1 : list[float]
        Antoine [A, B, C] for the more-volatile component (P in bar, T in K).
    par_c2 : list[float]
        Antoine [A, B, C] for the less-volatile component.
    n_points : int
        Number of intervals (returns n_points+1 points).

    Returns
    -------
    x : ndarray   liquid mole fraction of component 1
    y : ndarray   vapour mole fraction of component 1
    P : ndarray   total pressure (bar)
    """
    x    = np.linspace(0.0, 1.0, n_points + 1)
    P1sat = antoine(T_sys_K, par_c1)
    P2sat = antoine(T_sys_K, par_c2)
    P    = x * P1sat + (1.0 - x) * P2sat
    y    = np.where(P > 0.0, x * P1sat / P, 0.0)
    return x, y, P
