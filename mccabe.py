# -*- coding: utf-8 -*-
"""
McCabe-Thiele stepping algorithms for binary distillation.

All functions are pure (no side effects, no UI imports).
"""

import numpy as np

MAX_STAGES = 100  # hard cap — prevents infinite loops on bad inputs


def int_XY(X, Y, x_int):
    """
    Linear interpolation: search X (ascending) for x_int, return interpolated Y.
    Passing (Y, X, y_val) finds the x on the equilibrium curve for a given y.
    """
    if x_int >= X[-1]:
        return Y[-1]
    idx    = 0
    x_eval = X[idx]
    while x_eval < x_int:
        idx   += 1
        x_eval = X[idx]
    return (Y[idx] - Y[idx - 1]) / (X[idx] - X[idx - 1]) * (x_int - X[idx - 1]) + Y[idx - 1]


def find_qc(xf, qq):
    """y-intercept of the q-line:  y = qq/(qq-1)*x + cq"""
    return xf - (qq / (qq - 1)) * xf


def find_xy_cross(RR, xd, qq, cq):
    """
    Intersection of the rectifying operating line and the q-line.
    Returns (x_cross, y_cross).
    """
    mr = RR / (RR + 1)
    cr = xd / (RR + 1)
    mq = qq / (qq - 1)
    x  = (cq - cr) / (mr - mq)
    y  = mq * x + cq
    return x, y


def find_strip(xb, x_c, y_c):
    """
    Stripping operating line through (xb, xb) and the q-line/rectifying crossing.
    Returns slope m and intercept c.
    """
    m = (y_c - xb) / (x_c - xb)
    c = y_c - m * x_c
    return m, c


def minimum_reflux(
    x_eq: np.ndarray,
    y_eq: np.ndarray,
    xF: float,
    xD: float,
    qq: float,
) -> "tuple[float, float, float]":
    """
    Minimum reflux ratio for a binary system (Underwood/pinch method).
    Assumes no tangent pinch — valid for ideal systems (e.g. benzene-toluene).

    Parameters
    ----------
    x_eq, y_eq : ndarray   equilibrium curve (ascending x)
    xF         : float     feed composition
    xD         : float     distillate composition
    qq         : float     feed quality (handles q=1 vertical q-line separately)

    Returns
    -------
    R_min   : float   minimum reflux ratio
    x_pinch : float   pinch point liquid composition
    y_pinch : float   pinch point vapour composition
    """
    if abs(qq - 1.0) < 1e-6:
        # Vertical q-line: pinch is at xF directly on the equilibrium curve
        x_pinch = float(xF)
        y_pinch = float(np.interp(xF, x_eq, y_eq))
    else:
        mq = qq / (qq - 1.0)
        cq = xF / (1.0 - qq)   # y-intercept: q-line is y = mq*x + cq

        # f(x) = y_eq(x) - y_qline(x); find where it crosses zero
        x_scan = np.linspace(0.001, 0.999, 2000)
        f      = np.interp(x_scan, x_eq, y_eq) - (mq * x_scan + cq)

        crossings = np.where(np.diff(np.sign(f)))[0]
        if len(crossings) == 0:
            x_pinch = float(xF)
            y_pinch = float(np.interp(xF, x_eq, y_eq))
        else:
            # Take the crossing nearest to xF
            i   = crossings[np.argmin(np.abs(x_scan[crossings] - xF))]
            x0, x1 = x_scan[i], x_scan[i + 1]
            f0, f1 = f[i], f[i + 1]
            x_pinch = float(x0 - f0 * (x1 - x0) / (f1 - f0))  # linear interpolation
            y_pinch = float(np.interp(x_pinch, x_eq, y_eq))

    denom = y_pinch - x_pinch
    R_min = float((xD - y_pinch) / denom) if abs(denom) > 1e-10 else float("inf")

    return R_min, x_pinch, y_pinch


def MCT_Reflux(X, Y, RR, xf, xb, xd, qq):
    """
    Step off theoretical stages at a finite reflux ratio.

    Parameters
    ----------
    X, Y : array-like  equilibrium x and y (ascending, length N)
    RR   : float       reflux ratio L/D
    xf   : float       feed composition
    xb   : float       bottoms composition
    xd   : float       distillate composition
    qq   : float       feed quality  (must not be exactly 1.0)

    Returns
    -------
    MCT_x, MCT_y : lists   staircase coordinates
    n_stages     : int     number of theoretical stages
    x_c, y_c     : float   q-line / rectifying-line crossing point
    """
    cq       = find_qc(xf, qq)
    x_c, y_c = find_xy_cross(RR, xd, qq, cq)
    ms, cs   = find_strip(xb, x_c, y_c)

    MCT_x = [xd]
    MCT_y = [xd]

    # Rectifying section — step down until we cross the q-line
    while MCT_x[-1] > x_c and len(MCT_x) < 2 * MAX_STAGES:
        y = MCT_y[-1]
        x = int_XY(Y, X, y)          # horizontal step to equilibrium curve
        MCT_x.append(x)
        MCT_y.append(y)
        if x < x_c:
            break
        MCT_x.append(x)              # vertical step down to rectifying line
        MCT_y.append((RR * x + xd) / (RR + 1))

    # Stripping section — step down to bottoms
    while MCT_x[-1] > xb and len(MCT_x) < 2 * MAX_STAGES:
        y = MCT_y[-1]
        x = int_XY(Y, X, y)          # horizontal step to equilibrium curve
        MCT_x.append(x)
        MCT_y.append(y)
        MCT_x.append(x)              # vertical step down to stripping line
        MCT_y.append(max(ms * x + cs, x))

    n_stages = int((len(MCT_x) - 1) / 2)
    return MCT_x, MCT_y, n_stages, x_c, y_c


def MCT_Rect_TotReflux(X, Y, xf, xd):
    """Rectifying section at total reflux (step up from xf to xd)."""
    MCT_x = [xf]
    MCT_y = [xf]
    while MCT_x[-1] < xd and len(MCT_x) < 2 * MAX_STAGES:
        x = MCT_x[-1]
        y = int_XY(X, Y, x)
        MCT_x.append(x)
        MCT_y.append(y)
        MCT_x.append(y)
        MCT_y.append(y)
    return MCT_x, MCT_y, int((len(MCT_x) - 1) / 2)


def MCT_Strp_TotReflux(X, Y, xf, xb):
    """Stripping section at total reflux (step down from xf to xb)."""
    MCT_x = [xf]
    MCT_y = [xf]
    while MCT_x[-1] > xb and len(MCT_x) < 2 * MAX_STAGES:
        y = MCT_x[-1]
        x = int_XY(Y, X, y)
        MCT_x.append(x)
        MCT_y.append(y)
        MCT_x.append(x)
        MCT_y.append(x)
    return MCT_x, MCT_y, int((len(MCT_x) - 1) / 2)


def MCT_TotReflux(X, Y, xf, xb, xd):
    """Full staircase at total reflux (stripping + rectifying combined)."""
    STRPx, STRPy, STRPs = MCT_Strp_TotReflux(X, Y, xf, xb)
    Rectx, Recty, Rects = MCT_Rect_TotReflux(X, Y, xf, xd)

    STRPx.reverse()
    STRPy.reverse()
    STRPx.pop()
    STRPy.pop()

    return STRPx + Rectx, STRPy + Recty, STRPs, Rects
