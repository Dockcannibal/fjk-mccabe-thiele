"""
McCabe-Thiele Distillation Design — Streamlit UI
Benzene (1) / Toluene (2) system, ideal Raoult's law, Antoine equation.

Run:  streamlit run app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import mccabe as MCT
import thermo as TF

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="McCabe-Thiele | Benzene–Toluene",
    layout="wide",
)

st.title("McCabe-Thiele Distillation Design")
st.caption("Benzene (1) / Toluene (2)  ·  Ideal Raoult's law  ·  Antoine equation")


# ── Cached computations ───────────────────────────────────────────────────────
@st.cache_data
def get_pxy(T_K: float):
    """Pxy equilibrium at temperature T_K (K). Returns numpy arrays."""
    return TF.compute_pxy(T_K, TF.BENZENE_ANTOINE, TF.TOLUENE_ANTOINE)


@st.cache_data
def get_rmin(
    x_eq: np.ndarray,
    y_eq: np.ndarray,
    xF: float,
    xD: float,
    qq: float,
):
    """Minimum reflux ratio and pinch point."""
    return MCT.minimum_reflux(x_eq, y_eq, xF, xD, qq)


@st.cache_data
def get_stages(
    x_eq: np.ndarray,
    y_eq: np.ndarray,
    RR: float,
    xF: float,
    xB: float,
    xD: float,
    qq: float,
):
    """McCabe-Thiele stage stepping. Returns staircase arrays + crossing point."""
    return MCT.MCT_Reflux(list(x_eq), list(y_eq), RR, xF, xB, xD, qq)


# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("System Temperature")
    T_degC = st.number_input(
        "Temperature (°C)  →  sets system pressure",
        min_value=65, max_value=135, value=90, step=2,
        help="Pxy diagram is computed at this temperature. "
             "The bubble-point pressure at xF is shown as 'System Pressure'.",
    )

    st.header("Column Compositions")
    xF = st.number_input("Feed composition  xF",        min_value=0.05,  max_value=0.90,  value=0.50, step=0.01,  format="%.2f")
    xD = st.number_input("Distillate composition  xD",  min_value=0.80,  max_value=0.999, value=0.95, step=0.005, format="%.3f")
    xB = st.number_input("Bottoms composition  xB",     min_value=0.001, max_value=0.20,  value=0.05, step=0.005, format="%.3f")

    st.header("Operating Parameters")
    RR = st.number_input("Reflux ratio  R  (L/D)", min_value=0.5, max_value=15.0, value=3.0, step=0.1,  format="%.1f")
    qq = st.number_input(
        "Feed quality  q",
        min_value=-0.5, max_value=1.5, value=1.0, step=0.05, format="%.2f",
        help="q = 1: saturated liquid  |  q = 0: saturated vapour  |  "
             "0 < q < 1: partial vapour  |  q > 1: subcooled liquid",
    )

# ── Input validation ──────────────────────────────────────────────────────────
errors = []
if not xB < xF:
    errors.append(f"xB ({xB:.3f}) must be less than xF ({xF:.2f}).")
if not xF < xD:
    errors.append(f"xF ({xF:.2f}) must be less than xD ({xD:.3f}).")
if errors:
    for msg in errors:
        st.error(msg)
    st.stop()

# Avoid q = 1 singularity in q-line slope
qq_calc = 0.9999999 if abs(qq - 1.0) < 1e-5 else qq
T_K     = T_degC + 273.15

# ── Run calculations ──────────────────────────────────────────────────────────
x_eq, y_eq, P_arr = get_pxy(T_K)

# Bubble-point pressure at feed composition — the column operating pressure
P_sys_bar = float(np.interp(xF, x_eq, P_arr))

R_min, x_pinch, y_pinch = get_rmin(x_eq, y_eq, xF, xD, qq_calc)

try:
    MCT_x, MCT_y, n_stages, Qx, Qy = get_stages(
        x_eq, y_eq, RR, xF, xB, xD, qq_calc
    )
    stage_error = None
except Exception as exc:
    stage_error = str(exc)
    n_stages    = 0
    MCT_x = MCT_y = []
    Qx = Qy = None

# ── Key metrics ───────────────────────────────────────────────────────────────
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric(
    label="System Pressure  (bubble point at xF)",
    value=f"{P_sys_bar:.3f} bar",
    delta=f"{P_sys_bar * 14.5038:.1f} psia",
    delta_color="off",
)
m_col2.metric(
    label="Theoretical Stages",
    value=str(n_stages) if not stage_error else "—",
)
m_col3.metric(
    label="Minimum Reflux  R_min",
    value=f"{R_min:.3f}" if R_min < 99 else ">99",
)
m_col4.metric(
    label="Temperature",
    value=f"{T_degC} °C",
    delta=f"{T_K:.1f} K",
    delta_color="off",
)

if RR < R_min * 1.05:
    st.warning(
        f"R = {RR:.2f} is within 5 % of R_min = {R_min:.3f}. "
        "Separation may require a very large number of stages."
    )
if stage_error:
    st.warning(
        f"Stage stepping failed: {stage_error}  \n"
        "Try increasing the reflux ratio or relaxing the compositions."
    )

# ── Figures ───────────────────────────────────────────────────────────────────
fig, (ax_pxy, ax_mct) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor("#0e1117")
for ax in (ax_pxy, ax_mct):
    ax.set_facecolor("#262730")
    ax.tick_params(colors="0.75")
    ax.xaxis.label.set_color("0.85")
    ax.yaxis.label.set_color("0.85")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("0.4")

# Left panel: Pxy diagram
ax_pxy.plot(x_eq, P_arr, color="#4c9be8", linewidth=2.5, label="Bubble curve  Px")
ax_pxy.plot(y_eq, P_arr, color="#e85c5c", linewidth=2.5, label="Dew curve  Py")
ax_pxy.axvline(xF, color="0.55", linestyle="--", linewidth=1, label=f"xF = {xF:.2f}")
ax_pxy.axhline(P_sys_bar, color="0.45", linestyle=":", linewidth=0.9)
ax_pxy.annotate(
    f"{P_sys_bar:.3f} bar",
    xy=(0.02, P_sys_bar),
    xycoords=("axes fraction", "data"),
    color="0.7", fontsize=8, va="bottom",
)
ax_pxy.set_xlabel("Composition — Benzene  (x, y)")
ax_pxy.set_ylabel("Pressure  (bar)")
ax_pxy.set_title(f"Pxy Diagram  at  {T_degC} °C")
ax_pxy.set_xlim(0.0, 1.0)
ax_pxy.set_ylim(0.0, 4.0)  # fixed axis — keeps scale consistent across temperatures for easy comparison
ax_pxy.legend(fontsize=9, facecolor="#1e1e2e", labelcolor="0.85", edgecolor="0.4")
ax_pxy.grid(True, alpha=0.25)

# Right panel: McCabe-Thiele diagram
ax_mct.plot([0, 1], [0, 1], color="0.5", linewidth=0.8)
ax_mct.plot(x_eq, y_eq, color="#e85c5c", linewidth=2.5, label="Equilibrium curve")
if not stage_error and MCT_x:
    ax_mct.plot(MCT_x, MCT_y, color="#4c9be8", linewidth=1.4,
                label=f"{n_stages} theoretical stages")
    ax_mct.plot([xF, Qx], [xF, Qy], color="#f0a500", linestyle="--",
                linewidth=1.4, label=f"q-line  (q = {qq:.2f})")
    ax_mct.plot([xD, Qx], [xD, Qy], color="#6fcf97", linestyle="--",
                linewidth=1.4, label=f"Rectifying  (R = {RR:.1f})")
    ax_mct.plot([xB, Qx], [xB, Qy], color="#bb86fc", linestyle="--",
                linewidth=1.4, label="Stripping")
ax_mct.plot(x_pinch, y_pinch, "o", color="#f0a500", markersize=7,
            zorder=5, label=f"Pinch  (R_min = {R_min:.3f})")
for xv, lbl in [(xB, "xB"), (xF, "xF"), (xD, "xD")]:
    ax_mct.axvline(xv, color="0.45", linestyle=":", linewidth=0.9)
    ax_mct.text(xv + 0.01, 0.03, lbl, color="0.6", fontsize=8, va="bottom")
ax_mct.set_xlabel("Liquid composition — Benzene  (x)")
ax_mct.set_ylabel("Vapour composition — Benzene  (y)")
ax_mct.set_title(f"McCabe-Thiele  |  R = {RR:.1f}   q = {qq:.2f}")
ax_mct.set_xlim(0.0, 1.0)
ax_mct.set_ylim(0.0, 1.0)
ax_mct.legend(fontsize=8, facecolor="#1e1e2e", labelcolor="0.85", edgecolor="0.4")
ax_mct.grid(True, alpha=0.25)

fig.tight_layout(pad=2.0)
st.pyplot(fig)
plt.close(fig)
