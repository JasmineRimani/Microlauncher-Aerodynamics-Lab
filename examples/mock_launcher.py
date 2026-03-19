"""
Mock Launcher Example
=====================
Demonstrates the drag prediction tool using a *fictitious* two-stage
small launch vehicle ("Alpha-1").  All geometry values are invented for
illustration purposes and do not correspond to any real launcher.

Run this file directly::

    python examples/mock_launcher.py

It produces a CD vs. Mach plot and prints a summary table.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.aerodynamics import (
    LauncherGeometry,
    BoattailGeometry,
    FinGeometry,
    compute_drag,
    angle_of_attack_increment,
)

# ---------------------------------------------------------------------------
# Mock launcher geometry — "Alpha-1"  (completely fictitious)
# ---------------------------------------------------------------------------

boattail = BoattailGeometry(
    length=0.6,            # [m]
    diameter_fore=1.0,     # forward (larger) end
    diameter_aft=1.3,      # aft (smaller) end — note: fore < aft here means
    ref_diameter=1.3,      # the frustum opens aft (typical for 1st/2nd junction)
)

launcher = LauncherGeometry(
    # Sections:  [nose, 1st stage, 2nd stage]
    stage_lengths=[1.8, 7.0, 3.5],
    stage_diameters=[1.2, 1.3, 1.0],
    surface_roughness=0.00025,      # smooth matte paint [m]
    interference_factor=1.04,
    nose_joint_half_angle_deg=18.0,
    ogive_factor=0.82,              # secant ogive
    nose_bluntness_ratio=0.07,
    fin_exists=False,
    boattail_exists=True,
    boattail=boattail,
)

# Mach sweep
mach = np.arange(0.1, 6.01, 0.1)

# ---------------------------------------------------------------------------
# Compute drag
# ---------------------------------------------------------------------------
result = compute_drag(launcher, mach_array=mach, altitude_m=0.0)

# Angle-of-attack sweep
alphas = np.array([5.0, 10.0, 15.0, 20.0])
delta_CD_alpha = angle_of_attack_increment(launcher, alphas)

CD_5  = result.CD_total_zero_lift + delta_CD_alpha[0]
CD_10 = result.CD_total_zero_lift + delta_CD_alpha[1]
CD_15 = result.CD_total_zero_lift + delta_CD_alpha[2]

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
print(f"\n{'Mach':>6}  {'CD_friction':>12}  {'CD_base':>10}  "
      f"{'CD_wave_t':>10}  {'CD_wave_s':>10}  {'CD_boattail':>11}  {'CD_total':>9}")
print("-" * 75)
for j in range(0, len(mach), 5):
    M  = result.mach[j]
    Cf = result.Cd_friction_total[j]
    Cb = result.Cd_base[j]
    Cwt = result.CD_wave_transonic[j]
    Cws = result.CD_wave_supersonic[j]
    Cbt = result.Cd_boattail[j]
    Ct  = result.CD_total_zero_lift[j]
    print(f"{M:6.2f}  {Cf:12.5f}  {Cb:10.5f}  "
          f"{Cwt:10.5f}  {Cws:10.5f}  {Cbt:11.5f}  {Ct:9.5f}")

# ---------------------------------------------------------------------------
# Plot: drag breakdown
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Alpha-1 Mock Launcher — Aerodynamic Drag Prediction",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.stackplot(
    mach,
    result.Cd_friction_total,
    result.Cd_base,
    result.CD_wave_transonic + result.CD_wave_supersonic,
    result.Cd_boattail,
    labels=["Skin Friction", "Base Drag", "Wave Drag (transonic+supersonic)",
            "Boattail"],
    alpha=0.75,
)
ax.plot(mach, result.CD_total_zero_lift, "k-", lw=1.8, label="Total $C_D$ (0°)")
ax.set_xlabel("Mach  [ – ]", fontsize=11)
ax.set_ylabel("Drag Coefficient  [ – ]", fontsize=11)
ax.set_title("Drag Component Breakdown (α = 0°)", fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.set_xlim(mach[0], mach[-1])
ax.grid(True, linestyle="--", alpha=0.5)

ax2 = axes[1]
ax2.plot(mach, result.CD_total_zero_lift, lw=2, label="0°")
ax2.plot(mach, CD_5,  lw=1.5, linestyle="--", label="5°")
ax2.plot(mach, CD_10, lw=1.5, linestyle="-.", label="10°")
ax2.plot(mach, CD_15, lw=1.5, linestyle=":",  label="15°")
ax2.set_xlabel("Mach  [ – ]", fontsize=11)
ax2.set_ylabel("Drag Coefficient  [ – ]", fontsize=11)
ax2.set_title("Total $C_D$ at Different Angles of Attack", fontsize=11)
ax2.legend(title="α", fontsize=9)
ax2.set_xlim(mach[0], mach[-1])
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("alpha1_drag_prediction.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to alpha1_drag_prediction.png")
plt.show()
