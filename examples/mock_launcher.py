"""
Mock Launcher Example
=====================
Demonstrates the drag prediction tool using a fictitious two-stage
small launch vehicle ("Alpha-1"). All geometry values are invented for
illustration purposes and do not correspond to any real launcher.

Run this file directly::

    python examples/mock_launcher.py

It produces a CD vs. Mach plot and prints a summary table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aerodynamics import (
    BoattailGeometry,
    LauncherGeometry,
    angle_of_attack_increment,
    compute_drag,
)

ANGLE_SWEEP_DEG = np.array([5.0, 10.0, 15.0])
MACH_SWEEP = np.arange(0.1, 6.1, 0.1)
OUTPUT_FILE = PROJECT_ROOT / "alpha1_drag_prediction.png"
SUMMARY_ROW_STEP = 5


def build_mock_launcher() -> LauncherGeometry:
    """Create the fictitious Alpha-1 launcher geometry used in the example."""
    boattail = BoattailGeometry(
        length=0.6,
        diameter_fore=1.0,
        diameter_aft=1.3,
        ref_diameter=1.3,
    )

    return LauncherGeometry(
        stage_lengths=[1.8, 7.0, 3.5],
        stage_diameters=[1.2, 1.3, 1.0],
        surface_roughness=0.00025,
        interference_factor=1.04,
        nose_joint_half_angle_deg=18.0,
        ogive_factor=0.82,
        nose_bluntness_ratio=0.07,
        boattail_exists=True,
        boattail=boattail,
    )


def build_angle_of_attack_curves(
    launcher: LauncherGeometry,
    zero_lift_drag: np.ndarray,
) -> dict[float, np.ndarray]:
    """Return total-drag curves for the configured angle-of-attack sweep."""
    drag_increments = angle_of_attack_increment(launcher, ANGLE_SWEEP_DEG)
    curves = {0.0: zero_lift_drag}

    for angle_deg, increment in zip(ANGLE_SWEEP_DEG, drag_increments):
        curves[float(angle_deg)] = zero_lift_drag + increment

    return curves


def print_summary_table(result) -> None:
    """Print a compact drag-breakdown table for every fifth Mach sample."""
    header = (
        f"\n{'Mach':>6}  {'CD_friction':>12}  {'CD_base':>10}  "
        f"{'CD_wave_t':>10}  {'CD_wave_s':>10}  {'CD_boattail':>11}  {'CD_total':>9}"
    )
    print(header)
    print("-" * len(header))

    for index in range(0, len(result.mach), SUMMARY_ROW_STEP):
        print(
            f"{result.mach[index]:6.2f}  "
            f"{result.Cd_friction_total[index]:12.5f}  "
            f"{result.Cd_base[index]:10.5f}  "
            f"{result.CD_wave_transonic[index]:10.5f}  "
            f"{result.CD_wave_supersonic[index]:10.5f}  "
            f"{result.Cd_boattail[index]:11.5f}  "
            f"{result.CD_total_zero_lift[index]:9.5f}"
        )


def plot_results(result, angle_curves: dict[float, np.ndarray]) -> None:
    """Plot the drag breakdown and total drag at several angles of attack."""
    mach = result.mach
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Alpha-1 Mock Launcher - Aerodynamic Drag Prediction",
        fontsize=13,
        fontweight="bold",
    )

    drag_components = [
        result.Cd_friction_total,
        result.Cd_base,
        result.CD_wave_transonic + result.CD_wave_supersonic,
        result.Cd_boattail,
    ]
    component_labels = [
        "Skin Friction",
        "Base Drag",
        "Wave Drag (transonic + supersonic)",
        "Boattail",
    ]

    axes[0].stackplot(
        mach,
        *drag_components,
        labels=component_labels,
        alpha=0.75,
    )
    axes[0].plot(mach, result.CD_total_zero_lift, "k-", lw=1.8, label="Total $C_D$ (0 deg)")
    axes[0].set_xlabel("Mach  [ - ]", fontsize=11)
    axes[0].set_ylabel("Drag Coefficient  [ - ]", fontsize=11)
    axes[0].set_title("Drag Component Breakdown (alpha = 0 deg)", fontsize=11)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_xlim(mach[0], mach[-1])
    axes[0].grid(True, linestyle="--", alpha=0.5)

    line_styles = {
        0.0: {"linewidth": 2.0, "linestyle": "-", "label": "0 deg"},
        5.0: {"linewidth": 1.5, "linestyle": "--", "label": "5 deg"},
        10.0: {"linewidth": 1.5, "linestyle": "-.", "label": "10 deg"},
        15.0: {"linewidth": 1.5, "linestyle": ":", "label": "15 deg"},
    }
    for angle_deg, curve in angle_curves.items():
        axes[1].plot(mach, curve, **line_styles[angle_deg])

    axes[1].set_xlabel("Mach  [ - ]", fontsize=11)
    axes[1].set_ylabel("Drag Coefficient  [ - ]", fontsize=11)
    axes[1].set_title("Total $C_D$ at Different Angles of Attack", fontsize=11)
    axes[1].legend(title="alpha", fontsize=9)
    axes[1].set_xlim(mach[0], mach[-1])
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {OUTPUT_FILE}")
    if "agg" not in plt.get_backend().lower():
        plt.show()


def main() -> None:
    """Run the worked example from geometry definition to plotting."""
    launcher = build_mock_launcher()
    result = compute_drag(launcher, mach_array=MACH_SWEEP, altitude_m=0.0)
    angle_curves = build_angle_of_attack_curves(
        launcher,
        result.CD_total_zero_lift,
    )

    print_summary_table(result)
    plot_results(result, angle_curves)


if __name__ == "__main__":
    main()
