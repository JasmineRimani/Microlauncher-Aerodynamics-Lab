# Launcher Aerodynamic Drag Prediction

A Python implementation of a **zero-lift and angle-of-attack drag coefficient
estimation tool** for axisymmetric launch vehicles, covering subsonic,
transonic, and supersonic flight regimes.

The methodology is described in:

> J. Rimani et al., *"Aerodynamic Drag Coefficient Prediction for Micro-Launchers"*,  
> Aerospace Systems, Springer, 2021.  
> https://doi.org/10.1007/s42401-021-00095-w

---

## Physical models

| Component | Model source |
|-----------|-------------|
| Skin-friction drag (body, fins, protuberances) | Compressible turbulent flat-plate + roughness limit (Stoney / Fleeman) |
| Base drag | Piecewise-polynomial fit anchored at M = 0.6 |
| Transonic wave drag | Drag-divergence / drag-rise fit (Walpot) |
| Supersonic wave drag | Ogive pressure coefficient (Fleeman) |
| Boattail drag | Frustum base-pressure model |
| AoA increment | Cross-flow + leading-edge suction (Fleeman) |

---

## Installation

```bash
pip install numpy scipy matplotlib
```

No other dependencies are required.

---

## Quick start

```python
import numpy as np
from src.aerodynamics import LauncherGeometry, BoattailGeometry, compute_drag

boattail = BoattailGeometry(
    length=0.6,
    diameter_fore=1.0,
    diameter_aft=1.3,
    ref_diameter=1.3,
)

geom = LauncherGeometry(
    stage_lengths=[1.8, 7.0, 3.5],      # [m]  nose + stages
    stage_diameters=[1.2, 1.3, 1.0],    # [m]
    boattail_exists=True,
    boattail=boattail,
)

mach   = np.arange(0.1, 6.1, 0.1)
result = compute_drag(geom, mach)

print(result.CD_total_zero_lift)
```

Run the full example with mock launcher geometry:

```bash
python examples/mock_launcher.py
```

---

## Repository layout

```
.
├── src/
│   └── aerodynamics.py      # core drag models
├── examples/
│   └── mock_launcher.py     # worked example with fictitious "Alpha-1" launcher
├── tests/
│   └── test_aerodynamics.py # pytest suite
└── README.md
```

---

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Input geometry description

All lengths in **metres**.

```
LauncherGeometry
├── stage_lengths        list[float]   axial length of each section [nose, stg1, stg2, …]
├── stage_diameters      list[float]   outer diameter of each section
├── surface_roughness    float         K_skin  [m]  (0 = smooth, 0.00025 = matte paint)
├── interference_factor  float         K_f  (typically 1.04 for rockets)
├── nose_joint_half_angle_deg  float   φ [deg]
├── ogive_factor         float         1.0 = tangent, 0.82 = secant, 0.5 = conical
├── nose_bluntness_ratio float         B_r  (0 = sharp tip, 0.05–0.30 typical)
├── fin_exists           bool
├── fins                 FinGeometry | None
├── boattail_exists      bool
├── boattail             BoattailGeometry | None
├── protuberance_exists  bool
└── protuberance         ProtuberanceGeometry | None
```

---

## Note on mock data

The example launcher ("Alpha-1") uses **completely fictitious** geometry to
demonstrate the tool.  No real vehicle data is included or implied.

---

## License

See `LICENSE` file.  The physical models themselves are based on open
literature references listed in the source code.
