"""
Launcher Aerodynamic Drag Coefficient Prediction
=================================================
Python implementation of the zero-lift and angle-of-attack drag estimation
methodology for axisymmetric launch vehicles, covering subsonic, transonic,
and supersonic flight regimes.

References
----------
- Stoney, W.E.: "Collection of zero-lift drag data on bodies of revolution
  from free-flight investigation", NACA TR-1218.
- Walpot, L.: "Drag Coefficient Prediction".
- Fleeman, E.L.: "Tactical Missile Design", AIAA Education Series.
- Onel et al.: "Drag Coefficient Modelling in the context of small launcher
  optimisation".
- Niskanen, S.: "OpenRocket Technical Documentation".

Author: translated/adapted for open publication (original MATLAB by J. Rimani,
        ESA/PoliTO, 2019).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence

M_TO_IN = 39.3701
FT_TO_M = 0.3048
DEG_TO_RAD = np.pi / 180.0
ISA_GAMMA = 1.4
ISA_GAS_CONSTANT = 287.05287
ISA_G0 = 9.80665
ISA_EARTH_RADIUS_M = 6_356_766.0
ISA_SUTHERLAND_T0 = 288.15
ISA_SUTHERLAND_MU0 = 1.7894e-5
ISA_SUTHERLAND_C = 110.4
BASE_DRAG_ANCHOR_MACH = 0.6
WAVE_DRAG_BLEND_WIDTH = 0.1
ISA_LAYER_BASE_ALTITUDES_M = np.array([0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0])
ISA_LAYER_TOP_ALTITUDE_M = 84_852.0
ISA_LAYER_LAPSE_RATES = np.array([-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020])


def _as_positive_vector(values: Sequence[float], name: str) -> np.ndarray:
    """Return a validated one-dimensional positive float vector."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence.")
    if np.any(vector <= 0):
        raise ValueError(f"{name} must contain only positive values.")
    return vector


def _ensure_positive(value: float, name: str) -> None:
    """Raise a clear error when a scalar parameter is not strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_optional_component(flag: bool, component: object | None, name: str) -> None:
    """Ensure optional geometry blocks are supplied when their flag is enabled."""
    if flag and component is None:
        raise ValueError(f"{name} geometry must be provided when {name.lower()}_exists is True.")


def _smoothstep(x: float) -> float:
    """Cubic smoothstep clipped to the unit interval."""
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _walpot_transition_blend(x: float) -> float:
    """Normalized Walpot transonic blend factor on x in [0, 1]."""
    raw_blend = (
        -8.3474 * x ** 5
        + 24.543 * x ** 4
        - 24.946 * x ** 3
        + 8.6321 * x ** 2
        + 1.1195 * x
    )
    blend_at_one = -8.3474 + 24.543 - 24.946 + 8.6321 + 1.1195
    return float(np.clip(raw_blend / blend_at_one, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Geometry dataclass
# ---------------------------------------------------------------------------

@dataclass
class LauncherGeometry:
    """
    Geometric description of a multi-stage axisymmetric launch vehicle.

    All lengths in metres.  The launcher is assumed to consist of a nose
    section plus an arbitrary number of cylindrical stages, with optional
    fins, boattail, and external protuberances.

    Parameters
    ----------
    stage_lengths : list[float]
        Axial length of each section [nose, stage-1, stage-2, …]  [m].
    stage_diameters : list[float]
        Outer diameter of each section [nose, stage-1, …]  [m].
    surface_roughness : float
        Equivalent sand-grain roughness K_skin [m].  Use 0 for smooth.
    interference_factor : float
        Interference drag factor K_f (typically 1.04 for rocket data).
    nose_joint_half_angle_deg : float
        Half-angle φ of the ogive-to-cylinder junction [deg].
    ogive_factor : float
        Nose ogive shape factor (1.0 = tangent, 0.82 = secant, 0.5 = conical).
    nose_bluntness_ratio : float
        Tip bluntness ratio B_r (0 = sharp, typical range 0.05–0.30).
    fin_exists : bool
        Whether fins are present.
    fins : FinGeometry | None
        Fin specification (required when fin_exists is True).
    boattail_exists : bool
        Whether a boattail inter-stage adapter is present.
    boattail : BoattailGeometry | None
        Boattail specification (required when boattail_exists is True).
    protuberance_exists : bool
        Whether external protuberances (cables, fairings) are modelled.
    protuberance : ProtuberanceGeometry | None
        Protuberance specification (required when protuberance_exists is True).
    """

    stage_lengths: Sequence[float]
    stage_diameters: Sequence[float]
    surface_roughness: float = 0.00025
    interference_factor: float = 1.04
    nose_joint_half_angle_deg: float = 20.0
    ogive_factor: float = 0.82
    nose_bluntness_ratio: float = 0.05
    fin_exists: bool = False
    fins: Optional["FinGeometry"] = None
    boattail_exists: bool = False
    boattail: Optional["BoattailGeometry"] = None
    protuberance_exists: bool = False
    protuberance: Optional["ProtuberanceGeometry"] = None

    # ------------------------------------------------------------------ #
    # Derived quantities (computed once on first access)                  #
    # ------------------------------------------------------------------ #

    def __post_init__(self):
        stages = _as_positive_vector(self.stage_lengths, "stage_lengths")
        diams = _as_positive_vector(self.stage_diameters, "stage_diameters")

        if stages.size != diams.size:
            raise ValueError("stage_lengths and stage_diameters must have the same length.")
        if stages.size < 2:
            raise ValueError("At least a nose section and one body stage must be defined.")
        if self.surface_roughness < 0:
            raise ValueError("surface_roughness cannot be negative.")
        _ensure_positive(self.interference_factor, "interference_factor")
        _ensure_positive(self.ogive_factor, "ogive_factor")
        if self.nose_joint_half_angle_deg < 0:
            raise ValueError("nose_joint_half_angle_deg cannot be negative.")
        if self.nose_bluntness_ratio < 0:
            raise ValueError("nose_bluntness_ratio cannot be negative.")

        _validate_optional_component(self.fin_exists, self.fins, "Fin")
        _validate_optional_component(self.boattail_exists, self.boattail, "Boattail")
        _validate_optional_component(
            self.protuberance_exists,
            self.protuberance,
            "Protuberance",
        )

        self.stage_lengths = stages.tolist()
        self.stage_diameters = diams.tolist()

        self.L_total = float(np.sum(stages))
        self.d_max = float(np.max(diams))
        self.d_base = float(diams[1])
        self.L_nose = float(stages[0])
        self.d_nose = float(diams[0])

        # Effective length for wave-drag (conservative: use full length)
        self.L_effective = self.L_total

        # Length from nose tip to station of maximum diameter
        # (for tapered vehicles this is non-zero)
        if self.d_nose != self.d_max:
            # interstage nose length is assumed to be stages[2] when present
            self.L_to_d_max = float(stages[2]) if len(stages) > 2 else 0.0
        else:
            self.L_to_d_max = self.L_total - self.L_nose

        # Imperial equivalents
        self.L_total_in = self.L_total * M_TO_IN
        self.d_max_in = self.d_max * M_TO_IN
        self.L_nose_in = self.L_nose * M_TO_IN

        # Wetted areas [m²]
        a_ogive = ((self.d_nose / 2) ** 2 + self.L_nose ** 2) / self.d_nose
        S_nose  = (2 * np.pi * a_ogive *
                   ((self.d_nose / 2 - a_ogive) *
                    np.arcsin(self.L_nose / a_ogive) + self.L_nose))
        S_stages = [2 * np.pi * (diams[k] / 2) * stages[k]
                    for k in range(1, len(stages))]
        self.S_wet_total = S_nose + sum(S_stages)
        self.S_wet_total_in = self.S_wet_total * M_TO_IN ** 2

        # Joint half-angle in radians
        self.phi = self.nose_joint_half_angle_deg * DEG_TO_RAD

        # Bluntness correction factor F_cr
        Br = self.nose_bluntness_ratio
        self.F_cr = 1.0 - 0.16 * Br + 0.46 * Br ** 2


@dataclass
class FinGeometry:
    """Trapezoidal fin specification."""
    N_fins: int          # number of fins
    root_chord: float    # C_r  [m]
    tip_chord: float     # C_t  [m]
    span: float          # half-span b/2  [m]
    max_thickness: float # t    [m]
    x_tc: float          # chordwise position of max thickness / C_r  [-]
    sweep_LE_deg: float = 0.0  # leading-edge sweep angle  [deg]

    def __post_init__(self):
        if self.N_fins < 1:
            raise ValueError("N_fins must be at least 1.")
        _ensure_positive(self.root_chord, "root_chord")
        _ensure_positive(self.tip_chord, "tip_chord")
        _ensure_positive(self.span, "span")
        _ensure_positive(self.max_thickness, "max_thickness")
        if not 0 <= self.x_tc <= 1:
            raise ValueError("x_tc must be between 0 and 1.")

        self.taper_ratio = self.tip_chord / self.root_chord
        self.S_fin = self.span * (self.root_chord + self.tip_chord) / 2
        self.S_fin_in = self.S_fin * M_TO_IN ** 2
        self.root_chord_in = self.root_chord * M_TO_IN
        self.max_thickness_in = self.max_thickness * M_TO_IN


@dataclass
class BoattailGeometry:
    """Inter-stage boattail (frustum) geometry."""
    length: float          # axial length  [m]
    diameter_fore: float   # forward-station diameter  [m]
    diameter_aft: float    # aft-station diameter  [m]
    ref_diameter: float    # reference diameter (= vehicle max diam)  [m]

    def __post_init__(self):
        _ensure_positive(self.length, "length")
        _ensure_positive(self.diameter_fore, "diameter_fore")
        _ensure_positive(self.diameter_aft, "diameter_aft")
        _ensure_positive(self.ref_diameter, "ref_diameter")
        if np.isclose(self.diameter_fore, self.diameter_aft):
            raise ValueError("diameter_fore and diameter_aft must differ for a boattail.")

        self.A_fore = np.pi * (self.diameter_fore / 2) ** 2
        self.A_aft = np.pi * (self.diameter_aft / 2) ** 2
        self.A_ref = np.pi * (self.ref_diameter / 2) ** 2


@dataclass
class ProtuberanceGeometry:
    """External protuberance (cable conduit, fairing strut, …)."""
    length: float            # streamwise length  [m]
    max_cross_section: float # maximum cross-section area  [m²]
    wetted_area: float       # total wetted area  [m²]

    def __post_init__(self):
        _ensure_positive(self.length, "length")
        _ensure_positive(self.max_cross_section, "max_cross_section")
        _ensure_positive(self.wetted_area, "wetted_area")

        self.length_in = self.length * M_TO_IN
        self.max_cross_section_in = self.max_cross_section * M_TO_IN ** 2
        self.wetted_area_in = self.wetted_area * M_TO_IN ** 2


# ---------------------------------------------------------------------------
# Atmosphere helper
# ---------------------------------------------------------------------------

def _atmosphere(altitude_m: float):
    """
    Return speed of sound [m/s] and kinematic viscosity [m²/s] at
    the given geometric altitude using the 1976 International Standard
    Atmosphere with Sutherland's law for viscosity.

    Parameters
    ----------
    altitude_m : float
        Geometric altitude [m]. Altitudes above the ISA model ceiling are
        clamped to the top layer because aerodynamic drag is negligible there.

    Returns
    -------
    a_m : float  Speed of sound [m/s].
    nu_m : float Kinematic viscosity [m²/s].
    """
    geometric_altitude_m = max(float(altitude_m), 0.0)
    geopotential_altitude_m = (
        ISA_EARTH_RADIUS_M * geometric_altitude_m / (ISA_EARTH_RADIUS_M + geometric_altitude_m)
    )
    geopotential_altitude_m = min(geopotential_altitude_m, ISA_LAYER_TOP_ALTITUDE_M)

    temperature_k = ISA_SUTHERLAND_T0
    pressure_pa = 101_325.0

    for layer_index, layer_base_altitude_m in enumerate(ISA_LAYER_BASE_ALTITUDES_M):
        lapse_rate = ISA_LAYER_LAPSE_RATES[layer_index]
        layer_top_altitude_m = (
            ISA_LAYER_BASE_ALTITUDES_M[layer_index + 1]
            if layer_index + 1 < len(ISA_LAYER_BASE_ALTITUDES_M)
            else ISA_LAYER_TOP_ALTITUDE_M
        )

        if geopotential_altitude_m <= layer_top_altitude_m:
            delta_h = geopotential_altitude_m - layer_base_altitude_m
            if np.isclose(lapse_rate, 0.0):
                pressure_pa *= np.exp(-ISA_G0 * delta_h / (ISA_GAS_CONSTANT * temperature_k))
            else:
                next_temperature_k = temperature_k + lapse_rate * delta_h
                pressure_pa *= (next_temperature_k / temperature_k) ** (
                    -ISA_G0 / (lapse_rate * ISA_GAS_CONSTANT)
                )
                temperature_k = next_temperature_k
            break

        delta_h = layer_top_altitude_m - layer_base_altitude_m
        if np.isclose(lapse_rate, 0.0):
            pressure_pa *= np.exp(-ISA_G0 * delta_h / (ISA_GAS_CONSTANT * temperature_k))
        else:
            next_temperature_k = temperature_k + lapse_rate * delta_h
            pressure_pa *= (next_temperature_k / temperature_k) ** (
                -ISA_G0 / (lapse_rate * ISA_GAS_CONSTANT)
            )
            temperature_k = next_temperature_k

    density_kg_m3 = pressure_pa / (ISA_GAS_CONSTANT * temperature_k)
    viscosity_pa_s = ISA_SUTHERLAND_MU0 * (
        (temperature_k / ISA_SUTHERLAND_T0) ** 1.5
        * (ISA_SUTHERLAND_T0 + ISA_SUTHERLAND_C)
        / (temperature_k + ISA_SUTHERLAND_C)
    )
    speed_of_sound_m_s = np.sqrt(ISA_GAMMA * ISA_GAS_CONSTANT * temperature_k)
    kinematic_viscosity_m2_s = viscosity_pa_s / density_kg_m3

    return speed_of_sound_m_s, kinematic_viscosity_m2_s


def _compressibility_reynolds_factor(mach: float) -> float:
    """Polynomial correction used in the Fleeman compressible Reynolds number."""
    return (
        1
        + 0.0283 * mach
        - 0.043 * mach ** 2
        + 0.2107 * mach ** 3
        - 0.03829 * mach ** 4
        + 0.002709 * mach ** 5
    )


def _compressibility_skin_friction_factor(mach: float) -> float:
    """Polynomial compressibility correction for turbulent skin friction."""
    return (
        1
        + 0.00798 * mach
        - 0.1813 * mach ** 2
        + 0.0632 * mach ** 3
        - 0.00933 * mach ** 4
        + 0.000549 * mach ** 5
    )


def _incompressible_skin_friction(reynolds_number: float) -> float:
    """Turbulent flat-plate skin-friction coefficient."""
    return 0.037036 * reynolds_number ** (-0.155079)


def _roughness_limited_skin_friction(
    reference_length_in: float,
    roughness_m: float,
    mach: float,
) -> float:
    """Return the fully turbulent rough-wall skin-friction limit."""
    if roughness_m <= 0:
        return 0.0

    cf_rough_inc = 1.0 / (1.89 + 1.62 * np.log10(reference_length_in / roughness_m)) ** 2.5
    return cf_rough_inc / (1 + 0.2044 * mach ** 2)


def _base_drag_subsonic(geom: LauncherGeometry, cd_friction_body: float) -> float:
    """Return the subsonic base-drag relation used to anchor the model."""
    length_ratio = geom.L_to_d_max / geom.d_max
    k_b = 0.0274 * np.arctan(length_ratio + 0.0116)
    exponent_n = 3.6542 * length_ratio ** (-0.2733)
    return k_b * (geom.d_base / geom.d_max) ** exponent_n / np.sqrt(max(cd_friction_body, 1e-12))


def _base_drag_scale_factor(mach: float) -> float:
    """Return Fleeman's piecewise base-drag scale factor relative to M = 0.6."""
    if mach < 1.0:
        return 1 + 215.8 * (mach - BASE_DRAG_ANCHOR_MACH) ** 6
    if mach <= 2.0:
        return (
            2.0881 * (mach - 1) ** 3
            - 3.7938 * (mach - 1) ** 2
            + 1.4618 * (mach - 1)
            + 1.8882917
        )
    if mach <= 2.5:
        return (
            0.297 * (mach - 2) ** 3
            - 0.7937 * (mach - 2) ** 2
            - 0.1115 * (mach - 2)
            + 1.64006
        )
    raise ValueError("Base-drag scale factor is only defined up to Mach 2.5.")


def _base_drag_anchor_value(
    geom: LauncherGeometry,
    altitude_m: float,
) -> float:
    """Compute the model anchor at M = 0.6 independently of the user Mach grid."""
    anchor_friction_body = skin_friction_drag(
        np.array([BASE_DRAG_ANCHOR_MACH]),
        geom,
        altitude_m,
    )[0][0]
    return _base_drag_subsonic(geom, anchor_friction_body)


def _high_mach_base_drag_polynomial(cd_base_06: float) -> np.poly1d:
    """Fit the high-Mach base-drag continuation from fixed anchor points."""
    mach_anchor = np.array([2.0, 2.25, 2.5, 5.0, 7.0, 9.0, 10.0])
    cd_anchor = np.array(
        [
            cd_base_06 * _base_drag_scale_factor(2.0),
            cd_base_06 * _base_drag_scale_factor(2.25),
            cd_base_06 * _base_drag_scale_factor(2.5),
            0.05,
            0.03,
            0.02,
            0.01,
        ]
    )
    return np.poly1d(np.polyfit(mach_anchor, cd_anchor, 4))


def _wave_drag_transition_bounds(geom: LauncherGeometry) -> tuple[float, float]:
    """Return drag-divergence and supersonic-transition Mach numbers."""
    r_d = geom.L_nose / geom.d_max
    mach_div = -0.0156 * r_d ** 2 + 0.136 * r_d + 0.6817

    rln = geom.L_nose / geom.L_effective
    if rln < 0.2:
        a_coef, b_coef = 2.4, -1.05
    else:
        a_coef = -321.94 * rln ** 2 + 264.07 * rln - 36.348
        b_coef = 19.634 * rln ** 2 - 18.369 * rln + 1.7434
    mach_final = a_coef * (geom.L_effective / geom.d_max) ** b_coef + 1.0275

    return mach_div, mach_final


def _supersonic_nose_wave_drag(
    geom: LauncherGeometry,
    mach: float,
    shape_factor: float,
    area_ratio: float,
) -> float:
    """Return Fleeman's supersonic ogive nose-wave drag coefficient."""
    if mach < 1.0:
        return 0.0

    cd_bp_nose = (
        2.1 * np.sin(geom.phi) ** 2
        + 0.5 * np.sin(geom.phi) / np.sqrt(max(mach ** 2 - 1, 1e-6))
    )
    return shape_factor * cd_bp_nose * geom.F_cr * area_ratio


# ---------------------------------------------------------------------------
# Component drag functions
# ---------------------------------------------------------------------------

def skin_friction_drag(
    mach_array: np.ndarray,
    geom: LauncherGeometry,
    altitude_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute skin-friction drag coefficient for the launcher body, fins,
    and protuberances (all referenced to the maximum cross-section area).

    Returns
    -------
    Cd_friction_body : ndarray  Body skin-friction CD vs. Mach.
    Cd_friction_total : ndarray Total (body + fins + protuberances) CD.
    """
    mach = np.asarray(mach_array, dtype=float)
    a_m, nu_m = _atmosphere(altitude_m)
    a_ft = a_m / FT_TO_M
    nu_ft = nu_m / FT_TO_M ** 2

    Cd_body = np.zeros_like(mach)
    Cd_fin = np.zeros_like(mach)
    Cd_prot = np.zeros_like(mach)
    Cd_excr = np.zeros_like(mach)

    L_in = geom.L_total_in
    d_in = geom.d_max_in
    S_in = geom.S_wet_total_in
    K_skin = geom.surface_roughness

    for j, M in enumerate(mach):
        poly_Re = _compressibility_reynolds_factor(M)
        poly_Cf = _compressibility_skin_friction_factor(M)
        Re_c = a_ft * M * (L_in / 12) / nu_ft * poly_Re
        Cf_comp = _incompressible_skin_friction(Re_c) * poly_Cf
        Cf_rough_comp = _roughness_limited_skin_friction(L_in, K_skin, M)

        Cf = max(Cf_comp, Cf_rough_comp)

        # Body form factor
        f_factor = (1 + 60.0 / (L_in / d_in) ** 3
                    + 0.0025 * (L_in / d_in))
        Cd_body[j] = Cf * f_factor * 4 * S_in / (np.pi * d_in ** 2)

        # --- Fins ---
        if geom.fin_exists and geom.fins is not None:
            fins = geom.fins
            Cr_in = fins.root_chord_in
            Re_c_fin = a_ft * M * (Cr_in / 12) / nu_ft * poly_Re
            Cf_comp_fin = _incompressible_skin_friction(Re_c_fin) * poly_Cf
            Cf_rough_comp_fin = _roughness_limited_skin_friction(Cr_in, K_skin, M)
            Cf_fin_val = max(Cf_comp_fin, Cf_rough_comp_fin)

            Re_inc_fin = a_ft * M * (Cr_in / 12) / nu_ft
            lam = fins.taper_ratio
            if lam == 0:
                Cf_fin_avg = Cf_fin_val * (1 + 0.5636 / np.log10(Re_inc_fin))
            else:
                log_Re = np.log10(Re_inc_fin)
                log_Re_lam = np.log10(Re_inc_fin * lam)
                Cf_fin_avg = (log_Re ** 2.6 / (lam ** 2 - 1) *
                              (lam ** 2 / log_Re_lam ** 2.6
                               - 1.0 / log_Re ** 2.6
                               + 0.5646 * lam ** 2 / log_Re_lam ** 3.6
                               - 1.0 / log_Re ** 3.6))

            t_c = fins.max_thickness_in / Cr_in
            x_tc = fins.x_tc
            d_fin_in = geom.d_base * M_TO_IN
            Cd_fin[j] = (
                Cf_fin_avg
                * (
                    1
                    + 60 * t_c ** 4
                    + 0.8 * (1 + 5 * x_tc ** 2) * (fins.max_thickness / fins.root_chord)
                )
                * 4
                * fins.N_fins
                * fins.S_fin_in
                / (np.pi * d_fin_in ** 2)
            )

        # --- Protuberances ---
        if geom.protuberance_exists and geom.protuberance is not None:
            p = geom.protuberance
            Re_c_p = a_ft * M * (p.length_in / 12) / nu_ft * poly_Re
            Cf_comp_p = _incompressible_skin_friction(Re_c_p) * poly_Cf
            Cf_rough_comp_p = _roughness_limited_skin_friction(p.length_in, K_skin, M)
            Cf_p = max(Cf_comp_p, Cf_rough_comp_p)

            sqrt_A = np.sqrt(p.max_cross_section_in)
            Cd_prot[j] = (Cf_p *
                          (1 + 1.798 * (sqrt_A / p.length_in) ** 1.5) *
                          4 * p.wetted_area_in / (np.pi * d_in ** 2))

        # --- Excrescencies (set to zero by default; can be re-enabled) ---
        if M < 0.78:
            K_e = 0.00038
        elif M > 1.04:
            K_e = 0.0002 * M ** 2 - 0.0012 * M + 0.0018
        else:
            K_e = (-0.4501 * M ** 4 + 1.5954 * M ** 3
                   - 2.1062 * M ** 2 + 1.2288 * M - 0.267171)
        Cd_excr[j] = 0.0 * K_e * 4 * S_in / (np.pi * d_in ** 2)  # disabled

    Cd_total = (
        Cd_body
        + geom.interference_factor * Cd_fin
        + geom.interference_factor * Cd_prot
        + Cd_excr
    )
    return Cd_body, Cd_total


def base_drag(
    mach_array: np.ndarray,
    geom: LauncherGeometry,
    Cd_friction_body: np.ndarray,
    altitude_m: float = 0.0,
) -> np.ndarray:
    """
    Compute base-pressure drag coefficient vs. Mach number.

    The subsonic baseline at M=0.6 anchors the transonic and supersonic
    correction via piecewise polynomial fits.

    Parameters
    ----------
    Cd_friction_body : ndarray
        Body skin-friction CD (used to compute the subsonic baseline).

    Returns
    -------
    Cd_base : ndarray
    """
    mach = np.asarray(mach_array, dtype=float)
    Cd_base = np.zeros_like(mach)
    cd_base_06 = _base_drag_anchor_value(geom, altitude_m)
    high_mach_poly = _high_mach_base_drag_polynomial(cd_base_06)

    for j, M in enumerate(mach):
        if M <= BASE_DRAG_ANCHOR_MACH:
            Cd_base[j] = _base_drag_subsonic(geom, Cd_friction_body[j])
        elif M <= 2.5:
            Cd_base[j] = cd_base_06 * _base_drag_scale_factor(M)
        else:
            Cd_base[j] = max(float(high_mach_poly(M)), 0.0)

    return Cd_base


def wave_drag(
    mach_array: np.ndarray,
    geom: LauncherGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute transonic and supersonic wave (pressure) drag on the nose.

    Uses the drag-divergence Mach / final Mach formulation from Walpot /
    Fleeman together with the ogive pressure coefficient model.

    Returns
    -------
    CD_wave_transonic  : ndarray  Transonic wave drag CD.
    CD_wave_supersonic : ndarray  Supersonic wave drag CD.
    """
    mach = np.asarray(mach_array, dtype=float)
    g = geom
    mach_div, mach_final = _wave_drag_transition_bounds(g)

    # Nose pressure drag at transonic onset
    A_ratio = (g.d_nose / g.d_max) ** 2
    Cd_bp_0 = 0.8 * np.sin(g.phi) ** 2
    shape_factor = 0.72 * (g.ogive_factor - 0.5) ** 2 + 0.82
    Cd_pb_0_blunt = shape_factor * Cd_bp_0 * g.F_cr * A_ratio

    cd_wave_transition_end = _supersonic_nose_wave_drag(
        g,
        mach_final,
        shape_factor,
        A_ratio,
    )

    CD_trans = np.zeros_like(mach)
    CD_super = np.zeros_like(mach)
    blend_start = mach_final - WAVE_DRAG_BLEND_WIDTH
    blend_span = max(2 * WAVE_DRAG_BLEND_WIDTH, 1e-6)

    for j, M in enumerate(mach):
        if M < mach_div:
            cd_transonic_model = Cd_pb_0_blunt
        else:
            x = (M - mach_div) / max(mach_final - mach_div, 1e-6)
            normalized_blend = _walpot_transition_blend(x)
            cd_transonic_model = Cd_pb_0_blunt + (
                cd_wave_transition_end - Cd_pb_0_blunt
            ) * normalized_blend

        cd_supersonic_model = _supersonic_nose_wave_drag(g, M, shape_factor, A_ratio)

        if M <= blend_start:
            CD_trans[j] = cd_transonic_model
            CD_super[j] = 0.0
        elif M >= mach_final + WAVE_DRAG_BLEND_WIDTH:
            CD_trans[j] = 0.0
            CD_super[j] = cd_supersonic_model
        else:
            blend = _smoothstep((M - blend_start) / blend_span)
            CD_trans[j] = (1.0 - blend) * cd_transonic_model
            CD_super[j] = blend * cd_supersonic_model

    return CD_trans, CD_super


def boattail_drag(
    mach_array: np.ndarray,
    geom: LauncherGeometry,
) -> np.ndarray:
    """
    Compute the boattail base-pressure drag contribution.

    Returns
    -------
    Cd_bp_bt : ndarray  (zeros if no boattail is defined)
    """
    mach = np.asarray(mach_array, dtype=float)
    if not geom.boattail_exists or geom.boattail is None:
        return np.zeros_like(mach)

    bt = geom.boattail
    sigma = bt.length / (bt.diameter_fore - bt.diameter_aft)
    Cd_bp_bt = np.zeros_like(mach)

    for j, M in enumerate(mach):
        if M < 1.0:
            Cd_bd_bt = 0.12 + 0.13 * M ** 2
        elif M >= 10.0:
            Cd_bd_bt = 0.13 / M
        else:
            Cd_bd_bt = 0.25 / M

        if M <= 0.8:
            Cd_bp_bt[j] = 0.0
        elif sigma < 1.0:
            Cd_bp_bt[j] = (bt.A_fore / bt.A_aft) * Cd_bd_bt * (bt.A_aft / bt.A_ref)
        elif 1.0 < sigma < 3.0:
            Cd_bp_bt[j] = ((bt.A_fore / bt.A_aft) * Cd_bd_bt
                           * (3 - sigma) / 2 * (bt.A_aft / bt.A_ref))
        else:
            Cd_bp_bt[j] = 0.0

    return Cd_bp_bt


# ---------------------------------------------------------------------------
# Top-level solver
# ---------------------------------------------------------------------------

@dataclass
class AeroResult:
    """Container for all drag components and total CD vs. Mach."""
    mach: np.ndarray
    Cd_friction_body: np.ndarray
    Cd_friction_total: np.ndarray
    Cd_base: np.ndarray
    CD_wave_transonic: np.ndarray
    CD_wave_supersonic: np.ndarray
    Cd_boattail: np.ndarray
    CD_total_zero_lift: np.ndarray


def compute_drag(
    geom: LauncherGeometry,
    mach_array: np.ndarray | None = None,
    altitude_m: float = 0.0,
) -> AeroResult:
    """
    Compute the total zero-lift drag coefficient for the given launcher
    geometry over a Mach number sweep.

    Parameters
    ----------
    geom : LauncherGeometry
    mach_array : ndarray, optional
        Mach numbers to evaluate.  Defaults to 0.1 … 6.0 in steps of 0.1.
    altitude_m : float
        Flight altitude [m] (affects atmosphere model).

    Returns
    -------
    AeroResult
    """
    if mach_array is None:
        mach_array = np.arange(0.1, 6.1, 0.1)

    Cd_f_body, Cd_f_total = skin_friction_drag(mach_array, geom, altitude_m)
    Cd_b  = base_drag(mach_array, geom, Cd_f_body, altitude_m)
    CD_wt, CD_ws = wave_drag(mach_array, geom)
    Cd_bt = boattail_drag(mach_array, geom)

    CD_tot = Cd_f_total + Cd_b + CD_wt + CD_ws + Cd_bt

    return AeroResult(
        mach=mach_array,
        Cd_friction_body=Cd_f_body,
        Cd_friction_total=Cd_f_total,
        Cd_base=Cd_b,
        CD_wave_transonic=CD_wt,
        CD_wave_supersonic=CD_ws,
        Cd_boattail=Cd_bt,
        CD_total_zero_lift=CD_tot,
    )


def angle_of_attack_increment(
    geom: LauncherGeometry,
    alpha_deg_array: np.ndarray,
) -> np.ndarray:
    """
    Estimate the incremental drag coefficient due to non-zero angle of attack
    (small-angle approximation).

    This correlation is intended for modest angles of attack, roughly up to
    15 degrees. It does not include high-angle nonlinear effects, normal-force
    build-up, or moment corrections.

    Parameters
    ----------
    alpha_deg_array : ndarray
        Angles of attack [deg] to evaluate.

    Returns
    -------
    delta_CD_alpha : ndarray  Shape (len(alpha_deg_array),).
    """
    alpha = np.asarray(alpha_deg_array, dtype=float) * np.pi / 180.0
    g = geom

    # Spline coefficients for δ (cross-flow) and η (leading-edge suction)
    alpha_knots = np.array([4, 6, 12, 20]) * np.pi / 180.0
    delta_knots = np.array([0.0, 0.85, 0.95, 0.97])
    eta_knots   = np.array([0.60, 0.63, 0.70, 0.753])

    from scipy.interpolate import CubicSpline
    spl_delta = CubicSpline(alpha_knots, delta_knots, extrapolate=True)
    spl_eta   = CubicSpline(alpha_knots, eta_knots,   extrapolate=True)

    delta_CD = np.zeros_like(alpha)
    for k, a in enumerate(alpha):
        delta_ = float(spl_delta(a))
        eta_   = float(spl_eta(a))
        CD_body_a = (2 * delta_ * a ** 2 +
                     (3.6 * eta_ * (1.36 * g.L_total - 0.55 * g.L_nose)
                      / (np.pi * g.d_max)) * a ** 3)

        if g.fin_exists and g.fins is not None:
            fins  = g.fins
            l_fin = fins.span
            d_ref = g.d_base
            R_s   = l_fin / d_ref
            kfb = 0.8065 * R_s ** 2 + 1.1552 * R_s
            kbf = 0.1935 * R_s ** 2 + 0.8174 * R_s + 1.0
            CD_fin_a = (a ** 2 * (1.2 * fins.S_fin ** 4 / np.pi / d_ref ** 2
                                  + 3.12 * (kfb + kbf - 1)
                                  * fins.S_fin ** 4 / np.pi / d_ref ** 2))
        else:
            CD_fin_a = 0.0

        delta_CD[k] = CD_body_a + CD_fin_a

    return delta_CD
