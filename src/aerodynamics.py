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
from dataclasses import dataclass, field
from typing import Optional


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

    stage_lengths: list[float]
    stage_diameters: list[float]
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
        # Unit conversions (internal calculations use imperial, matching
        # the empirical correlations from Stoney / Fleeman which were
        # developed with imperial data)
        M2FT = 3.28084
        M2IN = 39.3701

        stages = np.asarray(self.stage_lengths, dtype=float)
        diams  = np.asarray(self.stage_diameters, dtype=float)

        self.L_total   = float(np.sum(stages))          # [m]
        self.d_max     = float(np.max(diams))            # [m]
        self.d_base    = float(diams[1])                 # aft diameter of 1st stage
        self.L_nose    = float(stages[0])
        self.d_nose    = float(diams[0])

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
        self.L_total_in  = self.L_total * M2IN
        self.d_max_in    = self.d_max   * M2IN
        self.L_nose_in   = self.L_nose  * M2IN

        # Wetted areas [m²]
        a_ogive = ((self.d_nose / 2) ** 2 + self.L_nose ** 2) / self.d_nose
        S_nose  = (2 * np.pi * a_ogive *
                   ((self.d_nose / 2 - a_ogive) *
                    np.arcsin(self.L_nose / a_ogive) + self.L_nose))
        S_stages = [2 * np.pi * (diams[k] / 2) * stages[k]
                    for k in range(1, len(stages))]
        self.S_wet_total    = S_nose + sum(S_stages)   # [m²]
        self.S_wet_total_in = self.S_wet_total * M2IN ** 2  # [in²]

        # Joint half-angle in radians
        self.phi = self.nose_joint_half_angle_deg * np.pi / 180.0

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
        M2IN = 39.3701
        self.taper_ratio = self.tip_chord / self.root_chord
        self.S_fin    = self.span * (self.root_chord + self.tip_chord) / 2   # [m²]
        self.S_fin_in = self.S_fin * M2IN ** 2
        self.root_chord_in = self.root_chord * M2IN
        self.max_thickness_in = self.max_thickness * M2IN


@dataclass
class BoattailGeometry:
    """Inter-stage boattail (frustum) geometry."""
    length: float          # axial length  [m]
    diameter_fore: float   # forward (larger) diameter  [m]
    diameter_aft: float    # aft (smaller) diameter  [m]
    ref_diameter: float    # reference diameter (= vehicle max diam)  [m]

    def __post_init__(self):
        self.A_fore = np.pi * (self.diameter_fore / 2) ** 2
        self.A_aft  = np.pi * (self.diameter_aft  / 2) ** 2
        self.A_ref  = np.pi * (self.ref_diameter  / 2) ** 2


@dataclass
class ProtuberanceGeometry:
    """External protuberance (cable conduit, fairing strut, …)."""
    length: float            # streamwise length  [m]
    max_cross_section: float # maximum cross-section area  [m²]
    wetted_area: float       # total wetted area  [m²]

    def __post_init__(self):
        M2IN = 39.3701
        self.length_in          = self.length          * M2IN
        self.max_cross_section_in = self.max_cross_section * M2IN ** 2
        self.wetted_area_in     = self.wetted_area     * M2IN ** 2


# ---------------------------------------------------------------------------
# Atmosphere helper
# ---------------------------------------------------------------------------

def _atmosphere(altitude_m: float):
    """
    Return speed of sound [m/s] and kinematic viscosity [m²/s] at
    the given geometric altitude using the piecewise-linear fits from
    the original Fleeman / Stoney correlations (imperial internally).

    Parameters
    ----------
    altitude_m : float
        Geometric altitude [m].

    Returns
    -------
    a_m : float  Speed of sound [m/s].
    nu_m : float Kinematic viscosity [m²/s].
    """
    FT2M = 0.3048
    alt_ft = altitude_m / FT2M

    # Speed of sound [ft/s]
    if alt_ft < 37_000:
        a_ft = -0.004 * alt_ft + 1_116.45
    elif alt_ft > 64_000:
        a_ft = 0.0007 * alt_ft + 924.99
    else:
        a_ft = 968.08

    # Kinematic viscosity [ft²/s]
    if alt_ft < 15_000:
        a_coef, b_coef = 0.00002503, 0.0
    elif alt_ft > 30_000:
        a_coef, b_coef = 0.00004664, -0.6882
    else:
        a_coef, b_coef = 0.00002760, -0.03417
    nu_ft = 0.000157 * np.exp(a_coef * alt_ft + b_coef)

    return a_ft * FT2M, nu_ft * FT2M ** 2


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
    mach  = np.asarray(mach_array, dtype=float)
    a_m, nu_m = _atmosphere(altitude_m)
    a_ft  = a_m  / 0.3048
    nu_ft = nu_m / 0.3048 ** 2

    Cd_body  = np.zeros_like(mach)
    Cd_fin   = np.zeros_like(mach)
    Cd_prot  = np.zeros_like(mach)
    Cd_excr  = np.zeros_like(mach)

    L_in   = geom.L_total_in
    d_in   = geom.d_max_in
    S_in   = geom.S_wet_total_in
    K_skin = geom.surface_roughness

    for j, M in enumerate(mach):
        # Compressible Reynolds number (Fleeman correction polynomial)
        poly_Re = (1 + 0.0283*M - 0.043*M**2 + 0.2107*M**3
                   - 0.03829*M**4 + 0.002709*M**5)
        Re_c    = a_ft * M * (L_in / 12) / nu_ft * poly_Re

        # Incompressible Cf (turbulent flat plate)
        Cf_inc = 0.037036 * Re_c ** (-0.155079)

        # Compressibility correction
        poly_Cf = (1 + 0.00798*M - 0.1813*M**2 + 0.0632*M**3
                   - 0.00933*M**4 + 0.000549*M**5)
        Cf_comp = Cf_inc * poly_Cf

        # Roughness-limited (fully turbulent rough wall)
        Cf_rough_inc  = 1.0 / (1.89 + 1.62 * np.log10(L_in / K_skin)) ** 2.5
        Cf_rough_comp = Cf_rough_inc / (1 + 0.2044 * M ** 2)

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
            Cf_inc_fin  = 0.037036 * Re_c_fin ** (-0.155079)
            Cf_comp_fin = Cf_inc_fin * poly_Cf

            Cf_rough_inc_fin  = 1.0 / (1.89 + 1.62 * np.log10(Cr_in / K_skin)) ** 2.5
            Cf_rough_comp_fin = Cf_rough_inc_fin / (1 + 0.2044 * M ** 2)
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

            t_c   = fins.max_thickness_in / Cr_in
            x_tc  = fins.x_tc
            d_fin_in = geom.d_base * 39.3701  # diameter where fins attach
            Cd_fin[j] = (Cf_fin_avg *
                         (1 + 60 * t_c ** 4 + 0.8 * (1 + 5 * x_tc ** 2) * (fins.max_thickness / fins.root_chord)) *
                         4 * fins.N_fins * fins.S_fin_in / (np.pi * d_fin_in ** 2))

        # --- Protuberances ---
        if geom.protuberance_exists and geom.protuberance is not None:
            p = geom.protuberance
            Re_c_p = a_ft * M * (p.length_in / 12) / nu_ft * poly_Re
            Cf_inc_p  = 0.037036 * Re_c_p ** (-0.155079)
            Cf_comp_p = Cf_inc_p * poly_Cf
            Cf_rough_inc_p  = 1.0 / (1.89 + 1.62 * np.log10(p.length_in / K_skin)) ** 2.5
            Cf_rough_comp_p = Cf_rough_inc_p / (1 + 0.2044 * M ** 2)
            Cf_p = max(Cf_comp_p, Cf_rough_comp_p)

            sqrt_A = np.sqrt(p.max_cross_section_in)
            Cd_prot[j] = (Cf_p *
                          (1 + 1.798 * (sqrt_A / p.length_in) ** 1.5) *
                          4 * p.wetted_area_in / (np.pi * d_in ** 2))

        # --- Excrescencies (set to zero by default — can be re-enabled) ---
        if M < 0.78:
            K_e = 0.00038
        elif M > 1.04:
            K_e = 0.0002 * M ** 2 - 0.0012 * M + 0.0018
        else:
            K_e = (-0.4501 * M ** 4 + 1.5954 * M ** 3
                   - 2.1062 * M ** 2 + 1.2288 * M - 0.267171)
        Cd_excr[j] = 0.0 * K_e * 4 * S_in / (np.pi * d_in ** 2)  # disabled

    Cd_total = (Cd_body
                + geom.interference_factor * Cd_fin
                + geom.interference_factor * Cd_prot
                + Cd_excr)
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
    g = geom

    Cd_base = np.zeros_like(mach)
    Cd_base_06 = None  # anchored at M=0.6

    for j, M in enumerate(mach):
        if M <= 0.6:
            K_b = 0.0274 * np.arctan(g.L_to_d_max / g.d_max + 0.0116)
            n   = 3.6542 * (g.L_to_d_max / g.d_max) ** (-0.2733)
            Cd_base[j] = (K_b * (g.d_base / g.d_max) ** n
                          / np.sqrt(max(Cd_friction_body[j], 1e-12)))
            if abs(M - 0.6) < 1e-9:
                Cd_base_06 = Cd_base[j]
        else:
            # Make sure anchor is available (interpolate if M array skips 0.6)
            if Cd_base_06 is None:
                Cd_base_06 = Cd_base[j - 1]

            if M < 1.0:
                f_b = 1 + 215.8 * (M - 0.6) ** 6
                Cd_base[j] = Cd_base_06 * f_b
            elif M <= 2.0:
                f_b = (2.0881 * (M - 1) ** 3 - 3.7938 * (M - 1) ** 2
                       + 1.4618 * (M - 1) + 1.8882917)
                Cd_base[j] = Cd_base_06 * f_b
            elif M <= 2.5:
                f_b = (0.297 * (M - 2) ** 3 - 0.7937 * (M - 2) ** 2
                       - 0.1115 * (M - 2) + 1.64006)
                Cd_base[j] = Cd_base_06 * f_b
            else:
                # Polynomial fit through anchor points at high Mach
                mask  = (mach >= 2.0) & (mach <= 2.5)
                Mach_anchor = np.append(mach[mask], [5.0, 7.0, 9.0, 10.0])
                CD_anchor   = np.append(Cd_base[mask], [0.05, 0.03, 0.02, 0.01])
                if len(Mach_anchor) >= 5:
                    p_poly = np.polyfit(Mach_anchor, CD_anchor, 4)
                    Cd_base[j] = np.polyval(p_poly, M)
                else:
                    Cd_base[j] = 0.01  # fallback for very sparse Mach arrays

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
    g    = geom

    r_d = g.L_nose / g.d_max
    mach_div = -0.0156 * r_d ** 2 + 0.136 * r_d + 0.6817

    # Parameters for the final (supersonic) Mach transition
    rln = g.L_nose / g.L_effective
    if rln < 0.2:
        a_coef, b_coef = 2.4, -1.05
    else:
        a_coef = -321.94 * rln ** 2 + 264.07 * rln - 36.348
        b_coef =   19.634 * rln ** 2 - 18.369 * rln +  1.7434
    mach_final = a_coef * (g.L_effective / g.d_max) ** b_coef + 1.0275

    # Nose pressure drag at transonic onset
    A_ratio = (g.d_nose / g.d_max) ** 2  # area ratio nose / max cross-section
    Cd_bp_0 = 0.8 * np.sin(g.phi) ** 2
    shape_factor = 0.72 * (g.ogive_factor - 0.5) ** 2 + 0.82
    Cd_pb_0_blunt = shape_factor * Cd_bp_0 * g.F_cr * A_ratio

    # Peak wave drag
    c_peak =  50.676 * rln ** 2 - 51.734 * rln + 15.642
    g_peak =  -2.2538 * rln ** 2 + 1.3108 * rln - 1.7344
    Cd_bp_nose_sup = (2.1 * np.sin(g.phi) ** 2
                      + 0.5 * np.sin(g.phi) / np.sqrt(max(mach_final ** 2 - 1, 1e-6)))
    delta_CD_max = shape_factor * Cd_bp_nose_sup * g.F_cr * A_ratio

    CD_trans = np.zeros_like(mach)
    CD_super = np.zeros_like(mach)

    for j, M in enumerate(mach):
        # --- Transonic ---
        if M < mach_div:
            CD_trans[j] = Cd_pb_0_blunt
        elif M > mach_final:
            CD_trans[j] = 0.0
        else:
            x = (M - mach_div) / (mach_final - mach_div)
            F = (-8.3474 * x ** 5 + 24.543 * x ** 4
                 - 24.946 * x ** 3 + 8.6321 * x ** 2 + 1.1195 * x)
            CD_trans[j] = delta_CD_max * F

        # --- Supersonic ---
        if M >= mach_final:
            Cd_bp_nose = (2.1 * np.sin(g.phi) ** 2
                          + 0.5 * np.sin(g.phi) / np.sqrt(M ** 2 - 1))
            CD_super[j] = shape_factor * Cd_bp_nose * g.F_cr * A_ratio
        else:
            CD_super[j] = 0.0

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
