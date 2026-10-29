"""
Tests for the central-body spherical-harmonics (SH) gravity field.

Reference values are from the float64 Python transcription of the recursion, checked against an
mpmath direct Legendre sum to <= 4e-15 (docs/spherical_harmonics.md). The coefficient files in
test-python/data are byte-exact degree-10 heads of GRGM1200A, GL0660B, EGM2008 and EGM96.

Kernels: $SPICE/lsk/naif0012.tls, $SPICE/spk/de440.bsp, and for the body-fixed frames
$SPICE/pck/moon_pa_de440_200625.bpc, $SPICE/fk/moon_de440_250416.tf (MOON_PA) and
$SPICE/pck/earth_latest_high_prec.bpc (ITRF93).
"""

import math
import os
import sys

import numpy as np
import pytest

if sys.platform == "win32":
    load_dir = "build/Release"
else:
    load_dir = "build/lib"

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..", load_dir
    )
)

import campanella


def furnsh_kernels():
    spice_dir = os.environ["SPICE"]
    for kernel in ["lsk/naif0012.tls", "spk/de440.bsp",
                   "pck/moon_pa_de440_200625.bpc", "fk/moon_de440_250416.tf",
                   "pck/earth_latest_high_prec.bpc"]:
        campanella.furnsh(os.path.join(spice_dir, kernel))


DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
MOON_FILE = os.path.join(DATA_DIR, "gggrx_1200a_sha_n10.tab")
MOON_JPL_FILE = os.path.join(DATA_DIR, "jggrx_0660b_sha_n10.tab")
EARTH_FILE = os.path.join(DATA_DIR, "EGM2008_n10.gfc")
EARTH96_FILE = os.path.join(DATA_DIR, "EGM96_n10.gfc")

ET0 = 740463475.5117936
MU_MOON = 4.902800582147760e+03          # DE440, as in test_gve.py
MU_EARTH = 398600.435507
GM_MOON_FILE = 4.9028001224453001e+03    # GRGM1200A header
GM_EARTH_FILE = 398600.4415              # EGM2008 header
R_MOON = 1738.0
R_EARTH = 6378.1363


def relerr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


def row_normalised_error(M, M_ref):
    """max over rows of max_j |M_ij - Mref_ij| / max_j |Mref_ij|"""
    return np.max(np.abs(M - M_ref).max(axis=1) / np.abs(M_ref).max(axis=1))


def final_row(traj, tf):
    """Final row of a trajectory, checking that the integration reached tf"""
    assert traj[-1][0] == tf, f"integration stopped at t = {traj[-1][0]}"
    return np.array(traj[-1][1:])


def stm_y0(x0):
    return list(x0) + list(np.eye(6).ravel())


# ----------------------------------------------------------------- parsers
def test_parse_shadr():
    info = campanella.load_spherical_harmonics(MOON_FILE, 10)
    assert info["R"] == 1738.0
    assert info["GM"] == GM_MOON_FILE
    assert info["file_degree"] == 1200
    assert info["nmax"] == 10 and info["mmax"] == 10
    assert info["tide_system"] == ""
    C, S = info["C"], info["S"]
    assert C[0][0] == 1.0
    assert C[1] == [0.0, 0.0] and S[1] == [0.0, 0.0]
    assert C[2][0] == -9.0884339347424299e-05
    assert C[2][2] == 3.4673096470696298e-05
    assert S[2][2] == 9.0791898343722900e-10
    assert S[3][3] == -1.7742221828590300e-06
    assert C[10][10] == 9.4793029911678696e-07
    assert S[10][10] == -1.7193934782143801e-06
    # JPL GL0660B: upper-case E, 0.xxxE+04 mantissas
    info = campanella.load_spherical_harmonics(MOON_JPL_FILE, 10)
    assert info["R"] == 1738.0
    assert info["GM"] == 0.4902800305555400e+04
    assert info["file_degree"] == 660


def test_parse_icgem():
    info = campanella.load_spherical_harmonics(EARTH_FILE, 10)
    # file units m^3/s^2 and m, stored in km^3/s^2 and km
    assert info["GM"] == pytest.approx(GM_EARTH_FILE, rel=1e-15)
    assert info["R"] == pytest.approx(R_EARTH, rel=1e-15)
    assert info["file_degree"] == 2190
    assert info["tide_system"] == "tide_free"
    C, S = info["C"], info["S"]
    assert C[0][0] == 1.0          # "1.0d0" (Fortran exponent)
    assert C[1] == [0.0, 0.0]      # no degree-1 rows in the file
    assert C[2][0] == -0.484165143790815e-03
    assert C[2][2] == 0.243938357328313e-05
    assert S[2][2] == -0.140027370385934e-05
    assert C[10][10] == 0.100435991936118e-06
    assert S[10][10] == -0.238596204211893e-07
    # EGM96 has no norm line (ICGEM default: fully normalized)
    info = campanella.load_spherical_harmonics(EARTH96_FILE, 10)
    assert info["file_degree"] == 360
    assert info["GM"] == pytest.approx(GM_EARTH_FILE, rel=1e-15)
    assert info["C"][2][0] == pytest.approx(-0.484165371736e-03, rel=1e-12)


def test_truncation():
    full = campanella.load_spherical_harmonics(EARTH_FILE, 10)
    low = campanella.load_spherical_harmonics(EARTH_FILE, 6, mmax=3)
    assert len(low["C"]) == 7
    for n in range(2, 7):
        for m in range(n + 1):
            expect_C = full["C"][n][m] if m <= 3 else 0.0
            expect_S = full["S"][n][m] if m <= 3 else 0.0
            assert low["C"][n][m] == expect_C and low["S"][n][m] == expect_S


def test_parse_errors(tmp_path):
    for path in [MOON_FILE, EARTH_FILE]:
        with pytest.raises(RuntimeError, match="missing"):
            campanella.load_spherical_harmonics(path, 11)       # file truncated at degree 10
        with pytest.raises(ValueError):
            campanella.load_spherical_harmonics(path, 1)
        with pytest.raises(ValueError):
            campanella.load_spherical_harmonics(path, 1201)
        with pytest.raises(ValueError):
            campanella.load_spherical_harmonics(path, 4, mmax=5)
    with pytest.raises(RuntimeError, match="cannot open"):
        campanella.load_spherical_harmonics(str(tmp_path / "missing.tab"), 4)
    bad = tmp_path / "bad.txt"
    bad.write_text("not a gravity field\n")
    with pytest.raises(RuntimeError, match="not a PDS SHADR"):
        campanella.load_spherical_harmonics(str(bad), 4)
    # nmax above the degree of the file
    small = tmp_path / "small.tab"
    with open(MOON_FILE, "rb") as fh:
        lines = fh.read().split(b"\r\n")
    lines[0] = lines[0].replace(b" 1200, 1200,", b"    3,    3,")
    small.write_bytes(b"\r\n".join(lines))
    with pytest.raises(ValueError, match="exceeds"):
        campanella.load_spherical_harmonics(str(small), 4)
    # unnormalized SHADR coefficients are rejected
    lines[0] = lines[0].replace(b"    3,    3,    1,", b"    3,    3,    0,")
    small.write_bytes(b"\r\n".join(lines))
    with pytest.raises(RuntimeError, match="normalization"):
        campanella.load_spherical_harmonics(str(small), 2)


# ----------------------------------------------------------------- body-fixed reference values
# docs/spherical_harmonics.md, nmax = 10; km, km/s^2, 1/s^2; G row-major
BF_CASES = [
    (MOON_FILE, [1200.0, -900.0, 1100.0], 0.00012491270174809247,
     [4.181362902917254e-07, 6.063642518707181e-07, -5.555515237145871e-07],
     [-2.9917594013311878e-09, -5.529999177924802e-10, -6.510452652905672e-12, -5.529999177924802e-10,
      2.0727751130057133e-09, -2.46776025461735e-09, -6.510452652905672e-12, -2.46776025461735e-09,
      9.189842883254745e-10]),
    (MOON_FILE, [0.0, 0.0, 1838.0], -0.00043506045364557886,
     [3.77357610461997e-07, -4.29341310566829e-08, 6.101803435197487e-07],
     [8.777083891800687e-10, -9.417549304551924e-12, -1.5916303197101456e-09, -9.417549304551924e-12,
      1.8446461462947565e-10, 2.75064105425279e-10, -1.5916303197101456e-09, 2.75064105425279e-10,
      -1.0621730038095441e-09]),
    (MOON_FILE, [-1500.0, 300.0, -1000.0], 3.928856929660568e-06,
     [-3.178880689037637e-07, 5.286999219572918e-07, 4.078755739202395e-07],
     [-6.934895750649399e-10, 2.0447181473933113e-09, 7.40765231495259e-11, 2.0447181473933113e-09,
      -1.700398983806282e-10, 9.422375624378005e-10, 7.40765231495259e-11, 9.422375624378005e-10,
      8.635294734455677e-10]),
    (EARTH_FILE, [4000.0, -3000.0, 4500.0], -0.009598122217824736,
     [9.375528599039297e-06, -6.8290114535640936e-06, -6.621650419262276e-06],
     [-4.854602091211307e-09, 5.201036819417314e-09, -4.36830363498496e-10, 5.201036819417314e-09,
      -1.6363621554809393e-09, 2.472421035643345e-10, -4.36830363498496e-10, 2.472421035643345e-10,
      6.490964246692245e-09]),
    (EARTH_FILE, [0.0, 0.0, 7000.0], -0.05099350264144708,
     [9.288149929460288e-08, -1.7600731895666085e-08, 2.1805738358692864e-05],
     [6.262736384583122e-09, -1.9501580474542972e-11, -9.580080589418933e-11, -1.9501580474542972e-11,
      6.1541447700287705e-09, 2.0729894340035105e-11, -9.580080589418933e-11, 2.0729894340035105e-11,
      -1.2416881154611893e-08]),
    (EARTH_FILE, [-5000.0, 4000.0, -2500.0], 0.01660570682402651,
     [2.9005683063593684e-06, -2.444805628697329e-06, 1.0282523656934686e-05],
     [-3.703022518705585e-10, -3.0260788136282583e-10, 4.857295775983681e-09, -3.0260788136282583e-10,
      -3.3926041903375625e-10, -3.887791380131979e-09, 4.857295775983681e-09, -3.887791380131979e-09,
      7.095626709043145e-10]),
]


@pytest.mark.parametrize("path, r_bf, U_ref, a_ref, G_ref", BF_CASES)
def test_bf_reference_values(path, r_bf, U_ref, a_ref, G_ref):
    U, a, G = campanella.sh_accel_bf(path, 10, r_bf)
    G = np.array(G)
    print(f"r_bf = {r_bf}: U {abs(U - U_ref)/abs(U_ref):.1e}, a {relerr(a, a_ref):.1e}, "
          f"G {relerr(G, G_ref):.1e}")
    assert abs(U - U_ref) <= 1e-13 * abs(U_ref)
    assert relerr(a, a_ref) <= 1e-13
    assert relerr(G, G_ref) <= 1e-12
    assert np.all(G.reshape(3, 3) == G.reshape(3, 3).T)
    assert abs(G[0] + G[4] + G[8]) <= 1e-14 * np.max(np.abs(G))


def test_bf_scaling():
    """Canonical units: a scales with LU/TU^2 and G with 1/TU^2 (mu = GM TU^2/LU^3, R = R/LU)"""
    LU, TU = 100000.0, 451624.90737307334
    r = [1200.0, -900.0, 1100.0]
    U, a, G = campanella.sh_accel_bf(MOON_FILE, 10, r)
    Uc, ac, Gc = campanella.sh_accel_bf(MOON_FILE, 10, list(np.array(r) / LU),
                                        mu=GM_MOON_FILE * TU**2 / LU**3, R=R_MOON / LU)
    assert relerr(np.array(ac) * LU / TU**2, a) <= 1e-14
    assert relerr(np.array(Gc) / TU**2, G) <= 1e-14
    assert abs(Uc * LU**2 / TU**2 - U) <= 1e-14 * abs(U)


def test_inside_reference_sphere():
    with pytest.raises(ValueError):
        campanella.sh_accel_bf(MOON_FILE, 10, [1000.0, 0.0, 1000.0])    # |r| < 1738 km
    # exactly on the sphere is allowed
    campanella.sh_accel_bf(MOON_FILE, 10, [0.0, 0.0, R_MOON])


# ----------------------------------------------------------------- J2 closed form
def j2_closed_form(r, GM, R, C20bar):
    J2 = -math.sqrt(5.0) * C20bar
    x, y, z = r
    r2 = x * x + y * y + z * z
    rn = math.sqrt(r2)
    U = -GM * J2 * R * R / (2 * rn**5) * (3 * z * z - r2)
    f = -1.5 * J2 * GM * R * R / rn**5
    a = f * np.array([x * (1 - 5 * z * z / r2), y * (1 - 5 * z * z / r2), z * (3 - 5 * z * z / r2)])
    k = -GM * J2 * R * R / 2
    X = np.array(r)
    G = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            dij = 1.0 if i == j else 0.0
            diz = 1.0 if i == 2 else 0.0
            djz = 1.0 if j == 2 else 0.0
            d2z2r5 = (2 * diz * djz * rn**-5 - 10 * z * (diz * X[j] + djz * X[i]) * rn**-7
                      + z * z * (-5 * dij * rn**-7 + 35 * X[i] * X[j] * rn**-9))
            d2r3 = -3 * dij * rn**-5 + 15 * X[i] * X[j] * rn**-7
            G[i, j] = k * (3 * d2z2r5 - d2r3)
    return U, a, G


@pytest.mark.parametrize("r", [[4000.0, -3000.0, 4500.0], [0.0, 0.0, 7000.0],
                               [7000.0, 0.0, 0.0], [-5000.0, 4000.0, -2500.0]])
def test_j2_closed_form(r):
    """nmax = 2, mmax = 0 keeps only C20: the J2 field"""
    C20 = campanella.load_spherical_harmonics(EARTH_FILE, 2)["C"][2][0]
    U, a, G = campanella.sh_accel_bf(EARTH_FILE, 2, r, mmax=0)
    U_ref, a_ref, G_ref = j2_closed_form(r, GM_EARTH_FILE, R_EARTH, C20)
    G_ref = G_ref.ravel()
    print(f"J2 at {r}: U {abs(U - U_ref)/abs(U_ref):.1e}, a {relerr(a, a_ref):.1e}, G {relerr(G, G_ref):.1e}")
    assert abs(U - U_ref) <= 1e-14 * abs(U_ref)
    assert relerr(a, a_ref) <= 1e-14
    assert relerr(G, G_ref) <= 1e-14
    if r[0] == 4000.0:
        np.testing.assert_allclose(a, [9.464073937831856e-06, -7.098055453373894e-06,
                                       -6.559363744858911e-06], rtol=1e-14)


# ----------------------------------------------------------------- finite differences
FD_POINTS = [
    (MOON_FILE, [1200.0, -900.0, 1100.0]),
    (MOON_FILE, [0.0, 0.0, 1838.0]),          # pole
    (MOON_FILE, [1e-9, 0.0, -1760.0]),        # next to the pole
    (EARTH_FILE, [4000.0, -3000.0, 4500.0]),
    (EARTH_FILE, [0.0, 0.0, -7000.0]),        # pole
]


@pytest.mark.parametrize("path, r", FD_POINTS)
def test_bf_finite_differences(path, r):
    """a vs central differences of U, G vs central differences of a (h = 1e-3 km)"""
    U, a, G = campanella.sh_accel_bf(path, 10, r)
    G = np.array(G).reshape(3, 3)
    assert np.all(np.isfinite(G))
    h = 1e-3
    a_fd = np.zeros(3)
    G_fd = np.zeros((3, 3))
    for j in range(3):
        rp, rm = np.array(r), np.array(r)
        rp[j] += h
        rm[j] -= h
        Up, ap, _ = campanella.sh_accel_bf(path, 10, list(rp))
        Um, am, _ = campanella.sh_accel_bf(path, 10, list(rm))
        a_fd[j] = (Up - Um) / (2 * h)
        G_fd[:, j] = (np.array(ap) - np.array(am)) / (2 * h)
    print(f"{os.path.basename(path)} {r}: a vs dU/dr {relerr(a, a_fd):.1e}, G vs da/dr {relerr(G, G_fd):.1e}")
    assert relerr(a, a_fd) <= 1e-7
    assert relerr(G, G_fd) <= 1e-7


# ----------------------------------------------------------------- inertial (J2000) values with SPICE
# docs/spherical_harmonics.md: frame J2000, et = ET0, nmax = 10
INERTIAL_CASES = {
    "MOON": dict(
        path=MOON_FILE, frame_bf="MOON_PA", r=[1200.0, -900.0, 1100.0],
        a=[3.5973511349444243e-07, -1.7587571041219314e-07, -1.1728171437280208e-07],
        G=[-6.145239976757624e-10, 8.102712981787004e-10, -3.3654856452129834e-10, 8.102712981787004e-10,
           3.0516681002086997e-10, -1.678507351992034e-10, -3.365485645212984e-10, -1.6785073519920343e-10,
           3.093571876548927e-10],
        LU=100000.0, TU=100000.0 / np.sqrt(MU_MOON / 100000.0),
        a_canon=[0.7337339291431145, -0.35872499292057164, -0.23921371552384216],
        G_canon=[-125.34142218906241, 165.26703148585858, -68.64414713230421, 165.2670314858586,
                 62.2433657881279, -34.23568476563924, -68.64414713230423, -34.23568476563925,
                 63.098056400934446],
        rtol=1e-12),
    # earth_latest_high_prec.bpc is updated daily and EOP revisions between releases change the
    # ITRF93 rotation at ~1e-10 (observed 7.7e-11 with a 4-month-older copy), hence 1e-9
    "EARTH": dict(
        path=EARTH_FILE, frame_bf="ITRF93", r=[4000.0, -3000.0, 4500.0],
        a=[9.527367341637467e-06, -7.231411374166767e-06, -6.381102306567685e-06],
        G=[-4.768823216360781e-09, 5.458236665744668e-09, -6.549457558444116e-10, 5.458236665744668e-09,
           -1.6465529837702443e-09, 5.166043093092149e-10, -6.549457558444117e-10, 5.166043093092149e-10,
           6.415376200131039e-09],
        LU=R_EARTH, TU=np.sqrt(R_EARTH**3 / MU_EARTH),
        a_canon=[0.0009723502566887701, -0.0007380280883223141, -0.0006512466921089657],
        G_canon=[-0.0031042367905091796, 0.003553006328894893, -0.00042633300058281124, 0.0035530063288948927,
                 -0.0010718137615599896, 0.00033628046801807145, -0.00042633300058281135,
                 0.00033628046801807145, 0.004176050552069177],
        rtol=1e-9),
}


@pytest.mark.parametrize("body", ["MOON", "EARTH"])
def test_inertial_reference_values(body):
    furnsh_kernels()
    c = INERTIAL_CASES[body]
    a, G = campanella.sh_accel_inertial(c["path"], 10, "J2000", c["frame_bf"], ET0, c["r"])
    print(f"{body}: a {relerr(a, c['a']):.1e}, G {relerr(G, c['G']):.1e}")
    assert relerr(a, c["a"]) <= c["rtol"]
    assert relerr(G, c["G"]) <= c["rtol"]
    LU, TU = c["LU"], c["TU"]
    a_c, G_c = campanella.sh_accel_inertial(c["path"], 10, "J2000", c["frame_bf"], ET0,
                                            list(np.array(c["r"]) / LU), LU=LU, TU=TU)
    print(f"{body} canonical: a {relerr(a_c, c['a_canon']):.1e}, G {relerr(G_c, c['G_canon']):.1e}")
    assert relerr(a_c, c["a_canon"]) <= c["rtol"]
    assert relerr(G_c, c["G_canon"]) <= c["rtol"]


@pytest.mark.parametrize("body", ["MOON", "EARTH"])
def test_inertial_gradient_fd(body):
    """G_in = T^T G_bf T vs central differences of a_in at fixed et"""
    furnsh_kernels()
    c = INERTIAL_CASES[body]
    _, G = campanella.sh_accel_inertial(c["path"], 10, "J2000", c["frame_bf"], ET0, c["r"])
    G = np.array(G).reshape(3, 3)
    h = 1e-3
    G_fd = np.zeros((3, 3))
    for j in range(3):
        rp, rm = np.array(c["r"]), np.array(c["r"])
        rp[j] += h
        rm[j] -= h
        ap, _ = campanella.sh_accel_inertial(c["path"], 10, "J2000", c["frame_bf"], ET0, list(rp))
        am, _ = campanella.sh_accel_inertial(c["path"], 10, "J2000", c["frame_bf"], ET0, list(rm))
        G_fd[:, j] = (np.array(ap) - np.array(am)) / (2 * h)
    print(f"{body}: G vs FD {relerr(G, G_fd):.1e}")
    assert relerr(G, G_fd) <= 1e-7


def test_inertial_errors():
    furnsh_kernels()
    with pytest.raises(RuntimeError, match="NOT_A_FRAME"):
        campanella.sh_accel_inertial(MOON_FILE, 10, "J2000", "NOT_A_FRAME", ET0, [1800.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        campanella.sh_accel_inertial(MOON_FILE, 10, "J2000", "MOON_PA", ET0, [1000.0, 0.0, 0.0])
    # the SPICE error state was reset
    campanella.sh_accel_inertial(MOON_FILE, 10, "J2000", "MOON_PA", ET0, [1800.0, 0.0, 0.0])


# ----------------------------------------------------------------- propagation
# Low lunar orbit: r = 1838 km (100 km altitude), circular, i = 60 deg in J2000; period ~ 2 h
GM_LIST = [MU_MOON, 398600.432896939, 132712197035.766]
BODIES = ["MOON", "EARTH", "SUN"]
SRP_ARGS = dict(A=34.675, m=1850.0, rhoS=0.9, rhoD=0.0)
NO_SRP_ARGS = dict(A=0.0, m=1850.0, rhoS=0.9, rhoD=0.0)
R_LLO = 1838.0
V_LLO = math.sqrt(MU_MOON / R_LLO)
INC = math.radians(60.0)
STATE_LLO = [R_LLO, 0.0, 0.0, 0.0, V_LLO * math.cos(INC), V_LLO * math.sin(INC)]
SH_MOON = dict(sh_file=MOON_FILE, sh_nmax=10, frame_bf="MOON_PA")
TF_LLO = 6 * 3600.0


def moon_prop(cls, sh=True, srp=False, LU=1.0, TU=1.0, gm=GM_LIST):
    args = {}
    if cls is campanella.PropagatorNbodySrp:
        args = SRP_ARGS
    elif cls is campanella.PropagatorGve:
        args = SRP_ARGS if srp else NO_SRP_ARGS
    return cls(GM=gm, BODIES=BODIES, frame="J2000", LU=LU, TU=TU,
               **args, **(SH_MOON if sh else {}))


def test_sh_off_reproduces_existing_results():
    """No sh_file, or an explicit empty one, gives exactly the results without SH"""
    furnsh_kernels()
    STATE0 = [-7505.7613773288285, 44128.19419987408, -55792.327508873765,
              0.07193326412942684, 0.018407794390508374, 0.004868437733535174]
    tspan = [0.0, 20 * 86400.0]
    off = dict(sh_file="", sh_nmax=0, sh_mmax=-1, frame_bf="")
    for cls, args in [(campanella.PropagatorNbody, {}), (campanella.PropagatorNbodySrp, SRP_ARGS),
                      (campanella.PropagatorGve, SRP_ARGS)]:
        p0 = cls(GM=GM_LIST, BODIES=BODIES, frame="J2000", LU=1.0, TU=1.0, **args)
        p1 = cls(GM=GM_LIST, BODIES=BODIES, frame="J2000", LU=1.0, TU=1.0, **args, **off)
        x0 = STATE0 if cls is not campanella.PropagatorGve else campanella.cart2mee(STATE0, MU_MOON)
        assert p0.solve_ivp(ET0, tspan, x0) == p1.solve_ivp(ET0, tspan, x0)
        if hasattr(p0, "solve_ivp_stm"):
            assert p0.solve_ivp_stm(ET0, tspan, stm_y0(x0)) == p1.solve_ivp_stm(ET0, tspan, stm_y0(x0))
        if cls is campanella.PropagatorGve:
            assert p0.eval_jacobian(ET0, 0.0, x0) == p1.eval_jacobian(ET0, 0.0, x0)
    # and the existing reference value (examples/solve_NbodySTM.cpp, test_gve.py)
    xf = final_row(campanella.PropagatorNbody(GM=GM_LIST, BODIES=BODIES, frame="J2000", LU=1.0, TU=1.0,
                                              **off).solve_ivp(ET0, tspan, STATE0), tspan[1])
    np.testing.assert_allclose(xf, [10171.343017143072, 33674.15294325228, -60877.5267967639,
                                    0.013967106146233203, -0.0805131196626756, 0.0033270015062299776],
                               rtol=1e-12)


def test_sh_argument_errors():
    furnsh_kernels()
    x0 = STATE_LLO
    # GM[0] must match the GM of the file (here: Earth field around the Moon)
    prop = campanella.PropagatorNbody(GM=GM_LIST, BODIES=BODIES, frame="J2000", LU=1.0, TU=1.0,
                                      sh_file=EARTH_FILE, sh_nmax=10, frame_bf="ITRF93")
    with pytest.raises(ValueError, match="GM"):
        prop.solve_ivp(ET0, [0.0, 60.0], x0)
    # unscaled GM[0] in canonical units
    prop = moon_prop(campanella.PropagatorNbody, LU=100000.0, TU=451624.90737307334)
    with pytest.raises(ValueError, match="GM"):
        prop.solve_ivp(ET0, [0.0, 1e-4], list(np.array(x0) / 1e5))
    prop = campanella.PropagatorNbody(GM=GM_LIST, BODIES=BODIES, frame="J2000", LU=1.0, TU=1.0,
                                      sh_file=MOON_FILE, sh_nmax=0, frame_bf="MOON_PA")
    with pytest.raises(ValueError, match="sh_nmax"):
        prop.solve_ivp(ET0, [0.0, 60.0], x0)
    prop = campanella.PropagatorNbody(GM=GM_LIST, BODIES=BODIES, frame="J2000", LU=1.0, TU=1.0,
                                      sh_file=MOON_FILE, sh_nmax=10)
    with pytest.raises(ValueError, match="frame_bf"):
        prop.solve_ivp(ET0, [0.0, 60.0], x0)
    gve = campanella.PropagatorGve(GM=GM_LIST, BODIES=BODIES, frame="J2000", LU=1.0, TU=1.0,
                                   **NO_SRP_ARGS, sh_file=MOON_FILE, sh_nmax=10, frame_bf="NOT_A_FRAME")
    with pytest.raises(RuntimeError, match="NOT_A_FRAME"):
        gve.eval_jacobian(ET0, 0.0, campanella.cart2mee(x0, MU_MOON))


def test_sh_effect_on_llo():
    """The SH field moves a 100 km lunar orbit by kilometres in 6 h; SRP adds on top of it"""
    furnsh_kernels()
    tspan = [0.0, TF_LLO]
    xf_sh = final_row(moon_prop(campanella.PropagatorNbody).solve_ivp(ET0, tspan, STATE_LLO), TF_LLO)
    xf_pm = final_row(moon_prop(campanella.PropagatorNbody, sh=False).solve_ivp(ET0, tspan, STATE_LLO), TF_LLO)
    dr = np.linalg.norm(xf_sh[:3] - xf_pm[:3])
    print(f"SH effect after 6 h: {dr:.3f} km")
    assert 0.1 < dr < 100.0
    xf_srp = final_row(moon_prop(campanella.PropagatorNbodySrp).solve_ivp(ET0, tspan, STATE_LLO), TF_LLO)
    dr_srp = np.linalg.norm(xf_srp[:3] - xf_sh[:3])
    print(f"SRP effect on top of SH after 6 h: {dr_srp:.3e} km")
    assert 0.0 < dr_srp < 1.0


def test_cartesian_stm_vs_fd_sh():
    """Cartesian STM with SH (Uxx += G_sh) vs central differences of solve_ivp"""
    furnsh_kernels()
    prop = moon_prop(campanella.PropagatorNbody)
    tspan = [0.0, TF_LLO]
    yf = final_row(prop.solve_ivp_stm(ET0, tspan, stm_y0(STATE_LLO)), TF_LLO)
    Phi = yf[6:].reshape(6, 6)
    xf = final_row(prop.solve_ivp(ET0, tspan, STATE_LLO), TF_LLO)
    assert np.max(np.abs(yf[:3] - xf[:3])) <= 1e-6

    steps = [1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6]
    Phi_fd = np.zeros((6, 6))
    for j in range(6):
        xp, xm = np.array(STATE_LLO), np.array(STATE_LLO)
        xp[j] += steps[j]
        xm[j] -= steps[j]
        Phi_fd[:, j] = (final_row(prop.solve_ivp(ET0, tspan, list(xp)), TF_LLO)
                        - final_row(prop.solve_ivp(ET0, tspan, list(xm)), TF_LLO)) / (2 * steps[j])
    # compare in scaled units so that the position and velocity blocks weigh alike
    D = np.diag([R_LLO] * 3 + [V_LLO] * 3)
    Dinv = np.linalg.inv(D)
    err = row_normalised_error(Dinv @ Phi @ D, Dinv @ Phi_fd @ D)
    # the point-mass STM differs from the SH STM by much more than the tolerance
    Phi_pm = final_row(moon_prop(campanella.PropagatorNbody, sh=False).solve_ivp_stm(
        ET0, tspan, stm_y0(STATE_LLO)), TF_LLO)[6:].reshape(6, 6)
    err_pm = row_normalised_error(Dinv @ Phi_pm @ D, Dinv @ Phi_fd @ D)
    print(f"Cartesian STM with SH vs FD: {err:.3e} (point-mass STM vs FD: {err_pm:.3e})")
    assert err <= 1e-6
    assert err_pm > 100 * err


def _mee_fd_steps(x0):
    return np.array([1e-6 * x0[0], 1e-7, 1e-7, 1e-7, 1e-7, 1e-7])


@pytest.mark.parametrize("srp", [False, True])
def test_gve_stm_vs_fd_sh(srp):
    """GVE STM with SH (G += G_sh) vs central differences of solve_ivp"""
    furnsh_kernels()
    prop = moon_prop(campanella.PropagatorGve, srp=srp)
    x0 = np.array(campanella.cart2mee(STATE_LLO, MU_MOON))
    tspan = [0.0, TF_LLO]
    yf = final_row(prop.solve_ivp_stm(ET0, tspan, stm_y0(x0)), TF_LLO)
    Phi = yf[6:].reshape(6, 6)
    steps = _mee_fd_steps(x0)
    Phi_fd = np.zeros((6, 6))
    for j in range(6):
        xp, xm = x0.copy(), x0.copy()
        xp[j] += steps[j]
        xm[j] -= steps[j]
        Phi_fd[:, j] = (final_row(prop.solve_ivp(ET0, tspan, list(xp)), TF_LLO)
                        - final_row(prop.solve_ivp(ET0, tspan, list(xm)), TF_LLO)) / (2 * steps[j])
    D = np.diag([x0[0], 1, 1, 1, 1, 1])
    err = row_normalised_error(np.linalg.inv(D) @ Phi @ D, np.linalg.inv(D) @ Phi_fd @ D)
    print(f"GVE STM with SH (srp={srp}) vs FD: {err:.3e}")
    assert err <= 1e-6


@pytest.mark.parametrize("t", [0.0, 3600.0])
def test_gve_jacobian_vs_fd_sh(t):
    furnsh_kernels()
    prop = moon_prop(campanella.PropagatorGve, srp=True)
    x = np.array(campanella.cart2mee(STATE_LLO, MU_MOON))
    _, A = prop.eval_jacobian(ET0, t, list(x))
    A = np.array(A).reshape(6, 6)
    steps = _mee_fd_steps(x)
    A_fd = np.zeros((6, 6))
    for m in range(6):
        xp, xm = x.copy(), x.copy()
        xp[m] += steps[m]
        xm[m] -= steps[m]
        A_fd[:, m] = (np.array(prop.eval_jacobian(ET0, t, list(xp))[0])
                      - np.array(prop.eval_jacobian(ET0, t, list(xm))[0])) / (2 * steps[m])
    scale = np.diag([x[0], 1, 1, 1, 1, 1])
    err = row_normalised_error(A @ scale, A_fd @ scale)
    print(f"GVE Jacobian with SH, t = {t}: {err:.3e}")
    assert err <= 1e-6


@pytest.mark.parametrize("srp", [False, True])
def test_gve_vs_cartesian_sh(srp):
    """GVE and Cartesian propagators see the same SH acceleration"""
    furnsh_kernels()
    tspan = [0.0, TF_LLO]
    x0 = campanella.cart2mee(STATE_LLO, MU_MOON)
    xf = final_row(moon_prop(campanella.PropagatorGve, srp=srp).solve_ivp(ET0, tspan, x0), TF_LLO)
    cls = campanella.PropagatorNbodySrp if srp else campanella.PropagatorNbody
    rv_f = final_row(moon_prop(cls).solve_ivp(ET0, tspan, STATE_LLO), TF_LLO)
    rv_gve = np.array(campanella.mee2cart(list(xf), MU_MOON))
    dr = np.linalg.norm(rv_gve[:3] - rv_f[:3])
    dv = np.linalg.norm(rv_gve[3:] - rv_f[3:])
    print(f"GVE vs Cartesian with SH (srp={srp}): |dr| = {dr:.3e} km, |dv| = {dv:.3e} km/s")
    assert dr <= 1e-6 and dv <= 1e-9


def test_canonical_units_sh():
    """1-day low lunar orbit with SH in km/s and in canonical units (test_canonical_moon.py scales)"""
    furnsh_kernels()
    LU = 100000.0
    VU = np.sqrt(MU_MOON / LU)
    TU = LU / VU
    gm_canon = [gm / MU_MOON for gm in GM_LIST]
    tf = 86400.0
    D = np.diag([LU] * 3 + [VU] * 3)
    x0_canon = np.array(STATE_LLO) / np.diag(D)
    xf_km = final_row(moon_prop(campanella.PropagatorNbody).solve_ivp(ET0, [0.0, tf], STATE_LLO), tf)
    canon = moon_prop(campanella.PropagatorNbody, LU=LU, TU=TU, gm=gm_canon)
    xf = D @ final_row(canon.solve_ivp(ET0, [0.0, tf / TU], list(x0_canon)), tf / TU)
    dr = np.linalg.norm(xf[:3] - xf_km[:3])
    dv = np.linalg.norm(xf[3:] - xf_km[3:])
    print(f"canonical vs km/s with SH: |dr| = {dr:.3e} km, |dv| = {dv:.3e} km/s")
    assert dr <= 1e-4 and dv <= 1e-7

    # GVE in canonical units
    gve_km = moon_prop(campanella.PropagatorGve)
    gve = moon_prop(campanella.PropagatorGve, LU=LU, TU=TU, gm=gm_canon)
    m_km = final_row(gve_km.solve_ivp(ET0, [0.0, tf], campanella.cart2mee(STATE_LLO, MU_MOON)), tf)
    m_c = final_row(gve.solve_ivp(ET0, [0.0, tf / TU], campanella.cart2mee(list(x0_canon), 1.0)), tf / TU)
    rv_km = np.array(campanella.mee2cart(list(m_km), MU_MOON))
    rv_c = D @ np.array(campanella.mee2cart(list(m_c), 1.0))
    print(f"GVE canonical vs km/s with SH: |dr| = {np.linalg.norm(rv_c[:3] - rv_km[:3]):.3e} km")
    assert np.linalg.norm(rv_c[:3] - rv_km[:3]) <= 1e-4


def test_earth_leo_sh():
    """EGM2008 n10 in ITRF93: LEO, GVE STM vs FD and GVE vs Cartesian"""
    furnsh_kernels()
    gm = [GM_EARTH_FILE, MU_MOON, 132712197035.766]
    bodies = ["EARTH", "MOON", "SUN"]
    sh = dict(sh_file=EARTH_FILE, sh_nmax=10, frame_bf="ITRF93")
    rv0 = [-2103.7154800484846, 3643.742096113127, 5308.456731666615,
           -6.647330004445163, -3.833624329580397, 0.004603821856452162]
    tf = 3 * 3600.0
    cart = campanella.PropagatorNbody(GM=gm, BODIES=bodies, frame="J2000", LU=1.0, TU=1.0, **sh)
    gve = campanella.PropagatorGve(GM=gm, BODIES=bodies, frame="J2000", LU=1.0, TU=1.0, **NO_SRP_ARGS, **sh)
    rv_f = final_row(cart.solve_ivp(ET0, [0.0, tf], rv0), tf)
    x0 = np.array(campanella.cart2mee(rv0, gm[0]))
    yf = final_row(gve.solve_ivp_stm(ET0, [0.0, tf], stm_y0(x0)), tf)
    rv_gve = np.array(campanella.mee2cart(list(yf[:6]), gm[0]))
    dr = np.linalg.norm(rv_gve[:3] - rv_f[:3])
    print(f"Earth LEO, GVE vs Cartesian with SH: |dr| = {dr:.3e} km")
    assert dr <= 1e-6
    # J2 dominates: the point-mass orbit is kilometres away
    rv_pm = final_row(campanella.PropagatorNbody(GM=gm, BODIES=bodies, frame="J2000", LU=1.0, TU=1.0)
                      .solve_ivp(ET0, [0.0, tf], rv0), tf)
    assert np.linalg.norm(rv_pm[:3] - rv_f[:3]) > 1.0

    Phi = yf[6:].reshape(6, 6)
    steps = _mee_fd_steps(x0)
    Phi_fd = np.zeros((6, 6))
    for j in range(6):
        xp, xm = x0.copy(), x0.copy()
        xp[j] += steps[j]
        xm[j] -= steps[j]
        Phi_fd[:, j] = (final_row(gve.solve_ivp(ET0, [0.0, tf], list(xp)), tf)
                        - final_row(gve.solve_ivp(ET0, [0.0, tf], list(xm)), tf)) / (2 * steps[j])
    D = np.diag([x0[0], 1, 1, 1, 1, 1])
    err = row_normalised_error(np.linalg.inv(D) @ Phi @ D, np.linalg.inv(D) @ Phi_fd @ D)
    print(f"Earth LEO, GVE STM with SH vs FD: {err:.3e}")
    assert err <= 1e-6



# ----------------------------------------------------------------- review fixes: domain, parsers, arguments
def _low_periapsis_state(rp=1733.0, ra=R_LLO):
    """Start at apoapsis ra; periapsis rp (km) is below the GRGM1200A reference radius 1738 km"""
    a = 0.5 * (ra + rp)
    v = math.sqrt(MU_MOON * (2 / ra - 1 / a))
    return [ra, 0.0, 0.0, 0.0, v * math.cos(INC), v * math.sin(INC)]


def test_below_reference_radius_raises():
    """Going below sh_rho_min * R raises ValueError instead of returning a truncated trajectory"""
    furnsh_kernels()
    x0 = _low_periapsis_state()
    tspan = [0.0, 20000.0]
    for cls in [campanella.PropagatorNbody, campanella.PropagatorNbodySrp]:
        with pytest.raises(ValueError, match="sh_rho_min"):
            moon_prop(cls).solve_ivp(ET0, tspan, x0)
    with pytest.raises(ValueError, match="sh_rho_min"):
        moon_prop(campanella.PropagatorNbody).solve_ivp_stm(ET0, tspan, stm_y0(x0))
    m0 = campanella.cart2mee(x0, MU_MOON)
    gve = moon_prop(campanella.PropagatorGve)
    with pytest.raises(ValueError, match="sh_rho_min"):
        gve.solve_ivp(ET0, tspan, m0)
    with pytest.raises(ValueError, match="sh_rho_min"):
        gve.solve_ivp_stm(ET0, tspan, stm_y0(m0))
    # at periapsis (L + pi) the position is below R
    m_peri = list(m0)
    m_peri[5] += math.pi
    with pytest.raises(ValueError, match="sh_rho_min"):
        gve.eval_jacobian(ET0, 0.0, m_peri)
    # without SH the same orbit still integrates to tf
    final_row(moon_prop(campanella.PropagatorNbody, sh=False).solve_ivp(ET0, tspan, x0), tspan[1])


def test_sh_rho_min_allows_low_arcs():
    furnsh_kernels()
    x0 = _low_periapsis_state()
    tspan = [0.0, 20000.0]
    kw = dict(GM=GM_LIST, BODIES=BODIES, frame="J2000", LU=1.0, TU=1.0, **SH_MOON, sh_rho_min=0.99)
    xf = final_row(campanella.PropagatorNbody(**kw).solve_ivp(ET0, tspan, x0), tspan[1])
    m0 = campanella.cart2mee(x0, MU_MOON)
    mf = final_row(campanella.PropagatorGve(**kw, **NO_SRP_ARGS).solve_ivp(ET0, tspan, m0), tspan[1])
    rv = np.array(campanella.mee2cart(list(mf), MU_MOON))
    assert np.linalg.norm(rv[:3] - xf[:3]) <= 1e-6
    # standalone evaluators
    r_low = [0.0, 0.0, 0.995 * R_MOON]
    with pytest.raises(ValueError):
        campanella.sh_accel_bf(MOON_FILE, 10, r_low)
    campanella.sh_accel_bf(MOON_FILE, 10, r_low, rho_min=0.99)
    campanella.sh_accel_inertial(MOON_FILE, 10, "J2000", "MOON_PA", ET0, r_low, rho_min=0.99)
    # sh_rho_min must be in (0, 1]
    for bad in [0.0, -1.0, 1.5]:
        kw["sh_rho_min"] = bad
        with pytest.raises(ValueError, match="sh_rho_min"):
            campanella.PropagatorNbody(**kw).solve_ivp(ET0, [0.0, 60.0], STATE_LLO)
        with pytest.raises(ValueError):
            campanella.sh_accel_bf(MOON_FILE, 10, STATE_LLO[:3], rho_min=bad)


def test_rho_min_boundary_rounding():
    """A point built as unit vector * R (possibly 1 ulp inside) is accepted"""
    rng = np.random.default_rng(3)
    LU = 100000.0
    for _ in range(200):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        campanella.sh_accel_bf(MOON_FILE, 10, list(u * R_MOON))
        campanella.sh_accel_bf(MOON_FILE, 10, list(u * (R_MOON / LU)),
                               mu=GM_MOON_FILE, R=R_MOON / LU)
        campanella.sh_accel_bf(MOON_FILE, 10, list(u * 0.99 * R_MOON), rho_min=0.99)
    with pytest.raises(ValueError):
        campanella.sh_accel_bf(MOON_FILE, 10, [0.0, 0.0, R_MOON * (1 - 1e-12)])


def test_sh_accel_bf_mu_and_R_together():
    r = [1200.0, -900.0, 1100.0]
    with pytest.raises(ValueError, match="mu and R"):
        campanella.sh_accel_bf(MOON_FILE, 10, r, mu=GM_MOON_FILE)
    with pytest.raises(ValueError, match="mu and R"):
        campanella.sh_accel_bf(MOON_FILE, 10, r, R=R_MOON)
    np.testing.assert_array_equal(campanella.sh_accel_bf(MOON_FILE, 10, r, mu=GM_MOON_FILE, R=R_MOON)[1],
                                  campanella.sh_accel_bf(MOON_FILE, 10, r)[1])


def _write_lines(path, lines, newline="\n"):
    with open(path, "w", newline="") as fh:
        fh.write(newline.join(lines) + newline)
    return str(path)


def _moon_rows():
    with open(MOON_FILE, "rb") as fh:
        return fh.read().decode().split("\r\n")


def test_non_finite_coefficient_rejected(tmp_path):
    lines = _moon_rows()
    for i, line in enumerate(lines):
        if line.startswith("    3,    1,"):
            lines[i] = "    3,    1,nan, 5.4545456359614202e-06, 0.0, 0.0"
    path = _write_lines(tmp_path / "nan.tab", lines, "\r\n")
    with pytest.raises(RuntimeError, match="non-finite"):
        campanella.load_spherical_harmonics(path, 10)
    # degree 2 alone does not reach the bad row
    campanella.load_spherical_harmonics(path, 2)
    lines = open(EARTH_FILE).read().splitlines()
    lines = [l.replace("0.141434926192941e-05", "inf") for l in lines]
    with pytest.raises(RuntimeError, match="non-finite"):
        campanella.load_spherical_harmonics(_write_lines(tmp_path / "inf.gfc", lines), 10)


def test_order_limited_files(tmp_path):
    """Degree 8, order 4 files: mmax defaults to the order of the file"""
    full = campanella.load_spherical_harmonics(MOON_FILE, 8)
    lines = _moon_rows()
    header = lines[0].replace(" 1200, 1200,", "    8,    4,")
    rows = [l for l in lines[1:] if l.strip() and int(l.split(",")[0]) <= 8 and int(l.split(",")[1]) <= 4]
    tab = _write_lines(tmp_path / "order4.tab", [header] + rows, "\r\n")
    info = campanella.load_spherical_harmonics(tab, 8)
    assert info["file_degree"] == 8 and info["file_order"] == 4 and info["mmax"] == 4
    for n in range(2, 9):
        for m in range(n + 1):
            assert info["C"][n][m] == (full["C"][n][m] if m <= 4 else 0.0)
    r = [1200.0, -900.0, 1100.0]
    U, a, G = campanella.sh_accel_bf(tab, 8, r)
    U_ref, a_ref, G_ref = campanella.sh_accel_bf(MOON_FILE, 8, r, mmax=4)
    assert U == U_ref and a == a_ref and G == G_ref
    assert campanella.load_spherical_harmonics(tab, 8, mmax=3)["mmax"] == 3
    with pytest.raises(ValueError, match="order"):
        campanella.load_spherical_harmonics(tab, 8, mmax=5)
    assert campanella.load_spherical_harmonics(tab, 3)["mmax"] == 3

    # ICGEM has no order keyword: an order-limited file needs an explicit mmax
    glines = open(EARTH_FILE).read().splitlines()
    glines = [l for l in glines if not (l.startswith("gfc") and int(l.split()[2]) > 4)]
    gfc = _write_lines(tmp_path / "order4.gfc", glines)
    with pytest.raises(RuntimeError, match="missing"):
        campanella.load_spherical_harmonics(gfc, 8)
    U, a, G = campanella.sh_accel_bf(gfc, 8, [4000.0, -3000.0, 4500.0], mmax=4)
    U_ref, a_ref, G_ref = campanella.sh_accel_bf(EARTH_FILE, 8, [4000.0, -3000.0, 4500.0], mmax=4)
    assert U == U_ref and a == a_ref and G == G_ref


def test_icgem_header_spec(tmp_path):
    """Long free-text preamble, begin_of_head, comments starting with keywords, indented end_of_head"""
    lines = open(EARTH_FILE).read().splitlines()
    head_end = next(i for i, l in enumerate(lines) if l.startswith("end_of_head"))
    preamble = [f"free text line {i}" for i in range(300)]
    preamble += ["radius of the Earth is not given here", "max_degree is defined below",
                 "earth_gravity_constant 1.0e+00",      # before begin_of_head: discarded
                 "begin_of_head ====="]
    header = ["   " + l for l in lines[:head_end]]
    header.insert(3, "radius of the reference sphere, in m:")
    header.insert(4, "norm of the coefficients is given below")
    body = ["  end_of_head ==========="] + lines[head_end + 1:]
    path = _write_lines(tmp_path / "spec.gfc", preamble + header + body)
    info = campanella.load_spherical_harmonics(path, 10)
    ref = campanella.load_spherical_harmonics(EARTH_FILE, 10)
    for key in ["GM", "R", "file_degree", "tide_system", "C", "S"]:
        assert info[key] == ref[key], key
    # an unnormalized model is still rejected
    bad = [l.replace("fully_normalized", "unnormalized") for l in preamble + header + body]
    with pytest.raises(RuntimeError, match="norm"):
        campanella.load_spherical_harmonics(_write_lines(tmp_path / "unnorm.gfc", bad), 10)
    # no end_of_head anywhere
    with pytest.raises(RuntimeError, match="not a PDS SHADR"):
        campanella.load_spherical_harmonics(_write_lines(tmp_path / "nohead.gfc", preamble + header), 10)


def test_cache_sees_file_changes(tmp_path):
    lines = open(EARTH_FILE).read().splitlines()
    path = _write_lines(tmp_path / "edit.gfc", lines)
    c20 = campanella.load_spherical_harmonics(path, 4)["C"][2][0]
    lines = [l.replace("-0.484165143790815e-03", "-0.484165143790800e-03") for l in lines]
    _write_lines(tmp_path / "edit.gfc", lines + ["", ""])   # different size, so a new key
    assert campanella.load_spherical_harmonics(path, 4)["C"][2][0] != c20


def test_cache_same_size_rewrites(tmp_path):
    """Same-size rewrites in a tight loop (same mtime tick) are re-read, not served from the cache"""
    lines = open(EARTH_FILE).read().splitlines()
    path = tmp_path / "loop.gfc"
    for k in range(50):
        c20 = f"-0.48416514379{k:04d}e-03"        # same width as the original value
        _write_lines(path, [l.replace("-0.484165143790815e-03", c20) for l in lines])
        assert campanella.load_spherical_harmonics(str(path), 4)["C"][2][0] == float(c20)


def test_cache_mtime_restored(tmp_path):
    """A rewrite whose mtime is set back to the old value (cp -p style) is re-read"""
    lines = open(EARTH_FILE).read().splitlines()
    path = tmp_path / "restore.gfc"
    _write_lines(path, lines)
    old = os.stat(path)
    past = (old.st_atime_ns - 10**11, old.st_mtime_ns - 10**11)   # 100 s ago
    os.utime(path, ns=past)
    c20 = campanella.load_spherical_harmonics(str(path), 4)["C"][2][0]
    _write_lines(path, [l.replace("-0.484165143790815e-03", "-0.484165143790999e-03") for l in lines])
    os.utime(path, ns=past)                                       # same size and mtime as before
    c20_new = campanella.load_spherical_harmonics(str(path), 4)["C"][2][0]
    assert c20 == -0.484165143790815e-03 and c20_new == -0.484165143790999e-03


def test_icgem_duplicate_keyword(tmp_path):
    lines = open(EARTH_FILE).read().splitlines()
    head_end = next(i for i, l in enumerate(lines) if l.startswith("end_of_head"))
    for dup in ["radius 6378137.0", "earth_gravity_constant 0.3986004418E+15", "max_degree 360",
                "norm fully_normalized", "tide_system zero_tide"]:
        bad = lines[:head_end] + [dup] + lines[head_end:]
        with pytest.raises(RuntimeError, match="more than once"):
            campanella.load_spherical_harmonics(_write_lines(tmp_path / "dup.gfc", bad), 4)
    # a duplicate before begin_of_head is free text, and a comment line is not a keyword line
    ok = ["radius 1.0", "begin_of_head"] + lines[:head_end] + ["radius of the sphere, in m"] + lines[head_end:]
    info = campanella.load_spherical_harmonics(_write_lines(tmp_path / "ok.gfc", ok), 4)
    assert info["R"] == pytest.approx(R_EARTH, rel=1e-15)


def test_bom_and_detection_messages(tmp_path):
    ref = campanella.load_spherical_harmonics(MOON_FILE, 10)
    with open(MOON_FILE, "rb") as fh:
        raw = fh.read()
    bom_tab = tmp_path / "bom.tab"
    bom_tab.write_bytes(b"\xef\xbb\xbf" + raw)
    info = campanella.load_spherical_harmonics(str(bom_tab), 10)
    assert info["C"] == ref["C"] and info["S"] == ref["S"] and info["GM"] == ref["GM"]
    bom_gfc = tmp_path / "bom.gfc"
    with open(EARTH_FILE, "rb") as fh:
        bom_gfc.write_bytes(b"\xef\xbb\xbf" + fh.read())
    assert campanella.load_spherical_harmonics(str(bom_gfc), 10)["C"] == \
        campanella.load_spherical_harmonics(EARTH_FILE, 10)["C"]
    # neither format: the message says why
    bad = tmp_path / "bad.txt"
    bad.write_text("1, 2, 3\nsome text\n")
    with pytest.raises(RuntimeError, match="8-field SHADR header"):
        campanella.load_spherical_harmonics(str(bad), 4)
    # a first line of 8 numbers is read as SHADR; header errors say so
    lines = raw.split(b"\r\n")
    lines[0] = lines[0].replace(b"    1, 0.0000", b"    0, 0.0000", 1)
    shadr_bad = tmp_path / "unnorm.tab"
    shadr_bad.write_bytes(b"\r\n".join(lines))
    with pytest.raises(RuntimeError, match="detected as PDS SHADR"):
        campanella.load_spherical_harmonics(str(shadr_bad), 4)
    with pytest.raises(ValueError, match="detected as PDS SHADR"):
        campanella.load_spherical_harmonics(str(_order_header(tmp_path, raw, 3)), 4)   # nmax > degree


def _order_header(tmp_path, raw, degree):
    """Degree-10 GRGM1200A head with the header degree and order set to degree"""
    lines = raw.split(b"\r\n")
    lines[0] = lines[0].replace(b" 1200, 1200,", f"{degree:5d},{degree:5d},".encode())
    path = tmp_path / f"deg{degree}.tab"
    path.write_bytes(b"\r\n".join(lines))
    return path


def test_shadr_rows_above_header_order_warn(tmp_path, capfd):
    with open(MOON_FILE, "rb") as fh:
        lines = fh.read().split(b"\r\n")
    lines[0] = lines[0].replace(b" 1200, 1200,", b"   10,    4,")       # order 4, but all rows present
    path = tmp_path / "rows_above_order.tab"
    path.write_bytes(b"\r\n".join(lines))
    capfd.readouterr()
    info = campanella.load_spherical_harmonics(str(path), 10)
    err = capfd.readouterr().err
    assert err.count("order above the order of the field") == 1
    assert info["mmax"] == 4
    full = campanella.load_spherical_harmonics(MOON_FILE, 10)
    assert info["C"][6][4] == full["C"][6][4] and info["C"][6][5] == 0.0
    # no warning for the regular file
    capfd.readouterr()
    with open(MOON_FILE, "rb") as fh:
        campanella.load_spherical_harmonics(str(_order_header(tmp_path, fh.read(), 10)), 10)
    assert "order above" not in capfd.readouterr().err

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
