"""Pure-Python copy of Open-Car-Dynamics' MF5.2 (lib/ocd_tire_models_cpp/src/mf_52.cpp, zero
camber) with the three tyre parameter sets used in ocd_first_lap: the shipped AV-21, sf70h_tum
(peak mu from TUM) and sf70h_f1 (plus PL14 slip stiffness / shape). Used only for the deck's
tyre figures; checked against docs/results/sf70h_f1/results.json (peak slips)."""
import math
import numpy as np

AV21 = dict(FNOMIN=3000., LMUX=.97, LMUY=.97, LKX=1., LKY=1.,
            PCX1=1.6, PDX1=1.5, PDX2=-.04, PEX1=.7, PEX2=-.17, PEX3=.023, PEX4=-.14,
            PKX1=30.7, PKX2=.27, PKX3=.13, PHX1=0., PHX2=0., PVX1=0., PVX2=0.,
            PCY1=1.5, PDY1=1.2, PDY2=-.09, PEY1=.1, PEY2=-1.2, PEY3=.05, PEY4=-8.1,
            PKY1=-75.5, PKY2=4.65, PKY4=2., PHY1=.003, PHY2=-.0006, PVY1=.04, PVY2=-.02,
            RBX1=17.4, RBX2=12.9, RCX1=1.1, RHX1=.001, REX1=0., REX2=0.,
            RBY1=20.6, RBY2=-23.5, RBY3=.001, RCY1=1., RHY1=-.02, RHY2=0., REY1=0., REY2=0.,
            RVY1=.16, RVY2=.03, RVY4=-29.7, RVY5=.03, RVY6=0.)
TUM_MU = dict(front=(1.65, 1.85), rear=(1.95, 2.15))   # mux, muy at 3 kN; dmu/dFz = -5e-5 /N


def sf70h_tum(axle):
    mux, muy = TUM_MU[axle]
    return {**AV21, 'PDX1': mux / .97, 'PDX2': -5e-5 * 3000 / .97,
            'PDY1': muy / .97, 'PDY2': -5e-5 * 3000 / .97}


def sf70h_f1(axle):
    pk = dict(front=(42.16885638827738, -0.9583830997335738),
              rear=(49.835921186145995, -0.38335323989343095))[axle]
    p = {**sf70h_tum(axle), 'PKX1': pk[0], 'PKX2': pk[1], 'PKX3': 0., 'PCX1': 1.9, 'PCY1': 1.9}
    for e in ('PEX1', 'PEX2', 'PEX3', 'PEX4', 'PEY1', 'PEY2', 'PEY3', 'PEY4'):
        p[e] = 0.
    return p


def fx0(p, fz, kappa):
    dfz = (fz - p['FNOMIN']) / p['FNOMIN']
    k = kappa + p['PHX1'] + p['PHX2'] * dfz
    C = p['PCX1']; D = (p['PDX1'] + p['PDX2'] * dfz) * p['LMUX'] * fz
    E = (p['PEX1'] + p['PEX2'] * dfz + p['PEX3'] * dfz ** 2) * (1 - p['PEX4'] * np.sign(k))
    K = fz * (p['PKX1'] + p['PKX2'] * dfz) * math.exp(p['PKX3'] * dfz) * p['LKX']
    B = K / (C * D)
    return D * np.sin(C * np.arctan(B * k - E * (B * k - np.arctan(B * k)))) + fz * (p['PVX1'] + p['PVX2'] * dfz) * p['LMUX']


def fy0(p, fz, alpha):
    dfz = (fz - p['FNOMIN']) / p['FNOMIN']
    Svy = fz * (p['PVY1'] + p['PVY2'] * dfz) * p['LMUY']
    Ky = p['PKY1'] * p['FNOMIN'] * math.sin(p['PKY4'] * math.atan(fz / (p['PKY2'] * p['FNOMIN']))) * p['LKY']
    a = alpha + p['PHY1'] + p['PHY2'] * dfz
    C = p['PCY1']; D = (p['PDY1'] + p['PDY2'] * dfz) * p['LMUY'] * fz
    B = Ky / (C * D)
    E = (p['PEY1'] + p['PEY2'] * dfz) * (1 - p['PEY3'] * np.sign(a))
    return D * np.sin(C * np.arctan(B * a - E * (B * a - np.arctan(B * a)))) + Svy


def combined(p, fz, kappa, alpha):
    """Fx, Fy with MF5.2 combined-slip weighting (alpha in OCD's sign: fy0 is called with -alpha)."""
    dfz = (fz - p['FNOMIN']) / p['FNOMIN']
    a = -alpha
    Fx0, Fy0 = fx0(p, fz, kappa), fy0(p, fz, a)
    Bxa = p['RBX1'] * np.cos(np.arctan(p['RBX2'] * kappa))
    Sh = p['RHX1']
    g = lambda B, C, E, x: np.cos(C * np.arctan(B * x - E * (B * x - np.arctan(B * x))))
    Gxa = g(Bxa, p['RCX1'], p['REX1'] + p['REX2'] * dfz, a + Sh) / g(Bxa, p['RCX1'], p['REX1'] + p['REX2'] * dfz, Sh)
    muy = (p['PDY1'] + p['PDY2'] * dfz) * p['LMUY']
    Dvyk = muy * fz * (p['RVY1'] + p['RVY2'] * dfz) * np.cos(np.arctan(p['RVY4'] * a))
    Svyk = Dvyk * np.sin(p['RVY5'] * np.arctan(p['RVY6'] * kappa))
    Byk = p['RBY1'] * np.cos(np.arctan(p['RBY2'] * (a - p['RBY3'])))
    Shk = p['RHY1'] + p['RHY2'] * dfz
    Eyk = p['REY1'] + p['REY2'] * dfz
    Gyk = g(Byk, p['RCY1'], Eyk, kappa + Shk) / g(Byk, p['RCY1'], Eyk, Shk)
    return Gxa * Fx0, Gyk * Fy0 + Svyk
