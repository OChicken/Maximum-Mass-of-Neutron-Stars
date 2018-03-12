# coding=utf-8
# author: Ma Seoyin, 2014 Applied Physics, SCUT
from numpy import *
from scipy.interpolate import interp1d
import scipy.constants as C


def TOV(EoS):
    Msunt0 = 1.989E30  # The mass of sun ---------------------------------------------------------------------------
    z1 = linspace(1E18, 4.5E18, 100)  # Central density: search range ----------------------------------------------
    col = len(z1)
    # Interpolate to the EoS ---------------------------------------------------------------------------------------
    rho = EoS[:, 0]
    P = EoS[:, 1]
    Interp_EoS = interp1d(rho, P, kind='linear')
    Interp_EoS_Inv = interp1d(P, rho, kind='linear')
    # Define -------------------------------------------------------------------------------------------------------
    zr = array(zeros((1, 1)))
    p = array(zeros((1, 1)))
    r = array(zeros((1, 1)))
    m = array(zeros((1, 1)))
    pcalculate = array(zeros(col))
    rsurface = array(zeros(col))
    Msunt = array(zeros(col))
    indexmax = array(zeros(col))
    # Begin to integrate the TOV equation with 4th order Rung-Kutta method
    x = 1
    for s in range(col):
        # Define the iteration parameters --------------------------------------------------------------------------
        i = -1
        n = int(1E18)
        h = 1
        # Define ---------------------------------------------------------------------------------------------------
        if s == 1:
            zr_col_len = shape(zr)[1]
            zr = vstack((zr, array(zeros((col - 1, zr_col_len)))))
            p = vstack((p, array(zeros((col - 1, zr_col_len)))))
            r = vstack((r, array(zeros((col - 1, zr_col_len)))))
            m = vstack((m, array(zeros((col - 1, zr_col_len)))))
        zr[s, 0] = z1[s]  # Initial central density ----------------------------------------------------------------
        z0 = log10(z1[s])
        p0 = Interp_EoS(z0)
        p[s, 0] = 10 ** p0
        r[s, 0] = 1  # Initial radius ------------------------------------------------------------------------------
        m[s, 0] = (4/3)*pi * zr[s, 0] * (r[s, 0]**3)  # Construct the initial m ------------------------------------
        p[s, 0] = p[s, 0] - (2/3)*pi * (zr[s, 0] * (C.c**2) + p[s, 0]) * (zr[s, 0] * (C.c**2) + x * 3 * p[s, 0]) * (C.G/(C.c**4) * (r[s, 0]**2))
        # Begin the cyclic integral --------------------------------------------------------------------------------
        while i < n:
            i += 1
            if s == 0:
                zr = hstack((zr, array([[0]])))
                p = hstack((p, array([[0]])))
                r = hstack((r, array([[0]])))
                m = hstack((m, array([[0]])))
            r[s, i + 1] = r[s, i] + h
            k1 = -(zr[s, i] * C.c**2 + p[s, i]) * (m[s, i] + x * 4 * pi * r[s, i]**3 * p[s, i] / C.c**2) * C.G / (C.c**2) / (r[s, i] * (r[s, i] - 2 * C.G / (C.c**2) * m[s, i]))
            l1 = 4 * pi * r[s, i]**2 * zr[s, i]
            p01 = p[s, i] + h * k1 / 2
            if p01 > 1E-10:
                p001 = log10(p01)
                z001 = Interp_EoS_Inv(p001)
                zr[s, i] = 10**z001
            k2 = -(zr[s, i] * C.c ** 2 + (p[s, i] + 0.5 * h * k1 / 2)) * ((m[s, i] + 0.5 * h * l1 / 2) + x * 4 * pi * (r[s, i] + h / 2) ** 3 * (p[s, i] + 0.5 * h * k1 / 2) / C.c ** 2) * C.G / (C.c ** 2) / ((r[s, i] + h / 2) * ((r[s, i] + h / 2) - 2 * C.G / (C.c ** 2) * (m[s, i] + 0.5 * h * l1 / 2)))
            l2 = 4 * pi * (r[s, i] + h / 2) ** 2 * zr[s, i]
            p02 = p[s, i] + h * k2 / 2
            if p02 > 1E-10:
                p002 = log10(p02)
                z002 = Interp_EoS_Inv(p002)
                zr[s, i] = 10**z002
            k3 = -(zr[s, i] * C.c**2 + (p[s, i] + 0.5 * h * k2 / 2)) * ((m[s, i] + 0.5 * h * l2 / 2) + x * 4 * pi * (r[s, i] + h / 2)**3 * (p[s, i] + 0.5 * h * k2 / 2) / C.c**2) * C.G / C.c**2 / ((r[s, i] + h / 2) * ((r[s, i] + h / 2) - 2 * C.G / C.c**2 * (m[s, i] + 0.5 * h * l2 / 2)))
            l3 = 4 * pi * (r[s, i] + h / 2)**2 * zr[s, i]
            p03 = p[s, i] + h * k3
            if p03 > 1E-10:
                p003 = log10(p03)
                z003 = Interp_EoS_Inv(p003)
                zr[s, i] = 10**z003
            k4 = -(zr[s, i] * C.c**2 + (p[s, i] + h * k3)) * ((m[s, i] + h * l3) + x * 4 * pi * (r[s, i] + h)**3 * (p[s, i] + h * k3) / C.c**2) * C.G / C.c**2 / ((r[s, i] + h) * ((r[s, i] + h) - 2 * C.G / C.c**2 * (m[s, i] + h * l3)))
            l4 = 4 * pi * (r[s, i] + h)**2 * zr[s, i]
            p[s, i + 1] = p[s, i] + 1 / 6 * h * (k1 + 2 * k2 + 2 * k3 + k4)
            m[s, i + 1] = m[s, i] + 1 / 6 * h * (l1 + 2 * l2 + 2 * l3 + l4)
            if p[s, i + 1] > 1E-10:
                p00 = log10(p[s, i + 1])
                z00 = Interp_EoS_Inv(p00)
                zr[s, i + 1] = 10**z00
            else:
                zr[s, i + 1] = zr[s, i]  # Density (corresponding to the given central density) --------------------
                pcalculate[s] = p[s, i + 1]  # Surface pressure ----------------------------------------------------
                p[s, i + 1] = 0  # Define the surface pressure as 0 ------------------------------------------------
                j = i + 1
                i = n + 1
        rsurface[s] = r[s, j]  # Radius of the NS ------------------------------------------------------------------
        Msunt[s] = m[s, j - 1] / Msunt0  # Mass of the NS (Msun as unit) -------------------------------------------
        indexmax[s] = j

    return [z1, rsurface, Msunt]  # Return the rho (density), r (radius), and M --------------------------------
