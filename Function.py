# coding=utf-8
# author: Ma Seoyin, 2014 Applied Physics, SCUT
from numpy import *
from scipy import stats
from scipy.interpolate import interp1d
import scipy.constants as C


#   --------------------------------------------------------------------------------------------------------------------
#   Randomly generate parameters in parameters' space
#   --------------------------------------------------------------------------------------------------------------------
def GeneratePara(t, Para0, D, P):
    # [rho_crit,   Vs] -------------------------------------------------------------------------------------------------
    Lolims = array([17.916, 0.6])  # Parameters' space, lower limits
    Uplims = array([18.229, 1])  # Parameters' space, upper limits
    #  In the t = 0 & 1 processes, generate Para from a large given range ----------------------------------------------
    if t <= 1:
        Para1 = random.uniform(Lolims, Uplims, 2)
    # In the t >= 2 processes, generate Para from the previous result, according to the Gaussian Distribution ----------
    else:
        Para1 = Para0 + array(matrix(P) * matrix(D) * transpose(matrix(random.normal(size=2 * 1)))).flatten()
        while (Para1 < Lolims).any() or (Uplims < Para1).any():
            Para1 = Para0 + array(matrix(P) * matrix(D) * transpose(matrix(random.normal(size=2 * 1)))).flatten()
    return Para1


#    -------------------------------------------------------------------------------------------------------------------
#   Convert Cov to D:
#   --------------------------------------------------------------------------------------------------------------------
def Cov2D(Cov):
    Cov = matrix(Cov)
    P = array(linalg.eig(Cov)[1])  # eigenvectors
    # square of diagonalized Cov -----------------------------------------------------------------------------------
    D = array(sqrt([linalg.eig(Cov)[0][i]*identity(len(Cov))[i] for i in range(len(Cov))]))
    # D = sqrt(linalg.inv(P) * Cov * P)  # square of diagonalized Cov
    return [D, P]


#   --------------------------------------------------------------------------------------------------------------------
#   Calculate the Hypothetical values, with TOV Confinement
#   --------------------------------------------------------------------------------------------------------------------
def TOV(Para, EoS):
    Msunt0 = 1.989E30  # The mass of sun -------------------------------------------------------------------------------
    z1 = linspace(1E18, 4.5E18, 100)  # Central density: search range --------------------------------------------------
    # z1 = array([2.55555555555553E18])  # Para me ---------------------------------------------------------------------
    col = len(z1)
    # Construct the new EoS --------------------------------------------------------------------------------------------
    rho_crit = Para[0]
    Vs_crit = Para[1]
    EoS0 = EoS.copy()
    rho = EoS0[:, 0]
    P = EoS0[:, 1]
    Interp_EoS = interp1d(rho, P, kind='linear')
    P_crit = Interp_EoS(rho_crit)
    min_index = argmin(abs(rho - rho_crit))
    if rho[min_index] <= rho_crit:
        min_index += 1
    P[min_index:] = log10((Vs_crit * C.c) ** 2 * (10 ** rho[min_index:] - 10 ** rho_crit) + 10 ** P_crit)
    # Interpolate to the EoS -------------------------------------------------------------------------------------------
    Interp_EoS = interp1d(rho, P, kind='linear')
    Interp_EoS_Inv = interp1d(P, rho, kind='linear')
    # Define -----------------------------------------------------------------------------------------------------------
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
        # Define the iteration parameters ------------------------------------------------------------------------------
        i = -1
        n = int(1E18)
        h = 1
        # Define -------------------------------------------------------------------------------------------------------
        if s == 1:
            zr_col_len = shape(zr)[1]
            zr = vstack((zr, array(zeros((col - 1, zr_col_len)))))
            p = vstack((p, array(zeros((col - 1, zr_col_len)))))
            r = vstack((r, array(zeros((col - 1, zr_col_len)))))
            m = vstack((m, array(zeros((col - 1, zr_col_len)))))
        zr[s, 0] = z1[s]  # Initial central density --------------------------------------------------------------------
        z0 = log10(z1[s])
        p0 = Interp_EoS(z0)
        p[s, 0] = 10 ** p0
        r[s, 0] = 1  # Initial radius ----------------------------------------------------------------------------------
        m[s, 0] = (4/3)*pi * zr[s, 0] * (r[s, 0]**3)  # Construct the initial m ----------------------------------------
        p[s, 0] = p[s, 0] - (2/3)*pi * (zr[s, 0] * (C.c**2) + p[s, 0]) * (zr[s, 0] * (C.c**2) + x * 3 * p[s, 0]) * (C.G/(C.c**4) * (r[s, 0]**2))
        # Begin the cyclic integral ------------------------------------------------------------------------------------
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
                zr[s, i + 1] = zr[s, i]  # Density (corresponding to the given central density) ------------------------
                pcalculate[s] = p[s, i + 1]  # Surface pressure --------------------------------------------------------
                p[s, i + 1] = 0  # Define the surface pressure as 0 ----------------------------------------------------
                j = i + 1
                i = n + 1
        rsurface[s] = r[s, j]  # Radius of the NS ----------------------------------------------------------------------
        Msunt[s] = m[s, j - 1] / Msunt0  # Mass of the NS (Msun as unit) -----------------------------------------------
        indexmax[s] = j
    return [z1, rsurface, Msunt]  # Return the rho (density), r (radius), and M ----------------------------------------


#   --------------------------------------------------------------------------------------------------------------------
#   Calculate the chi^2: Hypothetical values and Data
#   --------------------------------------------------------------------------------------------------------------------
def chi2(Para, Data, EoS):
    xD = Data[0]  # Observational mass 2.01
    sigma = Data[1]
    TOV_output = TOV(Para, EoS)
    xH = max(TOV_output[2])
    Chi2 = sum((xH - xD)**2 / sigma**2)
    return Chi2


#   --------------------------------------------------------------------------------------------------------------------
#   Confidence interval
#   --------------------------------------------------------------------------------------------------------------------
def CI(df, chi2):
    re = 1 - stats.chi2.cdf(chi2, df=df)
    return re


#   --------------------------------------------------------------------------------------------------------------------
#   Confidence interval index
#   --------------------------------------------------------------------------------------------------------------------
def CIindex(CI, Accept):
    Len = len(Accept)
    N = int(floor(sum(Accept) / CI))
    ProbabilityDensity = Accept / N  # Sum ProbabilityDensity will obtain CI
    Probability = array([sum(ProbabilityDensity[i:]) for i in range(Len)])
    CI_Normal_1sigma = 1-2*(1-stats.norm.cdf(1, 0, 1))
    i = 0
    while Probability[i] > CI_Normal_1sigma:
        i += 1
    index = i - 1
    return [index, ProbabilityDensity, Probability]
