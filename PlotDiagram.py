# coding=utf-8
# author: Ma Seoyin, 2014 Applied Physics, SCUT
from numpy import *
import matplotlib.pyplot as plt
import os, Function
from matplotlib import rc
rc('text', usetex=True)


#   --------------------------------------------------------------------------------------------------------------------
#   Construct 1sigma 2sigma 3siama parameter list
#   --------------------------------------------------------------------------------------------------------------------
def Construct123sigma(Para_Uplims, Para_Lolims):
    Para_Uplims = [Para_Uplims[i] for i in range(2)]
    Para_Uplims1 = array([eval('{}{}{}'.format(Para[i], Para_Uplims[i], 1 * Deviation[i])) for i in range(2)])
    Para_Uplims2 = array([eval('{}{}{}'.format(Para[i], Para_Uplims[i], 2 * Deviation[i])) for i in range(2)])
    Para_Uplims3 = array([eval('{}{}{}'.format(Para[i], Para_Uplims[i], 3 * Deviation[i])) for i in range(2)])
    Para_Uplims = [Para_Uplims1, Para_Uplims2, Para_Uplims3]

    Para_Lolims = [Para_Lolims[i] for i in range(2)]
    Para_Lolims1 = array([eval('{}{}{}'.format(Para[i], Para_Lolims[i], 1 * Deviation[i])) for i in range(2)])
    Para_Lolims2 = array([eval('{}{}{}'.format(Para[i], Para_Lolims[i], 2 * Deviation[i])) for i in range(2)])
    Para_Lolims3 = array([eval('{}{}{}'.format(Para[i], Para_Lolims[i], 3 * Deviation[i])) for i in range(2)])
    Para_Lolims = [Para_Lolims1, Para_Lolims2, Para_Lolims3]

    return [Para_Uplims, Para_Lolims]


#   --------------------------------------------------------------------------------------------------------------------
#   Get the r and M sets, corresponding to best-fit Para, 1sigma Para, 2sigma Para, 3sigma Para
#   --------------------------------------------------------------------------------------------------------------------
def Get_r_M(Para, Para_Uplims, Para_Lolims):
    TOV_output = Function.TOV(Para, EoS)
    r_NS = TOV_output[1]
    M_NS = TOV_output[2]
    TOV_output = Function.TOV(Para_Uplims[0], EoS)
    r_NS_Uplims1 = TOV_output[1]
    M_NS_Uplims1 = TOV_output[2]
    TOV_output = Function.TOV(Para_Lolims[0], EoS)
    r_NS_Lolims1 = TOV_output[1]
    M_NS_Lolims1 = TOV_output[2]
    TOV_output = Function.TOV(Para_Uplims[1], EoS)
    r_NS_Uplims2 = TOV_output[1]
    M_NS_Uplims2 = TOV_output[2]
    TOV_output = Function.TOV(Para_Lolims[1], EoS)
    r_NS_Lolims2 = TOV_output[1]
    M_NS_Lolims2 = TOV_output[2]
    TOV_output = Function.TOV(Para_Lolims[2], EoS)
    r_NS_Lolims3 = TOV_output[1]
    M_NS_Lolims3 = TOV_output[2]
    r = [r_NS, r_NS_Uplims1, r_NS_Lolims1, r_NS_Uplims2, r_NS_Lolims2, r_NS_Lolims3]
    M = [M_NS, M_NS_Uplims1, M_NS_Lolims1, M_NS_Uplims2, M_NS_Lolims2, M_NS_Lolims3]
    return [r, M]


#   --------------------------------------------------------------------------------------------------------------------
#   Plot the r and M
#   --------------------------------------------------------------------------------------------------------------------
def rTheoExpPlot(r, M):
    Plot = plt.figure('Maximum Mass 2.01M_sun of Neutron Stars')
    Plot.set_size_inches(8.4, 5.2)
    ax = Plot.add_subplot(111)
    ax.plot(r[0], M[0], label='Best fit')
    ax.plot(r[1], M[1], label='1$\sigma$, Up')
    ax.plot(r[2], M[2], label='1$\sigma$, Lo')
    ax.plot(r[3], M[3], label='2$\sigma$, Up')
    ax.plot(r[4], M[4], label='2$\sigma$, Lo')
    ax.plot(r[5], M[5], label='3$\sigma$, Lo')
    ax.legend(loc='lower left', ncol=1)
    return Plot


#   --------------------------------------------------------------------------------------------------------------------
#   Plot the chi^2-Probability Density and chi^2-Probability
#   --------------------------------------------------------------------------------------------------------------------
def MonteCarloAnalysis(chi2Set, ProbabilityDensity, Probability):
    MCAnalysisPlot = plt.figure('Monte Carlo Analysis')
    MCAnalysisPlot.set_size_inches(10.4, 5.2)

    # chi^2 - Probability Density --------------------------------------------------------------------------------------
    ax1 = MCAnalysisPlot.add_subplot(121)
    ax1.set_xlim(-1, 20)
    ax1.set_xlabel('$\\chi^2$')
    ax1.set_ylim(0, 0.06)
    ax1.set_ylabel('Probability Density')
    ax1.plot(chi2Set, ProbabilityDensity, linewidth=2)

    # exp(-chi^2/2) - Probability --------------------------------------------------------------------------------------
    ax2 = MCAnalysisPlot.add_subplot(122)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('$\\exp\\left(-\\chi^2/2\\right)$')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Probability')
    ax2.plot(exp(-(1/2)*chi2Set), Probability)
    return MCAnalysisPlot


ResultDir = os.getcwd() + '/Result/'   # Add '_Answer' to see the answer, or remain 'Result' only to see my result

EoS = loadtxt('APR.txt')  # APR EoS
Para = loadtxt(ResultDir + 'Para.txt')
Deviation = loadtxt(ResultDir + 'Deviation.txt')
Para_Uplims = open(ResultDir + 'Para_Uplims.txt').read()
Para_Lolims = open(ResultDir + 'Para_Lolims.txt').read()
[Para_Uplims, Para_Lolims] = Construct123sigma(Para_Uplims, Para_Lolims)
[r, M] = Get_r_M(Para, Para_Uplims, Para_Lolims)
Diagram = rTheoExpPlot(r, M)
plt.savefig(ResultDir + 'Maximum Mass 2.01M_sun of Neutron Stars.png')
plt.savefig(ResultDir + 'Maximum Mass 2.01M_sun of Neutron Stars.eps')

chi2Set = loadtxt(ResultDir + 'chi2Set.txt')
ProbabilityDensity = loadtxt(ResultDir + 'ProbabilityDensity.txt')
Probability = loadtxt(ResultDir + 'Probability.txt')
MCAnalysisPlot = MonteCarloAnalysis(chi2Set, ProbabilityDensity, Probability)
MCAnalysisPlot.savefig(ResultDir + 'Monte_Carlo_Analysis.png')
MCAnalysisPlot.savefig(ResultDir + 'Monte_Carlo_Analysis.pdf')

plt.show((Diagram, MCAnalysisPlot))
