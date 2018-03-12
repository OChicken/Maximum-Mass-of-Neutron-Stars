# coding=utf-8
# author: Ma Seoyin, 2014 Applied Physics, SCUT
from numpy import *
import Function


#   ----------------------------------------------------------------------------------------------------------------
#   Rearrange the chi2Set, ParaSet, DistSet
#   ----------------------------------------------------------------------------------------------------------------
def Rearrange(ParaSet, chi2Set, Accept):
    AcceptLen = len(Accept)
    Dict_Para = {}
    Dict_Accept = {}
    for i in range(AcceptLen):
        Dict_Para[chi2Set[i]] = ParaSet[i]
        Dict_Accept[chi2Set[i]] = Accept[i]
    chi2Set = sort(chi2Set)
    chi2Set = array(chi2Set)
    ParaSet = [Dict_Para[chi2Set[i]] for i in range(AcceptLen)]
    ParaSet = array(ParaSet)
    Accept = [Dict_Accept[chi2Set[i]] for i in range(AcceptLen)]
    Accept = array(Accept)
    return [ParaSet, chi2Set, Accept]


#   ----------------------------------------------------------------------------------------------------------------
# Renew the Cov, i.e., the D & P
#   ----------------------------------------------------------------------------------------------------------------
def RenewCov(ParaSet, Accept):
    ParaLen = shape(ParaSet)[1]
    AcceptLen = len(Accept)
    AcceptP = Accept / sum(Accept)
    mu = array(matrix(AcceptP) * matrix(ParaSet)).flatten()
    Cov = matrix(zeros((ParaLen, ParaLen)))
    for i in range(ParaLen):
        for j in range(ParaLen):
            Covij = zeros(AcceptLen)
            for k in range(AcceptLen):
                Covij[k] = (ParaSet[k, i] - mu[i]) * (ParaSet[k, j] - mu[j]) * AcceptP[k]
            Cov[i, j] = sum(Covij)
    Cov = array(Cov)
    [D, P] = Function.Cov2D(Cov)
    return [D, P, mu, Cov]


#   ----------------------------------------------------------------------------------------------------------------
#   The main MCMC
#   ----------------------------------------------------------------------------------------------------------------
def MarkovChain(Data, EoS, Para, chi2, D, P, ParaSet, chi2Set, Accept, t):
    # In individual process, process S steps to renew the Cov ------------------------------------------------------
    S = 100
    # Metropolis process -------------------------------------------------------------------------------------------
    s = 1  # In the 1st step, Do the Accept and ParaSet initialization
    while s <= S:
        Para1 = Function.GeneratePara(t, Para, D, P)
        chi21 = Function.chi2(Para1, Data, EoS)
        if chi21 < chi2:
            alpha = 1  # Will accept with probability alpha = 1
        else:
            alpha = e ** (-(1 / 2) * (chi21 - chi2))  # Accept with probability alpha < 1
        beta = random.rand()
        if beta < alpha:
            # Accept -----------------------------------------------------------------------------------------------
            Para = Para1; chi2 = chi21
            Accept = hstack((1, Accept))
            chi2Set = hstack((chi2, chi2Set))
            ParaSet = vstack((Para, ParaSet))
        else:
            # Reject -----------------------------------------------------------------------------------------------
            Accept[0] += 1
        s += 1
    # Return the results -------------------------------------------------------------------------------------------
    [ParaSet, chi2Set, Accept] = Rearrange(ParaSet, chi2Set, Accept)  # The returned arguments had been rearranged
    Para = ParaSet[0]
    chi2 = chi2Set[0]
    [D, P, mu, Cov] = RenewCov(ParaSet, Accept)
    return [Para, chi2, D, P, ParaSet, chi2Set, Accept, mu, Cov]
