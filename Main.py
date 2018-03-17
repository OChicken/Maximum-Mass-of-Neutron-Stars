# coding=utf-8
# author: Ma Seoyin, 2014 Applied Physics, SCUT
"""Remember to delete 'winsound' from the import and comment the last code containing of 'winsound.Beep()'
   if you want to run me on a Linux machine"""
from numpy import *
from itertools import product
import os, time, winsound, Function, MCMC


#   --------------------------------------------------------------------------------------------------------------------
#   Find the Para_Uplims and Para_Lolims that enabling the y^Theo to be draw in upper and lower limits
#   --------------------------------------------------------------------------------------------------------------------
def Para_Uplims_Lolims(Para, Deviation, EoS, M_NS_max):
    ParaLen = len(Para)
    Sign = ['+', '-']
    Para_uplims_lolims = array([Para] * (2**ParaLen))
    M_NS_max_lims = zeros(2**ParaLen)
    Y_uplims = ones(2**ParaLen)*1e9
    Y_lolims = ones(2**ParaLen)*(-1e9)
    i = 0
    for Iter in product(range(2), repeat=ParaLen):
        for j in range(ParaLen):
            Para_uplims_lolims[i, j] = eval('{}{}{}'.format(Para_uplims_lolims[i, j], Sign[Iter[j]], Deviation[j]))
        i += 1
    for i in range(2**ParaLen):
        Para_Temp = Para_uplims_lolims[i, :]
        TOV_output = Function.TOV(Para_Temp, EoS)
        M_NS_max_lims[i] = max(TOV_output[2])
        Yerror = M_NS_max_lims[i] - M_NS_max
        if Yerror >= 0:
            Y_uplims[i] = Yerror
        if Yerror <= 0:
            Y_lolims[i] = Yerror
    Uplims = argmin(Y_uplims)
    Lolims = argmax(Y_lolims)
    Para_Uplims = list(Para_uplims_lolims[Uplims, :] - Para)
    M_NS_max_Uplims = M_NS_max_lims[Uplims] - M_NS_max
    Para_Lolims = list(Para_uplims_lolims[Lolims, :] - Para)
    M_NS_max_Lolims = M_NS_max_lims[Lolims] - M_NS_max
    for i in range(ParaLen):
        if Para_Uplims[i] > 0:
            Para_Uplims[i] = '+'
            Para_Lolims[i] = '-'
        else:
            Para_Uplims[i] = '-'
            Para_Lolims[i] = '+'
    return [Para_Uplims, Para_Lolims, M_NS_max_Uplims, M_NS_max_Lolims]


#   --------------------------------------------------------------------------------------------------------------------
# Write Result
#   --------------------------------------------------------------------------------------------------------------------
def WriteResult(A, file):
    if type(A) == ndarray:
        A = A.tolist()
    dim = ndim(A)
    if dim == 0:
        file.write(str(A))
    elif dim == 1:
        for i in range(shape(A)[0]):
            if type(A[0]) == str:
                file.write(A[i])
            else:
                file.write(str(A[i]) + '\n')
    elif dim == 2:
        for i in range(shape(A)[0]):
            for j in range(shape(A)[1]):
                file.write(str(A[i][j]) + '\t')
                if j == range(shape(A)[1])[-1]:
                    file.write('\n')
    file.close()


DataDir = os.getcwd() + '/'
Data = loadtxt(DataDir + 'Data.txt')  # Initialize Data
EoS = loadtxt(DataDir + 'APR.txt')  # APR EoS
ParaAnswer = loadtxt(DataDir + '/ResultAnswer/' + 'Para.txt')

Start = time.clock()
T = 3  # The length of MCMC is T
t = 0
Para = Function.GeneratePara(t, Para0=None, D=None, P=None)  # Initialize Para
D = None; P = None  # Get D1 and P1
df = 1
# Begin to find the best Para and its relevant parameters D, chi2 Cov etc-----------------------------------------------
t += 1
ParaSet = array(Para)  # The set of the accepted Para
chi2 = Function.chi2(Para, Data, EoS)
chi2Set = array([chi2])  # The set of the chi2 corresponding to the accepted Para
Accept = array([1])  # The set of the Accept corresponding to the accepted Para
while t <= T:
    [Para, chi2, D, P, ParaSet, chi2Set, Accept, mu, Cov] = \
        MCMC.MarkovChain(Data, EoS, Para, chi2, D, P, ParaSet, chi2Set, Accept, t)
    t += 1
CI = Function.CI(df, chi2)
[index, ProbabilityDensity, Probability] = Function.CIindex(CI, Accept)
Deviation = abs(Para - ParaSet[index])
# The NS's maximum mass and its upper limits and lower limits ----------------------------------------------------------
TOV_output = Function.TOV(Para, EoS)
M_NS_max = max(TOV_output[2])
[Para_Uplims, Para_Lolims, M_NS_max_Uplims, M_NS_max_Lolims] = Para_Uplims_Lolims(Para, Deviation, EoS, M_NS_max)
End = time.clock()
TimeUsed = End - Start

# Output Result --------------------------------------------------------------------------------------------------------
ResultDir = os.getcwd() + '/Result/'

print(TimeUsed)
file = open(ResultDir + 'TimeUsed.txt', 'w+')
WriteResult(TimeUsed, file)
print(chi2)
file = open(ResultDir + 'chi2.txt', 'w+')
WriteResult(chi2, file)
print(CI)
file = open(ResultDir + 'CI.txt', 'w+')
WriteResult(CI, file)
print(Para)
file = open(ResultDir + 'Para.txt', 'w+')
WriteResult(Para, file)
print(Deviation)
file = open(ResultDir + 'Deviation.txt', 'w+')
WriteResult(Deviation, file)
print(M_NS_max)
file = open(ResultDir + 'M_NS_max.txt', 'w+')
WriteResult(M_NS_max, file)
print(M_NS_max_Uplims)
file = open(ResultDir + 'M_NS_max_Uplims.txt', 'w+')
WriteResult(M_NS_max_Uplims, file)
print(M_NS_max_Lolims)
file = open(ResultDir + 'M_NS_max_Lolims.txt', 'w+')
WriteResult(M_NS_max_Lolims, file)

file = open(ResultDir + 'D.txt', 'w+')
WriteResult(D, file)
file = open(ResultDir + 'P.txt', 'w+')
WriteResult(P, file)
file = open(ResultDir + 'ParaSet.txt', 'w+')
WriteResult(ParaSet, file)
file = open(ResultDir + 'chi2Set.txt', 'w+')
WriteResult(chi2Set, file)
file = open(ResultDir + 'ProbabilityDensity.txt', 'w+')
WriteResult(ProbabilityDensity, file)

file = open(ResultDir + 'index.txt', 'w+')
WriteResult(index, file)
file = open(ResultDir + 'mu.txt', 'w+')
WriteResult(mu, file)
file = open(ResultDir + 'Cov.txt', 'w+')
WriteResult(Cov, file)

print(Para_Uplims)
file = open(ResultDir + 'Para_Uplims.txt', 'w+')
WriteResult(Para_Uplims, file)
print(Para_Lolims)
file = open(ResultDir + 'Para_Lolims.txt', 'w+')
WriteResult(Para_Lolims, file)

# Beep when finished ---------------------------------------------------------------------------------------------------
winsound.Beep(2500, 5000)

