#!/usr/bin/env python

import sys
import random

from MurmurHash2 import MurmurHash2

class HashCoFi:
    def __init__(self, infile, outfile):
        self.infile = infile
        self.outfile = outfile

        self.t0 = 1000000
        self.lam = .05
        self.nfactors = 128
        
        self.compressed = HashMatrix(18, 128)
        self.loss = SquaredLoss()

    def computeMatrix(self):
        trainFile = open(self.infile, 'r')
        for line in trainFile:
            eta = 1./pow(self.t0, .5)
            self.t0 = self.t0 + 1

            splitLine = line.split(',')
            user = splitLine[0]
            item = splitLine[1]
            score = splitLine[2]
            
            F_ik = self.compressed.decompressMatrixFactorization(0, user, item,
                                                       range(0, self.nfactors))
            gamma = eta * self.loss.FirstDerivative(F_ik, float(score))
            mu = 1. - eta * self.lam

            self.compressed.updateWeightVectors(gamma, mu, user, item,
                                                self.nfactors)

        trainFile.close()

    def writeWeights(self):
        outFile = open(self.outfile, 'w')
        for i in range(0, self.compressed.weightLength):
            outLine = []
            outLine.append(str(i))
            outLine.append(str(self.compressed.userWeights[i]))
            outLine.append(str(self.compressed.itemWeights[i]))
            outFile.write(','.join(outLine) + '\n')

        outFile.close()

    def reset(self):
        self.compressed.resetWeights()

    def set_t0(self, new_t):
        self.t0 = new_t

    def set_lambda(self, new_lam):
        self.lam = new_lam

class HashMatrix:

    def __init__(self, nbits, nfactors):
        self.weightLength = 1 << nbits
        self.nbits = nbits
        self.nfactors = nfactors

        self.userWeights = self.getRandomWeights()
        self.itemWeights = self.getRandomWeights()
        
        self.murmur1 = MurmurHash2()
        self.murmur2 = MurmurHash2(104706279373)

    def hash1(self, a):
        return self.murmur1.hash32(a) % self.weightLength

    def hash2(self, a):
        return self.murmur2.hash32(a) % self.weightLength

    def rademacher1(self, item):
        hashedval = self.hash1(item)
        if hashedval % 2 == 0:
            return 1
        else:
            return 0

    def rademacher2(self, item):
        hashedval = self.hash2(item)
        if hashedval % 2 == 0:
            return 1
        else:
            return 0

    def decompressMatrixFactorization(self, startVal, user, item, factors):
        if len(factors) < 1:
            return startVal

        j = factors[0]
        userFactorString = user + ":" + str(j)
        userIndex = self.hash1(userFactorString)
        userSigma = self.rademacher1(userFactorString)
        userWeight = self.userWeights[userIndex]

        itemFactorString = item + ":" + str(j)
        itemIndex = self.hash2(itemFactorString)
        itemSigma = self.rademacher2(itemFactorString)
        itemWeight = self.itemWeights[itemIndex]

        factorSum = startVal + userSigma*itemSigma*userWeight*itemWeight

        return self.decompressMatrixFactorization(factorSum, user, item,
                                                  factors[1:])

    def updateWeightVectors(self, gamma, mu, user, item, numFactors):

        for j in range(0, numFactors):
            userFactorString = user + ":" + str(j)
            itemFactorString = item + ":" + str(j)
            
            userIndex = self.hash1(userFactorString)
            itemIndex = self.hash2(itemFactorString)

            userSigma = self.rademacher1(userFactorString)
            itemSigma = self.rademacher2(itemFactorString)

            oldUserWeight = self.userWeights[userIndex]
            oldItemWeight = self.itemWeights[itemIndex]

            self.userWeights[userIndex] = mu*oldUserWeight - gamma*itemSigma\
            *userSigma*oldItemWeight

            self.itemWeights[itemIndex] = mu*oldItemWeight - gamma*itemSigma\
            *userSigma*oldUserWeight
    
    def resetWeights(self):
        self.userWeights = self.getRandomWeights()
        self.itemWeights = self.getRandomWeights()

    def getRandomWeights(self):
        return [(random.random()-.5)/1000. for x in range(0, self.weightLength)]

class SquaredLoss:
    
    def __init__(self):
        pass

    def FirstDerivative(self, prediction, actual):
        return prediction - actual


if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]

    hcf = HashCoFi(infile, outfile)

