#!/usr/bin/env python

import sys
import random

from FnvHash import fnv32

class HashCoFi:
    """ Class that performs "Collaborative Filtering on a Budget".
    Usage:
        1. Initialize an instance, hcf = HashCoFi()
        2. Set parameters if desired, e.g., hcf.t0 = 1000000000
        3. Run on input file, hcf.updateMatrixFromFile("datafile.txt")
        4. Write out weight vectors, hcf.writeWeightsFoFile("weight_output.txt")
        5. Iterate again, hcf.singlePass().  Each invocation
           passes over the dataset once
        6. Iterate 10 times over the data, hcf.nPasses(10).  Do this until algo
           converges.
        7. Check if the algorithm has converged, hcf.checkConverge(10)
           Iterates through the first 10 points in the input file and computes
           the mean absolute parameter change
        8. Get prediction, hcf.predict('user', 'item')"""
 
    def __init__(self):
        self.infile = None
        self.outfile = None

        # Learning rate.  At each instance, take a step in the direction of the
        # gradient of the loss, proportional to 1/(t0 + n), where n is the
        # number of observations seen so far.
        self.t0 = 1000000
        # Regularization parameter
        self.lam = .05
        # Number of latent factors
        self.nfactors = 128
        
        self.compressed = HashMatrix(18, 128)
        self.loss = SquaredLoss()
    
    def updateMatrixFromFile(self, infile):
        self.infile = infile
        self.computeMatrix()

    def writeWeightsToFile(self, outfile):
        self.outfile = outfile
        self.writeWeights()

    def singlePass(self):
        # Make a single pass over the data in self.infile
        self.computeMatrix()

    def nPasses(self, n):
        # Make n passes over the data in self.infile
        for i in range(0, n):
            self.computeMatrix()

    def predict(self, user, item):
        return self.compressed.decompressMatrixFactorization(
            0, str(user), str(item), range(0, self.nfactors)
        )

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

    def checkConvergence(self, n=10):
        trainFile = open(self.infile, 'r')
        absChange = 0.
        for i in range(0, n):
            eta = 1./pow(self.t0, .5)

            splitLine = trainFile.readline().split(',')
            user = splitLine[0]
            item = splitLine[1]
            score = splitLine[2]

            F_ik = self.compressed.decompressMatrixFactorization(
                0, user, item, range(0, self.nfactors)
            )
            gamma = eta * self.loss.FirstDerivative(F_ik, float(score))
            mu = 1. - eta * self.lam

            absChange = absChange + self.compressed.updateWeightVectors(
                gamma, mu, user, item, self.nfactors
            )

        return absChange/float(n)


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
        
        self.fnv = fnv32() 

    def hash1(self, a):
        return self.fnv.fnv1(a) % self.weightLength

    def hash2(self, a):
        return self.fnv.fnv1a(a) % self.weightLength

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
        
        updateAmt = 0.

        for j in range(0, numFactors):
            userFactorString = user + ":" + str(j)
            itemFactorString = item + ":" + str(j)
            
            userIndex = self.hash1(userFactorString)
            itemIndex = self.hash2(itemFactorString)

            userSigma = self.rademacher1(userFactorString)
            itemSigma = self.rademacher2(itemFactorString)

            oldUserWeight = self.userWeights[userIndex]
            oldItemWeight = self.itemWeights[itemIndex]

            self.userWeights[userIndex] = self.getNewCoefficient(
                gamma, mu, oldUserWeight, oldItemWeight, userSigma, itemSigma
            )

            self.itemWeights[itemIndex] = self.getNewCoefficient(
                gamma, mu, oldItemWeight, oldUserWeight, itemSigma, userSigma
            )

            updateAmt = updateAmt + abs(oldUserWeight - self.userWeights[userIndex])
            updateAmt = updateAmt + abs(oldItemWeight - self.itemWeights[itemIndex])

        return updateAmt

    def getNewCoefficient(self, gamma, mu, val1, val2, sig1, sig2):
        return mu*val1 - gamma*sig1*sig2*val2
   
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

class EpsInsensLoss:

    def __init__(self, eps = 1, multiplier = 1):
        self.eps = abs(eps)
        self.m = multiplier

    def FirstDerivative(self, prediction, actual):
        if abs(prediction - actual) > self.eps:
            if (prediction - actual) > 0:
                return 1. * self.m
            else:
                return -1. * self.m
        else:
            return 0.

class SmoothedEpsInsensLoss:

    def __init__(self, eps):
        self.eps = abs(eps)

    def FirstDerivative(self, prediction, actual):
        firstDenom = 1. + pow(math.e, eps - actual + prediction)
        secondDenom = 1. + pow(math.e, eps - prediction + actual)
        return (1./firstDenom) - (1./secondDenom)

class HubersRobustLoss:
    
    def __init__(self, sigma = 1, multiplier = 1):
        self.sigma = abs(sigma)
        self.m = multiplier

    def FirstDerivative(self, prediction, actual):
        if abs(prediction - actual) <= sigma:
            return (1./sigma) * (prediction - actual) * self.m
        else:
            if (prediction - actual) > 0:
                return 1 * self.m
            else:
                return -1 * self.m

if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]

    hcf = HashCoFi(infile, outfile)

