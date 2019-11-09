####################################################################
# Natalie Leon Guerrero
# CS 445 Machine Learning
# Programming Assignment 2 - Naive Bayes Classification Algorithm
#
#   To run this code, use this command in the terminal
#     python bayes.py
#   This algorithm should not take long to complete.
#   Results will print to the console, as well as be put into a csv 
#   file bayes.csv
####################################################################

import numpy as np
import random
import math
import csv

class NaiveBayes:
    def __init__(self, dataTrain):
        self.data = dataTrain               # Save copy of training data
        self.nSets = len(self.data)         # Number of examples/sets in data
        self.nFeat = len(self.data[0]) - 1  # Number of features per example
        self.nOnes = self.cntClass(1)       # Number of examples with positive outputs
        self.nZeros = self.cntClass(0)      # Number of examples with negative outputs
        self.prior1 = float(self.nOnes)/self.nSets  # Probability of positive outputs
        self.prior0 = float(self.nZeros)/self.nSets # Probability of negative outputs
        self.calcStats()                    # Find means and standard deviations

    def cntClass(self, output):
        ''' Returns the number of examples with the specified output '''
        cnt = 0
        for i in range(self.nSets):
            if self.data[i,self.nFeat] == output:
                cnt += 1
        return cnt

    def calcStats(self):
        ''' Calculates the mean and standard deviation of each feature for each output (1 and 0) '''
        # Initialize mean and standard deviation arrays
        self.mu1Arr = np.full((self.nFeat), -1.0)
        self.mu0Arr = np.full((self.nFeat), -1.0)
        self.sd1Arr = np.full((self.nFeat), -1.0)
        self.sd0Arr = np.full((self.nFeat), -1.0)
        # For each feature, calculate the mean and standard deviation. Each output will have
        # a separate array
        for i in range(self.nFeat):
            self.mu1Arr[i] = self.calcMean(self.data[:self.nOnes,i])
            self.sd1Arr[i] = self.calcSd(self.data[:self.nOnes,i], self.mu1Arr[i])
            self.mu0Arr[i] = self.calcMean(self.data[self.nOnes:,i])
            self.sd0Arr[i] = self.calcSd(self.data[self.nOnes:,i], self.mu0Arr[i])
        return
    
    def calcMean(self, xs):
        ''' Returns the mean of the specified array '''
        total = len(xs)
        val = 0
        for i in range(total):
            val += xs[i]
        return float(val/total)

    def calcSd(self, xs, mu):
        ''' Returns the standard deviation of the specified array '''
        n = 0
        for i in range(len(xs)):
            n += (xs[i]-mu)**2
        sd = (n/len(xs))**0.5
        # Ensures sd is not 0 as it will cause problems in the model and will likely
        # predict a wrong output
        if sd == 0:
            return 0.0001
        else:
            return float(sd)

    def test(self, testData):
        ''' Tests the model on the remaining test data '''
        # Check to ensure the test data has the same number of features with an 
        # expected output
        if (len(testData[0]) != (self.nFeat + 1)):
            print "Cannot test this data set"
            return 

        #Initialize csv file
        with open('bayes.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(["Percentage Spam"] + [self.prior1])
            writer.writerow(["Percentage Not Spam"] + [self.prior0])

        TP,TN,FP,FN = 0,0,0,0
        for i in range(len(testData)):
            prediction = self.predictClass(testData[i,:(self.nFeat)])
            target = testData[i,self.nFeat]
            # True Positive
            if prediction == target and prediction == 1:
                TP += 1
            # True Negative
            elif prediction == target and prediction == 0:
                TN += 1
            # False Positive
            elif prediction != target and prediction == 1:
                FP += 1
            # False Negative
            else:
                FN += 1
        # Confusion Matrix
        confuseM = np.array([[TP,FN],[FP,TN]])
        # Calcualtes accuracy, precision and recall
        accuracy = float(TP+TN)/(TP+TN+FP+FN)
        precision = float(TP)/(TP+FP)
        recall = float(TP)/(TP+FN)
        # Prints results to the console
        print "accuracy: %", accuracy*100
        print "precision: %", precision*100
        print "recall: %", recall*100
        print "Confusion Matrix"
        print confuseM
        
        with open('bayes.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(["Accuracy"] + [accuracy*100])
            writer.writerow(["Precision"] + [precision*100])
            writer.writerow(["Recall"] + [recall*100])
            writer.writerow(["Confusion Matrix"])
            for i in range(2):
                writer.writerow(confuseM[i,:])
            writer.writerow([""])
            writer.writerow([""])
        
    def predictClass(self, xs):
        ''' Returns prediction based on the model created from training data.
            The natural log is used to avoid underflow in exponential numbers. '''
        # Probability of the following outputs from the training data
        prob1 = math.log(self.prior1)
        prob0 = math.log(self.prior0)
        # For each feature of the specific test example, find the log of 
        # the normal distribution and add it to the probability of the
        # specific output
        for i in range(self.nFeat):
            prob1 += self.lognormpdf(xs[i],self.mu1Arr[i],self.sd1Arr[i])
            prob0 += self.lognormpdf(xs[i],self.mu0Arr[i],self.sd0Arr[i])
        # Return the highest probablility as the prediction
        if prob1 >= prob0:
            return 1
        else:
            return 0

    def lognormpdf(self, x, mu, sd):
        ''' Returns the log of the normal distribution '''
        num = -((x-mu)**2)/(2*(sd**2))
        denom = math.log(((2*math.pi)**0.5)*sd)
        return num-denom

# ==========================================================================
# Main

def readFile(filename):
    ''' Reads data from file and saves data into matrix '''
    data = list(csv.reader(open(filename), delimiter=','))
    data = np.array(data).astype("float")
    return data

def splitData(data):
    ''' Splits data into test and training data. Roughly 40% Spam and 60% Not Spam. '''
    nSetsTotal = len(data)          # Number of total examples/sets in the data
    nSetsHalf = nSetsTotal/2        # Number of half the data examples/sets
    nSetsSpam = int(nSetsHalf*0.38) # Number of spam (roughly 40%)
    # Initilize matrices for testing and training data
    testData = np.full((nSetsHalf,len(data[0])), -1)
    trainData = np.full((nSetsTotal-nSetsHalf,len(data[0])), -1)
    # Index array to keep track of examples copied over to another matrix
    indexArr = np.zeros(nSetsTotal)
    # The next 3 lines will count the total number of spam examples in the data
    nSpamTotal = 0
    while data[nSpamTotal,57] == 1:
        nSpamTotal +=1
    # The next 6 lines of code will randomly choose indices of spam examples from data to 
    # add to the test data matrix. Roughly 40% of the matrix will be spam examples.
    i = 0
    while i <= nSetsSpam:
        # Randomly chooses an index of spam examples
        index = random.randint(0,nSpamTotal)
        if indexArr[index] == 0:
            # If the index of the index array is 0, the example has not been copied yet
            # and will now be copied to the test matrix
            testData[i,:] = data[index,:]
            # Change the index of the index array to 1 now that the example has been copied
            indexArr[index] = 1
            i += 1
    # The next 6 lines of code will randomly choose indices of examples that are not spam 
    # from data to add to the test data matrix. Roughly 60% of the matrix will be not be 
    # spam examples.
    while i < len(testData):
        # Randomly chooses an index of examples that are not spam
        index = random.randint(nSpamTotal,(nSetsTotal-1))
        if indexArr[index] == 0:
            # If the index of the index array is 0, the example has not been copied yet
            # and will now be copied to the test matrix
            testData[i,:] = data[index,:]
            # Change the index of the index array to 1 now that the example has been copied
            indexArr[index] = 1
            i += 1
    # Next 5 lines of code will copy the remaining examples/sets to the test matrix
    # If the example is not already copied (the index arr at that index is 0) then
    # copy the examples to the test matrix
    # i will iterate through the entire data matrix, while index will iterate through the
    # training data matrix
    index = 0
    for i in range(nSetsTotal):
        if indexArr[i] == 0:
            trainData[index,:] = data[i,:]
            index += 1
    # Return the training and testing matrices
    return trainData, testData


if __name__ == '__main__':
    # Open the csv file with the entire data sets
    data = readFile("spambase.csv")

    with open('bayes.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["Naive Bayes Learning Model"])
        writer.writerow([""])

    for i in range(10):
        dataTrain, dataTest = splitData(data)
        nbayes = NaiveBayes(dataTrain)
        nbayes.test(dataTest)

    
