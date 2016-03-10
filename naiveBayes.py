from __future__ import division
import numpy as np
import csv
import math as m
import tools

from scipy.misc import comb
from sklearn.cross_validation import KFold

'''
Read and structure the given data into a ndarray
Input: File Name
Return: np ndarray
'''

def readTextData(dataFile, labelFile, binomial = True):
      data = np.zeros((400, 2500))
      with open(dataFile) as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                  row = map(int, row)
                  if binomial:
                        data[row[0] - 1, row[1] - 1] = 1
                  else:
                        data[row[0] - 1, row[1] - 1] = row[2]
      label = np.loadtxt(labelFile, dtype = 'int')
      data = np.column_stack((data, label))
      return data

'''
Compute Bernoulli parameters.
Input: Data and Label matrix as np.ndarray
Return: np.array of P(j|y=1) for all j and Prior of class 1
'''

def computeBinomialParameters(data):
      likelihood = np.zeros((2,2500))
      likelihood += 0.005 #Epsilon for LaPlace Smoothing
      no_of_samples = [0.01, 0.01] #2*Epsilon for LaPlace smoothing
      for row in data:
            if row[-1] == 0:
                  likelihood[0] += row[:-1]
                  no_of_samples[0] += 1
            else:
                  likelihood[1] += row[:-1]
                  no_of_samples[1] += 1
      likelihood[0] /= no_of_samples[0]
      likelihood[1] /= no_of_samples[1]
      prior = np.asarray(no_of_samples)/len(data)
      return likelihood, prior
   

'''
Classification function based on a binomial distribution.
Uses Naive-Bayes assumption
Input: shuffled data matrix with labels in the last column
Returns: Precsion and Recall Values
'''

def classifyBinomial(data):
      kf = KFold(len(data), n_folds=10, shuffle=True)
      precision = 0
      recall = 0
      f_measure = 0
      accuracy = 0
      for train, test in kf:
            likelihood, prior = computeBinomialParameters(data[train])
            y_hat = []
            y = []
            membership = [0,0]
            for i in test:
                  for _class in [0,1]:
                        g_x = 0
                        for j in range(len(likelihood[_class,:])):
                              g_x += (data[i,j] * m.log(likelihood[_class,j])) + \
                                          ((1-data[i,j])*(m.log(1 - likelihood[_class,j])))
                        g_x += m.log(prior[_class])
                        membership[_class] = g_x
                  y_hat.append(membership.index(max(membership)))
                  y.append(data[i, -1])
                  p, r, f, a = tools.createConfusion(y_hat, y, [0,1])
            precision += p
            recall += r
            f_measure += f
            accuracy += a
      precision /= 10
      recall /= 10
      f_measure /= 10
      accuracy /= 10
      for _class in [0,1]:
            print "Class: %s\nPrecision: %s\nRecall: %s\nF-measure: %s\n" \
                        % (_class, precision[_class], recall[_class], f_measure[_class])
            print "\nAccuracy: %s\n======================" % accuracy

'''
Compute Bernoulli Parameters
Input: data and list of total words
Return: np array of P(j|y=1) and prior class probabilities
'''

def computeBernoulliParameters(data, p):
      likelihood = np.zeros((2,2500))
      likelihood += 0.005 #Epsilon for LaPlace Smoothing
      total_words = [0.01, 0.01] #2*Espilon for Laplace Smoothing
      no_of_samples = [0,0]
      for i in range(len(data)):
            if data[i,-1] == 0:
                  likelihood[0] += data[i,:-1]
                  total_words[0] += p[i]
                  no_of_samples[0] += 1
            else:
                  likelihood[1] += data[i,:-1]
                  total_words[1] += p[i]
                  no_of_samples[1] += 1
            likelihood[0] /= total_words[0]
            likelihood[1] /= total_words[0]
            prior = np.asarray(no_of_samples)/len(data)
      return likelihood, prior

'''
Classification function based on a Bernoulli Distribution.
Uses naive-Bayes assumption
Input: shuffled data matrix with labels in the last column
Returns: Precision and Recall values
'''

def classifyBernoulli(data):
      kf = KFold(len(data), n_folds=10, shuffle=True)
      p = np.zeros(len(data))
      for i in range(len(data)):
            p[i] = np.sum(data[i,:-1])
      precision = 0
      recall = 0
      f_measure = 0
      accuracy = 0
      
      print p
      for train, test in kf:
            likelihood, prior = computeBernoulliParameters(data, p)
            y_hat = []
            y = []
            membership = [0,0]
            for i in test:
                  p = np.sum(data[i,:-1])
                  for _class in [0,1]:
                        g_x = 0
                        for j in range(len(likelihood[_class,:])):
                              g_x += comb(p, data[i,j]) * \
                                    m.pow(likelihood[_class, j], data[i,j]) * \
                                    m.pow((1 - likelihood[_class, j]), (p - data[i,j]))
                        g_x += m.log(prior[_class])
                        membership[_class] = g_x
                  y_hat.append(membership.index(max(membership)))
                  y.append(data[i, -1])
                  p,r,f,a = tools.createConfusion(y_hat, y, [0,1])
            precision += p
            recall += r
            f_measure += f
            accuracy += a
      precision /= 10
      recall /= 10
      f_measure /= 10
      accuracy /= 10

      for _class in [0,1]:
            print "Class: %s\nPrecision: %s\nRecall: %s\nF-Measure: %s\n" % \
                        (_class, precision[_class], recall[_class], f_measure[_class])
            print "Accuracy: %s\n=========================\n" % accuracy



'''
The label data transitions from 0 to 1 at sample 200
We, thus, use the sample position 200 to evaluate the parameters for the two classes
The data set has equal priors for both classes
Main Function follows
'''

if __name__ == "__main__":
      data = readTextData("train-features-400.txt", "train-labels-400.txt")
      classifyBinomial(data)
      data = readTextData("train-features-400.txt", "train-labels-400.txt", binomial=False)
      classifyBernoulli(data)
#      computeBinomialParameters(data, label)

