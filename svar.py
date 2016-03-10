from __future__ import division
import numpy as np
import pandas as pd
import csv
import tools
import math as m
import os
import glob
import matplotlib.pyplot as plt

from sklearn.cross_validation import KFold
from numpy.linalg import inv, det

#np.set_printoptions(threshold='nan')


'''
Read the data from a specified file and compile into a single table
This is used for CSV files with comma separator for thousands
Input: File Name
Return: np.ndarray
'''

def readData_1D(dataFile):
      tempData = pd.read_csv(dataFile, thousands=',', \
                  header=None)
      formattedData = []
      for row in tempData.iterrows():
            index, data = row
            y = data[0]
            i = iter(data[1:])
            for element in i:
                  formattedData.append([element, y])
      return np.asarray(formattedData).astype(int)


'''
Read the data from a specified file and compile into a single table
This method is used for tab delimited text files with numerical values
Input: File Name
Return: np.ndarray
'''

def readData_nD2k(dataFile):
      data = []
      with open(dataFile) as f:
            reader = csv.reader(f, delimiter = '\t')
            for row in reader:
                  if row[-1] == '1':
                        row[-1] = '0'
                  else:
                        row[-1] = '1'
                  data.append(row)

      return np.asarray(data).astype(int)


'''
Read the data from all files in a specified directory into a single table
This method is used for data spread in multiple CSV files
Input: prectory path
Return: np.ndarray
'''

def readData_nD(dataPath):
      data = []
      dataFiles = glob.glob(os.path.join(dataPath, '*.csv'))
      for dataFile in dataFiles:
            with open(dataFile, 'r') as f:
                  reader = csv.reader(f, delimiter = ',')
                  for row in reader:
                        if row[-1] != '0':
                              data.append(row[1:])
      return np.asarray(data).astype(int)


'''
Classifier function for general case of k-classes, n-features.
Creates a decriminant function using the final classified examples
Makes use of ten fold cross validation for error analysis
Input: Data matrix as an numpy array with the label in the last column
Returns: A descriminant function output
'''

def classify(data, classSet, threshold = 1.0):
      kf = KFold(len(data), n_folds=10, shuffle = True)
      
      precision = np.zeros(len(classSet))
      recall = np.zeros(len(classSet))
      f_measure = np.zeros(len(classSet))
      accuracy = 0

      for train, test in kf:
            sigmaSet, classMeans, classCount = tools.computeParameters(data[train,:], classSet)
            y_hat = []
            y = []
            for i in test:
                  X = data[i,:-1]
                  membership = []
                  _class = 0
                  for _class in classSet:
                        diff = X - classMeans[_class,:]
                        sigma = sigmaSet[_class,:,:]
                        e = np.dot(np.dot(diff.T, inv(sigma)), diff)
                        g_x = m.log(classCount[_class]/len(data)) - m.log(det(sigma)) - e
                        membership.append(g_x)
                  if threshold != 1.0:
                        if membership[0]/membership[1] >= threshold:
                              y_hat.append(1)
                        else:
                              y_hat.append(0)
                  else:
                        y_hat.append(\
                                    membership.index(max(membership)))
                  y.append(data[i,-1])
            p, r, f, a = tools.createConfusion(y_hat, y, classSet)
            precision = precision + p
            recall = recall + r
            f_measure = f_measure + f
            accuracy = accuracy + a
      precision /= 10
      recall /= 10
      f_measure /= 10
      accuracy /= 10
      
      _class = 0
      for _class in classSet:
            print "Class: %s\nPrecision: %s\nRecall: %s\nF-measure: %s\n" \
                        % (_class, precision[_class], recall[_class], f_measure[_class])
      print "Accuracy: %s\n=======================\n" % accuracy
      return precision, recall


if __name__ == "__main__":
      
      # 1D 2-class data set
      print "\n1D Data Set - 2 Classes\n-----------------------\n"
      data = readData_1D("perfume_data.csv")
      two_classSet = [0,1]
      classify(data, two_classSet)

      # nD 2-class data set
      print "\nnD Data Set - 2 Classes\n-----------------------\n"
      data = readData_nD2k("Skin_NonSkin.txt")
      precision = []
      recall = []
      for i in np.arange(0.1, 1, 0.1):
            print "*****-T = %s-*****" % i
            p, r = classify(data, two_classSet, threshold=i)
            precision.append(p)
            recall.append(r)
      precision = np.asarray(precision)
      recall = np.asarray(recall)
      plt.plot(recall[:,1], precision[:,1], color='blue', label='1')
      plt.xlabel('Recall')
      plt.ylabel('Precision')
      plt.title('ROC Plot')
      plt.legend()
      plt.show()
      

      # nD k-class data set
      print "\nnD Data Set - k Classes\n-----------------------\n"
      data = readData_nD("activity_data/")
      for i in range(len(data)):
            data[i,-1] -= 1
      n_classSet = np.arange(7)
      classify(data, n_classSet)
