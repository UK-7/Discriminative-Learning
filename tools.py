from __future__ import division
import numpy as np
import math

'''
Compute the confusion matrix for a general class of classification
Hope you appreciate the pun ;)
//ToDO
'''

def createConfusion(y_hat, y, classSet):
      confusion = np.zeros((len(classSet), len(classSet)))
      for i in range(len(y_hat)):
            confusion[y_hat[i],y[i]] += 1
      precision = np.zeros(len(classSet))
      recall = np.zeros(len(classSet))
      f_measure = np.zeros(len(classSet))
      for _class in classSet:
            precision[_class] = \
                        confusion[_class, _class] / \
                        np.sum(confusion, axis=1, dtype='float')[_class]
            if math.isnan(precision[_class]):
                  precision[_class] = 0
            recall[_class] = \
                        confusion[_class, _class] / \
                        np.sum(confusion, axis=0, dtype='float')[_class]
            if math.isnan(recall[_class]):
                  recall[_class] = 0
            f_measure[_class] = 2*(precision[_class] * recall[_class])/\
                        (precision[_class] + recall[_class])
            if math.isnan(f_measure[_class]):
                  f_measure[_class] = 0
      accuracy = np.trace(confusion)/np.sum(confusion)
      return precision, recall, f_measure, accuracy
            
'''
Compute parameter values for each feature for all classes
Class labels must be numeric starting from 0 and must be the last label in data
Input: Data, class list
Return: np.array as ["Class", "Mean", "Sigma"]
'''

def computeParameters(data, classList):
      classMeans=[]
      classCount=[]
      classList.sort()
      dataRows, dataCols = data.shape
      zeroParams = []
      i = 0;
      # Initializa all nd arrays to zeros
      for i in range(dataCols-1):
            zeroParams.append(0)
      for _class in classList:
            classMeans.append(zeroParams)
            classCount.append(0)
      classMeans = np.asarray(classMeans).astype(int)
      classCount = np.asarray(classCount).astype(int)
      
      # Calculate mean
      for rowIndex in range(dataRows):
            _class = data[rowIndex, -1]
            record = data[rowIndex, 0:-1]
            classCount[_class] += 1
            i = 0;
            for i in range(dataCols-1):
                  classMeans[_class,i] += record[i]
      _class = 0
      for _class in classList:
            classMeans[_class,:] = \
                        [x/classCount[_class] for x in classMeans[_class,:]]
      
      # Allocate a 3d nd array for sigma matrices
      sigma = np.zeros((dataCols-1, dataCols-1))
      sigmaSet = []
      _class = 0
      for _class in classList:
            sigmaSet.append(sigma)
      sigmaSet = np.asarray(sigmaSet)

      # Re-itrate the data set to evaluate sigma matrices for each class
      for rowIndex in range(dataRows):
            _class = data[rowIndex, -1]
            record = data[rowIndex, 0:-1]
            var = record - classMeans[_class,:]
            var = np.outer(var, var)
            sigmaSet[_class,:,:] += var
      _class = 0
      for _class in classList:
            sigmaSet[_class,:,:] /= classCount[_class]
      return sigmaSet, classMeans, classCount

            
