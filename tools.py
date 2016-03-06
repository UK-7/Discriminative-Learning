import numpy as np

'''
Compute the confusion matrix for a general class of classification
Hope you appreciate the pun ;)
//ToDO
'''

def createConfusion():
      print "Confusion implementation pending"
      #Implementation pending

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
      for i in range(dataCols-1):
            zeroParams.append(0)
      for _class in classList:
            classMeans.append(zeroParams)
            classCount.append(0)
      classMeans = np.asarray(classMeans).astype(int)
      classCount = np.asarray(classCount).astype(int)

      for rowIndex in range(dataRows):
            _class = data[rowIndex, -1]
            record = data[rowIndex, 0:-1]
            classCount[_class] += 1
            i = 0;
            for i in range(dataCols-1):
                  classMeans[_class,i] += record[i]
      for _class in classList:
            classMeans[_class,:] = \
                        [x/classCount[_class] for x in classMeans[_class,:]]
      print classMeans
