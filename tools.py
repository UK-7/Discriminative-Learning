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
      for _class in classList:
            classMeans[_class,:] = \
                        [x/classCount[_class] for x in classMeans[_class,:]]
      
      # Allocate a 3d nd array for sigma matrices
      sigma = np.zeros((dataCols-1, dataCols-1))
      sigmaSet = []
      for _class in classList:
            sigmaSet.append(sigma)

      # Re-itrate the data set to evaluate sigma matrices for each class
      for rowIndex in range(dataRows):
            _class = data[rowIndex, -1]
            record = data[rowIndex, 0:-1]
            var = record - classMeans[_class,:]
            var = np.outer(var, var)
            sigmaSet[_class,:,:] += var
      print "Class 0\n%s" % sigmaSet[0,:,:]
      print "Class 1\n%s" % sigmaSet[1,:,:]

            
