import numpy as np
import pandas as pd
import csv
import tools

'''
Read the data from a specified file and compile into a single table
Input: File Name
Return: np.matrix
'''

def readData(dataFile):
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


if __name__ == "__main__":
      data = readData("perfume_data.csv")
      print data
      tools.computeParameters(data, [0,1])
