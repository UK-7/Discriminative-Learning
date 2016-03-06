import numpy as np
import csv

"""
Read the data from a specified file and compile into a single table
Input: File Name
Return: np.matrix
"""

def readData(dataFile):
      tempData = []
      data = []
      with open(dataFile) as f:
            reader = csv.reader(f)
            for row in reader:
                  y = row[0]
                  i = iter(row[1:])
                  for element in i:
                        data.append([y, element])
      return np.asarray(data)

if __name__ == "__main__":
      data = readData("perfume_data.csv")
      i = np.nditer(data)
      for element in i:
            print type(element)
