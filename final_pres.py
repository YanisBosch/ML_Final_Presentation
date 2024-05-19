import os
import pandas as pd
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model

#-------IMPORTING DATA-------

def import_data():
    cwd = os.getcwd()   #go into current working directory
    header = []         #store names of columns
    with open(cwd + "\Data\README.txt") as file:    #we suppose the Data is in a folder called Data in the cwd
        reader = csv.reader(file, delimiter=' ')    #get csv reader object
        for row in reader:                          #iterate through rows
            if len(row) != 0:                       #to avoid error when a row is empty
                if row[0] == "@attribute":          #extract attribute names
                    header.append(row[1])
    data =  pd.read_csv(cwd + "\Data\communities.data", sep=",", names = header)    #read .csv, with header as
                                                                                    #computed before
    data.replace("?",float('NaN'),inplace = True)   #replace ? with NaN to be able to convert to float                                           
    return([data,header])

temp = import_data()    #used to split grouped return from the function
df = temp[0]
header = temp[1]
del(temp)

#----------------------------

#--------------SPLITTING DATA INTO PREDICTORS AND RESPONDERS--------------

pred = header[5:-1]     #predictors are all columns except violent crime, and exluding the first five columns
                        #which are non predictive
resp = header[-1]

for col in pred:                        
    df[col].astype(float)               #convert values in df to float
df[header[-1]].astype(float)

#----------------------------

#-------EXPLORING THE DATASET-------

#First idea: See which columns contain the most missing values, if one column is almost empty we may exclude it

num_NaN = df.isna().sum().sort_values(ascending = False)
#print(num_NaN[num_NaN >= 1])

#Many columns are almost empty -> idea: remove them
#OtherPerCap has only one missing value, we want to remove the row containing that one missing value to lose less
#information

to_remove_col = num_NaN[num_NaN >= 2].index.tolist()
to_remove_row = num_NaN[num_NaN == 1].index.tolist()

for col in to_remove_col:
    try:
        i = pred.index(col)             #exclude columns with many missing values
        del pred[i]
    except:
        pass

X = df[pred]                            #create dataframe containing the predictor columns
Y = df[resp]                            #create dataframe containing the responding column
    
for row in to_remove_row:               #remove row from X and Y containg NaN in the columns with only one missing value
    mask = X[row].notna()
    X = X[mask]
    Y = Y[mask]

#--------------









