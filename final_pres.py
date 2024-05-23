import os
import pandas as pd
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
import warnings
warnings.filterwarnings('ignore')

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

#----------------------------

#-------EXPLORING THE DATASET-------

#First idea: See which columns contain the most missing values, if one column is almost empty we may exclude it

df = import_data()[0]

num_rows = len(df.index)
num_NaN = df.isna().sum().sort_values(ascending = False)
print("-------MISSING VALUES-------")
print(num_NaN[num_NaN > 0])
print("----------------------------")

#Many columns are almost empty -> idea: remove them
#OtherPerCap has only one missing value, we want to remove the row containing that one missing value to lose less
#information

def clean_data(arr, threshold = 1):
    df = arr[0]
    header = arr[1]

    pred = header[5:-1]     #predictors are all columns except violent crime, and exluding the first five columns
                            #which are non predictive
    resp = header[-1]

    header = header[5:]

    for col in pred:                        
        df[col].astype(float)               #convert values in df to float
        df[header[-1]].astype(float)

    num_NaN = df.isna().sum().sort_values(ascending = False)
    to_remove_col = num_NaN[num_NaN > threshold].index.tolist()
    to_remove_row = num_NaN[num_NaN == threshold].index.tolist()
    num_NaN = df.isna().sum().sort_values(ascending = False)

    for col in to_remove_col:
        try:
            i = pred.index(col)             #exclude columns with many missing values
            del pred[i]
            del header[i]
        except:
            pass
    
    temp = pred + [resp]
    df = df[temp]
    del temp

    # Convert all predictor columns to float, ensuring the changes are saved
    df[pred] = df[pred].apply(pd.to_numeric, errors='coerce')

    # Convert the response column to float
    df[resp] = pd.to_numeric(df[resp], errors='coerce')

    X = df[pred]                            #create dataframe containing the predictor columns
    Y = df[resp]                            #create dataframe containing the responding column
    
    for row in to_remove_row:               #remove row from X and Y containg NaN in the columns with only one missing value
        mask = X[row].notna()
        X = X[mask]
        Y = Y[mask]
    return([df,X,Y,header])

#-------

temp = clean_data(import_data())
df = temp[0]
X = temp[1]
Y = temp[2]
header = temp[3]
del temp

#-------MARIE-------

# Convert the Series to a 2D numpy array
target_correlation_2d = target_correlation.values[:, None]


#Plot the correlations with the target variable
plt.figure(figsize=(10, 15))
sns.heatmap(target_correlation_2d, annot=False, cmap='coolwarm', cbar=True, linewidths=0.5,
            yticklabels=target_correlation.index)
plt.title("Correlation with Violent Crimes per 100K Population")
plt.show()












