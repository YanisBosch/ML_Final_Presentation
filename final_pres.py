import os
import pandas as pd
import csv
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
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

#-------CORRELATION PLOTTING-------

def plot_corr(df,threshold = 5):
    """Plots top and bottom threshold values in order of correlation"""
    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Focus on the correlation of features with the target variable 'ViolentCrimesPerPop'

    target_correlation = correlation_matrix['ViolentCrimesPerPop'].sort_values(ascending=False)

    # Convert the Series to a 2D numpy array
    if threshold < (target_correlation.values.size/2):
        indexes = [x+1 for x in range(threshold)] + [-x-1 for x in range(threshold)][::-1]
        
        target_correlation = target_correlation.take(indexes)
    target_correlation_2d = target_correlation.values[:, None]


    #Plot the correlations with the target variable
    plt.figure(figsize=(10, 15))
    sns.heatmap(target_correlation_2d, annot=False, cmap='coolwarm', cbar=True, linewidths=0.5,
            yticklabels=target_correlation.index)
    plt.title("Correlation with Violent Crimes per 100K Population")
    plt.yticks(rotation=0)
    plt.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)
    plt.show()
    #plt.savefig("correlation.png")

#plot_corr(df,5)

#--------------

#-------RANDOM FOREST REGRESSION-------

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#-------Models setup

models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

#-------Testing their performance

rmse_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse_scores[name] = mean_squared_error(y_test, y_pred, squared=False)
print(rmse_scores)

#-------Ridge regression

#Define the model with specific or default parameters

Model = RandomForestRegressor(n_estimators=100, random_state=42)

#Train the model

results = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse_scores = mean_squared_error(y_test, y_pred, squared=False)
print(rmse_scores)

#-------Cross-Validation to Assess Model Performance, and Hyperparameter Tuning with Grid Search-------

#Define the base model

random_forest = RandomForestRegressor(random_state=42)

#Perform simple cross-validation first
#Marie provide some explanations on number of estimators, random state , and also cv
#what happen if we increase or reduces the number of estimators, random state , and also cv

cv_scores = cross_val_score(random_forest, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print("CV RMSE scores:", cv_rmse)
print("Mean CV RMSE:", np.mean(cv_rmse))

#Setup parameter grid for Grid Search

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}

#Setup Grid Search with cross-validation

grid_search = GridSearchCV(random_forest, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

#Fetch the best model

best_rf = grid_search.best_estimator_
print("Best parameters from Grid Search:", grid_search.best_params_)

#Evaluate the best model on the test set

test_predictions = best_rf.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
print("Test RMSE with the best model:", test_rmse)
















