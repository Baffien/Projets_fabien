# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:30:47 2019

@author: fabie
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Import dataset
print("Loading dataset...")
dataset = pd.read_excel("googleplaystore.xlsx")
print("...Done.")
print()


#median_installs=dataset[3].median()
print("Create is_popular column with 0s and 1s...")
print(dataset.head())
median_installs=np.mean([100000,500000])

def encode_Y(x):
    if x < median_installs : 
        return 0
    else :
        return 1

dataset.loc[:,'is_popular'] = dataset['Installs'].apply(encode_Y)
print("...Done.")
print(dataset.head())
print(dataset.loc[:5,['Installs','is_popular']])

"""
for i in range(0,10840):
    if dataset.loc[i,"Installs"] < median_installs :
        dataset.loc[i,"Check_Installs"]=0
    else:
        dataset.loc[i,"Check_Installs"]=1
"""

# Basic stats
data_desc = dataset.describe(include='all')

# Separate target variable Y from features X
print("Separating labels from features...")
features_list = ["Category", "Rating", "Type"]
X = dataset.loc[:,features_list]
Y = dataset.loc[:,"is_popular"]
print("...Done.")
print()

# Divide dataset Train set & Test set 
print("Dividing into train and test sets...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, 
                                                    random_state=0
                                                    , stratify=Y)
print("...Done.")
print()

# Convert pandas DataFrames to numpy arrays before using scikit-learn
print("Convert pandas DataFrames to numpy arrays...")
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()
print()
print(X_train[0:5,:])
print()
print(X_test[0:5,:])
print()
print(Y_train[0:5])
print()
print(Y_test[0:5])
print()
print("...Done")


### Training pipeline ###
print("--- Training pipeline ---")
print()

# Missing values
print("Imputing missing values...")
print(X_train[0:5,:])
print()
imputer = SimpleImputer(strategy="mean")
X_train[:,[1]] = imputer.fit_transform(X_train[:,[1]])
imputer2 = SimpleImputer(strategy="most_frequent")
X_train[:,[0,2]] = imputer2.fit_transform(X_train[:,[0,2]])
print("...Fini!")
print(X_train[0:5,:]) 
print()  

# Encoding categorical features and standardizing numeric features
print("Encoding categorical features and standardizing numerical features...")
print()
print(X_train[0:5,:])

numeric_features = [1]
numeric_transformer = StandardScaler()

categorical_features = [0,2]
categorical_transformer = OneHotEncoder(categories='auto',sparse=False)

featureencoder = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),    
        ('num', numeric_transformer, numeric_features)
        ]
    )

X_train = featureencoder.fit_transform(X_train)
print("...Done")
print(X_train[0:5,:])

# Train model
print("Train model...")
regressor = LogisticRegression(random_state = 0, class_weight="balanced")
regressor.fit(X_train, Y_train)
print("...Done.")

# Predictions on training set
print("Predictions on training set...")
Y_train_pred = regressor.predict(X_train)
print("...Done.")
print(Y_train_pred)
print()


### Test pipeline ###
print("--- Test pipeline ---")

# Missing values
print("Imputing missing values...")
print(X_test[0:1,:])
print()
X_test[:,[1]] = imputer.transform(X_test[:,[1]])
X_test[:,[0,2]] = imputer2.transform(X_test[:,[0,2]])
print("...Fini!")
print(X_test[0:1,:]) 
print()  

# Encoding categorical features and standardizing numeric features
print("Encoding categorical features and standardizing numerical features...")
print()
print(X_test[0:1,:])

X_test = featureencoder.transform(X_test)
print("...Done")
print(X_test[0:1,:])

# Predictions on test set
print("Predictions on test set...")
Y_test_pred = regressor.predict(X_test)
print("...Done.")
print(Y_test_pred)
print()


# Print F1 scores
print("F1 score on training set : ", f1_score(Y_train, Y_train_pred))
print("F1 score on test set : ", f1_score(Y_test, Y_test_pred))


print("Confusion matrix on train set : ")
print(confusion_matrix(Y_train, Y_train_pred))
print("Confusion matrix on test set : ")
print(confusion_matrix(Y_test, Y_test_pred))
print()



