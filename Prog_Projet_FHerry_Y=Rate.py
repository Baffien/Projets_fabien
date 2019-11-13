import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Import dataset
print("Loading dataset...")
dataset = pd.read_excel("googleplaystore_original.xls")
print("...Done.")
print(dataset.shape)
print(dataset.describe(include="all"))
print()    


# Basic stats
data_desc = dataset.describe(include='all')


# Suppression des NaN dans la dataset
dataset = dataset.dropna(subset=['Rating'])


# Separate target variable Y from features X
print("Separating labels from features...")
features_list = dataset.iloc[:,[1,3,4,5,6,7,8,10,11,12]]
X = features_list
Y = dataset.loc[:,"Rating"]
print("...Done.")
print()


# Divide dataset Train set & Test set 
print("Dividing into train and test sets...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, 
                                                    random_state=0)
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
print(X_train[0:5,:]) 

# Missing values
print("Imputing missing values...")
print(X_train[0:5,:])
print()
imputer = SimpleImputer(strategy="mean")
X_train[:,[1,2,5]] = imputer.fit_transform(X_train[:,[1,2,5]])
imputer2 = SimpleImputer(strategy="most_frequent")
X_train[:,[0,3,4,6,7]] = imputer2.fit_transform(X_train[:,[0,3,4,6,7]])
print("...Fini!")
print(X_train[0:5,:]) 
print()  

# Encoding categorical features and standardizing numeric features
print("Encoding categorical features and standardizing numerical features...")
print()
print(X_train[0:5,:])

numeric_features = [1,2]
numeric_transformer = StandardScaler()

categorical_features = [0]
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
DecisionTree = DecisionTreeRegressor(max_depth=20)
DecisionTree.fit(X_train, Y_train)
print("...Done.")

# Predictions on training set
print("Predictions on training set...")
Y_train_pred = DecisionTree.predict(X_train)
print("...Done.")
print(Y_train_pred)
print()


### Test pipeline ###
print("--- Test pipeline ---")

# Missing values
print("Imputing missing values...")
print(X_test[0:1,:])
print()
X_test[:,[1,2,5]] = imputer.transform(X_test[:,[1,2,5]])
X_test[:,[0,3,4,6,7]] = imputer2.transform(X_test[:,[0,3,4,6,7]])
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
Y_test_pred = DecisionTree.predict(X_test)
print("...Done.")
print(Y_test_pred)
print()

plt.scatter(X_test[:,33], Y_test,  color='black')

# Print R^2 scores
print("R2 score on training set : ", r2_score(Y_train, Y_train_pred))
print("R2 score on test set : ", r2_score(Y_test, Y_test_pred))

dataset.to_excel("datasetgoogle-model.xlsx")


