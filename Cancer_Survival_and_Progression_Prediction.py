# -*- coding: utf-8 -*-
"""Cancer Survival Prediction"""

## Data Management

# import required dependencies 
import pandas as pd
import os

# load train and test data 
train = pd.read_csv('Tumor Information_train.csv')
test = pd.read_csv('Tumor Information_test.csv')

# Set all the column names in test and train data to lower case 
train.columns = map(str.lower, train.columns)
test.columns = map(str.lower, test.columns)

# Put all the files in the set directory into a list data structure with each file name as an element in the list
file_list = os.listdir()

# Remove the train and test files from the list as they're already defined
file_list.remove('Tumor Information_train.csv')
file_list.remove('Tumor Information_test.csv')

# Remove the questionnaire file from the list because it provides little information
file_list.remove('DBBR Questionnaire Data Dictionary.csv')

# View file_list to see the 17 files we have to work with
print(file_list)

# Left merge all the files in the list to the train and test sets on the qbarcode column
for data in file_list:
  train = train.merge(pd.read_csv(data).rename(columns=str.lower), on='qbarcode', how='left')
  test = test.merge(pd.read_csv(data).rename(columns=str.lower), on='qbarcode', how='left')

# Concatenate the train and test sets to create the master dataset
all_data = pd.concat([train, test])

test.head()

# View the first five entries of the master dataset
all_data.head()

print(all_data['age'])

##Data Pre-processing

# Import preprocessing utilities from scikit-learn
from sklearn import preprocessing

# Create a list of all columns that aren't patientstatus or qbarcode
cols = [col for col in all_data.columns if col not in ['patientstatus', 'qbarcode']]

types = all_data.dtypes
cat_columns = [t[0] for t in types.iteritems() if ((t[1] not in ['int64', 'float64']))]

# Encode the non-numerical data
lbl = preprocessing.LabelEncoder()
for col in cat_columns:
    all_data[col] = lbl.fit_transform(all_data[col].astype(str))

# Impute missing data with the mode
all_data = all_data.fillna(all_data.mode().iloc[0])

# Seperate the processed data into test and train 
train_processed_data = all_data.iloc[:len(train)]
test_processed_data = all_data.iloc[len(train):]

# Print the dimensions of the train and test data 
print('Below are the shapes of our of our train and test sets:')
print('Train:', train_processed_data.shape)
print('Test:', test_processed_data.shape)

# Print the frequency of each class: survival versus non survival
print('The frequencies of each class:')
print(train['patientstatus'].value_counts())

"""## Study Design"""

#Create histograms of patients' identifying information
#Such as age, gender, race, cancer types

n_bins=100

var_1 = all_data['age'].values
var_2 = lbl.fit_transform(all_data['primarysite'])


figsize(12,4)
ylim([-5,80])

# Add title and axis names 
plt.title('Histogram of ages') 
plt.xlabel('Age') 
plt.ylabel('Frequency')

pyplot.hist(var_1, bins=n_bins)

ylim([-5,400])

# Add title and axis names 
plt.title('Histogram of primary sites') 
plt.xlabel('Primary site code') 
plt.ylabel('Frequency')

pyplot.hist(var_2, bins=n_bins, color='red')

all_data['primarysite_encoded'] = var_2
all_data['primarysite_encoded'].value_counts()

#compute summary statistics of data
all_data.describe()

# View the first five entries of the train_processed_data
train_processed_data.head()

# View the first five entries of the test_processed_data
test_processed_data.head()

## Validation Techniques

# import required model and reporting dependencies 
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import classification_report

# Create a list of all columns that aren't patientstatus or qbarcode
cols = [col for col in train_processed_data.columns if col not in ['patientstatus', 'qbarcode']]

# Turn data into numpy arrays 
train = train_processed_data[cols].values
labels = train_processed_data['patientstatus'].values

# Split the train data into traning and validation data (80/20 split)
x_train, x_val, y_train, y_val = train_test_split(train, labels, test_size=0.2, random_state=42)

## Model Training, Tuning, and Validation

# Initailize the model and tune the parameters
bst = xgb.XGBClassifier(max_depth=2, n_estimators=100, learning_rate=0.1, objective='reg:logistic', gamma=0, colsample_bytree=0.3)

# fit the model 
bst.fit(x_train, y_train)
print("Model Training Accuracy:", bst.score(x_train, y_train)*100, "%")

## Model Performance and Interpretability

# Generate predictions 
y_hat = bst.predict(x_val)

# Print performance metrics 
print('Accuracy Score:', bst.score(x_val, y_val))
print('Area under ROC:', roc_auc_score(y_val, y_hat))

# Print the classification report to check the type 1 (false positive) and type 2 (false negative) error rates
print(classification_report(y_val, y_hat))

from xgboost import plot_tree
plot_tree(bst)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(150, 100)
#fig.savefig('tree.png')

#Plot the ROC curve.
# Get ROC curve metrics 
false_positive_rate, true_positive_rate, threshold = roc_curve(y_val, y_hat)

# Plot ROC curve
plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - Gradient Boosting')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Turn the unseen test data into a numpy array
test= test_processed_data[cols].values

# Generate predictions
predictions = bst.predict(test)

# Match patients with their prediction
out_df = pd.DataFrame(
        {"qbarcode": test_processed_data.qbarcode.values,
         "PatientStatus": predictions
        }
    )

# View the predictions of the first 20 patients
print(out_df.head(20))

#out_df.to_csv('submission_aim.csv', index=False)

## Tumor Stage Progression Prediction


prog_cols = [col for col in all_data.columns if col not in ['grade', 'qbarcode']]

types = all_data.dtypes
cat_prog_columns = [t[0] for t in types.iteritems() if ((t[1] not in ['int64', 'float64']))]

print('Label encoding categorical columns:', cat_prog_columns)
for col in cat_prog_columns:
    all_data[col] = lbl.fit_transform(all_data[col].astype(str))

all_data = all_data.fillna(all_data.mode().iloc[0])

train_prog_processed_data = all_data.iloc[:len(train)]
test_prog_processed_data = all_data.iloc[len(train):]

print(all_data['primarysite'].value_counts())
prostate_data = all_data.loc[all_data['primarysite'] == 85]
prostate_data = prostate_data.loc[prostate_data['grade'] != 0]
prostate_data['grade'] = prostate_data['grade'] - 1
print(prostate_data['grade'].value_counts())
print(prostate_data.head())

prostate_cols = [col for col in prostate_data.columns if col not in ['grade', 'qbarcode']]

# Turn data into numpy arrays 
train_prog = prostate_data[prostate_cols].values
labels_prog = prostate_data['grade'].values

# Split the train data into traning and validation data (80/20 split)
x_prog_train, x_prog_val, y_prog_train, y_prog_val = train_test_split(train_prog, labels_prog, test_size=0.2, random_state=42)

bst_prog = xgb.XGBClassifier(max_depth=2, n_estimators=100, learning_rate=0.1, objective='reg:logistic', gamma=0, colsample_bytree=0.3)

# fit the model 
bst_prog.fit(x_prog_train, y_prog_train)
print("Model Training Accuracy:", bst_prog.score(x_prog_train, y_prog_train)*100, "%")

y_prog_hat = bst_prog.predict(x_prog_val)

# Print performance metrics 
print('Accuracy Score:', bst_prog.score(x_prog_val, y_prog_val))
print('Area under ROC:', roc_auc_score(y_prog_val, y_prog_hat))

# Print the classification report to check the type 1 (false positive) and type 2 (false negative) error rates
print(classification_report(y_prog_val, y_prog_hat))

#Plot decision tree
from xgboost import plot_tree
plot_tree(bst_prog)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(150, 100)

# Get ROC curve metrics 
false_prog_positive_rate, true_prog_positive_rate, threshold = roc_curve(y_prog_val, y_prog_hat)

# Plot ROC curve
plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - Gradient Boosting')
plt.plot(false_prog_positive_rate, true_prog_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

all_data.head()

