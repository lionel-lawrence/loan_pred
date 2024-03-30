# -*- coding: utf-8 -*-
### Importing libraries & functions


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

"""### Importing dataset"""

dataset=pd.read_excel("a_Dataset_CreditScoring.xlsx", skiprows=1)

"""### Data preparation"""

# shows count of rows and columns
dataset.shape

#shows first few rows of the code
dataset.head()

#dropping customer ID column from the dataset
dataset=dataset.drop('ID',axis=1)
dataset.shape

# explore missing values
dataset.isna().sum()

# list(zip(dataset.mean().values, dataset.median().values))

# filling missing values with mean
dataset=dataset.fillna(dataset.median())

# explore missing values post missing value fix
dataset.isna().sum()

# # count of good loans (0) and bad loans (1)
# dataset['TARGET'].value_counts()

# # data summary across 0 & 1
# dataset.groupby('TARGET').mean()

"""### Train Test Split"""

y = dataset.iloc[:, 0].values
X = dataset.iloc[:, 1:29].values

# splitting dataset into training and test (in ratio 80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=y)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Exporting Normalisation Coefficients for later use in prediction
import joblib
joblib.dump(sc,'f2_Normalisation_CreditScoring')

"""### Risk Model building"""

classifier =  LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Exporting Logistic Regression Classifier for later use in prediction

joblib.dump(classifier, 'f1_Classifier_CreditScoring')

"""### Model *performance*"""

print(confusion_matrix(y_test,y_pred))

print(accuracy_score(y_test, y_pred))

"""### Writing output file"""

predictions = classifier.predict_proba(X_test)
predictions

# writing model output file

df_prediction_prob = pd.DataFrame(predictions, columns = ['prob_0', 'prob_1'])
df_prediction_target = pd.DataFrame(classifier.predict(X_test), columns = ['predicted_TARGET'])
df_test_dataset = pd.DataFrame(y_test,columns= ['Actual Outcome'])

dfx=pd.concat([df_test_dataset, df_prediction_prob, df_prediction_target], axis=1)

dfx.to_csv("c1_Model_Prediction.xlsx", sep=',', encoding='UTF-8')

dfx.head()

"""### Coding ends here!"""
