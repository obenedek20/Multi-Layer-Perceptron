# Starter code for CS 165B MP3
import random

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, KBinsDiscretizer, StandardScaler
from sklearn.decomposition import PCA



import sklearn

np.random.seed(0)

def compute_metric(labels, expected):
    tp = np.sum(labels[expected == 1])
    fp = np.sum(labels[expected == 0])
    tn = np.sum(1-labels[expected == 0])
    fn = np.sum(1-labels[expected == 1])
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    error_rate = (fp+fn)/(tp+fp+tn+fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    precision = tp/(tp+fp)
    f1 = 2*tp/(2*tp+fp+fn)

    return {
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "tpr": tpr,
        "fpr": fpr,
        "error_rate": error_rate,
    }

def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    """
    | Feature Name        | Meaning                  | Feature Type             |
    | ------------------- | ------------------------ |  ----------------------- |  
    | CODE_GENDER         | Gender                   | binary (F/M)             |
    | FLAG_OWN_CAR        | Is there a car           | binary (Y/N)             |
    | FLAG_OWN_REALTY     | Is there a property      | binary (Y/N)             | 
    | CNT_CHILDREN        | Number of children       | categorical              |
    | AMT_INCOME_TOTAL    | Annual income            | continuous               |
    | NAME_INCOME_TYPE    | Income category          | categorical              | 
    | NAME_EDUCATION_TYPE | Education level          | categorical              |  
    | NAME_FAMILY_STATUS  | Marital status           | categorical              | 
    | NAME_HOUSING_TYPE   | Way of living            | categorical              | 
    | DAYS_BIRTH          | Birthday                 | continuous               |
    | DAYS_EMPLOYED       | Start date of employment | continuous               |
    | FLAG_MOBIL          | Is there a mobile phone  | binary (0/1)             |
    | FLAG_WORK_PHONE     | Is there a work phone    | binary (0/1)             |
    | FLAG_PHONE          | Is there a phone         | binary (0/1)             |
    | FLAG_EMAIL          | Is there an email        | binary (0/1)             |
    | OCCUPATION_TYPE     | Occupation               | categorical              |
    | CNT_FAM_MEMBERS     | Family size              | categorical              |
    | QUANTIZED_INC       | quantized income         | categorical              |
    | QUANTIZED_AGE       | quantized age            | categorical              |
    | QUANTIZED_WORK_YEAR | quantized employment year| categorical              |
    | target              | whether the user is a risky customer | binary (0/1)    
    """
    le = LabelEncoder()

    le.fit(train["NAME_INCOME_TYPE"])
    train["NAME_INCOME_TYPE"] = le.transform(train["NAME_INCOME_TYPE"])
    test["NAME_INCOME_TYPE"] = le.transform(test["NAME_INCOME_TYPE"])

    le.fit(train["NAME_EDUCATION_TYPE"])
    train["NAME_EDUCATION_TYPE"] = le.transform(train["NAME_EDUCATION_TYPE"])
    test["NAME_EDUCATION_TYPE"] = le.transform(test["NAME_EDUCATION_TYPE"])

    le.fit(train["NAME_FAMILY_STATUS"])
    train["NAME_FAMILY_STATUS"] = le.transform(train["NAME_FAMILY_STATUS"])
    test["NAME_FAMILY_STATUS"] = le.transform(test["NAME_FAMILY_STATUS"])

    le.fit(train["NAME_HOUSING_TYPE"])
    train["NAME_HOUSING_TYPE"] = le.transform(train["NAME_HOUSING_TYPE"])
    test["NAME_HOUSING_TYPE"] = le.transform(test["NAME_HOUSING_TYPE"])

    le.fit(train["QUANTIZED_INC"])
    train["QUANTIZED_INC"] = le.transform(train["QUANTIZED_INC"])
    test["QUANTIZED_INC"] = le.transform(test["QUANTIZED_INC"])

    le.fit(train["QUANTIZED_AGE"])
    train["QUANTIZED_AGE"] = le.transform(train["QUANTIZED_AGE"])
    test["QUANTIZED_AGE"] = le.transform(test["QUANTIZED_AGE"])

    le.fit(train["QUANTIZED_WORK_YEAR"])
    train["QUANTIZED_WORK_YEAR"] = le.transform(train["QUANTIZED_WORK_YEAR"])
    test["QUANTIZED_WORK_YEAR"] = le.transform(test["QUANTIZED_WORK_YEAR"])

    est = KBinsDiscretizer(n_bins=5, strategy='uniform', encode='ordinal')
    cols = ["DAYS_EMPLOYED", "DAYS_BIRTH", "AMT_INCOME_TOTAL"]
    est.fit(train[cols])
    train[cols] = est.transform(train[cols])
    test[cols] = est.transform(test[cols])

    train = train.drop(["OCCUPATION_TYPE"], axis=1)
    test = test.drop(["OCCUPATION_TYPE"], axis=1)
    return train, test


def run_train_test(training_data: pd.DataFrame, testing_data: pd.DataFrame) -> List[int]:
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data:
        testing_data: the same as training_data with "target" removed.

    Output:
        testing_prediction: List[int]
    Example output:
    return random.choices([0, 1, 2], k=len(testing_data))
    """
    preprocessed_train, preprocessed_test = preprocess(training_data, testing_data)
    X_train = preprocessed_train.iloc[:,:-1]
    y_train = preprocessed_train.iloc[:,-1]

    #https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    model = MLPClassifier(hidden_layer_sizes=100, batch_size=64, learning_rate='constant', learning_rate_init=0.001, random_state=1, max_iter=400).fit(X_train, y_train)
    predict = model.predict(preprocessed_test)

    return predict


if __name__ == '__main__':

    training = pd.read_csv('data/train.csv')
    development = pd.read_csv('data/dev.csv')

    target_label = development['target']
    development.drop('target', axis=1, inplace=True)
    prediction = run_train_test(training, development)
    target_label = target_label.values
    status = compute_metric(prediction, target_label)
    print(status)
