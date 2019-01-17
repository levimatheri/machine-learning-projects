import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from cross_validation import *
from predict import *

def init():
    names = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
             'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
             'Property_Area', 'Loan_Status']
    df = pd.read_csv('train.csv')



    # normalize applicantIncome
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['TotalIncome_norm'] = np.log(df['TotalIncome'])
    df['LoanAmount_norm'] = np.log(df['LoanAmount'])

    # fill_missing
    fill_missing(df)

    df = df.drop(columns=['ApplicantIncome', 'CoapplicantIncome', 'TotalIncome', 'LoanAmount'])

    df = df[['Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_norm','TotalIncome_norm', 'Loan_Status']]



    # categorize
    encode(df, ['Self_Employed', 'Property_Area', 'Loan_Status'])

    dataset = df.values

    X = dataset[:, 0:6]
    Y = dataset[:, 6]


    # create cv set
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.33)


    # cross_validation checks
    test_models(X_train, Y_train)

    # outcome_var = 'Loan_Status'
    # model = LogisticRegression()
    # predictor_var = ['Credit_History']

    # classification_model(model, df, predictor_var, outcome_var)

    # test = pd.read_csv('test.csv')
    #
    # test_df = test[['Loan_ID','Gender','Married','Dependents','Education','Self_Employed',
    #                 'ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term',
    #                 'Credit_History','Property_Area']]
    # X_test = test_data[:, ]
    #
    # predict()
    #predict(X_train, Y_train, X_val, Y_val)


def fill_missing(df):
    #df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Education'].fillna(df['Education'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    df['Property_Area'].fillna(df['Property_Area'].mode()[0], inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['TotalIncome_norm'].fillna(df['TotalIncome_norm'].mean(), inplace=True)
    df['LoanAmount_norm'].fillna(df['LoanAmount_norm'].mean(), inplace=True)


def encode(df, attr_list):
    encoder = LabelEncoder()
    for i in attr_list:
        df[i] = encoder.fit_transform(df[i])
