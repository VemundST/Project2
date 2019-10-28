from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



def load_data(scaler='standard'):
    directory = os.getcwd()
    filename = directory + '/cred_card.xls'
    nanDict = {} # fjerner NaN
    dataframe = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)


    dataframe.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

    # Features and targets
    X = dataframe.loc[:, dataframe.columns != 'defaultPaymentNextMonth'].values
    y = dataframe.loc[:, dataframe.columns == 'defaultPaymentNextMonth'].values


    # Categorical variables to one-hot's
    onehotencoder = OneHotEncoder(categories="auto", sparse = False)
    X = ColumnTransformer(
        [("", onehotencoder, [1,2,3,5,6,7,8,9,10]),],
        remainder="passthrough"
    ).fit_transform(X)



        # Remove instances with zeros only for past bill statements or paid amounts
    '''
    dataframe = dataframe.drop(dataframe[(dataframe.BILL_AMT1 == 0) &
                    (dataframe.BILL_AMT2 == 0) &
                    (dataframe.BILL_AMT3 == 0) &
                    (dataframe.BILL_AMT4 == 0) &
                    (dataframe.BILL_AMT5 == 0) &
                    (dataframe.BILL_AMT6 == 0) &
                    (dataframe.PAY_AMT1 == 0) &
                    (dataframe.PAY_AMT2 == 0) &
                    (dataframe.PAY_AMT3 == 0) &
                    (dataframe.PAY_AMT4 == 0) &
                    (dataframe.PAY_AMT5 == 0) &
                    (dataframe.PAY_AMT6 == 0)].index)

    dataframe = dataframe.drop(dataframe[(dataframe.BILL_AMT1 == 0) &
                    (dataframe.BILL_AMT2 == 0) &
                    (dataframe.BILL_AMT3 == 0) &
                    (dataframe.BILL_AMT4 == 0) &
                    (dataframe.BILL_AMT5 == 0) &
                    (dataframe.BILL_AMT6 == 0)].index)

    dataframe = dataframe.drop(dataframe[(dataframe.PAY_AMT1 == 0) &
                    (dataframe.PAY_AMT2 == 0) &
                    (dataframe.PAY_AMT3 == 0) &
                    (dataframe.PAY_AMT4 == 0) &
                    (dataframe.PAY_AMT5 == 0) &
                    (dataframe.PAY_AMT6 == 0)].index)
    '''


    if scaler=='standard':
        scaler = StandardScaler()
    if scaler=='minmax':
        scaler = MinMaxScaler()

    scaler.fit(X[:,[77,78,79,80,81,82,83,84,85,86,87,88,89,90]])
    scaled = scaler.transform(X[:,[77,78,79,80,81,82,83,84,85,86,87,88,89,90]])
    X[:,77:91]=scaled
    return X,y
