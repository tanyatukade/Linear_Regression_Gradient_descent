#####################################################################################################################
#   CS 6375.003 - Assignment 1, Linear Regression using Gradient Descent
#   This is a simple starter code in Python 3.6 for linear regression using the notation shown in class.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   test - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn import preprocessing


def preProcess_nmr(data):
    df = pd.read_csv(data)
    df = df.dropna()  # Removing null values
    df.drop_duplicates()  # Removing duplicates
    df['carName'] = df['carName'].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(
        lambda x: x.cat.codes)  # categorical to numerical
    # scaler = preprocessing.MinMaxScaler()
    # df[df.columns] = scaler.fit_transform(df[df.columns])
    df = df.drop(['carName'], axis=1)
    return df

def split_training_data(df):
    train = df.sample(frac=0.8, random_state=300)
    test = df.drop(train.index)
    return (train, test)

df = preProcess_nmr("dataset.csv")
print(df.shape[0])
train, test = split_training_data(df)


class LinearRegression_nmr:

    def __init__(self, train):
        df = train
        np.random.seed(1)
        # train refers to the training dataset
        # stepSize refers to the step size of gradient descent
        df.insert(1, 'X0', 1)
        self.nrows, self.ncols = df.shape[0], df.shape[1]
        self.X = df.iloc[:, 1:(self.ncols)].values.reshape(
            self.nrows, self.ncols-1)
        self.y = df.iloc[:, 0].values.reshape(self.nrows, 1)
        self.W = np.random.rand(self.ncols-1).reshape(self.ncols-1, 1)
    # TODO: Perform pre-processing for your dataset. It may include doing the following:
    #   - getting rid of null values
    #   - converting categorical to numerical values
    #   - scaling and standardizing attributes
    #   - anything else that you think could increase model performance
    # Below is the pre-process function

    # Below is the training function

    def train(self, epochs=10, learning_rate=0.1):
        # Perform Gradient Descent
        for i in range(epochs):
            # Make prediction with current weights
            h = np.dot(self.X, self.W)
            # Find error
            error = h - self.y

            # print(error)
            self.W = self.W - (1 / self.nrows) * \
                learning_rate * np.dot(self.X.T, error)
            # if i == epochs - 1:
            # print("i= ", i , "h= ", h , "W= " ,self.W, " y = ", self.y, "error= ", error )
        return self.W, error

    # predict on test dataset
    def predict(self, txt):
        #         print(test.shape[0])
        if(txt == "test"):
            testDF = test
            testDF.insert(1, "X0", 1)
        else:
            testDF = train
        nrows, ncols = testDF.shape[0], testDF.shape[1]
        print("rows, columns =", nrows, ncols)
        testX = testDF.iloc[:, 1:ncols].values.reshape(nrows, ncols - 1)
        testY = testDF.iloc[:, 0].values.reshape(nrows, 1)
        pred = np.dot(testX, self.W)
        error = pred - testY
        #print("r2= ", metrics.r2_score(testY, pred))
        mse = (1/(2.0*nrows)) * np.dot(error.T, error)
        # print("error = ", error, "error.T = ", error.T)
        return mse


if __name__ == "__main__":
    model = LinearRegression_nmr(train)
    W, e = model.train()
    # print(W, e)
    mse = model.predict("test")
    print("Testing mse= ", mse)
    mse = model.predict("train")
    print("Training mse= ", mse)
