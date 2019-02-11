# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def preProcess(data):
        df = pd.read_csv(data)
        df = df.dropna() # Removing null values
        df.drop_duplicates()    # Dropping Dduplicate rows
        df['carName'] = df['carName'].astype('category')
        cat_columns = df.select_dtypes(['category']).columns
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes) # Categorical to Numerical 
        scaler = preprocessing.MinMaxScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns])   # Scaling 
        df = df.drop(['carName'],axis=1)
        return df
  
# DIR='/content/gdrive/My Drive/'
df = preProcess("dataset.csv")

nrows, ncols = df.shape[0], df.shape[1]
X =  df.iloc[:, 1:].values.reshape(nrows, ncols-1)
y = df.iloc[:, 0].values.reshape(nrows, 1)
# W = np.random.rand(self.ncols-1).reshape(self.ncols-1, 1)

from sklearn.linear_model import LinearRegression

# X = StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2,random_state=50)

model = LinearRegression()

model.fit(X_train,y_train)

y_predicted = model.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true=y_test,y_pred=y_predicted)
print(mse)