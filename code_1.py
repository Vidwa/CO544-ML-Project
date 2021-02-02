# importing required libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# read the train dataset
train_data = pd.read_csv('trainData.csv')
print('Shape of data :',train_data.shape)

train_data['A1'].replace('?',np.nan,inplace =True)
train_data['A2'].replace('?',np.nan,inplace=True)
train_data['A3'].replace('?',np.nan,inplace=True)
train_data['A4'].replace('?',np.nan,inplace=True)
train_data['A5'].replace('?',np.nan,inplace=True)
train_data['A6'].replace('?',np.nan,inplace=True)
train_data['A7'].replace('?',np.nan,inplace=True)
train_data['A8'].replace('?',np.nan,inplace=True)
train_data['A9'].replace('?',np.nan,inplace=True)
train_data['A10'].replace('?',np.nan,inplace=True)
train_data['A11'].replace('?',np.nan,inplace=True)
train_data['A12'].replace('?',np.nan,inplace=True)
train_data['A13'].replace('?',np.nan,inplace=True)
train_data['A14'].replace('?',np.nan,inplace=True)
train_data['A15'].replace('?',np.nan,inplace=True)
train_data.dropna(inplace=True)

train_data['A2'] = train_data.A2.astype(float)
train_data['A5'] = train_data.A5.astype(float)
train_data['A7'] = train_data.A7.astype(float)
train_data['A10'] = train_data.A10.astype(float)
train_data['A12'] = train_data.A12.astype(float)
train_data['A14'] = train_data.A14.astype(float)

# shape of the dataset
print('Shape of training data :',train_data.shape)

df = pd.DataFrame(train_data)

#encode the labels in the column
le_A1 = LabelEncoder()


df['A1_n'] = le_A1.fit_transform(df['A1'])
#df['A2_n'] = le_A1.fit_transform(df['A2'])
df['A3_n'] = le_A1.fit_transform(df['A3'])
df['A4_n'] = le_A1.fit_transform(df['A4'])
#df['A5_n'] = le_A1.fit_transform(df['A5'])
df['A6_n'] = le_A1.fit_transform(df['A6'])
#df['A7_n'] = le_A1.fit_transform(df['A7'])
df['A8_n'] = le_A1.fit_transform(df['A8'])
df['A9_n'] = le_A1.fit_transform(df['A9'])
#df['A10_n'] = le_A1.fit_transform(df['A10'])
df['A11_n'] = le_A1.fit_transform(df['A11'])
#df['A12_n'] = le_A1.fit_transform(df['A12'])
df['A13_n'] = le_A1.fit_transform(df['A13'])
#df['A14_n'] = le_A1.fit_transform(df['A14'])
df['A15_n'] = le_A1.fit_transform(df['A15'])
#df['A16_n'] = le_A1.fit_transform(df['A16'])

df = df.drop(['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15'],axis='columns')


columns = df.columns.tolist()

columns = [c for c in columns if c not in ["A16"]]

target = "A16"

train_x = df[columns]
train_y = df[target]

scaler = MinMaxScaler()
train_x = scaler.fit_transform(train_x)


model = RandomForestClassifier()


# fit the model with the training data
model.fit(train_x,train_y)

# predict the target on the train dataset
predict_train = model.predict(train_x)
#print('Target on train data',predict_train)

# Accuray Score on train dataset
accuracy_train = accuracy_score(train_y,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)


#test dataset...............

# read the train  dataset
test_data = pd.read_csv('testdata.csv')

train_data['A2'] = train_data.A2.astype(float)
train_data['A5'] = train_data.A5.astype(float)
train_data['A7'] = train_data.A7.astype(float)
train_data['A10'] = train_data.A10.astype(float)
train_data['A12'] = train_data.A12.astype(float)
train_data['A14'] = train_data.A14.astype(float)

#encode the labels in the column
le_A1_test = LabelEncoder()

#fit and transform encoded columns
test_data['A1_test'] = le_A1_test.fit_transform(test_data['A1'])
#test_data['A2_test'] = le_A1_test.fit_transform(test_data['A2'])
test_data['A3_test'] = le_A1_test.fit_transform(test_data['A3'])
test_data['A4_test'] = le_A1_test.fit_transform(test_data['A4'])
#test_data['A5_test'] = le_A1_test.fit_transform(test_data['A5'])
test_data['A6_test'] = le_A1_test.fit_transform(test_data['A6'])
#test_data['A7_test'] = le_A1_test.fit_transform(test_data['A7'])
test_data['A8_test'] = le_A1_test.fit_transform(test_data['A8'])
test_data['A9_test'] = le_A1_test.fit_transform(test_data['A9'])
#test_data['A10_test'] = le_A1_test.fit_transform(test_data['A10'])
test_data['A11_test'] = le_A1_test.fit_transform(test_data['A11'])
#test_data['A12_test'] = le_A1_test.fit_transform(test_data['A12'])
test_data['A13_test'] = le_A1_test.fit_transform(test_data['A13'])
#test_data['A14_test'] = le_A1_test.fit_transform(test_data['A14'])
test_data['A15_test'] = le_A1_test.fit_transform(test_data['A15'])

test_data = test_data.drop(['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15'],axis='columns')

test_data = scaler.transform(test_data.values[:,0:15])

#test prediction results
predict_test = model.predict(test_data)
print('Test prediction results',predict_test)
