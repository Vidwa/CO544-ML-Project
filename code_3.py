# CO544 project
# SVM


import pandas as pd
import numpy as np
import statistics

# read the train dataset
data_read = pd.read_csv('trainData.csv')

df = pd.DataFrame(data_read)

#df = df.replace("?", np.NaN)

df['A2']=df['A2'].replace('?',np.NAN)
df['A2']=df['A2'].astype(float)
A2_median = df['A2'].median()
df['A2']=df['A2'].fillna(A2_median)

df['A5']=df['A5'].replace('?',np.NAN)
df['A5']=df['A5'].astype(float)
A5_median = df['A5'].median()
df['A5']=df['A5'].fillna(A5_median)

df['A7']=df['A7'].replace('?',np.NAN)
df['A7']=df['A7'].astype(float)
A7_median = df['A7'].median()
df['A7']=df['A7'].fillna(A7_median)

df['A10']=df['A10'].replace('?',np.NAN)
df['A10']=df['A10'].astype(float)
A10_median = df['A10'].median()
df['A10']=df['A10'].fillna(A10_median)

df['A12']=df['A12'].replace('?',np.NAN)
df['A12']=df['A12'].astype(float)
A12_median = df['A12'].median()
df['A12']=df['A12'].fillna(A12_median)

df['A14']=df['A14'].replace('?',np.NAN)
df['A14']=df['A14'].astype(float)
A14_median = df['A14'].median()
df['A14']=df['A14'].fillna(A14_median)


from sklearn.preprocessing import LabelEncoder

#encode the data for using for further analysings
le_A1 = LabelEncoder()

df['A1'] = le_A1.fit_transform(df['A1'])
df['A3'] = le_A1.fit_transform(df['A3'])
df['A4'] = le_A1.fit_transform(df['A4'])
df['A6'] = le_A1.fit_transform(df['A6'])
df['A8'] = le_A1.fit_transform(df['A8'])
df['A9'] = le_A1.fit_transform(df['A9'])
df['A11'] = le_A1.fit_transform(df['A11'])
df['A13'] = le_A1.fit_transform(df['A13'])
df['A15'] = le_A1.fit_transform(df['A15'])
#df['A16'] = le_A1.fit_transform(df['A16'])


from sklearn.model_selection import train_test_split

xValues=df.values[:,0:15]
yValues=data_read.values[:,15]

train_y = data_read['A16']

#separating the given data set into train data set and test data set for analysing futher
train_x, test_x, train_y, test_y =train_test_split(xValues,yValues,random_state=0)


from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

#preprocessing using MinMaxScaler by rescaling data
min_max_scaler= preprocessing.MinMaxScaler()

train_x=min_max_scaler.fit_transform(train_x)
test_x=min_max_scaler.transform(test_x)



from sklearn.svm import SVC

print('Using SVM algorithm:\n')
model_SVM = SVC(gamma='scale')
model_SVM.fit(train_x,train_y)

Accuracy_train_svm = model_SVM.score(train_x,train_y)*100
print('Accuracy on train set: {:.2f}'.format(Accuracy_train_svm))

Accuracy_test_svm = model_SVM.score(test_x,test_y)*100
print('Accuracy on test set: {:.2f}'.format(Accuracy_test_svm))



#read the given test data csv file that needs to be predicted

test_data = pd.read_csv('testdata.csv')

df_1=pd.DataFrame(test_data)

df_1['A2']=df_1['A2'].replace('?',np.NAN)
df_1['A2']=df_1['A2'].astype(float)
A2_median = df_1['A2'].median()
df_1['A2']=df_1['A2'].fillna(A2_median)

df_1['A5']=df_1['A5'].replace('?',np.NAN)
df_1['A5']=df_1['A5'].astype(float)
A5_median = df_1['A5'].median()
df_1['A5']=df_1['A5'].fillna(A5_median)

df_1['A7']=df_1['A7'].replace('?',np.NAN)
df_1['A7']=df_1['A7'].astype(float)
A7_median = df_1['A7'].median()
df_1['A7']=df_1['A7'].fillna(A7_median)

df_1['A10']=df_1['A10'].replace('?',np.NAN)
df_1['A10']=df_1['A10'].astype(float)
A10_median = df_1['A10'].median()
df_1['A10']=df_1['A10'].fillna(A10_median)

df_1['A12']=df_1['A12'].replace('?',np.NAN)
df_1['A12']=df_1['A12'].astype(float)
A12_median = df_1['A12'].median()
df_1['A12']=df_1['A12'].fillna(A12_median)

df_1['A14']=df_1['A14'].replace('?',np.NAN)
df_1['A14']=df_1['A14'].astype(float)
A14_median = df_1['A14'].median()
df_1['A14']=df_1['A14'].fillna(A14_median)


df_1['A1'] = le_A1.fit_transform(df_1['A1'])
df_1['A3'] = le_A1.fit_transform(df_1['A3'])
df_1['A4'] = le_A1.fit_transform(df_1['A4'])
df_1['A6'] = le_A1.fit_transform(df_1['A6'])
df_1['A8'] = le_A1.fit_transform(df_1['A8'])
df_1['A9'] = le_A1.fit_transform(df_1['A9'])
df_1['A11'] = le_A1.fit_transform(df_1['A11'])
df_1['A13'] = le_A1.fit_transform(df_1['A13'])
df_1['A15'] = le_A1.fit_transform(df_1['A15'])


scaled_data=min_max_scaler.transform(df_1.values[:,0:15])

predict_test = model_SVM.predict(scaled_data)
print(predict_test)

data_final = pd.read_csv('testdata.csv')

data_final['A16'] = predict_test

df_final = pd.DataFrame(data_final)
test_predictions = pd.DataFrame(data_final['A16'])

test_predictions.to_csv('testdata1.csv',sep=',')
