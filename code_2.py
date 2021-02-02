
# importing required libraries
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# read the train dataset
train_data = pd.read_csv('trainData.csv')


# seperate the independent and target variable on training data
train_x = train_data.drop(columns=['A16'],axis=1)
train_y = train_data['A16']


# handling the missing values
train_data = train_data.replace("?", np.NaN)
train_data = train_data.dropna(how ='any')

#encode the labels in the column
le_A1 = LabelEncoder()


train_x['A1_n'] = le_A1.fit_transform(train_x['A1'])
train_x['A2_n'] = le_A1.fit_transform(train_x['A2'])
train_x['A3_n'] = le_A1.fit_transform(train_x['A3'])
train_x['A4_n'] = le_A1.fit_transform(train_x['A4'])
train_x['A5_n'] = le_A1.fit_transform(train_x['A5'])
train_x['A6_n'] = le_A1.fit_transform(train_x['A6'])
train_x['A7_n'] = le_A1.fit_transform(train_x['A7'])
train_x['A8_n'] = le_A1.fit_transform(train_x['A8'])
train_x['A9_n'] = le_A1.fit_transform(train_x['A9'])
train_x['A10_n'] = le_A1.fit_transform(train_x['A10'])
train_x['A11_n'] = le_A1.fit_transform(train_x['A11'])
train_x['A12_n'] = le_A1.fit_transform(train_x['A12'])
train_x['A13_n'] = le_A1.fit_transform(train_x['A13'])
train_x['A14_n'] = le_A1.fit_transform(train_x['A14'])
train_x['A15_n'] = le_A1.fit_transform(train_x['A15'])

train_x = train_x.drop(['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15'],axis='columns')


x_train,x_test,y_train,y_test=train_test_split(train_x,train_y,test_size=0.2)

# shape of the dataset
print('Shape of train data :',x_train.shape)
print('Shape of test data :',x_test.shape)



model = GaussianNB()

# fit the model with the training data
model.fit(x_train,y_train)

# predict the target on the train dataset
predict_train = model.predict(x_train)
#print('Target on train data',predict_train)

# Accuray Score on train dataset
accuracy_train = accuracy_score(y_train,predict_train)
#print('accuracy_score on train dataset : ', accuracy_train)



# predict the target on the train dataset
predict_test = model.predict(x_test)
#print('Target on test data',predict_test)
# Accuray Score on test dataset
accuracy_test = accuracy_score(y_test,predict_test)
print('accuracy_score on test dataset : ', accuracy_test)


#predictions for given test dataset...............


#print('predictions for given test dataset:')
# read the train  dataset
test_data = pd.read_csv('testdata.csv')

#encode the labels in the column
le_A1_test = LabelEncoder()

#fit and transform encoded columns
test_data['A1_test'] = le_A1_test.fit_transform(test_data['A1'])
test_data['A2_test'] = le_A1_test.fit_transform(test_data['A2'])
test_data['A3_test'] = le_A1_test.fit_transform(test_data['A3'])
test_data['A4_test'] = le_A1_test.fit_transform(test_data['A4'])
test_data['A5_test'] = le_A1_test.fit_transform(test_data['A5'])
test_data['A6_test'] = le_A1_test.fit_transform(test_data['A6'])
test_data['A7_test'] = le_A1_test.fit_transform(test_data['A7'])
test_data['A8_test'] = le_A1_test.fit_transform(test_data['A8'])
test_data['A9_test'] = le_A1_test.fit_transform(test_data['A9'])
test_data['A10_test'] = le_A1_test.fit_transform(test_data['A10'])
test_data['A11_test'] = le_A1_test.fit_transform(test_data['A11'])
test_data['A12_test'] = le_A1_test.fit_transform(test_data['A12'])
test_data['A13_test'] = le_A1_test.fit_transform(test_data['A13'])
test_data['A14_test'] = le_A1_test.fit_transform(test_data['A14'])
test_data['A15_test'] = le_A1_test.fit_transform(test_data['A15'])

test_data = test_data.drop(['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15'],axis='columns')

#test prediction results 
predict_test = model.predict(test_data)
print('Test prediction results',predict_test)
