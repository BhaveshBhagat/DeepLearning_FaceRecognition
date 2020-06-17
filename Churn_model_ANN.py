import pandas as pd
import numpy as np

#reading the churn dataset
dataset = pd.read_csv('/content/drive/My Drive/Colab Notebook/Churn_Modelling.csv')

x = dataset.iloc[:,3:13]
y = dataset.iloc[:,13]

#convert categorical variable using one hot method
geography = pd.get_dummies(x['Geography'],drop_first=True)
gender = pd.get_dummies(x['Gender'],drop_first=True)

x = pd.concat([x , gender , geography], axis =1)
x.drop(['Geography', 'Gender'],axis=1, inplace=True)


#diving into training and tesing dataset
from sklearn.model_selection import train_test_split
X_train , X_test ,Y_train ,Y_test = train_test_split(x , y ,test_size = 0.2 , random_state = 0)


#Scale down the dataset for ANN model
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#creating Sequential ANN model
import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Dropout

classifier = Sequential()

classifier.add(Dense(units = 8 ,init = 'he_uniform' , activation = 'relu' , input_dim = 11))

classifier.add(Dense(units = 10 ,init = 'he_uniform' , activation = 'relu'))

classifier.add(Dense(units = 8 ,init = 'he_uniform' , activation = 'relu'))

classifier.add(Dense(units = 1 , init = 'glorot_uniform' , activation = 'sigmoid'))

classifier.summary()

#finally complie all layers and fit the model 
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

model_history = classifier.fit(X_train , Y_train , validation_split = 0.33 , batch_size=10 , epochs = 100)


# now predicitng values for test dataset
ypred = classifier.predict(X_test)
ypred = (ypred > 0.5)


#checking accuracy and confusion matrix
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(Y_test , ypred)

score = accuracy_score(ypred , Y_test)

print(cm ,score)
