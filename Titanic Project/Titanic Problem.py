import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import preprocessing
#Import data
trainingData = pd.read_csv('train.csv')
y = trainingData['Survived']
testingData = pd.read_csv('test.csv')

#Adjust data
trainingData = trainingData.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
x_categoricalFeatures = ['Pclass','Sex','SibSp','Embarked']
x_categoricalFeatures = pd.get_dummies(trainingData[x_categoricalFeatures])
x_numericalFeatures = trainingData.loc[:,['Age','Fare']]
X_train = pd.concat([x_categoricalFeatures,x_numericalFeatures],axis = 1)
X_train = X_train.values


idList = testingData['PassengerId'] 
testingData = testingData.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
y_categoricalFeatures = ['Pclass','Sex','SibSp','Embarked']
y_categoricalFeatures = pd.get_dummies(testingData[y_categoricalFeatures])
y_numericalFeatures = testingData.loc[:,['Age','Fare']]
X_test = pd.concat([y_categoricalFeatures,y_numericalFeatures],axis = 1)
X_test = X_test.values


#Replace Missing data
from sklearn import impute
imp_mean = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(X_train)
X_train = imp_mean.transform(X_train)
imp_mean.fit(X_test)
X_test = imp_mean.transform(X_test)

#Scale data
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X_train)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=1)
model.fit(X_train, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': idList, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
