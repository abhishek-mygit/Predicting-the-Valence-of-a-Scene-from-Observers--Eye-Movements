
import numpy as np
from scipy.io import loadmat
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

features = loadmat('./features.mat')
X=features['features']

classLabels = loadmat('./imageLabel.mat');
Y = classLabels['imageLabel']
Y = np.ravel(Y)

#Splits the 70% data for training and 30% data for testing
X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size=0.3,random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))

from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier

svm = SVC(kernel="linear", C=1.0, random_state=0,probability=True)
knn = KNeighborsClassifier(n_neighbors=4)
etc = ExtraTreesClassifier(n_estimators=10)

ensemble = VotingClassifier(estimators=[('svm', svm), ('knn', knn), ('etc',etc)], voting='hard')
ensemble.fit(X_train, y_train)

#Calculates the accuracy score by comparing the number of differences between the actual label and the predicted label
accuracy_percentage = ensemble.score(X_test,y_test)*100

predicted_values = ensemble.predict(X_test)

print("Accuracy is "+str(accuracy_percentage)+"%")

