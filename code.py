import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing , svm
from sklearn.metrics import accuracy_score

df = pd.read_csv('data.csv')

#Data pre-processing
Ydf = df['Churn']
Xdf = df.drop('Churn', 1)

#Encode Data
enc = preprocessing.OrdinalEncoder()
enc.fit(Xdf)

X = enc.transform(Xdf)

Y = Ydf.replace(to_replace=['No', 'Yes'], value=[0, 1])

#Train Data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

#optimization.
#you can skip this process but it will take too much time and process especially in SVM
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

#predict

# Naive Bayes method
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test) # X_test is encoded data and y_pred is the predict result (yes as 1 or no as 0)

y_pred_list = y_pred.tolist()

print('Predection percentage for yes = ', int( round(y_pred_list.count(1)/len(y_pred_list)*100) ), '%')
print('Predection percentage for no = ', int( round(y_pred_list.count(0)/len(y_pred_list)*100) ), '%')

print("Naive Bayes Accuracy Score = ",int( round(accuracy_score(y_pred, y_test)*100) ),'%')


# support vector machine (SVM) method
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(X_train,y_train)
predictions_SVM = SVM.predict(X_test)

print('Predection percentage for yes = ', int( round(y_pred_list.count(1)/len(y_pred_list)*100) ), '%')
print('Predection percentage for no = ', int( round(y_pred_list.count(0)/len(y_pred_list)*100) ), '%')

print("SVM Accuracy Score -> ",int( round(accuracy_score(predictions_SVM, y_test)*100) ),'%')

