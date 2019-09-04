#K-nearest neighbours
import numpy as np
import pandas
from sklearn import model_selection, neighbors
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

filename =  r'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class_predicted']
dataframe = pandas.read_csv(filename, names=names)
print(dataframe.head())

# Survival_Status is the classifier and rest are predictor features
x = np.array(dataframe.drop(['class_predicted'],1))
y = np.array(dataframe.class_predicted)

x_train,x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.80, test_size=0.20)

# knn = neighbors.KNeighborsClassifier(n_neighbors=8)
temp = 0
temp2 = 0
for i in range(1, 20):
    print("For K = ", i , " Neighbours")
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    # score or R^2 = 1-(u/v) where u is (y-ypred)^2 and v is (y-ymean)^2
    score = knn.score(x_test, y_test)
    print("Success rate or score : ", score)
    if (temp<score):
        temp = score
        temp2 = i

print("Maximum score when K is checked between 1 to 20 is : ", temp, " for K = ", temp2)
cnf_mat = confusion_matrix(y_test, y_pred)

# True negative rate, specificity = TN/(TN+FP)
specificity = cnf_mat[1][1] / (cnf_mat[1][1] + cnf_mat[0][1])

# True positive rate, sensistivity = TP/(TP+FN)
sensitivity = cnf_mat[0][0] / (cnf_mat[0][0] + cnf_mat[1][0])

#accuracy = (TP+TN)/ (TP+FP+TN+FN)
accuracy = (cnf_mat[0][0] + cnf_mat[1][1]) / (cnf_mat[0][0] + cnf_mat[0][1] + cnf_mat[1][0] + cnf_mat[1][1])

print("Specificity i.e True negative rate :", specificity)
print("Sensitivity i.e True positve rate :", sensitivity)
print("Accuracy :", accuracy)
print("Classfication report is :")
print(classification_report(y_test, y_pred))
