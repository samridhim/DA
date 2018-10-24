from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
#load csv, display histograms and boxplots if required
diabetes = pd.read_csv('/home/samridhi/Desktop/LP1/DA/pima-indians-diabetes.csv', header = None, names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Pedigree", "Age", "Class"])
plt.hist(diabetes["Age"], bins = 50)
plt.show()

#normalising using min max normaliser

min_max_scaler = preprocessing.MinMaxScaler()
diabetes[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = min_max_scaler.fit_transform(diabetes[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]])
diabetes["Pregnancies"]=diabetes["Pregnancies"].replace(0,diabetes["Pregnancies"].median())
diabetes["Glucose"]=diabetes["Glucose"].replace(0,diabetes["Glucose"].median())
diabetes["BloodPressure"]=diabetes["BloodPressure"].replace(0,diabetes["BloodPressure"].median())
diabetes["SkinThickness"]=diabetes["SkinThickness"].replace(0,diabetes["SkinThickness"].median())
diabetes["Insulin"]=diabetes["Insulin"].replace(0,diabetes["Insulin"].median())
diabetes["BMI"]=diabetes["BMI"].replace(0,diabetes["BMI"].median())
diabetes["Pedigree"]=diabetes["Pedigree"].replace(0,diabetes["Pedigree"].median())
diabetes["Age"]=diabetes["Age"].replace(0,diabetes["Age"].median())

diabetes.to_csv("mycsv.csv")

#train and test split
training = diabetes.sample(frac = 0.75)
test = diabetes.drop(training.index)
training_Y = training["Class"]
test_Y = test["Class"]
sns.scatterplot(diabetes["Age"], diabetes["BMI"], hue = diabetes["Class"])
plt.show()
training.drop(["Class"], axis = 1, inplace = True)
test.drop(["Class"], axis = 1, inplace = True)
scatter_matrix(training, figsize = (8, 8),diagonal = "kde")
plt.show()
#gaussian
clf = GaussianNB()
print "Starting training"
start = time.time()
clf.fit(training, training_Y)
print clf
print "Training took " + str(time.time() - start) + "s"
pred = clf.predict(test)
#print pred
#print clf.score(test, test_Y)
print accuracy_score(pred, test_Y)

#multinomial
clf = MultinomialNB()
clf.fit(training, training_Y)
print clf
pred = clf.predict(test)
#print pred
#print clf.score(test, test_Y)
print accuracy_score(pred, test_Y)

#complement
clf = ComplementNB()
clf.fit(training, training_Y)
print clf
pred = clf.predict(test)
#print pred
#print clf.score(test, test_Y)
print accuracy_score(pred, test_Y)
