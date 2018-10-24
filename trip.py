from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import time
from sklearn.metrics import accuracy_score,confusion_matrix
trip = pd.read_csv("/home/samridhi/Downloads/2010-capitalbikeshare-tripdata.csv")
trip.drop(["Start date", "End date", "Start station", "End station"], axis = 1, inplace = True)
#print trip["Start station"].value_counts()
#print trip["Start station"].nunique()	
#print trip["End station"].nunique()	
print list(trip)
membertype = {'Member': 1,'Casual': 0, 'Unknown' : 0 } 

le= LabelEncoder()
trip["Bike number"] = le.fit_transform(trip["Bike number"])
trip["Member type"] = [membertype[item] for item in trip["Member type"]] 
training = trip.sample(random_state = 200, frac = 0.7)
training_Y = training["Member type"]
training.drop(["Member type"], axis = 1, inplace = True)
test = trip.drop(training.index)
test_Y = test["Member type"]
test.drop(["Member type"], axis = 1, inplace = True)

trip.to_csv("mycsv.csv")
#print training.head()
#print training_Y

#logisticRegressor
reg = LogisticRegression()
start = time.time()
print "Starting training..."
reg.fit(training, training_Y)
print "Training took " +  str(time.time() - start) + "s"
pred = reg.predict(test)
print accuracy_score(pred, test_Y)

#RFClassifier

clf = RandomForestClassifier(n_estimators = 15, min_samples_leaf = 3, n_jobs = 4)
start = time.time()
print "Starting training..."
clf.fit(training, training_Y)
print "Training took " +  str(time.time() - start) + "s"
pred = clf.predict(test)

print accuracy_score(pred, test_Y)
sns.scatterplot(trip["Duration"], trip["Start station number"], hue = trip["Member type"])
plt.show() 
plt.hist(trip["Member type"])
plt.show()
