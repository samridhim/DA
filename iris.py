from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statistics
sepal_length=[]
sepal_width=[]
petal_length=[]
petal_width=[]


#plotting
dataf = pd.read_csv("/home/samridhi/Desktop/LP1/DA/iris.csv")
sepal_length = dataf["Sepal.Length"]
petal_length = dataf["Petal.Length"]
petal_width = dataf["Petal.Width"]
sepal_width = dataf["Sepal.Width"]
plt.hist(sepal_length,bins = 100)
plt.show()	
plt.boxplot([sepal_length,sepal_width,petal_length, petal_width], labels = ["sepal_length", "sepal_width", "petal_length", "petal_width"])
plt.show()
sns.scatterplot(sepal_length,sepal_width,hue=dataf["Species"]);
plt.show()
#computing mean, median and mode

#mode
#sepal_length_mode = max(sepal_length, key = sepal_length.count)
sepal_length = sorted(sepal_length)
#sepal_length_median = statistics.median(sepal_length)
sepal_length_median = (sepal_length[len(sepal_length)/2]);
sepal_length_median += (sepal_length[(len(sepal_length)+ 1)/2]);
sepal_length_median/=2

mean_sepal_length =0.0
for i in range(len(sepal_length)):
	mean_sepal_length+=sepal_length[i]
mean_sepal_length = mean_sepal_length/len(sepal_length)

var_sepal_length=0.000
for k in sepal_length:
	var_sepal_length += pow((k - mean_sepal_length),2)
var_sepal_length/=len(sepal_length)

print "Mean, Median, Variance, Std.Dev, Range"
print mean_sepal_length, sepal_length_median, var_sepal_length, pow(var_sepal_length, 0.5), max(sepal_length) - min(sepal_length)


print statistics.mean(sepal_length)
print statistics.variance(sepal_length)
print statistics.stdev(sepal_length)


#copy the same for all features
#quartiles - 25th, 50th(is median) and 75th	
lower_half_sepal_length = sepal_length[0:len(sepal_length)/2]
upper_half_sepal_length = sepal_length[len(sepal_length)/2:]

print lower_half_sepal_length[len(lower_half_sepal_length)/2]
print upper_half_sepal_length[len(upper_half_sepal_length)/2]


