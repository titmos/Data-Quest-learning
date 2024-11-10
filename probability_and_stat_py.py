
#Sampling Error

import pandas as pd
wnba = pd.read_csv('wnba.csv')
print(wnba.shape) #Find the number of rows and columns
#print(wnba.head()) #Check  first five rows
#print(wnba.tail()) #Check last five rows
parameter = wnba['Games Played'].max() #This is a population metric
sample = wnba.sample(n = 30, random_state = 1)#Returns a random sample of items from an axis of object.We can use random_state for reproducibility.
statistic = sample['Games Played'].max() #This is the sample metric
sampling_error = parameter - statistic

#Simple Random Sampling
import pandas as pd
import matplotlib.pyplot as plt

wnba = pd.read_csv('wnba.csv')

population_mean = wnba['PTS'].mean()
sample_means = []
#Using simple random sampling, take 100 samples of 10 values each from our WNBA data set, and for each sample, measure the average points scored by a player during the 2016-2017 season.
for item in range (100):
    sample = wnba['PTS'].sample(n = 10, random_state = item)
    sample_means.append(sample.mean())
    
#Display the discrepancy between the parameter of interest (the mean of the PTS column) and the statistics obtained in the previous step
plt.scatter(range(1, 101), sample_means)
plt.axhline(population_mean)
plt.show() #On the scatter plot from the previous screen, we can see that the sample means vary a lot around the population mean.
# there's a good chance we get a sample that isn't representative of the population
