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
#We can solve this problem by increasing the sample size. As we increase the sample size, the sample means vary less around the population mean, and the chances of getting an unrepresentative sample decrease.


#Stratified Sampling
wnba['Pts_per_game'] = wnba['PTS'] / wnba['Games Played'] #creates a new column for points/game
print(wnba['Pos'].unique()) #to view the strata in the Pos column

# Stratifying the position data in five strata using boolean 
stratum_G = wnba[wnba.Pos == 'G']
stratum_F = wnba[wnba.Pos == 'F']
stratum_C = wnba[wnba.Pos == 'C']
stratum_GF = wnba[wnba.Pos == 'G/F']
stratum_FC = wnba[wnba.Pos == 'F/C']

points_per_position = {} #Empty dictionary to save sample mean per stratum
#Loop to iterate the stratum(done by boolean filter)
for stratum, position in [(stratum_G, 'G'), (stratum_F, 'F'), (stratum_C, 'C'), (stratum_GF, 'G/F'), (stratum_FC, 'F/C')]:
    sample = stratum['Pts_per_game'].sample(n = 10, random_state = 0)## simple random sampling on each stratum
    points_per_position[position] = sample.mean() #appends mean of each strata tp the dictionary by its position 
    
position_most_points = max(points_per_position, key = points_per_position.get)#To get the strata with Max points 


#Proportional Stratified Sampling

print(round(wnba['Games Played'].value_counts(bins = 3))) #Check Proportionality

#stratifying the DF by proportion
stratum_1 = wnba[wnba['Games Played'] <= 12]
stratum_2 = wnba[(wnba['Games Played'] > 12) & (wnba['Games Played'] <= 22)]
stratum_3 = wnba[wnba['Games Played'] > 22]

proportional_sampling_means = []
#performing the sampling based on the strata
for item in range(100):
    sample_under_12 = stratum_1['PTS'].sample(1, random_state = item)
    sample_btw_13_22 = stratum_2['PTS'].sample(2, random_state = item)
    sample_over_23 = stratum_3['PTS'].sample(7, random_state = item)
    #join the stratas and find the mean
    final_sample = pd.concat([sample_under_12, sample_btw_13_22, sample_over_23])
    proportional_sampling_means.append(final_sample.mean())

plt.scatter(range(1,101), proportional_sampling_means)
plt.axhline(wnba['PTS'].mean())
plt.show()


# Cluster Sampling

print(wnba['Team'].unique()) #View disitinct teams in the Dataset

clusters = pd.Series(wnba['Team'].unique()).sample(4, random_state = 0)#Randomly picks 4 clusters from the unique teams available

sample = pd.DataFrame()#creates an empty DataFrame

for cluster in clusters:
    cluster_data = wnba[wnba['Team'] == cluster]
    sample = pd.concat([sample, cluster_data]) #appends each cluster data to the DF 
    #the above automatically separates the clusters from the main wnba DataFrame

#the construct blow computes the Sampling error (parameter - statistic)    
sampling_error_height = wnba['Height'].mean() - sample['Height'].mean()
sampling_error_age = wnba['Age'].mean() - sample['Age'].mean()
sampling_error_BMI = wnba['BMI'].mean() - sample['BMI'].mean()
sampling_error_points = wnba['PTS'].mean() - sample['PTS'].mean()


#Frequency Distribution Tables
#To generate a frequency distribution table using Python, we can use the Series.value_counts() method. 
freq_distro_pos = wnba['Pos'].value_counts()

freq_distro_height = wnba['Height'].value_counts()

print(wnba['Height'].mean(), wnba['Height'].min(), wnba['Height'].max())
