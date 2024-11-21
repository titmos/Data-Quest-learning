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

#Sorting Frequency Distribution Tables
age_ascending = wnba['Age'].value_counts().sort_index()

age_descending = wnba['Age'].value_counts().sort_index(ascending = False)
#wnba['Height'].value_counts() returns a Series object with the measures of height as indices. This allows us to sort the table by index using the Series.sort_index() method:
print(wnba['Height'].value_counts().sort_index())
#We can also sort the table by index in a descending order using wnba['Height'].value_counts().sort_index(ascending = False)


#Proportions and percentages
#It's slightly faster though to use Series.value_counts() with the normalize parameter set to True 
print(wnba['Pos'].value_counts(normalize = True))
#To find percentages, we just have to multiply the proportions by 100:

#Proportions and percentages

print(wnba['Age'].value_counts(normalize = True))
draft = wnba['Age'].value_counts(normalize = True).sort_index()#

proportion_25 = draft[25] #proportion of players are 25 years old
percentage_30 = draft[30] * 100 #percentage of players are 30 years old
#Note the need to sort the index for the loc to be effective, Hence the need to use sort_index
percentage_over_30 = draft.loc[30:].sum() * 100 #What percentage of players are 30 years or older
percentage_below_23 = draft.loc[:23].sum() * 100 #percentage of players are 23 years or younger


#Percentiles and Percentile Ranks 
#using scipy
from scipy.stats import percentileofscore
#percentage of players are 23 years or younger?"
print(percentileofscore(a=wnba['Age'], score=23, kind='weak'))

#We need to use kind = 'weak' to indicate that we want to find the percentage of values that are equal to or less than the value we specify in the score parameter
#What percentage of players played half the number of games or less 
percentile_rank_half_less = percentileofscore(a=wnba['Games Played'], score=17, kind='weak')
#What percentage of players played more than half the number of games
percentage_half_more = 100 - percentileofscore(a=wnba['Games Played'], score=17, kind='weak')


# Finding Percentiles with pandas
#we can use the Series.describe() method, which returns by default the 25th, the 50th, and the 75th percentiles
print(wnba['Age'].describe())#we can add this to show only the percentiles.iloc[4:7])
print(wnba['Age'].describe().iloc[4:7])
#We may be interested to find the percentiles for percentages other than 25%, 50%, or 75%. For that, we can use the percentiles parameter of Series.describe()
print(wnba['Age'].describe(percentiles=[0.1, 0.15, 0.33, 0.5, 0.592, 0.85, 0.99]).iloc[3:])

#Note Percentiles don't have a single standard definition, so don't be surprised if you get very similar (but not identical) values if you use different functions (especially if the functions come from different libraries).
percentiles = wnba['Age'].describe(percentiles = [0.5, 0.75, 0.95])
#We can use the methods below to extract each percentile
age_upper_quartile = percentiles['75%']
age_middle_quartile = percentiles['50%']
age_95th_percentile =percentiles['95%']
question1 = True
question2 = False
question3 = True


#Grouped Frequency Distribution Tables

grouped_freq_table = wnba['PTS'].value_counts(bins=10, normalize = True).sort_index(ascending = False) * 100
#When we generate grouped frequency distribution tables, there's an inevitable information loss
#Information Loss
#because we grouped the values, we lost more granular information 
#To get back this granular information, we can increase the number of class intervals. However, if we do that, we end up again with a table that's lengthy and very difficult to analyze.

#On the other side, if we decrease the number of class intervals, we lose even more information
#We can conclude there is a trade-off between the information in a table, and how comprehensible the table is

#As a rule of thumb, 10 is a good number of class intervals to choose because it offers a good balance between information and comprehensibility.
