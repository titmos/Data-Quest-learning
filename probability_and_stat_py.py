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


# Readability for Grouped Frequency Tables
# we can define the intervals ourselves. For the table above, we can define six intervals of 100 points each, and then count how many values fit in each interval. 
#To achieve this, let's look at one way to code the intervals. We start with creating the intervals using the pd.interval_range() function
intervals = pd.interval_range(start=0, end=600, freq=100)

#Next, we pass the intervals variable to the bins parameter
gr_freq_table = wnba["PTS"].value_counts(bins = intervals).sort_index()
print(gr_freq_table)
print(gr_freq_table.sum())
#Note that we're not restricted by the minimum and maximum values of a variable when we define intervals. The minimum number of points is 2, and the maximum is 584, but our intervals range from 1 to 600.

#Visualizing Frequency Distributions
#Bar Plots
#For variables measured on a nominal or an ordinal scale it's common to use a bar plot to visualize their distribution
import matplotlib.pyplot as plt
#The Series.plot.bar() method generates a vertical bar plot with the frequencies on the y-axis, and the unique values on the x-axis. To generate a horizontal bar plot, we can use the Series.plot.barh() method

wnba['Pos'].value_counts().plot.bar ()
plt.show()

import matplotlib.pyplot as plt
print(wnba['Exp_ordinal'].value_counts())#Generate a frequency table for the Exp_ordinal variable.
#Sort the table by unique labels in an ascending logical order 
print(wnba['Exp_ordinal'].value_counts().iloc[[3,0,2,1,4]])
#Generate a bar plot using the Series.plot.bar() method
wnba['Exp_ordinal'].value_counts().iloc[[3,0,2,1,4]].plot.bar ()
plt.show()

#Horizontal Bar Plots
import matplotlib.pyplot as plt

wnba['Exp_ordinal'].value_counts().iloc[[3,0,2,1,4]].plot.bar(rot=45) #make the tick labels readable using rot
plt.show()

#We can also use a horizontal bar plot to visualize for readability
wnba['Pos'].value_counts().plot.barh(title='Number of players in WNBA by position')
plt.show()


#Pie Charts
#We can generate pie charts using the Series.plot.pie() method
import matplotlib.pyplot as plt

wnba['Exp_ordinal'].value_counts().plot.pie()
plt.show()

#Customizing a Pie Chart
#If we want to give the pie chart the proper shape, we need to specify equal values for height and width in the figsize parameter

#To remove the lable we use plt.ylabel('')
# To show the proportions or Percentages on the chart we use the autopct parameter and string formating '%.1f%%'
#% initailises string fmt, .1 for decimal precision, f number as fixed point, % indicates the use of percentages, % uses percentage symbol
# the percentages were automatically determined under the hood, which means we don't have to transform to percentages ourselves using Series.value_counts(normalize = True) * 100
#https://docs.python.org/3/library/string.html#format-specification-mini-language
#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html autopct

#Trial 
import matplotlib.pyplot as plt
wnba['Exp_ordinal'].value_counts().plot.pie(figsize=(6,6), autopct='%.2f%%', title = 'Percentage of players in WNBA by level of experience')
plt.ylabel('')
plt.show()


#Histograms
import matplotlib.pyplot as plt
print(wnba['PTS'].describe())
#Because of the special properties of variables measured on interval and ratio scales, we can describe distributions in more elaborate ways.

wnba['PTS'].plot.hist()
plt.show()

#The Statistics Behind Histograms
print(wnba['PTS'].value_counts(bins=10).sort_index())
'''
Each bar in the histogram corresponds to one class interval. To show this is true, we'll generate below the same histogram as in the previous screen, but this time:

We'll add the values of the x-ticks manually using the xticks parameter.
The values will be the limits of each class interval.
We use the arange() function from numpy to generate the values and avoid spending time with typing all the values ourselves.
We start at 2, not at 1.417, because this is the actual minimum value of the first class interval (we discussed about this in more detail in the previous lesson).
We'll add a grid line using the grid parameter to demarcate clearly each bar.
We'll rotate the tick labels of the x-axis using the rot parameter for better readability.
'''
import matplotlib.pyplot as plt
import numpy as np

wnba['PTS'].plot.hist(grid=True, xticks=np.arange(2,585,58.2), rot=30)
plt.show()


#Binning for Histograms
#To modify the number of class intervals used for a histogram, we can use the bins parameter of Series.plot.hist()
#Also, we'll often want to avoid letting pandas work out the intervals, and use instead intervals that we think make more sense. 
#We start with specifying the range of the entire distribution using the range parameter of Series.plot.hist()
import matplotlib.pyplot as plt

wnba['PTS'].plot.hist(range=(1,600), bins=3)
plt.show()


import matplotlib.pyplot as plt

wnba['Games Played'].plot.hist(range=(1,32), bins=8, title = 'The distribution of players by games played') #Each bin must cover an interval of 4 games. The first bin must start at 1, the last bin must end at 32.
plt.xlabel('Games played')
plt.show()


#Skewed Distributions
#If the tail points to the left, then the distribution is said to be left skewed. When it points to the left, the tail points at the same time in the direction of negative numbers, and for this reason the distribution is sometimes also called negatively skewed.
#If the tail points to the right, then the distribution is right skewed. The distribution is sometimes also said to be positively skewed because the tail points in the direction of positive numbers.

import matplotlib.pyplot as plt

wnba['AST'].plot.hist(range=(1,32), bins=8)
plt.show()
print(wnba['FT%'].describe())
wnba['FT%'].plot.hist(range=(0,100), bins=8)
plt.show()
assists_distro = 'right skewed'
ft_percent_distro = 'left skewed'


# Symmetrical Distributions

#divide the histogram in two halves that are mirror images of one another.
#This pattern is specific to what we call a normal distribution (also called Gaussian distribution)
#Another common symmetrical distribution is one where the values are distributed uniformly across the entire range. This pattern is specific to a uniform distribution.
#When we say that the distribution above resembles closely a normal distribution, we mean that most values pile up somewhere close to the middle and decrease in frequency more or less gradually toward both ends of the histogram.
#A similar reasoning applies to skewed distributions. We don't see very often clear-cut skewed distributions, and we use the left and right skewed distributions as baselines for comparison.
import matplotlib.pyplot as plt

wnba['Age'].plot.hist(range=(21,36), bins=12, title = 'Age')
plt.show()
print(wnba['Height'].describe())
wnba['Height'].plot.hist(range=(165,206), bins=12, title = 'Height')
plt.show()
wnba['MIN'].plot.hist(range=(12,1018), bins=12, title = 'MIN')
plt.show()
normal_distribution = 'Age'


#Comparing Frequency Distributions
#graphs we can use to compare multiple frequency distributions at once
#Segment the players in the data set by level of experience.
rookies = wnba[wnba['Exp_ordinal'] == 'Rookie']
little_xp = wnba[wnba['Exp_ordinal'] == 'Little experience']
experienced = wnba[wnba['Exp_ordinal'] == 'Experienced']
very_xp = wnba[wnba['Exp_ordinal'] == 'Very experienced']
veterans =  wnba[wnba['Exp_ordinal'] == 'Veteran']

#For each segment, generate a frequency distribution table for the Pos variable.
rookie_distro = rookies['Pos'].value_counts()
little_xp_distro = little_xp['Pos'].value_counts()
experienced_distro = experienced['Pos'].value_counts()
very_xp_distro = very_xp['Pos'].value_counts()
veteran_distro = veterans['Pos'].value_counts()

#Analyze the frequency distributions comparatively.
print(rookie_distro, '\n\n', little_xp_distro, '\n\n', experienced_distro, '\n\n',
      very_xp_distro, '\n\n', veteran_distro)


#Grouped Bar Plots
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='Exp_ordinal', hue='Pos', data=wnba)
#x — specifies as a string the name of the column we want on the x-axis
#hue — specifies as a string the name of the column we want the bar plots generated for.
#data - specifies the name of the variable which stores the data set. 
plt.show()
plt.figure().set_figwidth(8)

#Grouped Bar Plots activity
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure().set_figwidth(8) #to prevent the x-axis labels from overlapping.
sns.countplot(x='Exp_ordinal', hue='Pos', data=wnba, order = ['Rookie', 'Little experience', 'Experienced', 'Very experienced', 'Veteran' ], hue_order = ['C', 'F', 'F/C', 'G', 'G/F']) 
#Using the order parameter of sns.countplot(), order the values on the x-axis in ascending order.
#Using the hue_order parameter, order the bars of each bar plot in ascending alphabetic order.
sns.set_theme()
plt.show()
