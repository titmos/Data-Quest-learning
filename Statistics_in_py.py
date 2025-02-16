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


#Challenge: Do Older Players Play Less?

#Let's hypothesize that older players generally play less than this average of 497 minutes, while younger players generally play more.
print(wnba['MIN'].mean())
print(wnba['Age'].mean())

wnba['age_mean_relative'] = wnba['Age'].apply(lambda x: 'old' if x >= 27 else 'young')
#We generate an ordinal variable for the age classes using lambda function
wnba['min_mean_relative'] = wnba['MIN'].apply(lambda x: 'average or above' if x >= 497 else
                                           'below average')
#We generate an ordinal variable for the min played class using lambda function
sns.countplot(x='age_mean_relative', hue='min_mean_relative', data=wnba)
sns.set_theme()
plt.show()
result = 'rejection' # to show if the hypothesis is true or not

#order = ['Rookie', 'Little experience', 'Experienced', 'Very experienced', 'Veteran' ], hue_order = ['C', 'F', 'F/C', 'G', 'G/F']


#Comparing Histograms
#The easiest way to compare two histograms is to superimpose one on top of the other.
wnba[wnba.Age >= 27]['MIN'].plot.hist(histtype='step', label='Old', legend=True)
wnba[wnba.Age < 27]['MIN'].plot.hist(histtype='step', label='Young', legend=True)
#We use the histtype parameter to visualize the differences in the two histogram (making it without fill)
plt.axvline(x = 497, label = 'Average') #This inserts a line on the specified point and shows the legend
plt.legend() #Display the label by running
plt.show()


#Kernel Density Estimate Plots
#We can smooth out our two histograms above for old and young players using the Series.plot.kde() function.
wnba[wnba.Age >= 27]['MIN'].plot.kde(label='Old', legend=True)
wnba[wnba.Age < 27]['MIN'].plot.kde(label='Young', legend=True)

plt.legend()
plt.show()
#Each of the smoothed histograms above is called a kernel density estimate plot or, shorter, kernel density plot.
#Unlike histograms, kernel density plots display densities on the y-axis instead of frequencies. The density values are actually probability values — which we'll be able to understand more about after the probability courses.

#More instances 
#Let's say we're interested in analyzing the distribution of player height as a function of player position. In other words, we want to figure out, for instance, whether centers are generally taller than forwards, whether forwards are generally shorter than guards, and so on. In the code below, we'll segment the data set by player position, and for each segment we'll generate a kernel density plot for the distribution of the Height variable:

wnba[wnba.Pos == 'F']['Height'].plot.kde(label='F', legend=True)
wnba[wnba.Pos == 'C']['Height'].plot.kde(label='C', legend=True)
wnba[wnba.Pos == 'G']['Height'].plot.kde(label='G', legend=True)
wnba[wnba.Pos == 'G/F']['Height'].plot.kde(label='G/F', legend=True)
wnba[wnba.Pos == 'F/C']['Height'].plot.kde(label='F/C', legend=True)
plt.show()


#Strip Plots
#This is one alternative we can use to visualize the distribution of heights as a function of player position:
#To generate the strip plot, we can use the sns.stripplot() function from the seaborn module. 
#sns.set_theme()
#sns.stripplot(x='Pos', y='Height', data=wnba, hue='Pos', jitter=False)

#Using strip plots, examine the distribution of player weight (not height) as a function of player position. 
sns.set_theme()
sns.stripplot(x='Pos', y='Weight', data=wnba, hue='Pos', jitter= True)
plt.show()


#Box plots
#Besides strip plots, there's another kind of graph we can use to display many distributions at once and make sure everything is still readable. Below, we use this kind of graph to plot again the distribution of player height as a function of player position:
#Each individual plot above shows a distribution. 
sns.set_theme()
sns.boxplot(x='Pos', y='Weight', hue='Pos', data=wnba)
plt.show()


#Outliers
#A value that is much lower or much larger than the rest of the values in a distribution is called an outlier.
#It's larger than the upper quartile by 1.5 times the difference between the upper quartile and the lower quartile (the difference is also called the interquartile range).
#It's lower than the lower quartile by 1.5 times the difference between the upper quartile and the lower quartile
sns.set_theme()
print(wnba['Games Played'].describe())

iqr = wnba['Games Played'].describe()['75%'] - wnba['Games Played'].describe()['25%']#interquartile range

lower_bound = 22 - (1.5 * iqr) 
upper_bound = 29 + (1.5 * iqr)
outliers_low = sum(wnba['Games Played'] < lower_bound) 
outliers_high = sum(wnba['Games Played'] > upper_bound)

sns.boxplot(wnba['Games Played'])
plt.show()




STATISTICS IN PY 2 (MEASURES)
#The Mean as a Balance Point
import numpy as np

equal_distances = 0

for x in range(0, 5000):#Generate 5000 different distributions
    np.random.seed(x)# the function ensures that the sequence of random numbers generated remains the same across multiple runs
    distribution = np.random.randint(0, 1000, 10) #Generate randomly a distribution of integers of 10 values in a range of 0 to 1000
    mean = sum(distribution) / len(distribution)
    
    above = []#measure the total distances above and below the mean
    below = []
    for i in distribution:
        if i == mean:
            continue
        if i < mean:
            below.append(mean - i)
        if i > mean:
            above.append(i - mean)
    sum_above = round(sum(above), 1)
    sum_below = round(sum(below), 1p)
    if (sum_above == sum_below):#measure the total distances above and below the mean
        equal_distances += 1

#An Alternative Definition of the mean 
distribution_1 = [42, 24, 32, 11]
distribution_2 = [102, 32, 74, 15, 38, 45, 22]
distribution_3 = [3, 12, 7, 2, 15, 1, 21]

def mean(list):#sums a sequence and returns the mean
    sum_list = 0
    for x in list:
        sum_list += x
    return sum_list/len(list)

mean_1 = mean(distribution_1)
mean_2 = mean(distribution_2)
mean_3 = mean(distribution_3)
    

#Estimating the Population Mean
#sampling_error = parameter - statistic

parameter = houses['SalePrice'].mean()
#initializations for the loop
n = 5
sampling_errors = []
sample_sizes = []
for i in range(101):
    sample = houses['SalePrice'].sample(n, random_state = i)
    sample_sizes.append(n)
    sampling_errors.append(parameter - sample.mean())
    n += 29
#print(sample_sizes, sampling_errors)
import matplotlib.pyplot as plt
plt.scatter(sample_sizes, sampling_errors)
plt.axhline(0)
plt.axvline(2930)
plt.xlabel('Sample size')
plt.ylabel('Sampling error')
plt.show()

#Estimates from Low-Sized Samples
#From the previous activity, we observed that we can also get better estimates for the population mean from lower sample sizes where sampling error is low  
#proof
means = []
for i in range(10000):
    sample = houses['SalePrice'].sample(3, random_state=i)
    means.append(sample.mean())

plt.hist(means)
plt.axvline(houses['SalePrice'].mean())
plt.show()
#we see thatt most sample mean cluster around the population mean from the plot.
#ACTIVITY
#Estimates from Low-Sized Samples

means = []
for i in range(10000):
    sample = houses['SalePrice'].sample(100, random_state=i)
    means.append(sample.mean())

plt.hist(means)
plt.axvline(houses['SalePrice'].mean())#vertical line to mark the population mean
plt.xlim(0, 500000) #Set the range of the x-axis to (0,500000)
plt.xlabel('Sample mean')
plt.ylabel('Frequency')
plt.show()
# we observed that with a sample size of 100 the sample means vary much less around the population mean than in the case of a sample size of 3.
#Generally, as we increase the sample size, there will be less and less variability around the population mean. If there's less variability, there are less chances to get a poor estimate for the population mean — the worst sample means we can get are fairly close to the population mean, which is good.

#The Sample Mean as an Unbiased Estimator; When a statistic is on average equal to the parameter it estimates.
#This also holds true if we sample with replacement,we sample one value, and then we put it back in the population. 
population = [3, 7, 2]
samples = [[3, 7], [3, 2],
           [7, 2], [7, 3],
           [2, 3], [2, 7]
          ]

sample_means = []

for sample in samples:
    sample_means.append(sum(sample) / len(sample))
    
population_mean = sum(population) / len(population)
mean_of_sample_means = sum(sample_means) / len(sample_means)

unbiased = (population_mean == mean_of_sample_means)


#Different Weights
houses_per_year['sum_per_year'] = houses_per_year['Mean Price'] * houses_per_year['Houses Sold']
all_sums_together = houses_per_year['sum_per_year'].sum()
total_n_houses = houses_per_year['Houses Sold'].sum()
weighted_mean = round(all_sums_together / total_n_houses, 10)
mean_original = round(houses['SalePrice'].mean(), 10)
difference = mean_original - weighted_mean


#Write a function that computes the weighted mean for any array of numbers
def w_function(x, y):
    sum_mean_weight = 0
    weight_sum = 0
    for i, j in zip(x, y):
        sum_mean_weight += (i * j)
        weight_sum += j
    return sum_mean_weight / weight_sum
#Use the function you wrote to compute the weighted mean
weighted_mean_function = w_function(houses_per_year['Mean Price'], houses_per_year['Houses Sold'])
import numpy as np
#compute the weighted mean using numpy
weighted_mean_numpy = np.average(houses_per_year['Mean Price'], weights = houses_per_year['Houses Sold'])

#To check if the value from the function and that from the numpy are same
equal = weighted_mean_function == weighted_mean_numpy


# Distributions with Even Number of Values
#Find the median value of the TotRms AbvGrd column.
for_use = houses['TotRms AbvGrd'].copy().replace({'10 or more': 10} ).astype(int). sort_values(axis = 0, ascending=True, ignore_index = True)

if len(for_use)%2 == 0:
    median = for_use.iloc[int(len(for_use)/2)] + for_use.iloc[int(len(for_use)/2)- 1]
    median = median/2
print(median)

#The Median as a Resistant Statistic
#When we compute the mean, we account equally for each value in the distribution — we sum up all the values in the distribution and then divide the total by the number of values we added
#When we compute the median,we only consider the middle value (or the middle two values).

# the mean takes into account every value in the distribution, while the median considers only the middle value. Because the median is so resistant to changes in the data, it's classified as a resistant or robust statistic.
import matplotlib.pyplot as plt
houses['Lot Area'].plot.box()
plt.show()
houses['SalePrice'].plot.box()
plt.show()

lotarea_difference = houses['Lot Area'].mean() - houses['Lot Area'].median()
saleprice_difference = houses['SalePrice'].mean() - houses['SalePrice'].median()
#This property makes the median ideal for finding reasonable averages for distributions containing outliers. 

#The Median for Ordinal Scales
#Because words like "fair" or "average" are coded with numbers, it becomes mathematically possible to compute the mean.
import matplotlib.pyplot as plt
print(houses['Overall Cond'].value_counts().sort_values(ignore_index=True))
#Find the mean and the median of the Overall Cond variable.
mean = houses['Overall Cond'].mean()
median = houses['Overall Cond'].median()
#Plot a histogram to visualize the distribution of the Overall Cond variable.
houses['Overall Cond'].plot.hist()
plt.show()

more_representative = 'mean'


#The Mode for Ordinal Variables

# function that takes in an array of values (including strings) and returns the mode of that array
def mode_fn(x):
    new_dic = {}
    for i in x:
        if i in new_dic:
            new_dic[i] += 1
        else: new_dic[i] = 1
    return max(new_dic, key = new_dic.get)

mode_function = mode_fn(houses['Land Slope'])
mode_method = houses['Land Slope'].mode()
same = mode_function == mode_method

#The Mode for Nominal Variables

# The function we wrote (you can copy-paste yours from the previous screen)
def mode(array):
    counts = {}
    
    for value in array:
        if value in counts:
            counts[value] += 1
        else:
            counts[value] = 1
    
    return (max(counts, key=counts.get), counts)
mode, value_counts = mode(houses['Roof Style']) #A Python function can return multiple values.
new_value_counts = houses['Roof Style'].value_counts()


#The Mode for Discrete Variables

print(houses['Bedroom AbvGr'].value_counts().sort_index())
bedroom_variable = 'discrete' #specifies the type of the Variables
bedroom_mode = houses['Bedroom AbvGr'].mode()

print(houses['SalePrice'].value_counts().sort_index())
price_variable = 'continuous'

#grouped frequency table for the SalePrice
intervals = pd.interval_range(start=0, end=800000, freq=100000)
gr_freq_table = pd.Series([0, 0, 0, 0, 0, 0, 0, 0], index=intervals)

for value in houses['SalePrice']:
    for interval in intervals:
        if value in interval:
            gr_freq_table.loc[interval] += 1
            break

maxi = max(gr_freq_table.values)

print(gr_freq_table)
mode = 150000
mean = houses['SalePrice'].mean()
median = houses['SalePrice'].median()
sentence_1 = True

sentence_2 = True


#Skewed Distributions
#When we plot a histogram or a kernel density plot to visualize the shape of a distribution, the mode will always be the peak of the Distribution
''' This distribution is clearly right skewed. Generally, the location of the mode, median and mean is predictable for a right-skewed distribution:

Most values are concentrated in the left body of the distribution where they will form a peak — this is where the mode will be.
Remember that the median divides a distribution in two halves of equal length. For this reason, the median is usually positioned slightly right from the peak (the mode) for a right-skewed distribution.
The mean takes into account each value in the distribution, and it will be affected by the outliers in the right tail. This will generally pull the mean to the right of the median.
So in a right-skewed distribution, the mean will usually be to the right of the median, and the median will be to the right of the mode. 
For a left-skewed distribution, the direction is simply reversed: the mean is positioned to the left of the median, and the median to the left of the mode. '''

distribution_1 = {'mean': 3021 , 'median': 3001, 'mode': 2947}
distribution_2 = {'median': 924 , 'mode': 832, 'mean': 962}
distribution_3 = {'mode': 202, 'mean': 143, 'median': 199}
shape_1 = 'right skew'
shape_2 = 'right skew'
shape_3 = 'left skew'


#Symmetrical Distributions
'''
The median divides the distribution in two equal halves. As a consequence, the median will always be at the center of a perfectly symmetrical distribution because only a line drawn at the center can divide the distribution in two equal halves.
 The mean is not pulled neither to the left, nor to the right, and stays instead in the center, at the same location as the median. The mean and the median are always equal for any perfectly symmetrical distribution.
It's possible to have a symmetrical distribution with more than one peak, which means that the mode won't be at the center
A uniform distribution doesn't have any peaks, which means it doesn't have any mode:


'''
houses['Mo Sold'].plot.kde(xlim = [1, 12])
plt.axvline(houses['Mo Sold'].mode()[0], color='green', label='Mode')
plt.axvline(houses['Mo Sold'].median(), color='orange', label='Median')
plt.axvline(houses['Mo Sold'].mean(), color='black', label='Mean')
plt.legend()
plt.show()



#Measures of Variability
#Range

import pandas as pd
houses = pd.read_table('AmesHousing_1.txt')

def range(x): #a function that takes in an array of numerical values and returns the range of that array
    return max(x) - min(x)
range_by_year = {} #Initialise an empty dictionary

for year in houses['Yr Sold'].unique(): #find the range per year
    q = houses['SalePrice'][houses['Yr Sold'] == year]
    range_by_year[year] = range(q)
print(range_by_year)

one = False
two = True
    

#The Average Distance
#problem range has is it considers only the two extreme values, and this makes it extremely sensitive to outliers
'''
To take into account each value when measuring variability we could:

#Take a reference value, and measure the distance of each value in the distribution from that reference value.
We can take the mean of the distribution as a reference value.
Then, we measure the distance between each value in the distribution and the mean.
#Find the mean of the distances.
We first need to sum up all the distances.
Then we need to divide the total by the number of distances.
'''

C = [1, 1, 1, 1, 1, 1, 1, 1, 1, 21]
#function that takes in a numerical array and returns the average distance
def avg_dis(x):
    mn_arr = sum(x)//len(x)
    em_lis = []
    for i in x:
        em_lis.append(i - mn_arr)
    return sum(em_lis)/len(em_lis)
avg_distance = avg_dis(C)
print(avg_distance)

#Mean Absolute Deviation
#We'll update the formula used previously to reflect the fact the we're summing up the absolute distances 
''' We call this measure of variability mean absolute distance. In statistical jargon, however, the distance of a value from the mean is called deviation. So the mean absolute distance is more commonly known as mean absolute deviation or average absolute deviation.'''
C = [1, 1, 1, 1, 1, 1, 1, 1, 1, 21]

#function that takes in a numerical array and returns the  mean absolute deviation
def mad_fn(x):
    mn_arr = sum(x)//len(x)
    em_lis = []
    for i in x:
        em_lis.append(abs(i - mn_arr))
    return sum(em_lis)/len(em_lis)
mad = mad_fn(C)
print(mad)

#Variance
#Another way to solve this problem is to square each distance and then find the mean of all the squared distances
#This measure of variability is sometimes called mean squared distance or mean squared deviation 
#However, it's more commonly known as variance
C = [1, 1, 1, 1, 1, 1, 1, 1, 1, 21]

#function that takes in a numerical array and returns the variance of that array
def var_dis(x):
    mn_arr = sum(x)//len(x)
    em_lis = []
    for i in x:
        em_lis.append((i - mn_arr)**2)
    return sum(em_lis)/len(em_lis)
variance_C = var_dis(C)
print(variance_C)

#Standard Deviation
from math import sqrt
C = [1, 1, 1, 1, 1, 1, 1, 1, 1, 21]
#The square root of variance is called standard deviation 
#function that takes in a numerical array and returns the standard deviation of that array
def std_dev(x):
    mn_arr = sum(x)//len(x)
    em_lis = []
    for i in x:
        em_lis.append((i - mn_arr)**2)
    return sqrt(sum(em_lis)/len(em_lis))
standard_deviation_C = std_dev(C)
print(standard_deviation_C)


# Average Variability Around the Mean
# we'll try to visualize this variability around the mean by:
#Generating a histogram for the distribution of the SalePrice variable.
#Using vertical lines to mark the mean and the average deviations above and below the mean.
mean = houses['SalePrice'].mean()
st_dev = standard_deviation(houses['SalePrice'])
houses['SalePrice'].plot.hist()
plt.axvline(mean, color='Black', label='Mean')
plt.axvline(mean - st_dev, color='Red', label='Below')
plt.axvline(mean + st_dev, color='Violet', label='Above')
plt.legend()
plt.show()


# Average Variability Around the Mean

def standard_deviation(array):
    reference_point = sum(array) / len(array)
    
    distances = []
    for value in array:
        squared_distance = (value - reference_point)**2
        distances.append(squared_distance)
    variance = sum(distances) / len(distances)
    return sqrt(variance)

#print(houses['Yr Sold'].unique())
dict = {}
for year in houses['Yr Sold'].unique():
    year_segment = houses['SalePrice'][houses['Yr Sold']==year]
    st_dev = standard_deviation(year_segment)
    dict[year] = st_dev
    
    #print(year, ': ', st_dev )
greatest_variability = max(dict, key= lambda x: dict[x]) #or try  max(years, key=years.get)
lowest_variability = min(dict, key= lambda x: dict[x]) #or try  min(years, key=years.get)
print(dict)
#print(houses['SalePrice'][houses['Yr Sold']==2009])


#A Measure of Spread
def standard_deviation(array):
    reference_point = sum(array) / len(array)
    
    distances = []
    for value in array:
        squared_distance = (value - reference_point)**2
        distances.append(squared_distance)
    
    variance = sum(distances) / len(distances)
    
    return sqrt(variance)


sample1 = houses['Year Built'].sample(50, random_state=1)
sample2 = houses['Year Built'].sample(50, random_state=2)
bigger_spread = 'sample 2'
st_dev1 = standard_deviation(sample1)
st_dev2 = standard_deviation(sample2)


#The Sample Standard Deviation
def standard_deviation(array):
    reference_point = sum(array) / len(array)
    
    distances = []
    for value in array:
        squared_distance = (value - reference_point)**2
        distances.append(squared_distance)
    
    variance = sum(distances) / len(distances)
    
    return sqrt(variance)

sam_dv = []
for x in range (0, 5001): #consider the data we have for SalePrice a population and sample it 5000 times
    sample = houses['SalePrice'].sample(10, random_state = x)
    sam_dv.append(standard_deviation(sample)) #find stn dev of each sample

plt.hist(sam_dv)#Histogram to visualize the distribution of the 5000 sample standard deviations
plt.axvline(standard_deviation(houses['SalePrice']))
plt.show()
#Notice that most sample standard deviations are clustered below the population standard deviation:
#This suggests that the sample standard deviation usually underestimates the population standard deviation. 

#Bessel's correction.
#This small correction we added to the sample standard deviation (dividing by n − 1 instead of n)
from math import sqrt
import matplotlib.pyplot as plt

def standard_deviation(array):
    reference_point = sum(array) / len(array)
    
    distances = []
    for value in array:
        squared_distance = (value - reference_point)**2
        distances.append(squared_distance)
    
    variance = sum(distances) / (len(distances) - 1)
    
    return sqrt(variance)

st_devs = []

for i in range(5000):
    sample = houses['SalePrice'].sample(10, random_state=i)
    st_dev = standard_deviation(sample)
    st_devs.append(st_dev)

plt.hist(st_devs)
plt.axvline(pop_stdev)
plt.show()

#Standard Notation
sample = houses.sample(100, random_state=1)
from numpy import std, var

pandas_stdev = sample['SalePrice'].std(ddof = 1) #ddof parameter is the degree of freedom used in finding more accurate sample std or var 
numpy_stdev = std(sample['SalePrice'], ddof = 1)
equal_stdevs = pandas_stdev == numpy_stdev

pandas_var = sample['SalePrice'].var(ddof=1) 
numpy_var = var(sample['SalePrice'], ddof=1) 
equal_vars = pandas_var == numpy_var
#we stated that statisticians agree that n−1 is better than n or n−2 for computing the sample standard deviation 
# a statistic is  an unbiased estimator when that statistic is equal on average to the parameter it estimate

#Sample Variance — Unbiased Estimator
from numpy import std, var
population = [0, 3, 6]
samples = [[0, 3], [0, 6],
           [3, 0], [3, 6],
           [6, 0], [6, 3]
          ]
li_std = []
li_var = []
for x in samples:
    li_std.append(std(x, ddof = 1))
    li_var.append(var(x, ddof = 1))
mean_var = sum(li_var)/len(li_var)
equal_var = mean_var == var(population)#If the sample variance is biased in this case, the result should be False
mean_std = sum(li_std)/len(li_std)
equal_stdev = mean_std == std(population) #If the sample variance is biased in this case, the result should be False



#Z-scores
#Individual Values

import pandas as pd
houses = pd.read_table('AmesHousing_1.txt')

#Generate a kernel density plot for the SalePrice variable to find out how far off $220,000 is from the mean.
houses['SalePrice'].plot.kde()
plt.xlim(houses['SalePrice'].min(), houses['SalePrice'].max())
plt.axvline(houses['SalePrice'].mean(), label = 'Mean', color = 'Black')
plt.axvline(houses['SalePrice'].std() + houses['SalePrice'].mean(), label = 'Standard deviation', color = 'Red')
plt.axvline(220000, label = '220000', color = 'Orange')
plt.legend()
plt.show()
very_expensive = False

#Z-scores
min_val = houses['SalePrice'].min()
mean_val = houses['SalePrice'].mean()
max_val = houses['SalePrice'].max() 

from numpy import std, var, mean
def fn(value, x): #function that takes in a value, the array then returns the z-score 
    mn = mean(x)
    st_dev = std(x)
    return (value - mn )/st_dev

min_z = fn(min_val, houses['SalePrice'])
mean_z = fn(mean_val, houses['SalePrice'])
max_z = fn(max_val, houses['SalePrice'])


# Locating Values in Different Distributions
#We're asked by our real estate specialists to analyze historical data on five neighborhoods:
''' Our goal is to find out for which of these neighborhoods a $200,000 house is average-priced. We can solve this task quickly by measuring the z-score for a $200,000 value for each of the five distributions — each of the five neighborhoods has its own distribution of sale prices with its own mean and standard deviation.
so our recommendation should be the neighborhood with the z-score closest to 0
'''
def z_score(value, array, bessel=0):
    mean = np.mean(array)
    st_dev = np.std(array, ddof=bessel)
    distance = value - mean
    z = distance / st_dev
    return z


#segments by location 
names = houses[houses['Neighborhood'] == 'NAmes']
collgcr = houses[houses['Neighborhood'] == 'CollgCr']
oldtown = houses[houses['Neighborhood'] == 'OldTown']
edwards = houses[houses['Neighborhood'] == 'Edwards']
somerst = houses[houses['Neighborhood'] == 'Somerst']

#Clever method by dataquest
# Find the z-score for 200000 for every location
z_by_location = {}
for data, neighborhood in [(names, 'NAmes'), (collgcr, 'CollgCr'),
                     (oldtown, 'OldTown'), (edwards, 'Edwards'),
                     (somerst, 'Somerst')]:
    
    z_by_location[neighborhood] = z_score(200000, data['SalePrice'],
                                          bessel=0)

# Find the location with the z-score closest to 0
print(z_by_location)
best_investment = 'College Creek'


# Transforming Distributions
#Z-scores are often used to transform entire distributions by converting all the values to z-scores

mean = houses['SalePrice'].mean()
st_dev = houses['SalePrice'].std(ddof=0)
#Convert each value to a z-score
houses['z_prices'] = houses['SalePrice'].apply(
    lambda x: ((x - mean) / st_dev)
    )
#Plot side by side the kernel density plot for the original values and the kernel density plot for the z-scores.
plt.figure(figsize=(11, 3.5))
plt.subplot(1, 2, 1)
houses['z_prices'].plot.kde(xlim=(houses['z_prices'].min(),
                                houses['z_prices'].max()
                                )
                        )
plt.subplot(1, 2, 2)
houses['SalePrice'].plot.kde(xlim=(houses['SalePrice'].min(),
                                    houses['SalePrice'].max()
                                    )
                            )
plt.tight_layout() # otherwise the plots will overlay partially
plt.show()

#Notice that the shape of the initial distribution is preserved perfectly in the new distribution of z-scores.

z_mean_price = houses['z_prices'].mean()
z_stdev_price = houses['z_prices'].std(ddof = 0)

#Transform the distribution of the Lot Area variable into a distribution of z-scores.
mean_la = houses['Lot Area'].mean()
st_dev_la = houses['Lot Area'].std(ddof=0)
houses['z_lot_area'] = houses['Lot Area'].apply(
    lambda x: ((x - mean_la) / st_dev_la)
    )
z_mean_area = houses['z_lot_area'].mean()
z_stdev_area = houses['z_lot_area'].std(ddof=0)
#for every distribution of z-scores, the mean is always 0 and the standard deviation is always 1. We got slightly different values in the previous exercise because of small rounding errors. 


#The Standard Distribution
#for every distribution of z-scores, the mean is always 0 and the standard deviation is always 1.
import numpy as np

population = [0, 8, 0, 8]

mean_pop = np.mean(population)
stdev_pop = np.std(population, ddof = 0)

standardized_pop = []
for value in population:
    z = (value - mean_pop) / stdev_pop
    standardized_pop.append(z)
mean_z = np.mean(standardized_pop)
stdev_z = np.std(standardized_pop, ddof = 0)


#Standardizing Samples
import numpy as np
sample = [0, 8, 0, 8]

x_bar = np.mean(sample)
s = np.std(sample, ddof=1)

standardized_sample = []
for value in sample:
    z = (value - x_bar) / s
    standardized_sample.append(z)
mean_st_sample = np.mean(standardized_sample)# mean of standardized_sample is 0
stdev_sample = np.std(standardized_sample, ddof = 1) #When we standardize a sample, the resulting distribution of z-scores is itself a sample. 


#Using Standardization for Comparisons
#Standardizing distributions can prove very useful when we need to compare values coming from different systems of measurement. 
'''
One thing the two systems have in common is that the index is directly proportional with the house quality — a lower index means lower quality, and a higher index means higher quality. To compare these indices coming from different measurement systems, we can:

Standardize each distribution of index values (transform each index to a z-score).
Then compare the z-scores.

Average houses will have z-scores around 0 in both distributions, good-quality houses will have z-scores significantly greater than 0, and low-quality houses will have z-scores significantly lower than 0
'''
mean_1 = houses['index_1'].mean()
st_dev_1 = houses['index_1'].std(ddof=0)
houses['z_1'] = houses['index_1'].apply(lambda x: ((x - mean_1) / st_dev_1))
mean_2 = houses['index_2'].mean()
st_dev_2 = houses['index_2'].std(ddof=0)
houses['z_2'] = houses['index_2'].apply(lambda x: ((x - mean_2) / st_dev_2))
print(houses[['z_1', 'z_2']].head(2))

better = 'first'

#Converting Back from Z-scores
#Z-scores may not be always straightforward to work with or communicate to non-technical audiences, but fortunately we can convert them to other values that are more intuitive
#For any standardized distribution we can also convert the z-scores back to the original values. All we have to do is use the initial values for mean and std dev
# However, it's more common in practice to transform the standardized distribution with convenient values for them

mean = 50
st_dev = 10
houses['distribution'] = houses['z_merged'].apply(lambda z: (z * st_dev + mean))

mean_transformed = houses['distribution'].mean()
stdev_transformed = houses['distribution'].std(ddof = 0)
'''
We found that z-scores are useful for comparing values coming from different distributions with different means and standard deviations.

We also learned to standardize distributions and discussed a few situations where that might be useful. It's worth mentioning that standardization is also useful for some machine learning algorithms that we'll learn about in later courses. Also, z-scores in general are key elements in understanding confidence intervals and hypothesis testing, which we'll discuss in later courses.
'''
