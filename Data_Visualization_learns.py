## Line Graphs and Time Series


#Import the pyplot submodule as plt
import matplotlib.pyplot as plt
month_number = [1, 2, 3, 4, 5, 6, 7]
new_deaths = [213, 2729, 37718, 184064, 143119, 136073, 165003]

#Plot a line graph using plt.plot(), where month_number gives the x-coordinates, and new_deaths gives the y_coordinates
plt.plot(month_number, new_deaths)
#Display the plot using plt.show()
plt.show()


#Customizing a Graph
import matplotlib.pyplot as plt

month_number = [1, 2, 3, 4, 5, 6, 7]
new_deaths = [213, 2729, 37718, 184064, 143119, 136073, 165003]

#plt.title(), add the title 
plt.plot(month_number, new_deaths)
plt.title('New Reported Deaths By Month (Globally)')
plt.xlabel('Month Number')
plt.ylabel('Number Of Deaths')
plt.show()


import pandas as pd
#Read in the WHO_time_series.csv file
who_time_series = pd.read_csv('WHO_time_series.csv')

#Modify the Date_reported column in who_time_series to a datetime data type

who_time_series['Date_reported'] = pd.to_datetime(who_time_series['Date_reported'])
print(who_time_series.head(5))
print('\n')
print(who_time_series.tail(5))
print('\n') 
print(who_time_series.info())


##Types of Line Graphs
#Plot a line graph Argentina.


#Isolate the data for Argentina in a variable named argentina
argentina = who_time_series[who_time_series['Country'] == 'Argentina']

#Plot the Cumulative_cases
plt.plot(argentina['Date_reported'], argentina['Cumulative_cases'])
plt.title('Argentina: Cumulative Reported Cases')
plt.xlabel('Date')
plt.ylabel('Number Of Cases')
plt.show()

#Determine the type of growth by examining the line graph.
#Assign your answer to the variable argentina_graph_type
argentina_graph_type = 'exponential'



'''
we're going to add a legend that shows which color corresponds to which country. In the code below, we first add a label argument to the plt.plot() function, and then we use the plt.legend() function:

When we use plt.plot() the first time, Matplotlib creates a line graph. When we use plt.plot() again, Matplotlib creates another line graph that shares the same x- and y-axis as the first graph. If we want Matplotlib to draw the second line graph separately, we need to close the first graph with the plt.show() function. '''

france = who_time_series[who_time_series['Country'] == 'France']
uk = who_time_series[who_time_series['Country'] == 'The United Kingdom']
italy = who_time_series[who_time_series['Country'] == 'Italy']


#Plot the evolution of cumulative cases for France, the United Kingdom, and Italy on the same graph.
#Add a legend to the graph using plt.legend(). Use the labels France, The UK, and Italy
plt.plot(france['Date_reported'], france['Cumulative_cases'],
         label='France')
plt.plot(uk['Date_reported'], uk['Cumulative_cases'],
         label='The UK')
plt.plot(italy['Date_reported'], italy['Cumulative_cases'],
         label='Italy')
plt.legend()
plt.show()
#Which country has the greatest number of cases at the end of July?
greatest_july = 'The UK'
#Which country has the lowest number of cases at the end of July?
lowest_july = 'France'
#Which country shows the greatest increase during March?
increase_march = 'Italy'



import pandas as pd
import matplotlib.pyplot as plt

bike_sharing = pd.read_csv('day.csv')
#When we pass a series of strings to plt.plot(), Matplotlib doesn't know how to handle that very well
bike_sharing['dteday'] = pd.to_datetime(bike_sharing['dteday'])

plt.plot(bike_sharing['dteday'], bike_sharing['cnt'])
plt.xticks(rotation=45)
plt.show()

#now the dates on the bottom of the graph are overlapping and we can barely read them. To fix this, we can rotate the labels using the plt.xticks() function


#Activity
import pandas as pd
import matplotlib.pyplot as plt

bike_sharing = pd.read_csv('day.csv')
#When we pass a series of strings to plt.plot(), Matplotlib doesn't know how to handle that very well
bike_sharing['dteday'] = pd.to_datetime(bike_sharing['dteday'])

#plot the dteday column on the x-axis and the casual column on the y-axis.
plt.plot(bike_sharing['dteday'], bike_sharing['casual'], label = 'Casual')
#plot dteday on the x-axis and registered on the y-axis.
plt.plot(bike_sharing['dteday'], bike_sharing['registered'], label = 'Registered')
#Rotate the x-ticks to an angle of 30 degrees using plt.xticks()
plt.xticks(rotation=30)
plt.ylabel('Bikes Rented')
plt.xlabel('Date')
plt.title('Bikes Rented: Casual vs. Registered')
plt.legend()
plt.show()

#now the dates on the bottom of the graph are overlapping and we can barely read them. To fix this, we can rotate the labels using the plt.xticks() function


#Scatter Plots
#Each point (also called a marker) on the scatter plot has an x-coordinate and an y-coordinate.

import pandas as pd
import matplotlib.pyplot as plt

bike_sharing = pd.read_csv('day.csv')

plt.scatter(bike_sharing['temp'], bike_sharing['cnt'])
plt.xlabel('Temperature')
plt.ylabel('Bikes Rented')
plt.show()


#Activity 
import pandas as pd
import matplotlib.pyplot as plt
bike_sharing = pd.read_csv('day.csv')
# What direction of the points do you expect to see considering that a strong wind can cause people to rent fewer bikes?
plt.scatter(bike_sharing['windspeed'], bike_sharing['cnt'])
plt.xlabel('Wind Speed')
plt.ylabel('Bikes Rented')
plt.show()



#correlation
#Activity
import pandas as pd
import matplotlib.pyplot as plt

bike_sharing = pd.read_csv('day.csv')
bike_sharing['dteday'] = pd.to_datetime(bike_sharing['dteday'])

plt.scatter(bike_sharing['atemp'], bike_sharing['registered'])
plt.show()
correlation = 'positive'



##Measuring Pearson's r

import pandas as pd
import matplotlib.pyplot as plt

bike_sharing = pd.read_csv('day.csv')
bike_sharing['dteday'] = pd.to_datetime(bike_sharing['dteday'])

#Calculate the Pearson's r between the temp and atemp columns
temp_atemp_corr = bike_sharing['temp'].corr(bike_sharing['atemp'])
#Calculate the Pearson's r between the windspeed and hum columns
wind_hum_corr = bike_sharing['windspeed'].corr(bike_sharing['hum'])

#Generate a scatter plot with the temp column on the x-axis and the atemp column on the y-axis
plt.scatter(bike_sharing['temp'], bike_sharing['atemp'])
plt.xlabel('Air Temperature')
plt.ylabel('Feeling Temperature')
plt.show()
#Generate a scatter plot with the windspeed column on the x-axis and the hum column on the y-axis
plt.scatter(bike_sharing['windspeed'], bike_sharing['hum'])
plt.xlabel('Wind Speed')
plt.ylabel('Humidity')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

bike_sharing = pd.read_csv('day.csv')
bike_sharing['dteday'] = pd.to_datetime(bike_sharing['dteday'])

#However, the 1 and 0 encoding is arbitrary. The creators of this dataset could have assigned 0 to a working day and 1 to a non-working day. Below, we make this change ourselves
bike_sharing['workingday'].replace({1:0, 0:1}, inplace=True)

#let's calculate workingday correlation with the casual and registered columns
print(bike_sharing.corr()['workingday'][['casual', 'registered']])


#Bar Plots

import matplotlib.pyplot as plt
working_days = ['Non-Working Day', 'Working Day']
registered_avg = [2959, 3978]

#Generate a bar plot 
plt.bar(working_days, registered_avg)
plt.show()


#Customizing Bar Plots

import pandas as pd
import matplotlib.pyplot as plt

bike_sharing = pd.read_csv('day.csv')
bike_sharing['dteday'] = pd.to_datetime(bike_sharing['dteday'])

bike_sharing['weekday'].value_counts().sort_index()
# We use Series.sort_index() to sort the index in ascending order

weekday_averages = bike_sharing.groupby('weekday').mean()[['casual', 'registered']].reset_index() # It's not essential to understand how this code works, we'll cover this in a later course

plt.bar(weekday_averages['weekday'], weekday_averages['registered'])
#The ticks parameter takes in the x-coordinates, and the labels parameter takes in the corresponding new labels
#Some of the x-tick labels are now overlapping. One thing we can do is leverage the rotation parameter 
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], rotation = 30)
plt.show()


#Frequency Tables
import matplotlib.pyplot as plt

unique_values = [1, 2, 3, 4]
weather_2011 = [226, 124, 15, 0]
weather_2012 = [237, 123, 6, 0]

#Generate a bar plot to display the weather patterns
plt.bar(unique_values, weather_2011)
plt.xticks(ticks=[1,2,3,4])
plt.title('Weather Patterns: 2011')
plt.ylabel('Frequency')
plt.xlabel('Unique Values')
plt.show()

plt.bar(unique_values, weather_2012)
plt.xticks(ticks=[1,2,3,4])
plt.title('Weather Patterns: 2012')
plt.ylabel('Frequency')
plt.xlabel('Unique Values')
plt.show()


#Grouped Frequency Tables

import pandas as pd
import matplotlib.pyplot as plt

bike_sharing = pd.read_csv('day.csv')
bike_sharing['dteday'] = pd.to_datetime(bike_sharing['dteday'])

#Generate a grouped frequency table with 10 intervals for the registered column
registered_freq = bike_sharing['registered'].value_counts(bins=10).sort_index() #normalize = True - this parametetr of Valu_counts uses proportion to display the counts, for percentage we multiply by 100
#The ( character in the bin value_count output indicates that the beginning of the interval isn't included, and the ] indicates that the endpoint is included.
#For example, (22.0, 32.0] means that 22.0 isn't included, while 32.0 is
##Generate a grouped frequency table with 10 intervals for the casual column
casual_freq = bike_sharing['casual'].value_counts(bins=10).sort_index()

''' The ( character indicates that the starting number is not included, while the ] indicates that the ending number is included.  '''


#Histograms

import pandas as pd
import matplotlib.pyplot as plt

bike_sharing = pd.read_csv('day.csv')
bike_sharing['dteday'] = pd.to_datetime(bike_sharing['dteday'])

#Generate a histogram for the casual column
plt.hist(bike_sharing['casual'])
plt.show()


##Pandas Visualization Methods
''' Series.plot.bar(): generates a vertical bar plot.
Series.plot.barh(): generates a horizontal bar plot.
Series.plot.line(): generates a line plot. '''

import matplotlib.pyplot as plt
import pandas as pd
traffic = pd.read_csv('traffic_sao_paulo.csv', sep=';')
traffic['Slowness in traffic (%)'] = traffic['Slowness in traffic (%)'].str.replace(',', '.')
traffic['Slowness in traffic (%)'] = traffic['Slowness in traffic (%)'].astype(float)

traffic['Slowness in traffic (%)'].plot.hist()
plt.title('Distribution of Slowness in traffic (%)')
plt.xlabel('Slowness in traffic (%)')
plt.show()


#Frequency of Incidents
import pandas as pd
import matplotlib.pyplot as plt

traffic = pd.read_csv('traffic_sao_paulo.csv', sep=';')
traffic['Slowness in traffic (%)'] = traffic['Slowness in traffic (%)'].str.replace(',', '.')
traffic['Slowness in traffic (%)'] = traffic['Slowness in traffic (%)'].astype(float)
# would narrow the traffic table down to just the Incidents
incidents = traffic.drop(['Hour (Coded)', 'Slowness in traffic (%)'], axis=1)

incidents.sum().plot.barh()
plt.show()


#Correlations with Traffic Slowness
import pandas as pd
import matplotlib.pyplot as plt

traffic = pd.read_csv('traffic_sao_paulo.csv', sep=';')
traffic['Slowness in traffic (%)'] = traffic['Slowness in traffic (%)'].str.replace(',', '.')
traffic['Slowness in traffic (%)'] = traffic['Slowness in traffic (%)'].astype(float)

#correlation between Slowness in traffic (%) and every other column
print(traffic.corr()['Slowness in traffic (%)'])
#To visualize the correlation between any two columns, we can use a scatter plot
traffic.plot.scatter(x='Slowness in traffic (%)',
                     y='Lack of electricity')
plt.show()

traffic.plot.scatter(x='Slowness in traffic (%)', y='Point of flooding')
plt.show()

traffic.plot.scatter(x='Slowness in traffic (%)', y='Semaphore off')
plt.show()


#Traffic Slowness Over 20%

import pandas as pd
import matplotlib.pyplot as plt

traffic = pd.read_csv('traffic_sao_paulo.csv', sep=';')
traffic['Slowness in traffic (%)'] = traffic['Slowness in traffic (%)'].str.replace(',', '.')
traffic['Slowness in traffic (%)'] = traffic['Slowness in traffic (%)'].astype(float)

#bool_filter
slowness_20_or_more = traffic[traffic['Slowness in traffic (%)'] >= 20]
#filter data frame
slowness_20_or_more = slowness_20_or_more.drop(['Slowness in traffic (%)', 'Hour (Coded)'], axis = 1)
incident_frequencies = slowness_20_or_more.sum()
incident_frequencies.plot.barh()
plt.show()



#How Traffic Slowness Change 

import pandas as pd
import matplotlib.pyplot as plt

traffic = pd.read_csv('traffic_sao_paulo.csv', sep=';')
traffic['Slowness in traffic (%)'] = traffic['Slowness in traffic (%)'].str.replace(',', '.')
traffic['Slowness in traffic (%)'] = traffic['Slowness in traffic (%)'].astype(float)

#to isolate the data for each day — from Monday to Friday
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
traffic_per_day = {} #empty dictionary to save the days (key, list pair)
#iterate over range() and days at the same time
#days is a generic list we created
#zip() function is used to combine two or more iterables (e.g., lists, tuples) element-wise, creating a new iterable with pairs of corresponding elements from the input iterables.
for i, day in zip(range(0, 135, 27), days):
    #The code inside the loop calculates a slice of the traffic list for each day, starting at index i and ending at index i + 27
    each_day_traffic = traffic[i:i+27]#uses iloc
    traffic_per_day[day] = each_day_traffic #saves it in the dictionary using the day as the Key
    traffic_per_day[day].plot.line(x='Hour (Coded)', y='Slowness in traffic (%)')
    plt.title(day)
    plt.ylim([0, 25]) #to make the range of the y-axis the same for all plots — this helps with comparison
    plt.show()



#Comparing Graphs
# to put all five line plots on the same graph.
import pandas as pd
import matplotlib.pyplot as plt

traffic = pd.read_csv('traffic_sao_paulo.csv', sep=';')
traffic['Slowness in traffic (%)'] = traffic['Slowness in traffic (%)'].str.replace(',', '.')
traffic['Slowness in traffic (%)'] = traffic['Slowness in traffic (%)'].astype(float)

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
traffic_per_day = {}
for i, day in zip(range(0, 135, 27), days):
    each_day_traffic = traffic[i:i+27]
    traffic_per_day[day] = each_day_traffic
    plt.plot(traffic_per_day[day]['Hour (Coded)'], traffic_per_day[day]['Slowness in traffic (%)'], label = day)

plt.legend()
plt.show()


#Grid Charts
''' there are portions on the graph where there are too many lines close to each other. This is making the graph harder to read. It's also hard to follow the line of a single day because it intersects with other lines. '''

'''A grid chart is a collection of similar graphs that usually share the same x- and y-axis range. The main purpose of a grid chart is to ease comparison.'''
import pandas as pd
import matplotlib.pyplot as plt

#The best approach in this case is to plot the five line plots both collectively and separately.

traffic = pd.read_csv('traffic_sao_paulo.csv', sep=';')
traffic['Slowness in traffic (%)'] = traffic['Slowness in traffic (%)'].str.replace(',', '.')
traffic['Slowness in traffic (%)'] = traffic['Slowness in traffic (%)'].astype(float)
#To create a grid chart, we start by creating the larger figure where we will plot all the graphs
plt.figure()
plt.subplot(3, 2, 1)
plt.subplot(3, 2, 2)
plt.subplot(3, 2, 3)
plt.subplot(3, 2, 4)
plt.subplot(3, 2, 5)
plt.subplot(3, 2, 6)
plt.show()



#Grid Charts (II)

import pandas as pd
import matplotlib.pyplot as plt

traffic = pd.read_csv('traffic_sao_paulo.csv', sep=';')
traffic['Slowness in traffic (%)'] = traffic['Slowness in traffic (%)'].str.replace(',', '.')
traffic['Slowness in traffic (%)'] = traffic['Slowness in traffic (%)'].astype(float)

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
traffic_per_day = {}
for i, day in zip(range(0, 135, 27), days):
    each_day_traffic = traffic[i:i+27]
    traffic_per_day[day] = each_day_traffic
    
''' The plots are overlapping, and the grid chart as a whole looks a bit too packed. To fix this, we're going to increase the size of the entire figure by using plt.figure(figsize=(width, height)) 

Matplotlib allows us to customize the subplots individually. All the Matplotlib code under a certain plt.subplot() function is targeted towards the particular subplot that function generates.

When we need to use the plt.subplot() function multiple times, it's better to use a for loop. Below, for instance, we create all the six plots with significantly less code.
'''
plt.figure(figsize=(10, 12))
for i, day in zip(range(1, 7), days) :
    plt.subplot(3, 2, i)
    plt.plot(traffic_per_day[day]['Hour (Coded)'], traffic_per_day[day]['Slowness in traffic (%)'])
    plt.ylim([0, 25])
    plt.title(day)

plt.show()


# Grid Charts (III)
#we generated the first five plots on our grid chart. Recall, however, that we still need to #add the line plot showing all the days on the same graph.
import pandas as pd
import matplotlib.pyplot as plt

traffic = pd.read_csv('traffic_sao_paulo.csv', sep=';')
traffic['Slowness in traffic (%)'] = traffic['Slowness in traffic (%)'].str.replace(',', '.')
traffic['Slowness in traffic (%)'] = traffic['Slowness in traffic (%)'].astype(float)

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
traffic_per_day = {}
for i, day in zip(range(0, 135, 27), days):
    each_day_traffic = traffic[i:i+27]
    traffic_per_day[day] = each_day_traffic
    
plt.figure(figsize=(10,12))

for i, day in zip(range(1,6), days):
    plt.subplot(3, 2, i)
    plt.plot(traffic_per_day[day]['Hour (Coded)'],
        traffic_per_day[day]['Slowness in traffic (%)'])
    plt.title(day)
    plt.ylim([0,25])
    
#Add a new subplot with index number 6 
#this is the plot that seeks to compare all the plots

plt.subplot(3, 2, 6)

for day in days:
    plt.plot(traffic_per_day[day]['Hour (Coded)'], traffic_per_day[day]['Slowness in traffic (%)'], label=day)
    plt.ylim([0,25])
    
plt.legend()
plt.show


#Relational Plots and Multiple Variables
#Seaborn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
housing = pd.read_csv('housing.csv')

#To switch to Seaborn defaults, we need to call
sns.set_theme()
#e need to call the sns.set_theme() function before generating the plot 
sns.relplot(data = housing, x = 'Gr Liv Area', y = 'SalePrice')
plt.show()
correlation = "positive"


# Variable Representation: Color
#on the scatter plot is blue. We can change the color intensity of the dots to represent a new variable.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#we use the hue parameter to add the Overall Qual
''' Seaborn matched lower ratings with lighter colors and higher ratings with darker colors. A pale pink represents a rating of one, while black represents a ten. Seaborn also generated a legend to describe which color describes which rating.
'''
housing = pd.read_csv('housing.csv')
sns.set_theme()
#We can find all the available color palettes in Matplotlib's documentation 
sns.relplot(data=housing, x='Gr Liv Area', y='SalePrice', hue = 'Overall Qual', palette='RdYlGn')
plt.show()


##Variable Representation: Size
'''
Another element we can use to represent values is size. A dot can have a color and x- and y-coordinates, but it can also be larger or smaller. Below, we use a size representation to add the Garage Area variable on the graph — we use the size parameter. Recall that Garage Area describes the garage area in square feet.

To make the size differences more visible, we'll increase the size range — the sizes parameter takes in a tuple specifying the minimum and maximum size.
The sizes parameter can take in a tuple only if the variable we represent is numerical 
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

housing = pd.read_csv('housing.csv')
sns.set_theme()
sns.relplot(data=housing, x='Gr Liv Area', y='SalePrice',
            hue='Overall Qual', palette='RdYlGn',
           size = 'Garage Area', sizes =(1, 300))
plt.show()
