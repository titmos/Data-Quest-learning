## Data Aggregation learns


import pandas as pd
happiness2015 = pd.read_csv('World_Happiness_2015.csv')
first_5 = happiness2015.head() #First five rows

#information about the DataFrame
print(happiness2015.info())


#Using Loops to Aggregate Data
#Create an empty dictionary named mean_happiness to store the results of this exercise.
mean_happiness = {}
#Use the Series.unique() method to create an array of unique values for the Region column
region = happiness2015['Region'].unique()

#Use a for loop to iterate over the unique region values from the Region column
for item in region:
    #Assign the rows belonging to the current region to a variable named region_group
    #filter and assign 
    region_group = happiness2015[happiness2015['Region'] == item]
    #Use the Series.mean() method to calculate the mean happiness score for region_group.
    region_mean = region_group['Happiness Score'].mean()
    #Assign the mean value to the mean_happiness dictionary
    mean_happiness[item] = region_mean


#Creating GroupBy Objects

'''
To create a GroupBy object, we use the DataFrame.groupby() method:
df.groupby('col')
"col" is the column you want to use to group the dataset.

We'll start by using the GroupBy.get_group() method to select data for a certain group.
'''
grouped = happiness2015.groupby('Region')
#to view a portion of the grouped data "get_group() method is called"
aus_nz = grouped.get_group('Australia and New Zealand')


#Exploring GroupBy Objects
grouped = happiness2015.groupby('Region')
print(grouped.groups)
# We obtained the Region and list of indexes of the coontries in the region using the above
#I tested the indexes using iloc
north_america = happiness2015.iloc[[4, 14]]
# used the get group to get the North america list
na_group = grouped.get_group('North America')
# compared results 
equal = na_group == north_america


#Common Aggregation Methods with Groupby
# Group by 'Region'
grouped = happiness2015.groupby('Region')

#A basic example of aggregation is computing the number of rows for each of the groups. We can use the GroupBy.size() method to confirm the size of each region group
print(grouped. size())
'''   it's often good practice to specify the numeric_only parameter for aggregation methods like mean(), sum(), min(), and max() to avoid TypeErrors. This parameter determines whether to include only numeric columns in the computation. By setting numeric_only=True, we ensure that only numeric columns are considered for the aggregation, avoiding potential errors with non-numeric data. '''
means = grouped.mean(numeric_only = True)
''' You may have noticed that Region appears in a different row than the rest of the column names. Because we grouped the DataFrame by region, the unique values in Region are used as the index. Up until now, we've mostly worked with DataFrames with a numeric index. '''


#Aggregating Specific Columns with Groupby

grouped = happiness2015.groupby('Region')
happy_grouped = grouped['Happiness Score']
happy_mean = happy_grouped.mean()
print(happy_mean)
