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


#Introduction to the Agg() Method
#to apply more than one kind of aggregation to a column at a time, we use the groupby.add() method 
import numpy as np
grouped = happiness2015.groupby('Region')
happy_grouped = grouped['Happiness Score']

def dif(group):
    return group.max() - group.mean()
happy_mean_max = happy_grouped.agg([np.mean, np.max])
mean_max_dif = happy_grouped.agg(dif)
print(happy_mean_max)

print(mean_max_dif)

#Computing Multiple and Custom Aggregations with the Agg() Method
#method chaining
happiness_means = happiness2015.groupby('Region')['Happiness Score'].mean()
''' Both approaches will return the same result. However, if you plan on computing multiple aggregations with the same GroupBy object, we recommend that you save the object to a variable first. (You may want to save it to a variable in every cases to make your code easier to understand. As we compute more complex aggregations, the syntax can become confusing!) '''


#Aggregation with Pivot Tables
#we use the df.pivot_table() method to perform the same aggregation as above.
pv_happiness = happiness2015.pivot_table(values='Happiness Score', index='Region', aggfunc=np.mean, margins = True) # If margins=True, special All columns and rows will be added with partial group aggregates across the categories on the rows and columns.
#All is the mean of the Happiness Score column in our instance
''' this method returns a DataFrame, so we can apply normal DataFrame filtering and methods to the result. For example, let's use the DataFrame.plot() method to create a visualization. Note that we exclude aggfunc below because the mean is the default aggregation function of df.pivot_table() '''
pv_happiness.plot(kind='barh', title='Mean Happiness Scores by Region', xlim=(0,10), legend=False)
plt.show()
world_mean_happiness = happiness2015['Happiness Score'].mean()


#Aggregating Multiple Columns and Functions with Pivot Tables

#The pivot_table method also allows us to aggregate multiple columns and apply multiple functions at once.

#To apply multiple functions, we can pass a list of the functions into the aggfunc parameter: aggfunc=[np.mean, np.min , np.max]
#Let's compare the results returned by the groupby operation and the pivot_table method.
gro = happiness2015.groupby('Region')
grouped = gro[['Happiness Score', 'Family']]
happy_family_stats = grouped.agg([np.min, np.max, np.mean])
print(happy_family_stats)

pv_happy_family_stats = happiness2015.pivot_table(['Happiness Score', 'Family' ], 'Region', aggfunc=[np.min, np.max , np.mean], margins=True)
print(pv_happy_family_stats)


#Combining Dataframes with the Concat Function
#We've already saved the subsets from happiness2015 and happiness2016

head_2015 = happiness2015[['Country','Happiness Score', 'Year']].head(3)
head_2016 = happiness2016[['Country','Happiness Score', 'Year']].head(3)

# combine head_2015 and head_2016 along axis = 0
concat_axis0 = pd.concat([head_2015, head_2016], axis = 0)

# combine head_2015 and head_2016 along axis = 1. 
concat_axis1 = pd.concat([head_2015, head_2016], axis = 1)

question1 = 6 #number of rows in concat_axis0
question2 = 3 #number of rows in concat_axis1


#Combining Dataframes with the Concat Function Continued
'''  when you use the concat() function to combine dataframes with the same shape and index, you can think of the function as "gluing" dataframes together.

However, what happens if the dataframes have different shapes or columns? Let's confirm the concat() function's behavior when we combine dataframes that don't have the same shape
'''

head_2015 = happiness2015[['Year','Country','Happiness Score', 'Standard Error']].head(4)
head_2016 = happiness2016[['Country','Happiness Score', 'Year']].head(3)

concat_axis0 = pd.concat([head_2015, head_2016], axis = 0)
rows = 7
columns = 4


#Combining Dataframes with Different Shapes Using the Concat Function
''' Also, notice again the indexes of the original dataframes didn't change. If the indexes aren't meaningful, it can be better to reset them. This is especially true when we create duplicate indexes, because they could cause errors as we perform other data cleaning tasks.

Luckily, the concat function has a parameter, ignore_index, that can be used to clear the existing index and reset it in the result. '''

head_2015 = happiness2015[['Year','Country','Happiness Score', 'Standard Error']].head(4)
head_2016 = happiness2016[['Country','Happiness Score', 'Year']].head(3)

concat_update_index = pd.concat([head_2015, head_2016], axis = 0, ignore_index = True)


#Joining Dataframes with the Merge Function

'''
pd.merge() function - a function that can execute high-performance database-style joins.
the merge function only combines dataframes horizontally (axis=1) and can only combine two dataframes at a time.

However, it can be valuable when we need to combine very large dataframes quickly and provides more flexibility in terms of how data can be combined.

With the merge() function, we'll combine dataframes on a key, a shared index or column. 
'''

three_2015 = happiness2015[['Country','Happiness Rank','Year']].iloc[2:5]
three_2016 = happiness2016[['Country','Happiness Rank','Year']].iloc[2:5]
#join three_2015 and three_2016 on the Country column
merged = pd.merge(left = three_2015, right = three_2016, on = 'Country')


#Joining on Columns with the Merge Function
three_2015 = happiness2015[['Country','Happiness Rank','Year']].iloc[2:5]
three_2016 = happiness2016[['Country','Happiness Rank','Year']].iloc[2:5]

#This way of combining, or joining, data is called an inner join. An inner join returns only the intersection of the keys, or the elements that appear in both dataframes with a common key.
''' There are actually four different types of joins:

Inner: only includes elements that appear in both dataframes with a common key
Outer: includes all data from both dataframes
Left: includes all of the rows from the "left" dataframe along with any rows from the "right" dataframe with a common key; the result retains all columns from both of the original dataframes
Right: includes all of the rows from the "right" dataframe along with any rows from the "left" dataframe with a common key; the result retains all columns from both of the original dataframes  

If the definition for outer joins sounds familiar, it's because we've already seen examples of outer joins! Recall that when we combined data using the concat function, it kept all of the data from all dataframes, no matter if missing values were created.
'''
merged = pd.merge(left=three_2015, right=three_2016, on='Country')

merged_left = pd.merge(left=three_2015, right=three_2016, on='Country', how = 'left')

merged_left_updated = pd.merge(left=three_2016, right=three_2015, on='Country', how = 'left')
