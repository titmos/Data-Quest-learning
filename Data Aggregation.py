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
