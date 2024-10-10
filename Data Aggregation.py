## Data Aggregation learns


import pandas as pd
happiness2015 = pd.read_csv('World_Happiness_2015.csv')
first_5 = happiness2015.head() #First five rows

#information about the DataFrame
print(happiness2015.info())
