import pandas as pd
f500 = pd.read_csv('f500.csv', index_col=0) #using the pandas CSV reader
f500.index.name = None

f500_type = type(f500) 
f500_shape = f500.shape #displays the shape of the dataset in row by column
print(f500_type)
print(f500_shape)

print(f500.head(3)) # would display the header and first 3 rows in the dataset
print(f500.tail(3)) # would display the last 3 rows in the dataset
f500_top_6 = f500.head(6)
f500_bottom_8 = f500.tail(8)


#overview of all the dtypes used in our DataFrame, along with its shape and other information, we can use the DataFrame.info() method
f500.info()


#There are two types of labels in pandas: Row Labels and Column Labels.
# df.loc[row_label, column_label]
# Select a Single Column
# dataFra.loc[:, "col_name"]
# we can also use f500_selection["rank"]
industries = f500["industry"]
industries_type = type(industries)
print(industries, '\n')
print(industries_type)


#Selecting Columns from a DataFrame by Label 
#this is for a single selection
countries = f500["country"]
#this works for multiple columns selection
revenues_years = f500[["revenues", "years_on_global_500_list"]]

#printing slices in a range, We will have to do the full specification using .loc
ceo_to_sector = f500.loc[:,"ceo" : "sector"]



"""Select the country column from f500 and save it to the variable countries.
Use the Series.value_counts() method to get the value counts for each unique non-null value in countries and save the results in the variable country_counts.
Print the variable country_counts to determine which country has the most companies on the Fortune 500 list and save that country's name as a string in the variable top_country."""

#Sample application of  the value_counts method on series data
countries = f500["country"]
country_counts = countries.value_counts()
print(country_counts)
top_country = 'USA'

hq_locations =f500["hq_location"]
hql_counts = hq_locations.value_counts()
print(hql_counts)
top_hq_city ='Beijing, China'

#Sample application of  the value_counts method on dataFrames

sectors_industries = f500[["sector", "industry"]]
print(type(sectors_industries))
si_value_counts = sectors_industries.value_counts()
print(si_value_counts)

#Series.describe() method tells us how many non-null values are contained in the series
#along with the mean, minimum, maximum

rank = f500["rank"]
rank_desc = rank.describe()
print(rank_desc)
prev_rank = f500["previous_rank"]
prev_rank_desc = prev_rank.describe()
print(prev_rank_desc)

#Method Chaining
#a way to combine multiple methods together in a single line.
prev_rank = f500["previous_rank"]
zero_previous_rank = prev_rank.value_counts().loc[0]

#using method chaining 
zero_using_chaining = f500["previous_rank"].value_counts().loc[0]


#Dataframe Exploration Methods
medians = f500[["revenues", "profits"]].median(axis=0)
# we could also use .median(axis="index")
print(medians)
#f500.info()
numeric_only = f500[["rank", "revenues", "revenue_change", "profits", "assets", "profit_change", "previous_rank", "years_on_global_500_list", "employees", "total_stockholder_equity"]]
max_f5001 = numeric_only.max(axis = 0)
max_f500 = f500.max(numeric_only = True)



#Dataframe Describe Method: can use to explore the dataframe more quickly
#By default, DataFrame.describe() will return statistics for only numeric columns.
# If we wanted to get just the object columns, we need to use the include=['O'] parameter:
describe = f500.describe(include=['O'])
f500_desc = f500.describe()
#Series.describe() method returns a series object, the DataFrame.describe() method returns a dataframe object



# Assignment with pandas
#top5_rank_revenue = f500[["rank", "revenues"]].head()
#print(top5_rank_revenue)
#top5_rank_revenue["revenues"] = 0
#print(top5_rank_revenue)
#top5_rank_revenue.loc["Sinopec Group", "revenues"] = 999
#print(top5_rank_revenue)
f500.loc["Dow Chemical", "ceo"] = "Jim Fitterling"


# Using Boolean Indexing with pandas Objects
#Create a boolean series, motor_bool, that compares whether the values in the industry column from the f500 dataframe are equal to "Motor Vehicles and Parts"

motor_bool = f500["industry"] == "Motor Vehicles and Parts"
#Use the motor_bool boolean series to index the country column. Assign the result to motor_countries
motor_countries = f500.loc[motor_bool, "country"]


# Using Boolean Arrays to Assign Values
import numpy as np
prev_rank_before = f500["previous_rank"].value_counts(dropna=False).head()

#Use boolean indexing to update values in the previous_rank column of the f500 dataframe:
prev_bool = f500["previous_rank"] == 0
#There should now be a value of np.nan where there previously was a value of 0
f500.loc[prev_bool, "previous_rank"] = np.nan
#Create a new pandas series, prev_rank_after, using the same syntax that was used to create the prev_rank_before series.
prev_rank_after = f500["previous_rank"].value_counts(dropna = False).head()


#Creating New Column (rank_change) in Pandas DataFrame
f500["rank_change"] = f500["previous_rank"] - f500["rank"]

rank_change_desc = f500["rank_change"].describe()



#Exploring Data with Pandas: Fundamentals
#Challenge: Top Performers by Country
top_2_countries = f500["country"].value_counts().head(2)
print(top_2_countries)

#Create a series, industry_usa, containing counts of the two most common values in the industry column for companies headquartered in the USA.
industry_usa_bool = f500["country"] == "USA"
industry_usa = f500.loc[industry_usa_bool, "industry"].value_counts().head(2)

#Create a series, sector_china, containing counts of the three most common values in the sector column for companies headquartered in China
sector_bool = f500["country"] == "China"
sector_china = f500.loc[sector_bool, "sector"].value_counts().head(3)



import pandas as pd
#The dataset has already been loaded into a pandas dataframe and assigned to the f500 variable. 
# read the dataset into a pandas dataframe
f500 = pd.read_csv("f500.csv", index_col=0)
f500.index.name = None

# replace 0 values in the "previous_rank" column with NaN
f500.loc[f500["previous_rank"] == 0, "previous_rank"] = np.nan

#Select the rank, revenues, and revenue_change columns in f500. Then, use the DataFrame.head() method to select the first five rows. Assign the result to f500_selection
f500_selection = f500[["rank", "revenues", "revenue_change"]].head()


f500 = pd.read_csv("f500.csv")
# To assign a Index name
f500.index.name = "Company"
#To assign a column name
f500.columns.name = "Metric"
f500.loc[f500["previous_rank"] == 0, "previous_rank"] = np.nan
#print(f500[["rank", "revenues"]])


# Using iloc to Select by Integer Location
fifth_row = f500.iloc[4] #Select just the fifth row of the f500 dataframe. Assign the result to fifth_row
company_value = f500.iloc[0, 0] #Select the value in first row of the company column
print(company_value)


#NOTE! With loc[], the end index is included.
#With iloc[], the end index is not included.
#Select the first three rows of the f500 dataframe. Assign the result to first_three_rows
first_three_rows = f500.iloc[0:3]
#Select the first and seventh rows while selecting the first five columns
first_seventh_row_slice = f500.iloc[[0, 6], :5]



# Using pandas Methods to Create Boolean Masks
#Use the Series.isnull() method to create a boolean mask of all rows from f500 that have a null value for the previous_rank column
prev_rank_null = f500["previous_rank"].isnull()
print(prev_rank_null.value_counts())
#Use the mask you created above to select the rows with null values in the previous_rank column while selecting the company, rank, and previous_rank columns. 
null_prev_rank = f500[prev_rank_null][["company", "rank", "previous_rank"]]



# Working with Integer Labels
#Given
prev_rank_is_null = f500[f500["previous_rank"].isnull()]
print(prev_rank_is_null)
#Assign the first five rows of the prev_rank_is_null dataframe to the variable top5_null_prev_rank by applying the correct method: loc[] or iloc[]
#top5_null_prev_rank = prev_rank_is_null.iloc[:5]
#Another way 
top5_null_prev_rank = prev_rank_is_null.loc[48 : 140]
#note we can alos use .head() Default is first 5 rows



#Pandas Index Alignment

'''previously_ranked = f500[f500["previous_rank"].notnull()]
rank_change = previously_ranked["previous_rank"] - previously_ranked["rank"]
print(rank_change.shape)
print(rank_change.tail())'''
#Above, we can see that our rank_change series has 467 rows. Since the last integer index label is 498, we know that our index labels no longer align with the integer positions.
#Suppose we decided to add the rank_change series to the f500 dataframe as a new column. Its index labels no longer match the index labels in f500, so how could this be done?
#Another powerful aspect of pandas is that almost every operation will align on the index labels. Let's look at an example.
print(f500.shape)
print(f500["profits"].notnull().value_counts())
profited = f500[f500["profits"].notnull()]
print(profited.shape)
costs = profited["revenues"] - profited["profits"]
#The pandas library will align on the index at every opportunity, no matter if our index labels are strings or integers. This makes working with data from different sources or working with data when we have removed, added, or reordered rows much easier than it would be otherwise.
#Assign the values in the costs to a new column in the f500 dataframe, "costs"
f500["costs"] = costs



# Using Boolean Operators
#Create a boolean array that selects the companies with revenues greater than 100 billion
large_revenue = f500["revenues"] > 100000
#Create a boolean array that selects the companies with profits less than 0
negative_profits = f500["profits"] < 0
#we use the & operator to combine the two boolean arrays using boolean "and" logic
#Combine large_revenue and negative_profits
combined = large_revenue & negative_profits
#we use the combined boolean array to make our selection on the entire dataframe
#Use the combined boolean mask to filter f500
big_rev_neg_profit = f500.loc[combined]




#Using Expression chaining and Boolean Operators
#Notice that we placed parentheses () around each of our boolean comparisons. 
#This is critical — our boolean operation will fail without wrapping each comparison in parentheses
#Select all companies whose country value is either "Brazil" or "Venezuela"
brazil_venezuela = f500[(f500["country"] == "Brazil") | (f500["country"] == "Venezuela")]

#Select the first five companies in the "Technology" "sector" for which the "country" is not "USA"
tech_outside_usa = f500[(~(f500["country"] == "USA")) & (f500["sector"] == "Technology")].head()


# Suppose we wanted to find the company that employs the most people in China. 
selected_rows = f500[f500["country"] == "China"]
#To sort the rows in descending order — from largest to smallest — we set the ascending parameter of the method to False
sorted_rows = selected_rows.sort_values("employees", ascending=False)
print(sorted_rows[["company", "country", "employees"]].head())


#Sorting Values
# Suppose we wanted to find the company that employs the most people in China. 
selected_rows = f500[f500["country"] == "China"]
#To sort the rows in descending order — from largest to smallest — we set the ascending parameter of the method to False
sorted_rows = selected_rows.sort_values("employees", ascending=False)
#print(sorted_rows[["company", "country", "employees"]].head())

#Trial Class Exercise
#Select only the rows of f500 that have a country equal to Japan and assign the resulting dataframe to the variable selected_rows
selected_rows = f500[f500["country"] == "Japan"]
#Use the DataFrame.sort_values() method on selected_rows to sort it by the profits column in descending order
sorted_rows = selected_rows.sort_values("profits", ascending = False)
#try by 
print(sorted_rows)
#Use DataFrame.iloc[] to select the first row of the sorted_rows dataframe and filter it to include just the company and profits columns.
top_japanese_company = sorted_rows.iloc[0, [0, 4]]
#print('\n', top_japanese_company)



# Create an empty dictionary to store the results
avg_rev_by_country = {}
#To identify the unique countries, we can use the Series.unique() method
#This method returns a NumPy array of unique values from any series
#Then, we can loop over that array and perform our operation. Here's what that looks like:

# Create an array of unique countries
countries = f500["country"].unique()

# Use a for loop to iterate over the unique countries
for c in countries:
    # Use a boolean comparison to select only rows that
    # correspond to the country for this iteration
    selected_rows = f500[f500["country"] == c]
    # Calculate the mean revenue for the selected rows
    mean = selected_rows["revenues"].mean()
    # Assign the mean value to the dictionary, using the
    # country name as the key
    avg_rev_by_country[c] = mean
print(avg_rev_by_country)


#Activity
# Create an empty dictionary to store the results
top_employer_by_country = {}
# Create an array of unique countries
#Use the Series.unique() method
countries = f500["country"].unique()

# Use a for loop to iterate over the unique countries
for c in countries:
    # Use a boolean comparison to select only rows that
    # correspond to the country for this iteration
    selected_rows = f500[f500["country"] == c]
#Use DataFrame.sort_values() to sort those rows by the employees column in descending order.
    sort = selected_rows.sort_values("employees", ascending = False)
#Select the first row from the sorted dataframe.
    top_employer = sort.iloc[0]
#Extract the company name from the company column of the first row.
    company = top_employer["company"]
#Assign the results to the top_employer_by_country dictionary
#country name as the key, and the company name as the value
    top_employer_by_country[c] = company


#ACtivity
f500["roa"] = f500["profits"] / f500["assets"]


top_roa_by_sector = {}

sectors = f500["sector"].unique()

for s in sectors:
    selected_rows = f500[f500["sector"] == s]
    sorted_rows = selected_rows.sort_values("roa", ascending = False)
    top_roa = sorted_rows.iloc[0]
    top_roa_extract = top_roa["company"]
    top_roa_by_sector[s] = top_roa_extract



#Cleaning Column Names
#efine a function, clean_col, which accepts a string argument, col and return a starndard label
def clean_col(col):
    #Removes any whitespace from the start and end of the string.
    col = col.strip()
    #Replaces the substring Operating System with the abbreviation os
    col = col.replace("Operating System", "OS")
    #Replaces all spaces with underscores
    col = col.replace(" ", "_")
    #Removes parentheses from the string
    col = col.replace("(","")
    col = col.replace(")","")
    #Makes the entire string lowercase
    col = col.lower()
    #Returns the modified string
    return col

#Use a for loop to apply the function to each item in the DataFrame.columns attribute for the laptops dataframe. Assign the result back to the DataFrame.columns attribute.
new_columns = []
for c in laptops.columns:
    clean_c = clean_col(c)
    new_columns.append(clean_c)
laptops.columns = new_columns
print(laptops.columns)


#Converting String Columns to Numeric
#Use the Series.unique() method to identify the unique values in the ram column of the laptops dataframe.

'''The Series.unique() method returns a numpy array, not a list or pandas series. This means that we can't use the Series methods we've learned so far, like Series.head(). If you want to convert the result to a list, you can use the tolist() method of the numpy array

unique_ram = laptops["ram"].unique().tolist()
'''
unique_ram = laptops["ram"].unique()
print(unique_ram)


#Removing Non-Digit Characters
#Use the Series.str.replace() method to remove the substring GB from the ram column
laptops["screen_size"] = laptops["screen_size"].str.replace('"', '')

laptops["ram"] = laptops["ram"].str.replace('GB', '')
print(laptops["ram"].unique())
print("`ram` dtype:", laptops["ram"].dtype)



#Converting Columns to Numeric dtypes
laptops["screen_size"] = laptops["screen_size"].astype(float)
#Use the Series.astype() method to cast the ram column to an int dtype.
laptops["ram"] = laptops["ram"].astype(int)
print(laptops["ram"])
#Use print() and the DataFrame.dtypes attribute to confirm that the screen_size and ram columns have been cast to numeric dtypes.
print("`ram` dtype:", laptops["ram"].dtype)


#Renaming Columns
laptops.rename({"screen_size": "screen_size_inches"}, axis=1, inplace=True)
#axis=1 parameter so pandas knows that we want to rename labels in the column axis 
#inplace=True or assign the result back to the dataframe

#use the DataFrame.rename() method to rename the column from ram to ram_gb
laptops.rename({"ram": "ram_gb"}, axis=1, inplace=True)
#Use the Series.describe() method to return a series of descriptive statistics for the ram_gb column
ram_gb_desc = laptops["ram_gb"].describe()
print(ram_gb_desc)


#Use the Series.str.split() method to split the gpu column into a list of words. Assign the result to gpu_split
gpu_split = laptops["gpu"].str.split()
#Use the Series.str accessor with [] to select the first element of each list of words.
laptops["gpu_manufacturer"] = gpu_split.str[0]
gpu_manufacturer_counts = laptops["gpu_manufacturer"].value_counts()

#Extract the manufacturer name from the cpu column and assign the results to a new column cpu_manufacturer of the laptops dataframe. Try to do it in one line of code; try not use an intermediate "cpu_split" variable
laptops["cpu_manufacturer"] = laptops["cpu"].str.split().str[0]
cpu_manufacturer_counts = laptops["cpu_manufacturer"].value_counts()


#Correcting Bad Values
#Use the Series.unique() method on the os column to display a list of all the unique values it contains

print(laptops['os'].unique())

#Create a dictionary called mapping_dict where each key is a unique value from the previous step, and the corresponding value is its replacement

''' When using the map() method, make sure that each unique value in the series is represented as a key in the dictionary being passed to the map() method, otherwise you'll get NaN values in your resulting series.
 If there are values in the series you don't want to change, ensure you set their keys and values equal to each other so that "no changes are mapped" but each unique value appears as a key in the dictionary. '''

mapping_dict = {"Windows" : "Windows", "No OS" : "No OS", "Linux" : "Linux", "Chrome OS" : "Chrome OS", "macOS" : "macOS", "Mac OS" : "macOS", "Android" : "Android"}

#Use the Series.map() method along with the mapping_dict dictionary from the previous step to correct the values in the os column.
laptops["os"] = laptops["os"].map(mapping_dict)
print(laptops["os"].value_counts())



#Dropping Missing Values
#Recall that we can use the DataFrame.isnull() method to identify missing values
print(laptops.isnull().sum())
#Use DataFrame.dropna() to remove any rows from the laptops dataframe that have null values. Assign the result to laptops_no_null_rows
laptops_no_null_rows = laptops.dropna(axis = 0)
#Use DataFrame.dropna() to remove any columns from the laptops dataframe that have null values. Assign the result to laptops_no_null_cols
laptops_no_null_cols = laptops.dropna(axis = 1)
print(laptops_no_null_rows)
print(laptops_no_null_rows.shape)
print(laptops_no_null_cols)
print(laptops_no_null_cols.shape)

#Note 
#DataFrame.dropna() method accepts an axis parameter, which indicates whether we want to drop along the index axis (axis=0) or the column axis (axis=1)



#Filling Missing Values
#While dropping rows or columns is the easiest approach to dealing with missing values, it may not always be the best approach
#.value_counts(dropna=False): This method generates a frequency table of unique values in the Series
#On default value_counts() does not return frequency of NAn/Missing values

#By setting dropna=False, the method includes missing values (NA) in the table.
value_counts_before = laptops.loc[laptops["os_version"].isnull(), "os"].value_counts()
laptops.loc[laptops["os"] == "macOS", "os_version"] = "X"
#Use a boolean array to select the rows that have the value No OS in the os column. Then, use loc[] and assignment to set the value Not Applicable in the os_version column for those selected rows.
laptops.loc[laptops["os"] == "No OS", "os_version"] = "Not Applicable"
#Use the provided syntax for value_counts_before to create a similar value_counts_after variable.
value_counts_after = laptops.loc[laptops["os_version"].isnull(), "os"].value_counts()
print(laptops["os_version"].value_counts(dropna=False))



#Challenge: Clean a String Column

#Convert the values in the weight column to numeric values.
laptops["weight"] = laptops["weight"].str.replace('kgs','').str.replace('kg', '').astype(float)
#Rename the weight column to weight_kg
laptops.rename({"weight" : "weight_kg"}, axis = 1, inplace = True)

#Use the DataFrame.to_csv() method to save the laptops dataframe to a CSV file /tmp/laptops_cleaned.csv without index labels


laptops.to_csv('/tmp/laptops_cleaned.csv', index=False)


##Transforming Data with Pandas
# we need to manipulate our data into a format that makes it easier to analyze. 
mapping = {'Economy (GDP per Capita)': 'Economy', 'Health (Life Expectancy)': 'Health', 'Trust (Government Corruption)': 'Trust' }
#Use the DataFrame.rename() method to change column names to the names specified in the mapping dictionary.
happiness2015 = happiness2015.rename(mapping, axis = 1)


#Apply a Function Element-wise Using the Map and Apply Methods

def label(element):
    if element > 1:
        return 'High'
    else:
        return 'Low'
#to apply the label function to the Economy column
economy_impact_map = happiness2015['Economy'].map(label)

# to apply the label function to the Economy column.
economy_impact_apply = happiness2015['Economy'].apply(label)
#to check if the methods produce the same result
equal = economy_impact_map.equals(economy_impact_apply)


#Apply a Function Element-wise Using the Map and Apply Methods Continued
#In the last exercise, we applied a function to the Economy column using the Series.map() and 
#Series.apply() methods and confirmed that both methods produce the same results.
''' Note that these methods don't modify the original series. If we want to work with the new series in the original dataframe, we must either assign the results back to the original column or create a new column. We recommend creating a new column,  
To create the Economy Impact column(A new column tahts stores the status of each element in the Economy column), map() and apply() iterate through the Economy column and pass each value into the label function. The function evaluates which range the value belongs to and assigns the corresponding value to the element in the new column.
'''
def label(element, x):
    if element > x:
        return 'High'
    else:
        return 'Low'
''' Since both map and apply can apply functions element-wise to a series, you may be wondering about the difference between them. Let's start by looking at a function with arguments.
In the label function, we arbitrarily split the values into 'High' and 'Low'. What if instead we allowed that number to be passed into the function as an argument?
When we try to apply the function to the Economy column with the map method, we get an error:
'''
#We learned in the last screen that the Series.map() method doesn't easily handle functions with additional arguments. The Series.apply() method, however, can be used for such functions
economy_impact_apply = happiness2015['Economy'].apply(label, x = 0.8)
#Apply a Function Element-wise to Multiple Columns Using Map method
''' 
So far, we've transformed just one column at a time. If we wanted to transform more than one column, we could use the Series.map() or Series.apply() method to transform them as follows:
'''
def label(element):
    if element > 1:
        return 'High'
    else:
        return 'Low'
#The construct below will create new columns and save the process from the label function
happiness2015['Economy Impact'] = happiness2015['Economy'].apply(label)
happiness2015['Health Impact'] = happiness2015['Health'].apply(label)
happiness2015['Family Impact'] = happiness2015['Family'].apply(label)

'''
However, it would be easier to just apply the same function to all of the factor columns (Economy, Health, Family, Freedom, Generosity, Trust) at once. Fortunately, however, pandas already has a method that can apply functions element-wise to multiple columns at once - the DataFrame.map() method.
At first it migt look like the same function, but the subtle difference is that now the function is DataFrame.map() instead of Series.map(). 
This distinction is important because DataFrame.map() applies the function to every element in the DataFrame, whereas Series.map() only works on a single column.
syntax to work with the DataFrame.map() method:
DataFrame[columns].map(function_name)
'''
