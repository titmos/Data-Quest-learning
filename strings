#string formatting in python

def artist_summary(name):
    num_artworks =artist_freq[name]
    print("There are {y} artworks by {z} in the dataset".format(z = name, y = num_artworks))
artist_summary("Henri Matisse")

#Inserting Variables into Strings
artist = "Pablo Picasso"
birth_year = 1881
template = "{artist}'s birth year is {birth_year}"
print(template.format(artist = "Pablo Picasso", birth_year = 1881))

#Create a template string that will insert the country name and population
pop_millions = [
    ["China", 1379.302771],
    ["India", 1281.935991],
    ["USA",  326.625791],
    ["Indonesia",  260.580739],
    ["Brazil",  207.353391],
]
for item in pop_millions:
    name = item[0]
    population = item[1]
    print("The population of {} is {:,.2f} million".format(name, population))

#Create a frequency table for the values in the Gender (row index 5) column.
#Loop over each key-value pair in the dictionary. Display a line of output in the format shown above summarizing each pair.

gender_freq = {}
for item in moma:
    gender = item[5]
    if gender in gender_freq:
        gender_freq[gender] += 1
    else:
        gender_freq[gender] = 1
#print(gender_freq)
for gender, frq in gender_freq.items():
    output = "There are {n:,} artworks by {g} artists".format(n = int(frq), g = gender)
    print(output)
