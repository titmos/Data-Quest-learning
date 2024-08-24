# CODE FROM THE PREVIOUS SCREEN
opened_file = open('AppleStore.csv')
from csv import reader
read_file = reader(opened_file)
apps_data = list(read_file)

def extract(index):
    column = []    
    for row in apps_data[1:]:
        value = row[index]
        column.append(value)    
    return column

def freq_table(a_list):
    new_dict = {}
    for item in a_list:
        if item in new_dict:
            new_dict[item] += 1
        else:
            new_dict[item] = 1
    return new_dict


genres = extract(11)
#print(genres)
genres_ft = freq_table(genres)
print(genres_ft)
