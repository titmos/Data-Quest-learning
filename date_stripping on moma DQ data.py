test_data = ["1912", "1929", "1913-1923",
             "(1951)", "1994", "1934",
             "c. 1915", "1995", "c. 1912",
             "(1988)", "2002", "1957-1959",
             "c. 1955.", "c. 1970's", 
             "C. 1990-1999"]

bad_chars = ["(",")","c","C",".","s","'", " "]

def strip_characters(string):
    for char in bad_chars:
        string = string.replace(char,"")
    return string

def process_date(string):
    if "-" in string:
        data_list = string.split("-")
        date_one = int(data_list[0])
        date_two = int(data_list[1])
        average = (date_one + date_two) / 2
        average = round(average)
    else:
        return int(string)
    return average


stripped_test_data = ['1912', '1929', '1913-1923',
                      '1951', '1994', '1934',
                      '1915', '1995', '1912',
                      '1988', '2002', '1957-1959',
                      '1955', '1970', '1990-1999']

processed_test_data = []
for item in stripped_test_data:
    processed_date = process_date(item)
    processed_test_data.append(processed_date)

for item in moma:
    date = item[6]
    stripped_date = strip_characters(date)
    processed_date = process_date(stripped_date)
    item[6] = processed_date
print(moma[:6])
