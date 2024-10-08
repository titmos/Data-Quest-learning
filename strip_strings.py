test_data = ["1912", "1929", "1913-1923",
             "(1951)", "1994", "1934",
             "c. 1915", "1995", "c. 1912",
             "(1988)", "2002", "1957-1959",
             "c. 1955.", "c. 1970's", 
             "C. 1990-1999"]

bad_chars = ["(",")","c","C",".","s","'", " "]

#this function strips the specified characters from given strings
def strip_characters(string):
    for content in bad_chars:
        string = string.replace(content, "")
    return string

#this module uses the function to strip strings in the list 
stripped_test_data = []
for item in test_data:
    string = strip_characters(item)
    stripped_test_data.append(string)
print(stripped_test_data)
