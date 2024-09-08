# The `potus` list of lists is available from
# the earlier screen where we created it
#The format of the app_start_date column is {month}/{day}/{two digit year} {hour 24hr time}:{minute}.
date_format = ("%m/%d/%y %H:%M")
for item in potus:
    appt_start_date = item[2]
    appt_start_date = dt.datetime.strptime(appt_start_date, date_format)
    item[2] = appt_start_date



# The `potus` list of lists is available from
# the earlier screen where we created it
#The format of the app_start_date column is {month}/{day}/{two digit year} {hour 24hr time}:{minute}.
import datetime as dt
date_format = "%m/%d/%y %H:%M"
for item in potus:
    appt_start_date = item[2]
    appt_start_date = dt.datetime.strptime(appt_start_date, date_format)
    #to prevent the datetime.datetime appearing in the list
    appt_start_date = appt_start_date.isoformat()
    item[2] = appt_start_date
print(potus[:10])




#Let's use the datetime.strftime() method to create a formatted frequency table and analyze the appointment dates in our data set. We'll do the following:
#1.Iterate over each of the datetime objects we created on the previous screen
#2.Create a string containing the month and year from each datetime object
#3.Create a frequency table for the month/year of the appointments
visitors_per_month = {}
for item in potus:
    appt_start_date = item[2]
    appt_start_date = appt_start_date.strftime("%B, %Y")
    if appt_start_date not in visitors_per_month:
        visitors_per_month[appt_start_date] = 1
    else:
        visitors_per_month[appt_start_date] += 1
print(visitors_per_month)



import datetime as dt
appt_times = []
for item in potus:
    appt_dt = item[2]
    appt_t = appt_dt.time()
    appt_times.append(appt_t)
print(appt_times)
"""Because we have already converted the app_start_date column to datetime objects, it's easy for us to convert them to time objects. Let's loop over our potus list of lists and create a list of appointment times that we can analyze on the following screen."""



'''calculate the time between dt_2 and dt_1, and assign the result to answer_1.
Add 56 days to dt_3, and assign the result to answer_2.
Subtract 3600 seconds from dt_4, and assign the result to answer_3.'''
dt_1 = dt.datetime(1981, 1, 31)
dt_2 = dt.datetime(1984, 6, 28)
dt_3 = dt.datetime(2016, 5, 24)
dt_4 = dt.datetime(2001, 1, 1, 8, 24, 13)
answer_1 = dt_2 - dt_1
answer_2 = dt_3 + dt.timedelta(days = 56)
answer_3 = dt_4 - dt.timedelta(seconds = 3600)
print(answer_1, '\n', answer_2, '\n', answer_3)


"""
We have provided code that converts the appt_end_date to datetime objects.
Calculate the minimum key in appt_lengths, and assign the result to min_length.
Calculate the maximum key in appt_lengths, and assign the result to max_length."""
appt_lengths = {}
for row in potus:
    start_date = row[2]
    end_date = row[3]
    end_date = dt.datetime.strptime(end_date, "%m/%d/%y %H:%M")
    row[2] = start_date
    row[3] = end_date
    length = end_date - start_date
    if length not in appt_lengths:
        appt_lengths[length] = 1
    else:
        appt_lengths[length] += 1
#print(appt_lengths)
min_length = min(appt_lengths)
max_length = max(appt_lengths)
