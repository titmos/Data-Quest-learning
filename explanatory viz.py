#The Familiarity Principle: favors simple graphs over complicated, eye-catching graphs. 
import pandas as pd
import matplotlib.pyplot as plt
top20_deathtoll = pd.read_csv('top20_deathtoll.csv')
plt.barh(top20_deathtoll['Country_Other'], top20_deathtoll['Total_Deaths'])
plt.show()


#The OO Interface
#we use the plt.subplots() function, which generates an empty plot and returns a tuple of two objects
#The Figure (the canvas)
#The Axes (the plot; don't confuse with "axis," which is the x- and y-axis of a plot).
#

import pandas as pd
import matplotlib.pyplot as plt

top20_deathtoll = pd.read_csv('top20_deathtoll.csv')

fig, ax = plt.subplots()
ax.barh(top20_deathtoll['Country_Other'], top20_deathtoll['Total_Deaths'])
plt.show()


#Mobile-Friendly proportions


import pandas as pd
import matplotlib.pyplot as plt

top20_deathtoll = pd.read_csv('top20_deathtoll.csv')

#To change the proportions, we can use the figsize parameter inside the plt.subplots(figsize=(width, height)) function
#ax = plt.subplots(figsize=(3, 5))

fig, ax = plt.subplots(figsize = (4.5, 6))
ax.barh(top20_deathtoll['Country_Other'], top20_deathtoll['Total_Deaths'])
plt.show()


#Mobile-Friendly proportions
#We know that a large part of our audience will read the article on a mobile device. This means our graph needs to have mobile-friendly proportions: small width, larger height

import pandas as pd
import matplotlib.pyplot as plt

top20_deathtoll = pd.read_csv('top20_deathtoll.csv')

#To change the proportions, we can use the figsize parameter inside the plt.subplots(figsize=(width, height)) function
#ax = plt.subplots(figsize=(3, 5))

fig, ax = plt.subplots(figsize = (4.5, 6))
ax.barh(top20_deathtoll['Country_Other'], top20_deathtoll['Total_Deaths'])
plt.show()


#Erasing Non-Data Ink
'''
To remove the axes (also called spines), we can use the Axes.spines[position].set_visible(bool) method, where position is a string indicating the location of the axis: 'left', 'right', 'top', and 'bottom'
fastest way is to use a for loop:

To remove the ticks, we can use the Axes.tick_params(bottom, top, left, right) method.
'''
import pandas as pd
import matplotlib.pyplot as plt

top20_deathtoll = pd.read_csv('top20_deathtoll.csv')

fig, ax = plt.subplots(figsize=(4.5, 6))
ax.barh(top20_deathtoll['Country_Other'],
         top20_deathtoll['Total_Deaths'])
#Remove all four spines from the horizontal bar plot.
for location in ['left', 'right', 'bottom', 'top']:
    ax.spines[location].set_visible(False)

#Remove the bottom and left ticks from the horizontal bar plot.
ax.tick_params(bottom=False, left=False)
plt.show()



#Erasing Redundant Data-Ink
'''
To make the bars less thick, we can use the height parameter inside the Axes.barh() method. The height parameter has a default value of 0.8

To remove some of the x-tick labels, we use the Axes.set_xticks method. Below, we only keep the labels 0, 100000, 200000, and 300000

'''
import pandas as pd
import matplotlib.pyplot as plt

top20_deathtoll = pd.read_csv('top20_deathtoll.csv')
#Reduce the thickness of each bar to a value of 0.45
fig, ax = plt.subplots(figsize=(4.5, 6))
ax.barh(top20_deathtoll['Country_Other'],
       top20_deathtoll['Total_Deaths'], height=0.45)
#Keep only 0, 150000, and 300000 as x-tick labels.
ax.set_xticks([0, 150000, 300000])
for location in ['left', 'right', 'top', 'bottom']:
    ax.spines[location].set_visible(False)
    
ax.tick_params(bottom=False, left=False)
plt.show()


#The Direction of Reading

'''
One problem with our graph is that the tick labels are located at the bottom. People will immediately see the country names, the bars, but they may get confused about the quantities.

To address that, we're going to move the tick labels at the top of the graph using the Axes.xaxis.tick_top() method: 
Then remove the top ticks

Right now, the first thing people will probably see are the x-tick labels. We want readers to focus on the data, so we'll do two things:

We'll color the x-tick labels in grey so they don't stand out visually so much.
We'll color the bars in a shade of red.
Axes.tick_params() method
However, we need to call it one more time because we only want to modify the ticks of the x-axis 
ax.tick_params(axis='x', colors='grey')

To change the colors of the bar, we use the color parameter in the Axes.barh(color) method. 
'''
import pandas as pd
import matplotlib.pyplot as plt

top20_deathtoll = pd.read_csv('top20_deathtoll.csv')

fig, ax = plt.subplots(figsize=(4.5, 6))
ax.barh(top20_deathtoll['Country_Other'],
        top20_deathtoll['Total_Deaths'],
        height=0.45, color='#af0b1e')

for location in ['left', 'right', 'top', 'bottom']:
    ax.spines[location].set_visible(False)
#Move the x-tick labels on top and make sure no ticks are visible.
ax.xaxis.tick_top()   
ax.tick_params(top = False, left=False)
#Color the x-tick labels in grey.
ax.tick_params(axis='x', colors='grey')
ax.set_xticks([0, 150000, 300000])
plt.show()



#Title and Subtitle

''' 
If someone looks at our graph, they won't be able to tell what the quantity means. They see the USA has almost 300,000 of something, but what is that something?
Instead of adding an x-axis label, we'll use the title and subtitle area to give the readers the necessary details.
Add a subtitle that explains what the quantity describes and when the data was collected.
Use the title to show readers more data 

To add a title and a subtitle, we're going to use the Axes.text() method
x and y: the coordinates that give the position of the text.
s: the text.

Axes.text() method has a size parameter we can use to control the text size. Also, it has a weight parameter that enables us to bold the texto0

'''
import pandas as pd
import matplotlib.pyplot as plt

top20_deathtoll = pd.read_csv('top20_deathtoll.csv')

fig, ax = plt.subplots(figsize=(4.5, 6))
ax.barh(top20_deathtoll['Country_Other'],
        top20_deathtoll['Total_Deaths'],
        height=0.45, color='#af0b1e')

for location in ['left', 'right', 'top', 'bottom']:
    ax.spines[location].set_visible(False)
    
ax.set_xticks([0, 150000, 300000])
ax.xaxis.tick_top()
ax.tick_params(top=False, left=False)
ax.tick_params(axis='x', colors='grey')
ax.text(x=-80000, y=23.5, s='The Death Toll Worldwide Is 1.5M+', size=17, weight='bold')
ax.text(x=-80000, y=22.5, s='Top 20 countries by death toll (December 2020)', size = 12)
plt.show()


#Final Touches
''' ext, we're going to left-align the y-tick labels (the country names) by applying a for loop over the country names using Python's zip function. Thereafter, we will leverage the flexibility of matplotlib's Axes.text() method. First, however, we're going to remove the current labels using the Axes.set_yticklabels() method. '''
import pandas as pd
import matplotlib.pyplot as plt

top20_deathtoll = pd.read_csv('top20_deathtoll.csv')

fig, ax = plt.subplots(figsize=(4.5, 6))
ax.barh(top20_deathtoll['Country_Other'],
        top20_deathtoll['Total_Deaths'],
        height=0.45, color='#af0b1e')

for location in ['left', 'right', 'top', 'bottom']:
    ax.spines[location].set_visible(False)
''' First, we'll make the y-tick labels easier to read. We'll add a comma to both 150000 and 300000 to make them more readable — so people don't have to struggle to tell whether it's a 30,000 or a 300,000, for instance we give ax.set_xticks([0, 150000, 300000]) a label.'''
ax.set_xticks([0, 150000, 300000])
#with
ax.set_xticklabels(['0', '150,000', '300,000'])
ax.xaxis.tick_top()
ax.tick_params(top=False, left=False)
ax.tick_params(axis='x', colors='grey')
ax.text(x=-80000, y=23.5,
        s='The Death Toll Worldwide Is 1.5M+',
        weight='bold', size=17)
ax.text(x=-80000, y=22.5,
        s='Top 20 countries by death toll (December 2020)',
        size=12)
''' Next, we're going to left-align the y-tick labels (the country names) by applying a for loop over the country names using Python's zip function. Thereafter, we will leverage the flexibility of matplotlib's Axes.text() method. First, however, we're going to remove the current labels using the Axes.set_yticklabels() method. '''
ax.set_yticklabels([]) # an empty list removes the labels
country_names = top20_deathtoll['Country_Other']
for i, country in zip(range(20), country_names):
    ax.text(x=-80000, y=i-0.15, s=country)
''' Readers who explore the graph will try to determine the approximate death toll for each country. To help them, we're going to draw a vertical line below the 150,000 value. To do that, we use the Axes.axvline(x) method, where x is the x-coordinate where the line begins: '''    
ax.axvline(x=150000, ymin=0.045, c='grey', alpha=0.5)
''' The color of the vertical line is too bright and stands out more than we want. Moreover, the line spans too far vertically and isn't on the same line with the Turkey label. To fix these problems, we're going to use the following:

The ymin parameter to make it shorter — where 0 is the bottom of the plot, and 1 is the top of the plot.
The c parameter to change the color to 'grey'.
The alpha parameter to add transparency to the line. '''

plt.show()
