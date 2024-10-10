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
