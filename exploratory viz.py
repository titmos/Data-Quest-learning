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

