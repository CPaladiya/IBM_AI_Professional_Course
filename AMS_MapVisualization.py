from cProfile import label
from lib2to3.pytree import Base
from sqlite3 import TimeFromTicks
import pandas as pd
import numpy as np
import geopy as gp
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import rcParams
import math

# if the preprocessed data is already available or not
PreProcessData = False

#if data is not preprocessed, we need to make sure it is already processed
if(PreProcessData):
    
    ## Data Fetching ----------------------------------------
    BaseData = pd.read_csv("Data/MapVisualizer/MotherJonesDataBase.csv") # Reading the database
    BaseData = BaseData[['location','date','total_victims']]  # Only fetching the values that is useful for this task

    ## Date Conversion & Data type preprocessing ----------------------------------------
    # converting dates into dateTime format, sorting it by date and fetching the year and month
    BaseData['date'] = pd.to_datetime(BaseData['date'])
    BaseData = BaseData.sort_values(by="date")
    BaseData['year'] = pd.DatetimeIndex(BaseData['date']).year
    BaseData['month'] = pd.DatetimeIndex(BaseData['date']).month
    BaseData = BaseData.drop("date", axis=1)#dropping the date column
    BaseData = BaseData.dropna()# Dropping rows that has any missing values

    # Correct numerical type is required,
    BaseData['total_victims'] = pd.to_numeric(BaseData['total_victims'], downcast='signed', errors = 'coerce')
    BaseData = BaseData.dropna() #removing any NAN rows
    BaseData['total_victims'] = BaseData['total_victims'].astype(np.int16)
    BaseData['location'] = BaseData['location'].astype("string") # location has to have all the string values,
    BaseData = BaseData.dropna(subset=['location','total_victims','year','month'])# any row that has NAN remove that

    ## Fetching location data for various addresses ----------------------------------------
    from geopy.extra.rate_limiter import RateLimiter
    geolocator = gp.Nominatim(user_agent="AA_Map_Visualizer.ipynb") # Setting up application to find lattitude and longitude of various places
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1) # since we will be doing repeated calls to nominatim, we need to use limiter to slow it down to avoid errors.
    BaseData['LocationObj'] = BaseData['location'].apply(geocode)# now let's use this geolocator to fetch latitude and longitude for all the cities we want to plot on the graph

    # Load latitude and longitude using LocationObj returned by Geocoder
    # See if the object type is of None, then just o/p NAN
    BaseData['latitude'] = BaseData['LocationObj'].apply(lambda x : x.latitude if x != None else np.NAN)
    BaseData['longitude'] = BaseData['LocationObj'].apply(lambda x : x.longitude if x != None else np.NAN)
    BaseData.drop('LocationObj', axis=1) # we don't need LocationObj anymore so remove it
    BaseData.dropna(subset=['latitude', 'longitude']) # Remove any row that has NAN values

else:
    BaseData = pd.read_csv("Data/MapVisualizer/preprocessedAMSData.csv")
    
    
## Setting up map of USA ----------------------------------------
from mpl_toolkits.basemap import Basemap as bmp
import matplotlib.animation as animation

# Setting up USA coordinates
llon=-130 # lower left hand map corner longitude
ulon=-65 # Upper right hand map corner longitude
llat=25
ulat=60

mapVisualizer, ax = plt.subplots(figsize=(12,8))
# creating a USA map
mp = bmp(projection='merc',resolution='i', area_thresh=100000 ,lon_0 = 0, lat_0 =90  ,llcrnrlon = llon, llcrnrlat = llat, urcrnrlon = ulon, urcrnrlat = ulat) 
# filling up land and water with different color to show boundaries and also drawing the coastlines
mp.drawmapboundary(fill_color='aqua')
mp.fillcontinents(color='white')
mp.drawcoastlines()
# drawing the boundaries of the states
mp.readshapefile('Data/MapVisualizer/USAShapeFile', 'States')

# gathering data to be plotted on the map in terms of longitude and latitude of the place where MS event took place
BaseData['ProcLons'], BaseData['ProcLats'] = mp(list(BaseData['longitude'].values), list(BaseData['latitude'].values))
BaseData.dropna(subset=['ProcLons', 'ProcLats'])
lons = list(BaseData['ProcLons'].values)
lats = list(BaseData['ProcLats'].values)
victims = list(BaseData['total_victims'].values)
#categorizing victim rows into bins of 5 and see which bins they fall into
victimsCategory = [int(x/5) for x in victims]
FRAMES = len(lons)

# generating sizes of dots based on the fatalities
stdSizes = (np.log(BaseData['total_victims']))*50000/BaseData['total_victims'].max()
# coloring with considering outliers and trying to associate colors from range 0 to 1
colors = plt.get_cmap('Reds')([x*0.062 if x<17 else 1 for x in victimsCategory]) 
edgecolor = [[0,0,0,1] for x in colors]
myscat = mp.scatter([], [], marker='o',zorder = 5, linewidths = 2)

# adding a color bar based on the number of victims falling into various bins
cmap = mpl.cm.Reds
bounds = [0,5,10,15,20,25,30,35,40,45,55,80,100,720]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
mp.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='Victims (~59% Injured + ~41% Passed Away)', ticks=bounds)

# All the fixed text on the map
HorzOffset = 250000
VertTextAlign = 4500000
plt.text(2100000, 5000000, "Mass Shootings in USA", weight='bold', size=20, color='black', rasterized=True, backgroundcolor='red')
plt.text(1000000 + HorzOffset, VertTextAlign, "Place", weight='bold', size=12, color='white', rasterized=True, backgroundcolor='black')
plt.text(1800000 + HorzOffset, VertTextAlign, "Date", weight='bold', size=12, color='white', rasterized=True, backgroundcolor='black')
plt.text(2600000 + HorzOffset, VertTextAlign, "Victims", weight='bold', size=12, color='white', rasterized=True, backgroundcolor='black')
plt.text(3700000 + HorzOffset, VertTextAlign, "Total Victims Since XYZ", weight='bold', size=12, color='white', rasterized=True, backgroundcolor='black')

plottedText = []
ModifiedSize = []
def update(i):
    
    x = lons[:i]
    y = lats[:i]
    # to animate latest plotted size with bigger dia
    ModifiedSize = stdSizes[:i]
    # reset the previously modified value
    if(len(ModifiedSize) > 1):
        ModifiedSize[i-2] /= 3
    # emplify the latest value to make it pop
    if(len(ModifiedSize) > 0):
        ModifiedSize[i-1] *= 3
    # updating the map with latest dots
    myscat.set_offsets(np.c_[x,y])
    myscat.set_sizes(ModifiedSize)
    myscat.set_color(colors[:i])
    myscat.set_edgecolor(edgecolor[:i])
    # updating the map with statistics
    return myscat,

# Running the animation
anim = animation.FuncAnimation(plt.gcf(), update, frames = FRAMES, interval=500)
plt.show()

# Add color bar
# Add month and date
# Add location
# Add number of victims at location
# Add total number of victims