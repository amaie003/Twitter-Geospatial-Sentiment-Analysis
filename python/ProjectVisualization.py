#!/usr/bin/env python
# coding: utf-8

# In[5]:


import requests
import geopandas as gpd
import pandas as pd
import mapclassify as mc
import zipfile
import io
import os
import matplotlib.pyplot as plt
from shapely.geometry import Point,Polygon
#%matplotlib inline


# In[6]:


from bokeh.tile_providers import STAMEN_TERRAIN, CARTODBPOSITRON_RETINA
from bokeh.io import output_notebook, show, output_file, save
from bokeh.plotting import figure
from bokeh.models import (ColumnDataSource,HoverTool,LogColorMapper)
from bokeh.palettes import RdYlBu11 as palette
import geopandas as gpd
import pysal as ps
#output_notebook()


# In[7]:


url = 'http://www2.census.gov/geo/tiger/TIGER2017/STATE/tl_2017_us_state.zip'

print('Downloading shapefile...')

r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
print("Done")
z.extractall(path='tmp/') # extract to folder - this is a pretty big file
filenames = [y for y in sorted(z.namelist()) for ending in ['dbf', 'prj', 'shp', 'shx'] if y.endswith(ending)] 
print(filenames)


# In[8]:


dbf, prj, shp, shx = [filename for filename in filenames]
usa = gpd.read_file('tmp/'+shp)
print("Shape of the dataframe: {}".format(usa.shape))
print("Projection of dataframe: {}".format(usa.crs))


# In[9]:


stfips = pd.read_csv('../dataset/ContiguousStFips.csv')
usa = usa[usa.STATEFP.astype('int').isin(stfips.STFIPS)].reset_index()
usa = usa.to_crs({'init': 'epsg:3857'})


# In[7]:



fig, ax = plt.subplots(figsize=(20,20), subplot_kw=dict(aspect='equal'))
usa.plot(column='index', cmap='binary',  ax=ax);
ax.set_axis_off()


# In[8]:


sentimentDF = pd.read_csv('../dataset/4.csv')
sentimentDF.columns = ['city','sentiment_mean','count','y','x']

geo = [Point(xy)for xy in zip(sentimentDF['x'],sentimentDF['y'])]

gdf = gpd.GeoDataFrame(
    sentimentDF, geometry=geo)


# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


gdf.head()


# In[10]:


geom =gdf.geometry.name


# In[11]:


gdf.head()


# In[12]:


usa = usa.to_crs({'init': 'epsg:4326'})#convert to 4326
gdf.crs = {'init' :'epsg:4326'}


# In[13]:


gdf.head()


# In[14]:


sentimentMin = gdf['sentiment_mean'].min()
sentimentMax = gdf['sentiment_mean'].max()
gdf['sentiment_MMN'] = (gdf['sentiment_mean']-sentimentMin)/(sentimentMax-sentimentMin)


# In[15]:


countMin = gdf['count'].min()
countMax = gdf['count'].max()
gdf['count_MMN'] = (gdf['count']-countMin)/(countMax-countMin)


# In[16]:


gdf.head(100)


# In[17]:


usa.head()


# In[18]:


gdf.head()


# In[19]:



fig1, ax1 = plt.subplots(figsize=(20,20), subplot_kw=dict(aspect='equal'))
usa.plot(column='GEOID', cmap='bone',  ax=ax1);
gdf.plot(column='sentiment_mean',cmap='RdYlGn',vmin=-1, vmax=1, ax=ax1, markersize=gdf['count_MMN']*2000);
ax1.set_axis_off()
plt.title('USA Not Normalized Sentiment Map')
plt.savefig('../result/NotNormalizedMapOutput.png')


# In[20]:


fig, ax = plt.subplots(figsize=(20,20), subplot_kw=dict(aspect='equal'))
usa.plot(column='GEOID', cmap='bone',  ax=ax);
gdf.plot(column='sentiment_MMN',cmap='RdYlGn',  ax=ax, markersize=gdf['count_MMN']*2000);
ax.set_axis_off()
plt.title('USA Normalized Sentiment Map')
plt.savefig('../result/NormalizedMapOutput.png')


# In[21]:


from bokeh.plotting import figure, show, output_file
from bokeh.tile_providers import get_provider, Vendors
import bokeh.models as bmo
from bokeh.models import LinearColorMapper
from bokeh.palettes import RdYlGn
import numpy as np
TOOLS="pan,wheel_zoom,box_zoom,reset,save"

gdf = gdf.to_crs({'init': 'epsg:3857'})
output_file("../result/InteractiveMapOutput.html")

cmaper= LinearColorMapper(palette = RdYlGn[3])
k = 6378137
gdf["x1"] = gdf['x'] * (k * np.pi/180.0)
gdf["y1"] = np.log(np.tan((90 + gdf['y']) * np.pi/360.0)) * k
        
        
tile_provider = get_provider(Vendors.CARTODBPOSITRON)
p = figure(x_range=(-14000000, -8000000), y_range=(3000000, 7000000),
           x_axis_type="mercator", y_axis_type="mercator")

source = ColumnDataSource(data=dict(longitude=gdf['x1'], nname = gdf['city'],nsize=gdf['count'],sentRound = round(gdf['sentiment_mean'], 2), nSS= gdf['sentiment_mean'],latitude=gdf['y1'],size = 4+(30*gdf['count_MMN'])))

p.add_tile(tile_provider)
circles = p.circle(x='longitude', y='latitude',size='size', color = {'field':'nSS','transform':cmaper},source=source)

p_hover = HoverTool(renderers=[circles])
p_hover.point_policy = "follow_mouse"
p_hover.tooltips=[
    ("City", "@nname"),
    ("tweet count", "@nsize"),
    ("sentiment Score","@sentRound")
]

p.add_tools(p_hover)


show(p)


# In[22]:


gdf[140:]

