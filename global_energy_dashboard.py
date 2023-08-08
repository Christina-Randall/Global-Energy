#!/usr/bin/env python
# coding: utf-8

# # Global Energy Heat Map (Production - Consumption)

# ## Prepping the Data

# In[1]:


#imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import os
import folium
import json


# In[2]:


#files
total_consumption = pd.read_pickle ('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/total_consumption.pkl')
total_production = pd.read_pickle ('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/total_production.pkl')


# In[3]:


total_consumption2 = total_consumption[['country',  'total_consumption']]
total_production2 = total_production[['country', 'total_production']]


# In[4]:


total_consumption2.head(2)


# In[5]:


country_list = total_production2['country'].tolist()
country_list


# In[6]:


total_production2['country'] = total_production2['country'].str.strip()


# In[7]:


country_list2 = total_production2['country'].tolist()
country_list2


# In[8]:


total_consumption2['country'] = total_consumption2['country'].str.strip()


# In[9]:


total_production2[total_production2['country'] == 'United States']


# In[10]:


total_consumption2[total_consumption2['country'] == 'United States']


# ## Making the DF

# In[11]:


total_energy_df = total_consumption2.merge(total_production2, on='country')
total_energy_df['net_balance'] = total_energy_df['total_production'] - total_energy_df['total_consumption']


# In[12]:


total_energy_df.columns


# ## Setting the Json Files

# In[13]:


f = open('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/countries (2).geojson')
countries_data = json.load(f)
# Iterating through the json list
for i in countries_data['features']:
    print(i)


# ## Coding the Heat Map

# In[ ]:


# Create a map with the desired size (800x400 pixels)
map = folium.Map(location=[0, 0], zoom_start=1.4, width=800, height=400)

# Choropleth maps bind Pandas DataFrames and JSON geometries. This allows us to quickly visualize data combinations
choropleth_map = folium.Choropleth(
    geo_data=countries_data,
    data=total_energy_df,
    columns=['country', 'net_balance'],
    key_on='feature.properties.ADMIN',  # Check your JSON file to see where the KEY is located
    fill_color='RdYlGn',  # Change the color scale here. Example: RdYlGn, YlGnBu, YlOrRd, etc.
    fill_opacity=0.6,
    line_opacity=0.1,
    legend_name="Net Balance",  # Change the legend name as desired
    map_title = "World Net Energy Balance"
    )

# Add the choropleth map to the main map
choropleth_map.add_to(map)

# Add layer control to the map
folium.LayerControl().add_to(map)


# # Bar Chart of 2021 Energy Production with Buttons

# In[15]:


import pandas as pd
import numpy as np
import panel as pn
pn.extension('tabulator')

import hvplot.pandas


# In[16]:


coal = pd.read_csv('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/Coal Map Data.csv')
petrol = pd.read_csv('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/2021 Petrol Country Production.csv')
nuclear = pd.read_csv('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/2021 Nuclear Country Production.csv')
natural_gas = pd.read_csv('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/2021 Natural Gas Country Production.csv')


# In[17]:


coal['coal_production'].nlargest(10)


# In[18]:


coal = coal.drop(columns = ['index'])


# In[19]:


petrol = petrol.drop(columns=petrol.columns[0])
nuclear = nuclear.drop(columns=nuclear.columns[0])
natural_gas = natural_gas.drop(columns=natural_gas.columns[0])


# In[20]:


coal


# In[21]:


df = pd.merge(coal, petrol, on='country')
df = pd.merge(df, nuclear, on='country')
df = pd.merge(df, natural_gas, on='country')


# In[22]:


df


# ## Coding the Bar Chart

# In[23]:


# Making DataFrame Pipeline Interactive
idf = df.interactive()


# In[24]:


yaxis_buttons = pn.widgets.RadioButtonGroup(
    name='Y axis', 
    options=['coal_production', 'petrol_production', 'nuclear_production', 'natural_gas_production'], 
    button_type='success'
)

#This is the line of code that needs to select the top ten countries for each column
continents_excl_world = df['country']

production_bar_pipeline = (
    idf[
        (idf.country.isin(continents_excl_world))
    ]
    .groupby(['country'])[yaxis_buttons].sum()
    .to_frame()
    .reset_index()
    .reset_index(drop=True)
)


# In[25]:


top_ten_countries = ( idf[idf.country.isin(continents_excl_world)] .groupby('country')[yaxis_buttons].sum() .nlargest(10) .reset_index())


# In[26]:


top_ten_countries


# In[27]:


production_bar_plot = top_ten_countries.hvplot(kind='bar', 
                                                     x='country', 
                                                     y=yaxis_buttons, 
                                                     title='2021 Energy Production by Country & Type',
                                                     rot = 45
                                              )
production_bar_plot


# In[28]:


# We have the production above so we will be focusing on the consumption in this part and then making a double bar chart in the next
consumption_df = pd.read_csv('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/2021 Counsumption Energy DF.csv')


# In[29]:


consumption_df


# In[30]:


consumption_df.columns


# In[31]:


consumption_df = consumption_df.drop(columns=['Unnamed: 0']).rename(columns={
    'nuclear_and_renewables': 'nuclear_consumption',
    'coal': 'coal_consumption',
    'petrol': 'petrol_consumption',
    'natural_gas': 'natural_gas_consumption'
})


# In[32]:


consumption_df


# # Historical Energy Line Chart

# ## Cleaning & Transforming the Data

# In[33]:


total_production = total_production.applymap(lambda x: x.strip() if isinstance(x, str) else x)
usa_prod = total_production[total_production['country'] == 'United States']
usa_prod = usa_prod.drop(['continent', 'total_production'], axis=1)
usa_prod = usa_prod.T.drop('country').reset_index()
usa_prod = usa_prod.rename(columns = {'index': 'year'})
usa_prod = usa_prod.rename(columns={usa_prod.columns[1]: 'energy_production'})


# In[34]:


usa_prod


# In[35]:


total_consumption = total_consumption.applymap(lambda x: x.strip() if isinstance(x, str) else x)
usa_con = total_consumption[total_consumption['country'] == 'United States']
usa_con = usa_con.drop(['continent', 'total_consumption'], axis=1)
usa_con = usa_con.T.drop('country').reset_index()
usa_con = usa_con.rename(columns = {'index': 'year'})
usa_con = usa_con.rename(columns={usa_con.columns[1]: 'energy_consumption'})


# In[36]:


usa_line_df = usa_prod.merge(usa_con, on='year')
usa_line_df['energy_production'] = usa_line_df['energy_production'].astype('float64')
usa_line_df['energy_consumption'] = usa_line_df['energy_consumption'].astype('float64')


# In[37]:


usa_line_df


# In[38]:


usa_line_df.dtypes


# ## Coding the Line Chart

# In[39]:


import pandas as pd
import hvplot.pandas  # Importing hvplot for pandas DataFrame support
import panel as pn

# Function to create the line chart using hvplot
def usa_line_chart3(data_df, x_column, y_column, title):
    line_chart = data_df.hvplot.line(
        x=x_column, 
        y=[y_column, 'energy_production'], 
        xlabel='Year', 
        ylabel='Energy in Quadrillion BTU', 
        title=title, 
        legend='bottom',  # Move the legend to the bottom of the plot
        width=800,  # Set the width to 800 pixels
        height=600,
        rot = 45
    )
    return line_chart

# Create the line chart using usa_line_chart3 function
chart = usa_line_chart3(usa_line_df, x_column='year', y_column='energy_consumption', title='US Energy Production vs Consumption')

# Display the line chart (it will be automatically shown in the notebook)
chart


# # Population Projections Line Chart

# In[40]:


us_population = pd.read_csv('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/united-states-population-2023-07-17.csv')


# In[41]:


us_population


# In[42]:


us_population.dtypes


# In[43]:


us_population_projection = pd.read_csv('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/Population Projections based on Immigration.csv')


# In[44]:


us_population_projection


# In[45]:


us_population_projection.dtypes


# #continued in Hvplot Charts tab

# # Hvplot Charts

# In[46]:


import hvplot.pandas
import holoviews as hv

hv.extension('bokeh')


# In[47]:


year = pn.widgets.IntSlider(name='year', start=1970, end=2022, step=4)


# In[48]:


import pandas as pd
import hvplot.pandas  

df1 = us_population
df2 = us_population_projection

# Creating a line chart using hvplot
line_chart = df1.hvplot.line(x='year', y='population_in_mil', xlabel='Year', ylabel='Population (in millions)', title='US Population Projection (1970-2060)') *              df2.hvplot.line(x='year', y='population_in_millions', xlabel='Year', ylabel='Population (in millions)', title='US Population Projection')

# Display the combined chart (it will be automatically shown in the notebook)
line_chart


# # Population Projection Energy Bar Chart

# ## Making the DataFrame

# In[49]:


energy_data = {
    'year': [2021, 2060],
    'energy_consumed': [97.907, 116.5]
}


# In[50]:


energy_data


# In[51]:


energy_projection = pd.DataFrame(energy_data)


# In[52]:


import pandas as pd
import matplotlib.pyplot as plt
import panel as pn

# Data for the DataFrame
data = {
    'year': [2021, 2060],
    'energy_consumed': [98, 117]
}

# Create the DataFrame
energy_projection = pd.DataFrame(data)

# Set 'year' column as the index to use it as the y-axis in the horizontal bar chart
energy_projection.set_index('year', inplace=True)

# Create the horizontal bar chart using pandas plot.barh()
fig, ax = plt.subplots(figsize=(8, 1.2))  # Increase the size of the figure here

bar_chart = energy_projection.plot.barh(
    color=['dodgerblue'],
    legend=False,
    width=0.5,
    ax=ax,  # Specify the axis to use for the plot
    
)

title = 'Projected U.S. Energy Demand by 2060'
bar_chart.set_title(title, loc='left')

# Add the total values at the end of each bar
for index, value in enumerate(energy_projection['energy_consumed']):
    bar_chart.annotate(str(value), xy=(value, index), xytext=(value + 0.5, index), ha='left', va='center')

plt.tight_layout()  # Adjust the layout to prevent overlapping


# In[53]:


from PIL import Image


# In[54]:


#imports
us = Image.open('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/PNGs/2021 Energy Consumption & Production for United States.png')
china = Image.open('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/PNGs/2021 Energy Consumption & Production for China.png')


# # Dashboard Layout

# In[55]:


#Dashboard 
header_text = "# United States Energy Analysis\n####How Stable is the U.S. Energy Market in terms of keeping up with U.S. Demand Indpendent of Other Energy Sources?"
header_pane = pn.pane.Markdown(header_text, style={'font-size': '20px', 'text-align': 'center'})

#Tabs
tabs = pn.Tabs()

#tab1-----------------------------------------------------
tab1_chart1 = pn.pane.plot.Folium(map, width=800, height=375)

tab1_title = "### On a global level how does the US rank in net energy indpendence?"
tab1_tstyle = pn.pane.Markdown(tab1_title, style={'font-size': '18px', 'text-align': 'center'})

tab1_text1 = '# Introduction \n U.S. Energy consumption vs production is an eye opening look into the stability of the U.S. Energy Market. If you were to look only at the last couple years you may not notice anything but looking at it from a global level you can see that the U.S. has ran a net deficit since 1970. Keeping this in mind, I decided to further explore the U.S. energy market and future energy demands to better understand how reliant the U.S. is on outside energy sources.'
tab1_text2 = '# Context \n This heat map shows the total sum of energy from each country (total production – total consumption.) starting from the 1970’s to 2022. You can see that the U.S. relies on energy imports to keep up with its demand. Saudi Arabia (across from the red sea) and Russia have had the most excess energy on balance in the world.'
tab1_text3 = '### Global Net Energy Heat Map (1970-2022)'
tab1_text4 = '# Insight \n This means that the U.S. will be dependent on energy imports until they either reduce the country’s demand or increase their energy production. Most of these imports will likely be from Saudi Arabia or Russia though Canada and Mexico may also be options as they share a border with the U.S.'
tab1_text5 = '# Dashboard Notes\n All Energy Units are in Quadrillion British Thermal Units. Citations are on the last page. Lastly, this is an interactive dashboard, feel free to further explore it using your mouse.'

tab1_tab2 = pn.Row(pn.Column(tab1_text1, tab1_text3, tab1_chart1), (pn.Column(tab1_text2, tab1_text4, tab1_text5)))
tab1_tab = pn.Column(tab1_tstyle, tab1_tab2)

#tab2-----------------------------------------------------
tab2_chart1 = chart

tab2_title = "### How does the U.S. net energy look historically?"
tab2_tstyle = pn.pane.Markdown(tab2_title, style={'font-size': '18px', 'text-align': 'center'})

tab2_text1 = '# Context \n This line chart shows U.S. production vs consumption from 1980 through 2021. We can see that energy consumption in the U.S. grows significantly faster until 2008 which is where U.S. Energy production significantly ramps up and starts to meet demand in 2019. It is only a very recently that the U.S. has been net energy positive.'
tab2_style1 = pn.pane.Markdown(tab2_text1, style={'font-size': '15px'})
tab2_text2 = '# Insight \n While the U.S. has only been energy independent recently this is likely due to historical factors such as cheaper energy costs and a stronger dollar making it not cost effective for the U.S. to produce their own energy. '
tab2_style2 = pn.pane.Markdown(tab2_text2, style={'font-size': '15px'})

tab2_tab1 = pn.Row(tab2_chart1, pn.Column(tab2_style1, tab2_style2))
tab2_tab = pn.Column(tab2_tstyle, tab2_tab1)

#tab3-----------------------------------------------------
tab3_title = "### Based on future population expectations, how much energy will the U.S. need in the future?"
tab3_tstyle = pn.pane.Markdown(tab3_title, style={'font-size': '18px', 'text-align': 'center'})

tab3_chart1 = line_chart.opts(width=800, height=400)
tab3_chart2 = pn.pane.Matplotlib(fig, width=800, height=100)

tab3_text1 = '# Context \n The US Population Projection line chart looks at both the U.S. historical population growth as well as expected future population growth as predicted by the U.S. Census Bureau in the main scenario. The population projection in red shows steady growth following the same trend from past growth.'
tab3_style1 = pn.pane.Markdown(tab3_text1, style={'font-size': '15px'})
tab3_text2 = 'The 98 QBtu to 117 QBtu increases between years represents a 19.4% increase in energy demand based on the U.S. population projections.'
tab3_style2 = pn.pane.Markdown(tab3_text2, style={'font-size': '15px'})
tab3_text3 = '# Insight \n It’s interesting to note that while energy demand shot up since the 1960’s, the U.S. population growth has been comparatively steady over the same period of time. The average energy use per person in the U.S. has increased given the advancements since the 1960’s. However, it makes it difficult to tell how much energy will be needed in future years. I have estimated the future amount below the chart, assuming that the average person usage is stagnant (based on 2021).'
tab3_style3 = pn.pane.Markdown(tab3_text3, style={'font-size': '15px'})

tab3_tab3 = pn.Column(tab3_chart1, tab3_chart2, tab3_style2)
tab3_tab2 = pn.Column(tab3_style1, tab3_style3)
tab3_tab1 =pn.Column(pn.Row(tab3_tab3, tab3_tab2))
tab3_tab = pn.Column(tab3_tstyle, tab3_tab1)

#tab4-----------------------------------------------------
tab4_title = "### If the U.S. is not able to keep up with its energy demand, who can we import from?"
tab4_tstyle = pn.pane.Markdown(tab4_title, style={'font-size': '18px', 'text-align': 'center'})

tab4_chart1 = production_bar_plot.opts(width=800, height=300)
tab4_chart2 = pn.pane.PNG(object=us, width=400, height=150)
tab4_chart3 = pn.pane.PNG(object=china, width=400, height=150)

tab4_text1 = '# Context \n This bar chart shows the top 10 producers for each energy type as of 2021. So far we have looked at total energy production vs total energy consumption (or net energy). This disregards that different energy types are needed by different sectors such as households or commercial. Here you can press on the different buttons to see which countries produce the most of a given energy. China, for example, greatly surpasses every other country in its coal production, due to its manufacturing type economy. Below you can see a few countries breakdown of their energy markets as pie charts. '
tab4_style1 = pn.pane.Markdown(tab4_text1, style={'font-size': '15px'})
tab4_text2 = '### Pie Charts'
tab4_style2 = pn.pane.Markdown(tab4_text2, style={'font-size': '18px'})
tab4_text3 = '# Insight \n If the U.S. needs to source from other countries it should focus on countries that are likely to have surpluses. However, for natural gas the U.S. focuses on producing their own as international pipelines can be tricky to maintain as seen with the recent Russian pipeline leak.'
tab4_style3 = pn.pane.Markdown(tab4_text3, style={'font-size': '15px'})

tab4_tab4 = pn.Row(tab4_chart2, tab4_chart3)
tab4_tab3 = pn.Column(tab4_chart1, tab4_style2, tab4_tab4)
tab4_tab2 = pn.Column(tab4_style1, tab4_style3)
tab4_tab1 = pn.Column(pn.Row(tab4_tab3, tab4_tab2))
tab4_tab = pn.Column(tab4_tstyle, tab4_tab1)

#tab5-----------------------------------------------------
tab5_title = "### What does this mean for the future of U.S. energy independence?"
tab5_tstyle = pn.pane.Markdown(tab5_title, style={'font-size': '18px', 'text-align': 'center'})

tab5_text1 ='# Energy Implications\nIf the US cannot keep up with its energy demands, they will be forced to buy energy from other countries. With the decline of globalism, this poses a greater risk for US energy security as energy prices and country relations remain volatile.'
tab5_style1 = pn.pane.Markdown(tab5_text1, width=700)
tab5_text4 = 'In addition, with global warming taking precedence in Europe, energy is poised to become increasingly expensive to import. Given that by 2060 the U.S. is likely to see a nearly 20% increase in energy demand, the US must increase its energy production over the next forty to ensure stable energy prices. It is paramount that the US keep up with its demand or risk energy outages and volatile prices.'
tab5_style4 = pn.pane.Markdown(tab5_text4, width=700)

tab5_text5 = '# Next Steps\n Based on this research the next steps to further look into the U.S. energy independence would be looking at each energy type more closely to determine which energy sources the U.S. is not producing enough of and determining wether or not the U.S. has the capabilities to produce enough of the energy to sustain future demand.'
tab5_style5 = pn.pane.Markdown(tab5_text5, width=700)

tab5_text2 = '# Further Resources'
tab5_style2 = pn.pane.Markdown(tab5_text2, width=700)

r1 = '[Global Energy Tracker]( https://globalenergymonitor.org/)'
r2 = '[Global Coal Tracker]( https://www.carbonbrief.org/mapped-worlds-coal-power-plants/)'

tab5_text3 = '# Citations'
tab5_style3 = pn.pane.Markdown(tab5_text3, width=700)

c1 = '[Kaggle : Energy Data](https://www.kaggle.com/datasets/akhiljethwa/world-energy-statistics) /n Sourced from the EIA.'
c2 = '[US Census Bureau : Population Projections](https://www.census.gov/data/datasets/2017/demo/popproj/2017-popproj.html)'
c3 = '[Json File : Heat Map](https://datahub.io/core/geo-countries#pandas)'
c4 = '[United Nations : US Population Data](https://www.macrotrends.net/countries/USA/united-states/population)'

tab5_citations = pn.Column(tab5_style3, c1, c2, c3, c4, tab5_style2, r2, r1 )
tab5_tab1 =pn.Row(pn.Column(tab5_style1, tab5_style4, tab5_style5),tab5_citations)
tab5_tab = pn.Column(tab5_tstyle, tab5_tab1)

# Append the tabs
tabs.append(('Global Net Energy', tab1_tab))
tabs.append(('US Energy Historically', tab2_tab))
tabs.append(('Population Projections', tab3_tab))
tabs.append(('Global Energy Production', tab4_tab))
tabs.append(('Energy Implications', tab5_tab))


#sizing the tabs
tabs.width = 1300
tabs.height = 700

#Footer
footer_text = "### By Christina Randall"
footer_pane = pn.pane.Markdown(footer_text, style={'font-size': '15px', 'text-align': 'center'})

# Show the dashboard
main = pn.Column(header_pane, tabs, footer_pane)

# Show the dashboard
main.servable()


# In[56]:


import gc

# Enable the garbage collector
gc.enable()

# Create some objects
class SampleObject:
    def __init__(self):
        pass

# Create references to objects
obj1 = SampleObject()
obj2 = SampleObject()

# Break references to objects
obj1 = None
obj2 = None

# Manually trigger garbage collection
collected = gc.collect()

# Analyze collected objects
print(f"Collected {collected} unreachable objects")

# List uncollectable objects
uncollectable = gc.garbage
print(f"Uncollectable objects: {uncollectable}")


# In[ ]:




