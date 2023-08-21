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

# In[14]:


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

# Display the map
map


# # Global Energy Production 2021 with Buttons Bar Chart

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


coal = coal.drop(columns = ['index'])


# In[18]:


petrol = petrol.drop(columns=petrol.columns[0])
nuclear = nuclear.drop(columns=nuclear.columns[0])
natural_gas = natural_gas.drop(columns=natural_gas.columns[0])


# In[19]:


coal


# In[20]:


df = pd.merge(coal, petrol, on='country')
df = pd.merge(df, nuclear, on='country')
df = pd.merge(df, natural_gas, on='country')


# In[21]:


df


# ## Coding the Bar Chart

# In[22]:


# Making DataFrame Pipeline Interactive
idf = df.interactive()


# In[23]:


yaxis_buttons = pn.widgets.RadioButtonGroup(
    name='Y axis', 
    options=['coal_production', 'nuclear_production', 'natural_gas_production'], 
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


# In[24]:


top_ten_countries = ( idf[idf.country.isin(continents_excl_world)] .groupby('country')[yaxis_buttons].sum() .nlargest(10) .reset_index())


# In[25]:


top_ten_countries


# In[26]:


production_bar_plot = top_ten_countries.hvplot(kind='bar', 
                                                     x='country', 
                                                     y=yaxis_buttons, 
                                                     title='2021 Energy Production by Country & Type',
                                                     rot = 45
                                              )
production_bar_plot


# # U.S. Energy Production vs Consumption Line Chart

# ## Cleaning & Transforming the Data

# In[27]:


total_production = total_production.applymap(lambda x: x.strip() if isinstance(x, str) else x)
usa_prod = total_production[total_production['country'] == 'United States']
usa_prod = usa_prod.drop(['continent', 'total_production'], axis=1)
usa_prod = usa_prod.T.drop('country').reset_index()
usa_prod = usa_prod.rename(columns = {'index': 'year'})
usa_prod = usa_prod.rename(columns={usa_prod.columns[1]: 'energy_production'})


# In[28]:


usa_prod


# In[29]:


total_consumption = total_consumption.applymap(lambda x: x.strip() if isinstance(x, str) else x)
usa_con = total_consumption[total_consumption['country'] == 'United States']
usa_con = usa_con.drop(['continent', 'total_consumption'], axis=1)
usa_con = usa_con.T.drop('country').reset_index()
usa_con = usa_con.rename(columns = {'index': 'year'})
usa_con = usa_con.rename(columns={usa_con.columns[1]: 'energy_consumption'})


# In[30]:


usa_line_df = usa_prod.merge(usa_con, on='year')
usa_line_df['energy_production'] = usa_line_df['energy_production'].astype('float64')
usa_line_df['energy_consumption'] = usa_line_df['energy_consumption'].astype('float64')


# In[31]:


usa_line_df


# In[32]:


usa_line_df.dtypes


# ## Coding the Line Chart

# In[33]:


import pandas as pd
import panel as pn
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

us_energy_history_line_chart = usa_line_df.hvplot.line(
    x='year', 
    y=['energy_consumption', 'energy_production'], 
    xlabel='Year', 
    ylabel='Energy in Quadrillion BTU', 
    title='US Energy Production vs Consumption', 
    legend='bottom',  
    width=800,  
    height=600,
    rot=45,
)


# Create the annotations
annotation = hv.Text(1990, 73, 'Rise of Natural Gas \n Production')
annotation1 = hv.Text(2004, 72.3, 'Energy Indpendence \n & Security Act')
annotation2 = hv.Text(2020, 90, 'Renewable \n Energy \n Overtakes \n Coal')
annotation3 = hv.Text(2013, 75, 'Fracking Takes \n Off in the U.S.')

# Create a black dot at (2012, 70)
black_dot = hv.Points([(1990, 70.6)], label='Black Dot').opts(size=5, fill_color='black', line_color='black')
black_dot1 = hv.Points([(2007, 71.3)], label='Black Dot').opts(size=5, fill_color='black', line_color='black')
black_dot2 = hv.Points([(2020, 96)], label='Black Dot').opts(size=5, fill_color='black', line_color='black')
black_dot3 = hv.Points([(2010, 75)], label='Black Dot').opts(size=5, fill_color='black', line_color='black')


# Combine the line chart, annotations, and the black dot using the overlay
overlay = (us_energy_history_line_chart * annotation * annotation1 * annotation2 * annotation3 * black_dot*black_dot1*black_dot2*black_dot3).opts(
    opts.Curve(width=600, height=400, tools=['hover']),
    opts.Text(text_font_size='8pt', text_font_style='bold', text_color='seagreen')
)

overlay


# # U.S. Population Projection from 1970 to 2060 (PreWork)

# In[34]:


us_population = pd.read_csv('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/united-states-population-2023-07-17.csv')


# In[35]:


us_population


# In[36]:


us_population.dtypes


# In[37]:


#Current population line chart
usa_pop_line = sns.lineplot(data = us_population, x = 'year',y = 'population_in_mil')


# In[38]:


us_population_projection = pd.read_csv('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/Population Projections based on Immigration.csv')


# In[39]:


us_population_projection


# In[40]:


us_population_projection.dtypes


# In[41]:


#projected population line chart
usa_pop_proj_line = sns.lineplot(data = us_population_projection, x = 'year',y = 'population_in_millions')


# In[42]:


#both together
usa_pop_line = sns.lineplot(data = us_population, x = 'year',y = 'population_in_mil')
usa_pop_proj_line = sns.lineplot(data = us_population_projection, x = 'year',y = 'population_in_millions')


# In[43]:


import matplotlib.pyplot as plt


# In[44]:


def usa_pop_chart(data_df, x_column, y_column, title):
    # Create the line plot
    usa_pop_line = sns.lineplot(data = us_population, x = 'year',y = 'population_in_mil', label = 'US Population')
    usa_pop_proj_line = sns.lineplot(data = us_population_projection, x = 'year',y = 'population_in_millions', label = 'US Projected Population')    
    plt.xlabel('Year')
    plt.ylabel('Population in Millions')
    plt.title(title)
    plt.show()

usa_pop_chart(us_population, x_column='', y_column='', title='US Population Projection')


# In[ ]:





# # U.S. Population Projection from 1970 to 2060

# In[45]:


import hvplot.pandas
import holoviews as hv

hv.extension('bokeh')


# In[46]:


year = pn.widgets.IntSlider(name='year', start=1970, end=2022, step=4)


# In[47]:


import pandas as pd
import hvplot.pandas  

df1 = us_population
df2 = us_population_projection

# Creating a line chart using hvplot
line_chart = df1.hvplot.line(x='year', y='population_in_mil', xlabel='Year', ylabel='Population (in millions)', title='US Population Projection (1970-2060)') *              df2.hvplot.line(x='year', y='population_in_millions', xlabel='Year', ylabel='Population (in millions)', title='US Population Projection')

# Display the combined chart (it will be automatically shown in the notebook)
line_chart


# # Estimated U.S. Energy Demand by 2060

# ## Making the DataFrame

# In[48]:


energy_data = {
    'year': [2021, 2060],
    'energy_consumed': [97.907, 116.5]
}


# In[49]:


energy_data


# In[50]:


energy_projection = pd.DataFrame(energy_data)


# In[51]:


import pandas as pd
import matplotlib.pyplot as plt

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
fig1, ax = plt.subplots(figsize=(11, 1.2))  

energy_projection.plot.barh(
    color=['dodgerblue'],
    legend=False,
    width=0.5,
    ax=ax,  # Specify the axis to use for the plot
)

title = 'Projected U.S. Energy Demand by 2060'
ax.set_title(title, loc='left', fontweight='bold')

# Add the total values at the end of each bar
for index, value in enumerate(energy_projection['energy_consumed']):
    ax.annotate(str(value), xy=(value, index), xytext=(value + 0.5, index), ha='left', va='center')

plt.tight_layout()

# Set a name for the chart
chart_name = 'Projected Energy Demand'
plt.gcf().canvas.set_window_title(chart_name)

plt.show()  # Show the plot


# In[ ]:





# # U.S. Energy Production Projection

# In[52]:


#Check Mexico and Canada (df is cumulative of years)
total_energy_df[total_energy_df['country'] == 'United States']


# In[53]:


#df is 2021 year only
df[df['country'] == 'United States']


# In[54]:


#df is 2021 year only
df[df['country'] == 'Canada']


# In[55]:


us_energy_projection = pd.read_csv ('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/energy_production_projection.csv')


# In[56]:


us_energy_line_chart = us_energy_projection.hvplot.line(
    x='year',
    y=('coal', 'petrol', 'natural_gas', 'nuclear', 'renewables'),
    title='U.S. Energy Production Projection',
    xlabel='Year',
    ylabel='QBtu',
    legend='top',
    width=800,
    height=600
)

us_energy_line_chart


# # Canada & Mexico Energy Supply

# In[57]:


canada_mexico = pd.read_csv('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Orginal Data/energy supply by source and country.csv')


# In[58]:


canada_mexico = (canada_mexico.rename(columns = {'Unnamed: 0' : 'country'})
                .drop(columns = ['unit'])
                .drop(index=2)
                )


# In[59]:


canada_mexico


# In[60]:


conversion_factor_tj_to_btu = 947817.12

# Conversion factor from TJ to QBTU
conversion_factor_tj_to_qbtu = 1e-6  # 1e15 / conversion_factor_tj_to_btu

# Create a new DataFrame with values converted to QBTU
qbtu_canada_mexico = canada_mexico.copy()
energy_columns = ['coal', 'natural_gas', 'nuclear', 'hydro', 'wind_solar', 'biofuels_waste', 'oil']

for column in energy_columns:
    qbtu_canada_mexico[column] = qbtu_canada_mexico[column] * conversion_factor_tj_to_qbtu

# Display the new DataFrame with values in QBTU
qbtu_canada_mexico


# In[ ]:





# In[61]:


qbtu_canada_mexico


# In[62]:


canada_mexico_chart = qbtu_canada_mexico.hvplot.bar(
    x='country', 
    y=['coal', 'nuclear', 'hydro', 'oil', 'natural_gas'],
    stacked=False,
    rot=45,
    legend='top_right',
    title='Canada & Mexico Energy Supply',
    xlabel='Country',
    ylabel='QBtu',
)

# Display the hvplot chartb
canada_mexico_chart


# In[63]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import plotly.express as px


# In[64]:


#Loading Overview & Consumption Data
world_energy_df = pd.read_pickle ('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/world_energy_df.pkl')
coal_consumption = pd.read_pickle ('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/coal_consumption.pkl')
natural_gas_consumption = pd.read_pickle ('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/natural_gas_consumption.pkl')
nuclear_and_renewables_consumption = pd.read_pickle ('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/nuclear_and_renewables_consumption.pkl')
petrol_consumption = pd.read_pickle ('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/petrol_consumption.pkl')
total_consumption = pd.read_pickle ('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/total_consumption.pkl')


# In[65]:


#Loading Production Data
coal_production = pd.read_pickle ('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/coal_production.pkl')
natural_gas_production = pd.read_pickle ('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/natural_gas_production.pkl')
nuclear_and_renewables_production = pd.read_pickle ('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/nuclear_and_renewables_production.pkl')
petrol_production = pd.read_pickle ('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/petrol_production.pkl')
total_production = pd.read_pickle ('/Users/christinarandall/Documents/Career Foundry Projects/Gobal Energy Prices/Data/Prepared Data/total_production.pkl')


# ## Transposing & Cleaning the Data

# ### Production Data

# In[66]:


# Coal Production
c_coal_production = coal_production.set_index('country')
c_coal_production = c_coal_production.drop(['continent', 'production_coal'], axis=1)
c_coal_production = c_coal_production.apply(pd.to_numeric, errors='coerce').fillna(0)
c_coal_production = c_coal_production.astype('int64')
c_coal_production_transposed = c_coal_production.T
coal_p = c_coal_production_transposed.rename(columns=lambda x: x.replace(' ', ''))

# Natural Gas Production
c_natural_gas_production = natural_gas_production.set_index('country')
c_natural_gas_production = c_natural_gas_production.drop(['continent', 'production_naturalgas'], axis=1)
c_natural_gas_production = c_natural_gas_production.apply(pd.to_numeric, errors='coerce').fillna(0)
c_natural_gas_production = c_natural_gas_production.astype('int64')
c_natural_gas_production_transposed = c_natural_gas_production.T
natural_gas_p = c_natural_gas_production_transposed.rename(columns=lambda x: x.replace(' ', ''))

# Nuclear and Renewables Production
c_nuclear_renewables_production = nuclear_and_renewables_production.set_index('country')
c_nuclear_renewables_production = c_nuclear_renewables_production.drop(['continent', 'production_neuclear'], axis=1)
c_nuclear_renewables_production = c_nuclear_renewables_production.apply(pd.to_numeric, errors='coerce').fillna(0)
c_nuclear_renewables_production = c_nuclear_renewables_production.astype('int64')
c_nuclear_renewables_production_transposed = c_nuclear_renewables_production.T
nuclear_renewables_p = c_nuclear_renewables_production_transposed.rename(columns=lambda x: x.replace(' ', ''))

# Petrol Production
c_petrol_production = petrol_production.set_index('country')
c_petrol_production = c_petrol_production.drop(['continent', 'production_petrolium'], axis=1)
c_petrol_production = c_petrol_production.apply(pd.to_numeric, errors='coerce').fillna(0)
c_petrol_production = c_petrol_production.astype('int64')
c_petrol_production_transposed = c_petrol_production.T
petrol_p = c_petrol_production_transposed.rename(columns=lambda x: x.replace(' ', ''))


# ### Consumption Data

# In[67]:


# Coal Consumption
c_coal_consumption = coal_consumption.set_index('country')
c_coal_consumption = c_coal_consumption.drop(['continent', 'consumption_coal'], axis=1)
c_coal_consumption = c_coal_consumption.apply(pd.to_numeric, errors='coerce').fillna(0)
c_coal_consumption = c_coal_consumption.astype('int64')
c_coal_consumption_transposed = c_coal_consumption.T
coal_c = c_coal_consumption_transposed.rename(columns=lambda x: x.replace(' ', ''))

# Natural Gas Consumption
c_natural_gas_consumption = natural_gas_consumption.set_index('country')
c_natural_gas_consumption = c_natural_gas_consumption.drop(['continent', 'consumption_naturalgas'], axis=1)
c_natural_gas_consumption = c_natural_gas_consumption.apply(pd.to_numeric, errors='coerce').fillna(0)
c_natural_gas_consumption = c_natural_gas_consumption.astype('int64')
c_natural_gas_consumption_transposed = c_natural_gas_consumption.T
natural_gas_c = c_natural_gas_consumption_transposed.rename(columns=lambda x: x.replace(' ', ''))

# Nuclear and Renewables Consumption
c_nuclear_renewables_consumption = nuclear_and_renewables_consumption.set_index('country')
c_nuclear_renewables_consumption = c_nuclear_renewables_consumption.drop(['continent', 'consumption_neuclear'], axis=1)
c_nuclear_renewables_consumption = c_nuclear_renewables_consumption.apply(pd.to_numeric, errors='coerce').fillna(0)
c_nuclear_renewables_consumption = c_nuclear_renewables_consumption.astype('int64')
c_nuclear_renewables_consumption_transposed = c_nuclear_renewables_consumption.T
nuclear_renewables_c = c_nuclear_renewables_consumption_transposed.rename(columns=lambda x: x.replace(' ', ''))

# Petrol Consumption
c_petrol_consumption = petrol_consumption.set_index('country')
c_petrol_consumption = c_petrol_consumption.drop(['continent', 'consumption_petrolium'], axis=1)
c_petrol_consumption = c_petrol_consumption.apply(pd.to_numeric, errors='coerce').fillna(0)
c_petrol_consumption = c_petrol_consumption.astype('int64')
c_petrol_consumption_transposed = c_petrol_consumption.T
petrol_c = c_petrol_consumption_transposed.rename(columns=lambda x: x.replace(' ', ''))


# ## Energy Breakdowns

# ### Concatting Production Data for 2021

# In[68]:


df_coal_p = coal_p.loc[['2021'], :]
df_coal_p = df_coal_p.T

df_natural_gas_p = natural_gas_p.loc[['2021'], :]
df_natural_gas_p = df_natural_gas_p.T

df_nuclear_renewables_p = nuclear_renewables_p.loc[['2021'], :]
df_nuclear_renewables_p = df_nuclear_renewables_p.T

df_petrol_p = petrol_p.loc[['2021'], :]
df_petrol_p = df_petrol_p.T


# In[69]:


combined_df = pd.concat([df_coal_p, df_petrol_p, df_nuclear_renewables_p, df_natural_gas_p], axis=1)


# In[70]:


combined_df = combined_df.reset_index()


# In[71]:


combined_df.columns.values[1] = 'coal'
combined_df.columns.values[2] = 'petrol'
combined_df.columns.values[3] = 'nuclear_and_renewables'
combined_df.columns.values[4] = 'natural_gas'


# In[72]:


combined_df


# ### 2021 U.S. Production Pie Chart

# In[73]:


# Filter the DataFrame to include only the 'UnitedStates' rows
us_data = combined_df[combined_df['country'] == 'UnitedStates']

# Select the columns of interest
columns_of_interest = ['coal', 'petrol', 'nuclear_and_renewables', 'natural_gas']

# Aggregate the values of the columns
aggregate_values = us_data[columns_of_interest].sum()

# Create a pie chart
plt.pie(aggregate_values, labels=columns_of_interest, autopct='%1.1f%%')

# Add a title
plt.title('Energy Production Breakdown for United States 2021')


# ### Concatting Consumption Data for 2021

# In[74]:


df_coal_c = coal_c.loc[['2021'], :]
df_coal_c = df_coal_c.T

df_natural_gas_c = natural_gas_c.loc[['2021'], :]
df_natural_gas_c = df_natural_gas_c.T

df_nuclear_renewables_c = nuclear_renewables_c.loc[['2021'], :]
df_nuclear_renewables_c = df_nuclear_renewables_c.T

df_petrol_c = petrol_c.loc[['2021'], :]
df_petrol_c = df_petrol_c.T


# In[75]:


combined_df2 = pd.concat([df_coal_c, df_petrol_c, df_nuclear_renewables_c, df_natural_gas_c], axis=1)


# In[76]:


combined_df2 = combined_df2.reset_index()


# In[77]:


combined_df2.columns.values[1] = 'coal'
combined_df2.columns.values[2] = 'petrol'
combined_df2.columns.values[3] = 'nuclear_and_renewables'
combined_df2.columns.values[4] = 'natural_gas'


# In[78]:


#consumption df for 2021
combined_df2


# ### 2021 U.S. Consumption Pie Chart

# In[79]:


# Filter the DataFrame to include only the 'UnitedStates' rows
us_data2 = combined_df2[combined_df2['country'] == 'UnitedStates']

# Select the columns of interest
columns_of_interest = ['coal', 'petrol', 'nuclear_and_renewables', 'natural_gas']

# Aggregate the values of the columns
aggregate_values = us_data2[columns_of_interest].sum()

# Create a pie chart
plt.pie(aggregate_values, labels=columns_of_interest, autopct='%1.1f%%')

# Add a title
plt.title('Energy Consumption Breakdown for United States 2021')

# Display the chart
plt.show()


# ### 2021 Energy Pie Charts by Country

# In[80]:


# Function to plot pie chart
def plot_pie_chart(data_df, title, subplot_position):
    # Filter the DataFrame to include only the 'UnitedStates' rows
    us_data = data_df[data_df['country'] == 'UnitedStates']

    # Select the columns of interest
    columns_of_interest = ['coal', 'petrol', 'nuclear_and_renewables', 'natural_gas']

    # Aggregate the values of the columns
    aggregate_values = us_data[columns_of_interest].sum()

    # Create a pie chart
    plt.subplot(subplot_position)
    plt.pie(aggregate_values, labels=columns_of_interest, autopct='%1.1f%%')

    # Adding a title
    plt.title(title)

# Create a figure with two subplots
pie_chart, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))

# Plot pie chart for production data
plot_pie_chart(combined_df, 'Energy Production for United States 2021', 121)

# Plot pie chart for consumption data
plot_pie_chart(combined_df2, 'Energy Consumption for United States 2021', 122)

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.5)

# Display the chart
plt.show()


# In[ ]:





# # Dashboard Layout

# In[81]:


#Dashboard Header
header_text = "# United States Energy Analysis\n####How Stable is the U.S. Energy Market in terms of keeping up with U.S. Demand Indpendent of Outside Energy Sources?"
header_pane = pn.pane.Markdown(header_text, style={'font-size': '20px', 'text-align': 'center'})

#Tabs
tabs = pn.Tabs()

#tab1-----------------------------------------------------
tab1_chart1 = pn.pane.plot.Folium(map, width=800, height=375)

tab1_title = "### On a global level how does the US rank in net energy indpendence?"
tab1_tstyle = pn.pane.Markdown(tab1_title, style={'font-size': '18px', 'text-align': 'center'})

tab1_text1 = '# Introduction \n Looking at U.S. energy consumption versus production is an eye-opening glimpse into the U.S. energy market’s stability. If you were to look only at the last few years, you might not notice anything, but from a global perspective you can see that the U.S. has run a net deficit since 1970. With this fact in mind, I decided to further explore the U.S. energy market and future energy demands to better understand how reliant the U.S. is on outside energy sources.'
tab1_text2 = '# Context \n This heat map shows each country’s sum total of energy (total production minus total consumption), from the 1970s to 2022. It indicates that the U.S. relies on energy imports to keep up with its demand, whereas Saudi Arabia and Russia, on balance, have had the most excess energy in the world.'
tab1_text3 = '### Global Net Energy Heat Map (1970-2022)'
tab1_text4 = '# Insight \n Clearly, based on this chart alone, the U.S. would be dependent on energy imports until they either reduce the country’s demand, increase their energy production, or both. In the meantime, most of the energy imports would likely come from Canada or Mexico, as they share geographical boundaries with the U.S., which cuts down on shipping costs. Russia and Saudi Arabia are also options, due to their large excess energy stores. However, as we will later see, this is not the case.'
tab1_text5 = '# Dashboard Notes\n All Energy Units are in Quadrillion British Thermal Units unless noted. Citations are on the last page. Lastly, this is an interactive dashboard, feel free to further explore it using your mouse.'

tab1_tab2 = pn.Row(pn.Column(tab1_text1, tab1_text3, tab1_chart1), (pn.Column(tab1_text2, tab1_text4, tab1_text5)))
tab1_tab = pn.Column(tab1_tstyle, tab1_tab2)

#tab2-----------------------------------------------------
tab2_chart1 = overlay.opts(width=700, height=600, shared_axes=False)

tab2_title = "### How does the U.S. net energy look historically?"
tab2_tstyle = pn.pane.Markdown(tab2_title, style={'font-size': '18px', 'text-align': 'center'})

tab2_text1 = '# Context \n This line chart shows U.S. production versus consumption from 1980 through 2021 and reveals that U.S. energy consumption grew quickly from 1980 until 2008. After 2008, U.S. energy production significantly ramped up, and it started to actually meet demand in 2019.'
tab2_style1 = pn.pane.Markdown(tab2_text1, style={'font-size': '15px'})
tab2_text2 = '# Insight \n The U.S. has only recently become energy independent. Our previous dependence was likely due to historical factors such as cheaper energy costs, a stronger dollar, and lack of intention. Together, these factors explain why the U.S. was reliant on other countries up until the Energy Act.'
tab2_style2 = pn.pane.Markdown(tab2_text2, style={'font-size': '15px'})
tab2_text3 = '####Tip: Use the reset button to refresh the chart'
tab2_style3 = pn.pane.Markdown(tab2_text3, style={'font-size': '13px'})

tab2_tab1 = pn.Row(tab2_chart1, pn.Column(tab2_style1, tab2_style2, tab2_style3))
tab2_tab = pn.Column(tab2_tstyle, tab2_tab1)

#tab3-----------------------------------------------------
tab3_title = "### Based on future population expectations, how much energy will the U.S. need in the future?"
tab3_tstyle = pn.pane.Markdown(tab3_title, style={'font-size': '18px', 'text-align': 'center'})

tab3_chart1 = line_chart.opts(width=800, height=400, shared_axes=False)
tab3_chart2 = pn.pane.Matplotlib(fig1)

tab3_text1 = '# Context \n The U.S. Population Projection line chart looks at both the United States’ historical and projected population growth as predicted by the U.S. Census Bureau in the main scenario. The population projection in red shows steady growth following the same general trend as past growth.'
tab3_style1 = pn.pane.Markdown(tab3_text1, style={'font-size': '15px'})
tab3_text2 = 'This represents a 19 percent increase in energy demand from 97.9 QBTU in 2021 to 116.5 QBTU in 2060.'
tab3_style2 = pn.pane.Markdown(tab3_text2, style={'font-size': '15px'})
tab3_text3 = '# Insight \n Interestingly, while energy demand has shot up since the 1960s, U.S. population growth has been comparatively steady over the same period of time. The disparity can be explained by the fact that the average energy use per person in the U.S. has increased substantially, largely due to the considerable technological advancements the world has seen since the 1960s. However, calculating how much energy will be needed in future years is difficult because of many unknown variables. I have estimated the future amount below the chart, assuming that the average person’s usage remains stagnant (based on 2021).'
tab3_style3 = pn.pane.Markdown(tab3_text3, style={'font-size': '15px'})

tab3_tab3 = pn.Column(tab3_chart1, tab3_chart2, tab3_style2)
tab3_tab2 = pn.Column(tab3_style1, tab3_style3)
tab3_tab1 =pn.Column(pn.Row(tab3_tab3, tab3_tab2))
tab3_tab = pn.Column(tab3_tstyle, tab3_tab1)

#tab4-----------------------------------------------------
tab4_title = "### If the U.S. is not able to keep up with its energy demand, who can we import from?"
tab4_tstyle = pn.pane.Markdown(tab4_title, style={'font-size': '18px', 'text-align': 'center'})

tab4_chart1 = production_bar_plot.opts(width=800, height=275, shared_axes=False)
tab4_chart2 = canada_mexico_chart.opts(width=400, height=275, shared_axes=False)

tab4_text1 = '# Context \n This bar chart shows the top 10 producers for each energy type as of 2021. So far, we have looked at total energy production versus total energy consumption (or net energy), which doesn’t account for the different energy types different sectors consume (i.e., households as opposed to commercial institutions). Here, you can click on the different buttons to see which countries produce the most of any given energy type. China, for example, greatly surpasses every other country in coal production due to its manufacturing economy.'
tab4_style1 = pn.pane.Markdown(tab4_text1, style={'font-size': '15px'})
tab4_text3 = '# Insight \n If the U.S. needs to source from other countries, it should focus on countries with energy surpluses. However, the U.S. should prioritize producing their own natural gas, since international pipelines can be tricky to maintain (as seen with the recent Russian pipeline leak).'
tab4_style3 = pn.pane.Markdown(tab4_text3, style={'font-size': '15px'})
tab4_text4 = '### U.S. Energy Trading Partners \n The United States’ primary trading partners, Canada and Mexico, have mostly filled our energy deficit. Canada relies on its U.S. energy exports for their GDP and has noted that it will have to expand its market as the U.S. becomes more energy independent. As the U.S. grows, however, they must keep strong trade connections with both countries because while it’s important to have an independent energy sector in the U.S., their energy will be strongest when they can source it both ways.'
tab4_style4 = pn.pane.Markdown(tab4_text4, style={'font-size': '13px'})

tab4_tab4 = pn.Row(tab4_chart2, tab4_style4)
tab4_tab3 = pn.Column(tab4_chart1, tab4_tab4)
tab4_tab2 = pn.Column(tab4_style1, tab4_style3)
tab4_tab1 = pn.Column(pn.Row(tab4_tab3, tab4_tab2))
tab4_tab = pn.Column(tab4_tstyle, tab4_tab1)

#tab5-----------------------------------------------------
tab5_title = "### What does this mean for the future of U.S. energy independence?"
tab5_tstyle = pn.pane.Markdown(tab5_title, style={'font-size': '18px', 'text-align': 'center'})

tab5_text1 ='# Energy Implications\n If the US cannot keep up with its energy demands (which projections indicate could happen), they will be forced to buy energy from other countries. The current decline of globalization poses a risk for US energy security in this regard, as energy prices and country relations remain volatile.' 
tab5_style1 = pn.pane.Markdown(tab5_text1, width=700)
tab5_text4 = 'In addition, with global warming taking its toll in Europe, energy is poised to become increasingly expensive to import. Given that by 2060 the U.S. is likely to see a nearly 20 percent increase in energy demand, they must increase their energy production over the next forty years to ensure stable energy prices. If the U.S. does not keep up with its own demand, they will risk energy outages and volatile prices.'
tab5_style4 = pn.pane.Markdown(tab5_text4, width=700)

tab5_text5 = '# Next Steps\n Based on this research, it is apparent that furthering the cause of U.S. energy independence will require the U.S. to scrutinize each energy type more closely and determine which ones they are capable of producing in enough quantity to sustain future demand. Once that information is determined, time and resources need to be concentrated in that/those area(s).' 
tab5_style5 = pn.pane.Markdown(tab5_text5, width=700)

tab5_text2 = '# Further Resources'
tab5_style2 = pn.pane.Markdown(tab5_text2, width=700)

r1 = '[Global Energy Tracker]( https://globalenergymonitor.org/)'
r2 = '[Global Coal Tracker]( https://www.carbonbrief.org/mapped-worlds-coal-power-plants/)'

tab5_text3 = '# Citations'
tab5_style3 = pn.pane.Markdown(tab5_text3, width=700)

c1 = '[Kaggle : Energy Data](https://www.kaggle.com/datasets/akhiljethwa/world-energy-statistics)'
c2 = '[US Census Bureau : Population Projections](https://www.census.gov/data/datasets/2017/demo/popproj/2017-popproj.html)'
c3 = '[Json File : Heat Map](https://datahub.io/core/geo-countries#pandas)'
c4 = '[United Nations : US Population Data](https://www.macrotrends.net/countries/USA/united-states/population)'

tab5_citations = pn.Column(tab5_style3, c1, c2, c3, c4, tab5_style2, r2, r1 )
tab5_tab1 =pn.Row(pn.Column(tab5_style1, tab5_style4, tab5_style5),tab5_citations)
tab5_tab = pn.Column(tab5_tstyle, tab5_tab1)

#tab6-----------------------------------------------------
tab6_chart1 = us_energy_line_chart.opts(width=800, height=600, shared_axes=False) 

tab6_title = "### What is the projected U.S. energy production?"
tab6_tstyle = pn.pane.Markdown(tab6_title, style={'font-size': '18px', 'text-align': 'center'})

tab6_text1 = '# Context \n The line chart shows the projected energy production through 2050, breaking it down into five categories: 1) coal, 2) petrol, 3) natural gas, 4) nuclear, and 5) renewables. Below it are U.S. production and consumption pie charts for 2021.'
tab6_style1 = pn.pane.Markdown(tab6_text1, style={'font-size': '15px'})
tab6_text2 = '# Insight \n Since the U.S. hit energy consumption levels in 2019 due to our rapid expansion in energy production, the IEA’s prediction for future growth looks stagnant. The only exception is the renewables category (biomass, geothermal, sunlight, water, and wind), which is driven mostly by private adoption. Given the United States’ current rate of consumption and growth, they may still be reliant on other countries for energy in the future.'
tab6_style2 = pn.pane.Markdown(tab6_text2, style={'font-size': '15px'})

tab6_tab1 = pn.Row(pn.Column(tab6_chart1), pn.Column(tab6_style1, tab6_style2))
tab6_tab = pn.Column(tab6_tstyle, tab6_tab1)

# Append the tabs
tabs.append(('Global Net Energy', tab1_tab))
tabs.append(('US Energy Historically', tab2_tab))
tabs.append(('Population Projections', tab3_tab))
tabs.append(('U.S. Energy Projections', tab6_tab))
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


# # Flask Deployment

# In[84]:


from flask import Flask, render_template


# In[85]:


app = Flask(__name__)


# In[91]:


@app.route('/')
def dashboard():
    # Create an instance of your Panel app
    main = pn.Column(header_pane, tabs, footer_pane)
    
    # Convert the Panel app to HTML
    html = main._repr_html_()
    
    return render_template('energy_dashboard.html', dashboard_html=html)


# In[90]:





# In[ ]:




