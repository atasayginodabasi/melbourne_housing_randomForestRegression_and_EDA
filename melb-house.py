import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import plotly.express as px
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

keras = tf.keras
warnings.filterwarnings('ignore')

# READ ME
'''''''''
Rooms: Number of rooms
Price: Price in dollars
Date: Date sold
Distance: Distance from CBD
Regionname: General Region
Propertycount: Number of properties that exist in the suburb.
Bathroom: Number of Bathrooms
Car: Number of carspots
Landsize: Land Size
BuildingArea: Building Size
'''''''''

# Reading the data
print(os.listdir(r'C:/Users/ata-d/'))
data = pd.read_csv(r'C:/Users/ata-d/OneDrive/Masaüstü/ML/Datasets/melb_data.csv')

# Count of the missing values of the dataset
count_NaN = data.isna().sum()

data = data.drop(['Address', 'Type', 'Method', 'SellerG', 'Date', 'CouncilArea'], axis=1)

# Dropping the highly NaN and not needed columns. BuildingArea, YearBuilt, CouncilArea, etc.
data = data.dropna(subset=['BuildingArea', 'YearBuilt'], axis=0)

# Filling the missing Car values with the mean of the Car value which is 1
data['Car'].fillna(np.floor(data['Car'].mean()), inplace=True)

# Count of the missing values of the dataset
count_NaN_updated = data.isna().sum()

# Creating a new dataset with the Latitude and Longitude
latitude = data['Lattitude']
longitude = data['Longtitude']

# -------------------------------------------------------------------------------------------------------

# Encoding the string dataframe (Regionname) and (Suburb) columns
ordinal_enc = OrdinalEncoder()
# data['Encoded_Regionname'] = ordinal_enc.fit_transform(data[['Regionname']])
data['Encoded_Suburb'] = ordinal_enc.fit_transform(data[['Suburb']])

# Encoded_Suburb = data[["Suburb", "Encoded_Suburb"]].drop_duplicates().sort_values(by='Encoded_Suburb')
# Encoded_Regionname = data[["Regionname", "Encoded_Regionname"]].drop_duplicates().sort_values(by='Encoded_Regionname')


dd = pd.get_dummies(data['Regionname'])

data = pd.concat([data, dd], axis=1)
# -------------------------------------------------------------------------------------------------------

# Data Visualization

# Heatmap of the House Prices (Plotly)
'''''''''
fig = px.density_mapbox(data, lat=latitude, lon=longitude, z=data['Price'],
                        center=dict(lat=-37.823002, lon=144.998001), zoom=10.5,
                        mapbox_style="stamen-terrain",
                        radius=20,
                        opacity=0.5)
fig.update_layout(title_text='Melbourne Heatmap of the House Prices', title_x=0.5, title_font=dict(size=32))
fig.show()
'''''''''

# Density Plots
'''''''''
# Density Plot of the Prices
plt.figure(figsize=(15, 8))
sns.distplot(data['Price'], hist=True, color='red')
plt.xlabel("Price (M)", fontsize=12)
plt.ylabel('Density',fontsize=12)
plt.title("Density Plot of the House Prices", fontsize=16)

# Density Plot of the Landsize
plt.figure(figsize=(15, 8))
sns.distplot(data['Landsize'], hist=True, color='blue')
plt.ylabel('Density',fontsize=12)
plt.xlabel("Landsize", fontsize=12)
plt.title("Density Plot of the Landsize", fontsize=16)

# Density Plot of the Regionname
plt.figure(figsize=(15, 8))
sns.distplot(data['Encoded_Regionname'], hist=True, color='darkblue')
plt.ylabel('Density',fontsize=12)
plt.xlabel("Regionname", fontsize=12)
plt.title("Density Plot of the Regionname", fontsize=16)

# Density Plot of the Distance from CBD (Central Business Sub District)
plt.figure(figsize=(15, 8))
sns.distplot(data['Distance'], hist=True, color='green')
plt.ylabel('Density',fontsize=12)
plt.xlabel("Distance from CBD", fontsize=12)
plt.title("Density Plot of the Distance from CBD", fontsize=16)

# Density Plot of the Rooms
plt.figure(figsize=(15, 8))
sns.distplot(data['Rooms'], hist=True, color='purple')
plt.ylabel('Density',fontsize=12)
plt.xlabel("Rooms", fontsize=12)
plt.title("Density Plot of the Rooms", fontsize=16)
'''''''''

# Bar plots
'''''''''
# Regions vs Price barplot
plt.figure(figsize=(15, 8))
sns.barplot(x=data['Regionname'], y=data['Price'])
plt.title("Graph of the Regions vs Price", fontsize=16)
plt.ylabel('Price (M)', fontsize=12)
plt.xlabel('Regionnames', fontsize=12)

# Rooms vs Price barplot
plt.figure(figsize=(15, 8))
sns.barplot(x=data['Rooms'], y=data['Price'])
plt.title("Graph of the Regions vs Price", fontsize=16)
plt.ylabel('Price (M)', fontsize=12)
plt.xlabel('Number of Rooms', fontsize=12)
'''''''''

# Scatter Plots
'''''''''
# The Relationship between the Distance vs Price
fig = px.scatter(data, x='Distance', y='Price', hover_data=['Price'])
fig.update_layout(title='The Relationship between the Distance and the Price', title_x=0.5, title_font=dict(size=30))
fig.show()

# The Relationship between the Suburb vs Price
fig = px.scatter(data, x='Regionname', y='Price', hover_data=['Price'])
fig.update_layout(title='The Relationship between the Regionname and the Price', title_x=0.5, title_font=dict(size=30))
fig.show()

# The Relationship between the Rooms vs Price
total_rooms = data['Rooms'] + data['Bedroom2'] + data['Bathroom']
fig = px.scatter(data, x=total_rooms, y='Price', hover_data=['Price'])
fig.update_layout(title='The Relationship between the Regionname and the Price', title_x=0.5, title_font=dict(size=30))
fig.update_layout(xaxis_title='Total Number of Rooms')
fig.show()

# The Relationship between the Landsize vs Price
fig = px.scatter(data, x='Landsize', y='Price', hover_data=['Price'])
fig.update_layout(title='The Relationship between the Regionname and the Price', title_x=0.5, title_font=dict(size=30))
fig.show()
'''''''''

# Correlation Analysis of the Dataset
'''''''''
plt.figure(figsize=(15, 8))
correlation = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True, linewidths=1, linecolor='black')
correlation.set_title('Correlation Graph of the Melbourne Dataset', fontdict={'fontsize': 24})
'''''''''

# -------------------------------------------------------------------------------------------------------

# After encoding the Suburb and Regionname columns, I don't need them for now
data = data.drop(['Suburb', 'Regionname'], axis=1)
y = np.log(data['Price'])
X = data.drop('Price', axis=1)

# -------------------------------------------------------------------------------------------------------

# Train and Text split
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=13)
trainX = pd.DataFrame(trainX)
trainY = pd.DataFrame(trainY)

# -------------------------------------------------------------------------------------------------------

# Grid Seach CV
'''''''''
param_grid = {
    'max_depth': [60, 70, 80, 90, 100],
    'max_features': [5, 6, 7],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [6, 8, 10],
    'n_estimators': [30, 40, 50, 60, 80],
    'max_leaf_nodes': [15, 20, 30, 40]
}
rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=3, n_jobs=2, verbose=2)
grid_search.fit(trainX, trainY)
best_params = grid_search.best_params_
best_grid = grid_search.best_estimator_
'''''''''

# -------------------------------------------------------------------------------------------------------

# Random Forest Model

rf = RandomForestRegressor(max_depth=60, max_features=6, min_samples_leaf=5,
                           min_samples_split=6, n_estimators=60, oob_score=True,
                           max_leaf_nodes=55)

rf.fit(trainX, trainY)


train_score = rf.score(trainX, trainY)
oob_score = rf.oob_score_
Adjusted_R2_train = 1 - (1-rf.score(trainX, trainY))*(len(trainY)-1)/(len(trainY)-trainX.shape[1]-1)
Adjusted_R2_test = 1 - (1-rf.score(testX, testY))*(len(testY)-1)/(len(testY)-testX.shape[1]-1)

print('Train Adjusted R2: %', Adjusted_R2_train*100)
print('Test Adjusted R2: %', Adjusted_R2_test*100)
print('OOB Score: %', oob_score*100)
# print('Test Score: ', rf.score(testX, testY))
print((Adjusted_R2_train-Adjusted_R2_test)*100)

# Feature Importance
'''''''''
plt.barh(X.columns, rf.feature_importances_)
plt.title("Melbourne Housing Snapshot Feature Importance", fontsize=20)
plt.xlabel('Relative Importance', fontsize=12)
'''''''''

# -------------------------------------------------------------------------------------------------------

dff = pd.read_csv(r'C:/Users/ata-d/OneDrive/Masaüstü/ML/Datasets/melb_data.csv')
dff = dff.drop(['Address', 'Type', 'Method', 'SellerG', 'Date'], axis=1)
dff = dff.dropna(subset=['BuildingArea', 'YearBuilt'], axis=0)

nan_values = dff[dff['CouncilArea'].isna()]
nan_values = nan_values[['Suburb', 'CouncilArea']]
nan_values_unique = pd.DataFrame(nan_values['Suburb'].unique())

# Data Binning
bins = np.linspace(min(data['Price']), max(data['Price']), 5)
group_names = ['Low', 'Medium', 'High', 'Very High']
data['bins'] = pd.cut(data['Price'], bins, labels=group_names, include_lowest=True)
