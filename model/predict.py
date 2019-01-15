# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:29:52 2018

@author: hardi
"""

import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats.stats import pearsonr  
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error, median_absolute_error
import xgboost as xgb

#regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

#evaluation metrics
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  # for classificationmodels=[RandomForestRegressor(),AdaBoostRegressor(),BaggingRegressor(),SVR(),KNeighborsRegressor()]

df = pd.read_csv("trip.csv")
weather = pd.read_csv("weather.csv")
stations = pd.read_csv("station.csv")
df.head()

df.isnull().sum()
df.duration.describe()
df.duration /= 60

#I want to remove major outliers from the data; trips longer than 6 hours. This will remove less than 0.5% of the data.
df['duration'].quantile(0.995)
df = df[df.duration <= 360]

#Convert to datetime so that it can be manipulated more easily
df.start_date = pd.to_datetime(df.start_date, format='%m/%d/%Y %H:%M')
df.start_date = df.start_date.dt.floor('h')
df.head()
df.start_station_id.value_counts()
df['date'] = df.start_date.dt.date

stations.rename(columns = {'id':'start_station_id'}, inplace = True)

#Good, each stations is only listed once
print (len(stations.name.unique()))
print (stations.shape)
weather.isnull().sum()
weather.date = pd.to_datetime(weather.date, format='%m/%d/%Y')
print (weather.shape)
weather.events.unique()
weather.loc[weather.events == 'rain', 'events'] = "Rain"
weather.loc[weather.events.isnull(), 'events'] = "Normal"

events = pd.get_dummies(weather.events)
weather = weather.merge(events, left_index = True, right_index = True)

#max_wind and max_gust are well correlated, so we can use max_wind to help fill the null values of max_gust
print (pearsonr(weather.max_wind_Speed_mph[weather.max_gust_speed_mph >= 0], 
               weather.max_gust_speed_mph[weather.max_gust_speed_mph >= 0]))

#For each value of max_wind, find the median max_gust and use that to fill the null values.
weather.loc[weather.max_gust_speed_mph.isnull(), 'max_gust_speed_mph'] = weather.groupby('max_wind_Speed_mph').max_gust_speed_mph.apply(lambda x: x.fillna(x.median()))
weather.isnull().sum()

#Change this feature from a string to numeric.
#Use errors = 'coerce' because some values currently equal 'T' and we want them to become NAs.
weather.precipitation_inches = pd.to_numeric(weather.precipitation_inches, errors = 'coerce')

#Change null values to the median, of values > 0, because T, I think, means True. 
#Therefore we want to find the median amount of precipitation on days when it rained.
weather.loc[weather.precipitation_inches.isnull(), 
            'precipitation_inches'] = weather[weather.precipitation_inches.notnull()].precipitation_inches.median()

weather.head()

df.head()
dates = {}
for d in df.start_date:
    if d not in dates:
        dates[d] = 1
    else:
        dates[d] += 1
#Create the data frame that will be used for training, with the dictionary we just created.
df2 = pd.DataFrame.from_dict(dates, orient = "index")
df2['date'] = df2.index
df2['trips'] = df2.iloc[:,0]
df2 = df2.iloc[:,1:3]
df2.reset_index(drop = True, inplace = True)
trips_agg = pd.merge(df, stations, on='start_station_id',how='left')
trips_agg.drop(['zip_code','name','installation_date'], axis=1,inplace=True)
trips_agg=trips_agg.groupby(['start_station_id','start_date','city','lat', 'long', 'dock_count']).size().reset_index(name='counts')
trips_agg= trips_agg.sort_values(by=['start_date'])
trips_agg.reset_index(drop=True, inplace=True)
zip_code=[]
for c in trips_agg.city:
    if c=='San Francisco':
        zip_code.append(94107)
    elif c=='San Jose':
        zip_code.append(95113)
    elif c=='Redwood City':
        zip_code.append(94063)
    elif c=='Palo Alto':
        zip_code.append(94301)
    elif c=='Mountain View':
        zip_code.append(94041)

trips_agg['zip_code']=zip_code

trips_agg['date']=trips_agg.start_date.dt.date
trips_agg['date']=pd.to_datetime(trips_agg['date'])
trips_agg.dtypes

train = pd.merge(trips_agg, weather, on=['date','zip_code'])
train.head()
  
train.drop(['city','zip_code'],1, inplace= True)

calendar = USFederalHolidayCalendar()
holidays = calendar.holidays(start=train.date.min(), end=train.date.max())

us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
business_days = pd.DatetimeIndex(start=train.date.min(), end=train.date.max(), freq=us_bd)
business_days = pd.to_datetime(business_days, format='%Y/%m/%d').date
holidays = pd.to_datetime(holidays, format='%Y/%m/%d').date
    

#A 'business_day' or 'holiday' is a date within either of the respected lists.
train['business_day'] = train.date.isin(business_days)
train['holiday'] = train.date.isin(holidays)

#Convert True to 1 and False to 0
train.business_day = train.business_day.map(lambda x: 1 if x == True else 0)
train.holiday = train.holiday.map(lambda x: 1 if x == True else 0)


#Convert date to the important features, year, month, weekday (0 = Monday, 1 = Tuesday...)
#We don't need day because what it represents changes every year.
train['year'] = pd.to_datetime(train['date']).dt.year
train['month'] = pd.to_datetime(train['date']).dt.month
train['weekday'] = pd.to_datetime(train['date']).dt.weekday
train['day'] = pd.to_datetime(train['date']).dt.day
train['hour'] = pd.to_datetime(train['start_date']).dt.hour

#corrMatt = subset[[
#        'max_temperature_f',
#       'mean_temperature_f', 'min_temperature_f', 'max_dew_point_f',
#       'mean_dew_point_f', 'min_dew_point_f', 'max_humidity', 'mean_humidity',
#       'min_humidity', 'max_sea_level_pressure_inches',
#       'mean_sea_level_pressure_inches', 'min_sea_level_pressure_inches',
#       'max_visibility_miles', 'mean_visibility_miles', 'min_visibility_miles',
#       'max_wind_Speed_mph', 'mean_wind_speed_mph', 'max_gust_speed_mph',
#       'precipitation_inches', 'cloud_cover', 'events', 'wind_dir_degrees',
#       'Fog', 'Fog-Rain', 'Normal', 'Rain', 'Rain-Thunderstorm', 
#       'business_day', 'holiday', 'year', 'month', 'weekday','counts']].corr()
#mask = np.array(corrMatt)
#mask[np.tril_indices_from(mask)] = False
#fig,ax= plt.subplots()
#fig.set_size_inches(20,10)
#sns.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
#print(corrMatt['counts'])

train.start_station_id.value_counts()

subset = train[train['start_station_id'] == 66]
labels = subset.counts

X_train, X_test, y_train, y_test = train_test_split(subset, labels, test_size=0.2, shuffle=False)
X_train.drop(['start_station_id','start_date','date','events','counts'],1, inplace= True)
X_test.drop(['start_station_id','start_date','date','events','counts'],1, inplace= True)
#feature selection here
X_train = X_train[['hour','weekday']]
X_test = X_test[['hour','weekday']]


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    
    errors = abs(np.round(predictions) - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} trips.'.format(np.mean(errors)))
    print('RMSE: ', np.sqrt(mean_squared_error(test_labels,predictions)))
    print("R2", r2_score(test_labels,predictions))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    d={
       'hour':X_test['hour'],  
       'actual':y_test,
       'pred':np.round(predictions)}
    ans=pd.DataFrame(d)
#    print(ans)
    return accuracy

rf = RandomForestRegressor(n_estimators = 55, min_samples_leaf = 4,random_state = 2)
rf.fit(X_train, y_train)
rf_accuracy = evaluate(rf, X_test, y_test)


gbr = GradientBoostingRegressor(learning_rate = 0.1,
                                n_estimators = 150,
                                max_depth = 8,
                                min_samples_leaf = 4,
                                random_state = 2)
#scoring(gbr)
gbr = gbr.fit(X_train, y_train)
gbr_accuracy = evaluate(gbr, X_test, y_test)

dtr = DecisionTreeRegressor(min_samples_leaf = 3,
                    max_depth = 8,
                    random_state = 2)
dtr = dtr.fit(X_train, y_train)
dtr_accuracy = evaluate(dtr, X_test, y_test)

abr = AdaBoostRegressor(n_estimators = 100,
                learning_rate = 0.1,
                loss = 'linear',
                random_state = 2)
abr = abr.fit(X_train, y_train)
abr_accuracy = evaluate(abr, X_test, y_test)


def plot_importances(model, model_name):
    importances = model.feature_importances_
    std = np.std([model.feature_importances_ for feature in model.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]    

    # Plot the feature importances of the forest
    plt.figure(figsize = (8,5))
    plt.title("Feature importances of " + model_name)
    plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

#     Print the feature ranking
print("Feature ranking:")

i = 0
for feature in X_train:
    print (i, feature)
    i += 1
    
plot_importances(rf, "Random Forest Regressor")


X_train, X_test, y_train, y_test = train_test_split(subset, labels, test_size=0.2, shuffle=False)

#X_test.drop(['start_station_id','start_date','date','events'],1, inplace= True)
hr=[]
mae=[]
for i in set(X_test['hour']):
    test_df = X_test[X_test['hour']==i]
    labl = test_df['counts']    
    test = test_df[['hour','weekday']]    
    hr.append(i)
    pred = rf.predict(test)
    err = np.mean(abs((pred) - labl))
    mae.append(err)
d={
'hour':hr,
'mae':mae}
err_df=pd.DataFrame(d)

x = err_df['hour']
y = err_df['mae']
plt.title('Variation in MAE across the day for station ID = 66')
#Axes.set_ylabel('MAE in # of trips')
plt.ylabel('MAE in # of trips')
plt.xlabel('hour')
plt.plot(x, y)

plt.show()

## Data
#df=pd.DataFrame({'x': range(1,11), 'y1': np.random.randn(10), 'y2': np.random.randn(10)+range(1,11), 'y3': np.random.randn(10)+range(11,21) })
#
## multiple line plot
#plt.plot( 'x', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
#plt.plot( 'x', 'y2', data=df, marker='', color='olive', linewidth=2)
#plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
#plt.legend()

    
d={
   'date':X_test['start_date'].tail(150),  
   'actual':y_test.tail(150),
   'pred':(rf.predict(X_test[['hour','weekday']].tail(150)))}
result=pd.DataFrame(d)

plt.plot( 'date', 'actual', data=result, marker='', color='blue', linewidth=2,label="actual")
plt.plot( 'date', 'pred', data=result, marker='', color='green', linewidth=2,  label="predicted")
plt.ylabel('count of trips')
plt.xticks( rotation='vertical')
plt.title("Actual vs Predicted for a particular station (66)")
plt.legend()
plt.show()

defaultPred=[]    
def default(defaultCountTest,defaultCount1,y_test):
    a=X_train['counts'].tolist()
    b=sum(a)
    outputMean=(b)/float(len(X_train))
    for i in range(len(defaultCountTest)):
        defaultPred.append(int(round(outputMean)))
    ytest=list(y_test)
    rms = np.mean(abs((y_test) - defaultPred))
    return rms
default(X_test,X_train,y_test)