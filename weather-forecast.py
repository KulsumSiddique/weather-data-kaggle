import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

city_attributes = pd.read_csv('./input/city_attributes.csv')
humidity = pd.read_csv('./input/humidity.csv')
pressure = pd.read_csv('./input/pressure.csv')
temperature = pd.read_csv('./input/temperature.csv')
weather_description = pd.read_csv('./input/weather_description.csv')
wind_direction = pd.read_csv('./input/wind_direction.csv')
wind_speed = pd.read_csv('./input/wind_speed.csv')

# we can reshape these using pd.melt
humidity = pd.melt(humidity, id_vars = ['datetime'], value_name = 'humidity', var_name = 'City')
pressure = pd.melt(pressure, id_vars = ['datetime'], value_name = 'pressure', var_name = 'City')
temperature = pd.melt(temperature, id_vars = ['datetime'], value_name = 'temperature', var_name = 'City')
weather_description = pd.melt(weather_description, id_vars = ['datetime'], value_name = 'weather_description', var_name = 'City')
wind_direction = pd.melt(wind_direction, id_vars = ['datetime'], value_name = 'wind_direction', var_name = 'City')
wind_speed = pd.melt(wind_speed, id_vars = ['datetime'], value_name = 'wind_speed', var_name = 'City')

# humidity.head()

# combine all of the dataframes created above 
weather = pd.concat([humidity, pressure, temperature, weather_description, wind_direction, wind_speed], axis = 1)
weather = weather.loc[:,~weather.columns.duplicated()] # indexing: every row, only the columns that aren't duplicates

# now we can merge this with the city attributes
weather = pd.merge(weather, city_attributes, on = 'City')
# weather.head()

# create a variable for binary classification 
weather['weather_binary'] = np.where(weather['weather_description'].isin(["sky is clear", "broken clouds", "few clouds", 
                                                  "scattered clouds", "overcast clouds"]), 'good', 'bad')

# create a variable for multi-classification
conditions = [
    (weather['weather_description'].isin(["drizzle", "freezing_rain", "heavy intensity drizzle", 
                                          "heavy intensity rain", "heavy intensity shower rain", 
                                          "light intensity drizzle", "light intensity drizzle rain", 
                                          "light intensity shower rain", "light rain", "light shower rain", 
                                          "moderate rain", "proximity moderate rain", "ragged shower rain", 
                                          "shower drizzle", "very heavy rain", "proximity shower rain"])),
    (weather['weather_description'].isin(["broken clouds", "overcast clouds", "scattered clouds", "few clouds"])),
    (weather['weather_description'].isin(["heavy snow", "light rain and snow", "light shower sleet", "light snow", 
                                          "rain and snow", "shower snow", "sleet", "snow", "heavy shower snow"])), 
    (weather['weather_description'].isin(["thunderstorm with drizzle", "thunderstorm with heavy drizzle", 
                                          "thunderstorm with light drizzle", "thunderstorm with rain", 
                                          "thunderstorm with light rain", "heavy thunderstorm", 
                                          "proximity thunderstorm", "proximity thunderstorm with drizzle", 
                                          "proximity thunderstorm with rain", "proximity thunderstorm", 
                                          "thunderstorm", "ragged thunderstorm"])),
    (weather['weather_description'].isin(["sky is clear"]))]
     
choices = ['rain', 'cloudy', 'snow', 'thunder', 'clear']
weather['weather_broad'] = np.select(conditions, choices, default='other')

# sklearn models won't work with NaN values. There are a whole suite of imputation techniques used to replace empty 
# values with the most appropriate estimate, but for the sake of these challenges, we'll just remove these cases.

weather = weather.dropna()
weather.head()
