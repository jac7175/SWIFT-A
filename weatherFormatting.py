import pandas as pd
import h5py
import numpy as np
from datetime import datetime
import os

# This code takes in a path 'pathFrom' to data as exported by https://uglos.mtu.edu/select_range.php?station=GLRCMET.
# Converts to standard CF format as outlined by CF-1.12: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.pdf
# Exports an HDF5 file to 'pathTo'

pathFrom = 'original/data/patj'
pathTo = 'path/to/write/new/file/to'

if os.path.exists(pathTo):
    os.remove(pathTo)

print('Processing weather data.')
weatherFrame = pd.read_csv(pathFrom)                                # reads in CSV as exported by https://uglos.mtu.edu/select_range.php?station=GLRCMET
weatherFrame[' DATE ()'] = pd.to_datetime(weatherFrame[' DATE ()']) # defines a dateTime object from the string given in the csv
weatherFrame.rename(columns={' DATE ()':'dateTime'}, inplace=True)  # renames dateTime column for easier calls
weatherFrame['dateTime'] = pd.to_datetime(weatherFrame['dateTime']) # converts to pandas datetime object
weatherFrame['dateTime'] = weatherFrame['dateTime'].dt.tz_localize('America/New_York') # assigns the New York timezone to the dateTime object
weatherFrame['dateTime'] = weatherFrame['dateTime'].dt.tz_convert('UTC') # converts the dateTime object to UTC

with h5py.File(pathTo, 'a') as f:
    f.attrs['title'] = 'SWIFT-A (Shallow Water Ice Fracture Tracking and Acoustics)'  # project title
    f.attrs['creator'] = 'John Arthur Case'  # project author
    f.attrs['description'] = 'Weather data to accompany SWIFT-A dataset' # description of data
    f.attrs['institution'] = 'Penn State University and Great Lakes Research Center, Michigan Technological University' # institutions involved
    f.attrs['source'] = 'https://uglos.mtu.edu/select_range.php?station=GLRCMET'
    f.attrs['Conventions'] = 'CF-1.12'  # CF standard (https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.pdf)

    f.create_dataset('datetime', data=[datetime.strftime(ii, "%Y-%m-%dT%H:%M:%S.%fUTC") for ii in weatherFrame['dateTime']])  # creates a dataset for datetime, assings values
    f['datetime'].attrs['units'] = 'YYYY-MM-DDThh:mm:ss.sTZD'  # ISO string unit convention
    f['datetime'].attrs['unit_convention'] = 'ISO 8601'        # ISO string unit convention
    f['datetime'].attrs['calendar'] = 'standard'               # calendar convention
    f['datetime'].attrs['long_name'] = 'calendar date'         # long name for CF convention

    f.create_dataset('latitude', data=np.float64(47.120138)) # latitude [deg north] of weather station dataset
    f['latitude'].attrs['units'] = 'degrees_north'
    f.create_dataset('longitude', data=np.float64(88.552871)) # longitude [deg west] of weather station dataset
    f['longitude'].attrs['units'] = 'degrees_west'

    f.create_dataset('air_temperature', data=weatherFrame[' ATMP1 (&deg;C)']) # adds air temperature dataset
    f['air_temperature'].attrs['units'] = 'degC'
    f['air_temperature'].attrs['units_metadata'] = 'on_scale'

    f.create_dataset('wind_from_direction', data=weatherFrame[' WDIR1 (&deg;)']) # adds wind from direction dataset
    f['wind_from_direction'].attrs['units'] = 'degree'

    f.create_dataset('wind_speed', data=weatherFrame[' WSPD1 (m/s)']) # adds wind speed dataset
    f['wind_speed'].attrs['units'] = 'm s-1'

    f.create_dataset('wind_speed_of_gust', data=weatherFrame[' GUST1 (m/s)']) # adds wind speed of gust dataset
    f['wind_speed_of_gust'].attrs['units'] = 'm s-1'

    f.create_dataset('relative_humidity', data=weatherFrame[' RRH (%)']) # adds relative humidity dataset
    f['relative_humidity'].attrs['units'] = '1'

    f.create_dataset('dew_point', data=weatherFrame[' DEWPT1 (&deg;C)']) # adds dewpoint dataset
    f['dew_point'].attrs['units'] = 'degC'

    f.create_dataset('air_pressure', data=weatherFrame[' BARO1 (hPa)'] * 100) # adds air pressure data set. Converts from hPa to Pa
    f['air_pressure'].attrs['units'] = 'Pa'

    f.create_dataset('solar_irradiance', data=weatherFrame[' SRAD1 (W/m^2)']) # adds solar irradiance dataset
    f['solar_irradiance'].attrs['units'] = 'W m-2'
print('Done.')