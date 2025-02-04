import pandas as pd
import h5py
import numpy as np
from datetime import datetime
import os

pathFrom = 'original/data/patj'
pathTo = 'path/to/write/new/file/to'

if os.path.exists(pathTo):
    os.remove(pathTo)

print('Processing weather data.')
weatherFrame = pd.read_csv(pathFrom)
weatherFrame[' DATE ()'] = pd.to_datetime(weatherFrame[' DATE ()'])
weatherFrame.rename(columns={' DATE ()':'dateTime'}, inplace=True)
weatherFrame['dateTime'] = pd.to_datetime(weatherFrame['dateTime'])
weatherFrame['dateTime'] = weatherFrame['dateTime'].dt.tz_localize('America/New_York')
weatherFrame['dateTime'] = weatherFrame['dateTime'].dt.tz_convert('UTC')
# weatherFrame.set_index('dateTime', inplace=True)
weatherFrame.drop(['ID ()', ' VBAT (V)', ' STATION ()', ' FM64III ()',
                   ' FM64XX ()', ' FM64K1 ()', ' FM64K2 ()', ' FM64K3 ()',
                   ' FM64K4 ()', ' FM64K6 ()', ' RAMOUNT ()', ' HAMOUNT ()',
                   ' PAR1 ()', ' MET ()', ' WCHILL ()'], axis=1, inplace=True)
weatherFrame.rename(columns={' WDIR1 (&deg;)': 'wdir', ' WSPD1 (m/s)': 'wspd',
                             ' GUST1 (m/s)': 'wgust', ' ATMP1 (&deg;C)': 'atmp',
                             ' RRH (%)': 'rrh', ' DEWPT1 (&deg;C)': 'dwpt',
                             ' BARO1 (hPa)': 'baro', ' SRAD1 (W/m^2)': 'srad1',
                             ' SRAD2 (&micro;mol s^-1 m^-2)': 'srad2'}, inplace=True)

lat = np.float64(47.120138)  # latitude [deg north], center weather station
lon = np.float64(88.552871)  # longitude [deg west], center weather station

with h5py.File(pathTo, 'a') as f:
    # Global attributes
    f.attrs['title'] = 'SWIFT-A (Shallow Water Ice Fracture Tracking and Acoustics)'
    f.attrs['creator'] = 'John Arthur Case'
    f.attrs['description'] = 'Weather data to accompany SWIFT-A dataset'
    f.attrs['institution'] = 'Penn State University and Great Lakes Research Center, Michigan Technological University'
    f.attrs['source'] = 'https://uglos.mtu.edu/select_range.php?station=GLRCMET'
    f.attrs['Conventions'] = 'CF-1.12'

    f.create_dataset('datetime', data=[datetime.strftime(ii, "%Y-%m-%dT%H:%M:%S.%fUTC") for ii in weatherFrame['dateTime']])  # creates a group for creationDateTime
    f['datetime'].attrs['units'] = 'YYYY-MM-DDThh:mm:ss.sTZD'  # ISO string unit convention
    f['datetime'].attrs['unit_convention'] = 'ISO 8601'  # ISO string unit convention
    f['datetime'].attrs['calendar'] = 'standard'  # calendar convention
    f['datetime'].attrs['long_name'] = 'calendar date'  # long name for CF convention

    f.create_dataset('latitude', data=lat)
    f['latitude'].attrs['units'] = 'degrees_north'
    f.create_dataset('longitude', data=lat)
    f['longitude'].attrs['units'] = 'degrees_west'

    f.create_dataset('air_temperature', data=weatherFrame['atmp'])
    f['air_temperature'].attrs['units'] = 'degC'
    f['air_temperature'].attrs['units_metadata'] = 'on_scale'

    f.create_dataset('wind_from_direction', data=weatherFrame['wdir'])
    f['wind_from_direction'].attrs['units'] = 'degree'

    f.create_dataset('wind_speed', data=weatherFrame['wspd'])
    f['wind_speed'].attrs['units'] = 'm s-1'

    f.create_dataset('wind_speed_of_gust', data=weatherFrame['wgust'])
    f['wind_speed_of_gust'].attrs['units'] = 'm s-1'

    f.create_dataset('relative_humidity', data=weatherFrame['rrh'])
    f['relative_humidity'].attrs['units'] = '1'

    f.create_dataset('dew_point', data=weatherFrame['dwpt'])
    f['dew_point'].attrs['units'] = 'degC'

    f.create_dataset('air_pressure', data=weatherFrame['baro'] * 100)
    f['air_pressure'].attrs['units'] = 'Pa'

    f.create_dataset('solar_irradiance', data=weatherFrame['srad1'])
    f['solar_irradiance'].attrs['units'] = 'W m-2'
print('Done.')