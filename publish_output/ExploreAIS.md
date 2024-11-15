# Alden Smith - UCSD Capstone Project (In Development)
## Abstract
Passive sonar is an excellent sensor for detecting and classifying objects surrounding the sensor.  With an adequate number and geoometry of hyrdrophones arranged in an array, the direction of sound in water can also become well resolved.  The weakness of passive sonar has always been it's inability to provide accurate, relaible range data.  

Most modern methods of range determination from passive sonar data either rely on theoretical modeling of sound propogation and dispersion paths in the surrounding water or they employ target motion analysis - a technique that uses time-series data of azimuthal bearing to geometrically constrain the  a target assumed to be operating at a constant speed.  

This study aims to take a data driven approach to better predict the range of a target by training ML models on sound data recorded by stationary hydrophones.  Various predictor variables will be extracted by pre-processed audio data available in the public domain.  Data for the target variable, range, will be determined by open-source AIS data of ships in known proximity of the chosen hydrophones. 

Accoustic Data employed in this project will be provided by the Integrated Ocean Observing System (IOOS) Sanctuary Soundscape (SANCTSOUND) project, administered  by the National Oceanagraphic and Atmospheric Administration (NOAA).  AIS data is provided by the Marine Cadastre project compiled and maintained by NOAA's Office for Coastal Management Bureau of Ocean Energy Management. 

## Proof of Concept - AIS Data
In the initital proof of concept, I chose one of the IOOS hydrophones known to be in proximity to shipping lanes off the coast of southern California.  I downloaded one day's worth of AIS data from the IOOS repository.  I then cull that data to filter out ships not expected to produce meaninghful interactions with the hydrophone based on their distance or speed (the data includes ships at anchor).  Next I filter based on time to find a small sample of isolated, single-ship interactions that can provide meaningful data for the continued study.  

Once this approach is refined it will be applied accross 30 days worth of data and 5 different hydrophone locations to ensure adequate data for model training, testing, and refinement.

### Import Dependencies


```python
# Import Dependencies
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import geopandas as gpd
import shapely
from shapely.geometry import Point
from geopy.distance import geodesic

```

### Load and Explore the AIS Data
The data was downloaded as csv from the Marine Cadastre (https://hub.marinecadastre.gov/pages/vesseltraffic) (https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2021/index.html) (https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2021/AIS_2021_01_01.zip)
This csv was converted to parqeut to improve loading speed


```python
# Load the Data Frame
ais = pd.read_parquet("D:\\UCSD_Capstone\\data\\ais_data_20210101.parquet")
ais.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6624812 entries, 0 to 6624811
    Data columns (total 17 columns):
     #   Column            Dtype  
    ---  ------            -----  
     0   MMSI              int64  
     1   BaseDateTime      object 
     2   LAT               float64
     3   LON               float64
     4   SOG               float64
     5   COG               float64
     6   Heading           float64
     7   VesselName        object 
     8   IMO               object 
     9   CallSign          object 
     10  VesselType        float64
     11  Status            float64
     12  Length            float64
     13  Width             float64
     14  Draft             float64
     15  Cargo             float64
     16  TranscieverClass  object 
    dtypes: float64(11), int64(1), object(5)
    memory usage: 859.2+ MB
    

## Preprocess Step 1 - Filter Out Anchored or Slow Moving Ships
These ships are not expected to contribute meaningful accoustic data for this study. 


```python
# Filter out boats at anchor or not moving
fast_ais = ais[ais.SOG > 3.5]
fast_ais.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 1152903 entries, 20 to 6624769
    Data columns (total 17 columns):
     #   Column            Non-Null Count    Dtype  
    ---  ------            --------------    -----  
     0   MMSI              1152903 non-null  int64  
     1   BaseDateTime      1152903 non-null  object 
     2   LAT               1152903 non-null  float64
     3   LON               1152903 non-null  float64
     4   SOG               1152903 non-null  float64
     5   COG               1152903 non-null  float64
     6   Heading           1152903 non-null  float64
     7   VesselName        991217 non-null   object 
     8   IMO               617845 non-null   object 
     9   CallSign          938772 non-null   object 
     10  VesselType        977348 non-null   float64
     11  Status            1021015 non-null  float64
     12  Length            941050 non-null   float64
     13  Width             830629 non-null   float64
     14  Draft             430135 non-null   float64
     15  Cargo             259807 non-null   float64
     16  TranscieverClass  1152903 non-null  object 
    dtypes: float64(11), int64(1), object(5)
    memory usage: 158.3+ MB
    

## Pre-Processing Step 2 - Filter Out Ships Too Distant from the Selected Hydrophone
This step is meant to incrementally cull the data set based on distance from the hydrophone.

First latitude and longitude will be used as a filter to reduce the number of calculations required to determine the distance to the hydrophone.

Next distance to the hydrophone was calculated from the remaining data

Finally this data was filtered based on a maximum distance to the hydrophone that was expected to produce meaningful accoustic data based on a review of environmental models provided on the IOOS website.  (https://sanctsound.portal.axds.co/#sanctsound/sanctuary/monterey-bay/site/MB02/method_type/prop-model)

Hydrophone location is provided in the meta-data for the accoustic files.  I have collected the meta-data for the accoustic files I expect to be using and consolidated it in a local JSON file for easy reference. 


```python
# convert the pandas dataframe to a geopandas dataframe
fast_ais_geo_df = gpd.GeoDataFrame(fast_ais, geometry=gpd.points_from_xy(fast_ais.LON, fast_ais.LAT), crs="EPSG:4326")

fast_ais_geo_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MMSI</th>
      <th>BaseDateTime</th>
      <th>LAT</th>
      <th>LON</th>
      <th>SOG</th>
      <th>COG</th>
      <th>Heading</th>
      <th>VesselName</th>
      <th>IMO</th>
      <th>CallSign</th>
      <th>VesselType</th>
      <th>Status</th>
      <th>Length</th>
      <th>Width</th>
      <th>Draft</th>
      <th>Cargo</th>
      <th>TranscieverClass</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>367382330</td>
      <td>2021-01-01T00:01:08</td>
      <td>38.27463</td>
      <td>-85.71606</td>
      <td>9.3</td>
      <td>231.2</td>
      <td>511.0</td>
      <td>GLENN R</td>
      <td>None</td>
      <td>WDI4715</td>
      <td>31.0</td>
      <td>15.0</td>
      <td>23.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A</td>
      <td>POINT (-85.71606 38.27463)</td>
    </tr>
    <tr>
      <th>22</th>
      <td>367167020</td>
      <td>2021-01-01T00:00:08</td>
      <td>29.61737</td>
      <td>-94.34092</td>
      <td>5.3</td>
      <td>244.4</td>
      <td>511.0</td>
      <td>EUGENIE</td>
      <td>IMO9091260</td>
      <td>WDF2441</td>
      <td>31.0</td>
      <td>12.0</td>
      <td>29.0</td>
      <td>9.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A</td>
      <td>POINT (-94.34092 29.61737)</td>
    </tr>
    <tr>
      <th>38</th>
      <td>518630000</td>
      <td>2021-01-01T00:00:10</td>
      <td>25.72142</td>
      <td>-77.75589</td>
      <td>10.4</td>
      <td>328.5</td>
      <td>332.0</td>
      <td>BETTY K VIII</td>
      <td>IMO8410328</td>
      <td>E5U2577</td>
      <td>70.0</td>
      <td>8.0</td>
      <td>77.0</td>
      <td>13.0</td>
      <td>4.1</td>
      <td>70.0</td>
      <td>A</td>
      <td>POINT (-77.75589 25.72142)</td>
    </tr>
    <tr>
      <th>63</th>
      <td>367709350</td>
      <td>2021-01-01T00:00:12</td>
      <td>27.99365</td>
      <td>-97.05918</td>
      <td>5.4</td>
      <td>79.0</td>
      <td>511.0</td>
      <td>ISLA MARGARET</td>
      <td>IMO7200013</td>
      <td>WDI5513</td>
      <td>31.0</td>
      <td>12.0</td>
      <td>19.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A</td>
      <td>POINT (-97.05918 27.99365)</td>
    </tr>
    <tr>
      <th>71</th>
      <td>366984210</td>
      <td>2021-01-01T00:00:14</td>
      <td>30.01366</td>
      <td>-90.83494</td>
      <td>11.7</td>
      <td>174.0</td>
      <td>173.0</td>
      <td>AMERICAN PILLAR</td>
      <td>None</td>
      <td>WDB9867</td>
      <td>31.0</td>
      <td>12.0</td>
      <td>55.0</td>
      <td>16.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A</td>
      <td>POINT (-90.83494 30.01366)</td>
    </tr>
  </tbody>
</table>
</div>




```python
# define the hydrophone point
hyd_lat = 36.6495
hyd_lon = -121.9084
hyd_point = Point(hyd_lon, hyd_lat)
```


```python
# filter down to points within 20 degrees of the hydrophone
fast_close_ais_df = fast_ais_geo_df[(fast_ais_geo_df.LAT < hyd_lat + 20) & (fast_ais_geo_df.LAT > hyd_lat - 20) & (fast_ais_geo_df.LON < hyd_lon + 20) & (fast_ais_geo_df.LON > hyd_lon - 20)].copy()
fast_close_ais_df.info()
```

    <class 'geopandas.geodataframe.GeoDataFrame'>
    Index: 181090 entries, 171 to 6615115
    Data columns (total 18 columns):
     #   Column            Non-Null Count   Dtype   
    ---  ------            --------------   -----   
     0   MMSI              181090 non-null  int64   
     1   BaseDateTime      181090 non-null  object  
     2   LAT               181090 non-null  float64 
     3   LON               181090 non-null  float64 
     4   SOG               181090 non-null  float64 
     5   COG               181090 non-null  float64 
     6   Heading           181090 non-null  float64 
     7   VesselName        153717 non-null  object  
     8   IMO               119253 non-null  object  
     9   CallSign          131945 non-null  object  
     10  VesselType        149899 non-null  float64 
     11  Status            146840 non-null  float64 
     12  Length            146114 non-null  float64 
     13  Width             134475 non-null  float64 
     14  Draft             92183 non-null   float64 
     15  Cargo             38609 non-null   float64 
     16  TranscieverClass  181090 non-null  object  
     17  geometry          181090 non-null  geometry
    dtypes: float64(11), geometry(1), int64(1), object(5)
    memory usage: 26.3+ MB
    


```python
# find the distance to the hydrophone
from geopy.distance import geodesic

fast_close_ais_df['distance_from_hyd'] = fast_close_ais_df.apply(
    lambda row: geodesic(
        (row.geometry.y, row.geometry.x),
        (hyd_point.y, hyd_point.x)
    ).meters,
    axis=1
)

fast_close_ais_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MMSI</th>
      <th>BaseDateTime</th>
      <th>LAT</th>
      <th>LON</th>
      <th>SOG</th>
      <th>COG</th>
      <th>Heading</th>
      <th>VesselName</th>
      <th>IMO</th>
      <th>CallSign</th>
      <th>VesselType</th>
      <th>Status</th>
      <th>Length</th>
      <th>Width</th>
      <th>Draft</th>
      <th>Cargo</th>
      <th>TranscieverClass</th>
      <th>geometry</th>
      <th>distance_from_hyd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>171</th>
      <td>338393441</td>
      <td>2021-01-01T00:00:48</td>
      <td>33.43823</td>
      <td>-117.65761</td>
      <td>10.8</td>
      <td>309.0</td>
      <td>511.0</td>
      <td>THE CURRENT</td>
      <td>IMO0000000</td>
      <td>None</td>
      <td>37.0</td>
      <td>NaN</td>
      <td>16.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>B</td>
      <td>POINT (-117.65761 33.43823)</td>
      <td>526519.327365</td>
    </tr>
    <tr>
      <th>193</th>
      <td>367754620</td>
      <td>2021-01-01T00:00:46</td>
      <td>32.72771</td>
      <td>-117.20546</td>
      <td>4.6</td>
      <td>88.2</td>
      <td>511.0</td>
      <td>TENACIOUS</td>
      <td>None</td>
      <td>WDJ2035</td>
      <td>90.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>B</td>
      <td>POINT (-117.20546 32.72771)</td>
      <td>612194.159085</td>
    </tr>
    <tr>
      <th>388</th>
      <td>338145694</td>
      <td>2021-01-01T00:00:47</td>
      <td>34.02874</td>
      <td>-118.52704</td>
      <td>20.9</td>
      <td>127.9</td>
      <td>121.0</td>
      <td>BAYWATCH 15</td>
      <td>None</td>
      <td>None</td>
      <td>90.0</td>
      <td>15.0</td>
      <td>9.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A</td>
      <td>POINT (-118.52704 34.02874)</td>
      <td>423069.842568</td>
    </tr>
    <tr>
      <th>410</th>
      <td>367360430</td>
      <td>2021-01-01T00:00:55</td>
      <td>32.70199</td>
      <td>-117.16374</td>
      <td>6.9</td>
      <td>226.5</td>
      <td>197.0</td>
      <td>SILVERGATE</td>
      <td>None</td>
      <td>WDE4978</td>
      <td>60.0</td>
      <td>0.0</td>
      <td>17.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>A</td>
      <td>POINT (-117.16374 32.70199)</td>
      <td>616954.638878</td>
    </tr>
    <tr>
      <th>455</th>
      <td>367014480</td>
      <td>2021-01-01T00:00:53</td>
      <td>33.72751</td>
      <td>-118.14938</td>
      <td>6.1</td>
      <td>269.4</td>
      <td>511.0</td>
      <td>DURANGO</td>
      <td>None</td>
      <td>WDC3807</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>18.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>52.0</td>
      <td>A</td>
      <td>POINT (-118.14938 33.72751)</td>
      <td>471410.362756</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Filter out vessels not within 45000 meters of the hydrophone
fast_close_ais_df_2 = fast_close_ais_df[fast_close_ais_df.distance_from_hyd < 45000]
fast_close_ais_df_2.info()
```

    <class 'geopandas.geodataframe.GeoDataFrame'>
    Index: 804 entries, 2118853 to 6613098
    Data columns (total 19 columns):
     #   Column             Non-Null Count  Dtype   
    ---  ------             --------------  -----   
     0   MMSI               804 non-null    int64   
     1   BaseDateTime       804 non-null    object  
     2   LAT                804 non-null    float64 
     3   LON                804 non-null    float64 
     4   SOG                804 non-null    float64 
     5   COG                804 non-null    float64 
     6   Heading            804 non-null    float64 
     7   VesselName         766 non-null    object  
     8   IMO                103 non-null    object  
     9   CallSign           615 non-null    object  
     10  VesselType         766 non-null    float64 
     11  Status             40 non-null     float64 
     12  Length             766 non-null    float64 
     13  Width              518 non-null    float64 
     14  Draft              0 non-null      float64 
     15  Cargo              0 non-null      float64 
     16  TranscieverClass   804 non-null    object  
     17  geometry           804 non-null    geometry
     18  distance_from_hyd  804 non-null    float64 
    dtypes: float64(12), geometry(1), int64(1), object(5)
    memory usage: 125.6+ KB
    

## Find Isolated Interactions
Reviewing the data set it seems like a number of ships are overlapping one another in time.  
Study involving multiple ships is certainly interesting and may evolve further in this project, my current intent is to simplify the data as much as possible for modeling purposes.  Therfore this data is broken down in time to remove accoustic interference from multiple, man-made sources.


```python
# Use a helper function to check the time spans of the ships in the data set and elimnate any that over lap with one another
# This will ensure that only one ship is interacting with the hydrophone during accoustic modeling.
import pandas as pd
import geopandas as gpd

def filter_isolated_ships(df):
    """
    Filter for ships that have time periods that don't overlap with any other ships.

    Parameters:
    df: GeoDataFrame with columns:
        - VesselName: identifier for each ship
        - BaseDateTime: datetime column
        - geometry: ship position
        - distance: distance to point of interest

    Returns:
    GeoDataFrame containing only ships with non-overlapping time periods
    """
    # Get time ranges for each ship
    ship_times = df.groupby('VesselName').agg({
        'BaseDateTime': ['min', 'max']
    }).reset_index()
    ship_times.columns = ['VesselName', 'start_time', 'end_time']

    # Function to check if two time ranges overlap
    def has_overlap(range1, range2):
        return (range1[0] <= range2[1]) and (range2[0] <= range1[1])

    # Find ships that don't overlap with any others
    isolated_ships = []

    for idx, ship in ship_times.iterrows():
        has_any_overlap = False

        # Compare with all other ships
        for other_idx, other_ship in ship_times.iterrows():
            if idx != other_idx:
                if has_overlap(
                    (ship['start_time'], ship['end_time']),
                    (other_ship['start_time'], other_ship['end_time'])
                ):
                    has_any_overlap = True
                    break

        if not has_any_overlap:
            isolated_ships.append(ship['VesselName'])

    # Filter original dataframe for isolated ships
    return df[df['VesselName'].isin(isolated_ships)]

isoloated_ship_df = filter_isolated_ships(fast_close_ais_df_2)
isoloated_ship_df.info()
```

    <class 'geopandas.geodataframe.GeoDataFrame'>
    Index: 2 entries, 2364611 to 2369781
    Data columns (total 19 columns):
     #   Column             Non-Null Count  Dtype   
    ---  ------             --------------  -----   
     0   MMSI               2 non-null      int64   
     1   BaseDateTime       2 non-null      object  
     2   LAT                2 non-null      float64 
     3   LON                2 non-null      float64 
     4   SOG                2 non-null      float64 
     5   COG                2 non-null      float64 
     6   Heading            2 non-null      float64 
     7   VesselName         2 non-null      object  
     8   IMO                2 non-null      object  
     9   CallSign           2 non-null      object  
     10  VesselType         2 non-null      float64 
     11  Status             2 non-null      float64 
     12  Length             2 non-null      float64 
     13  Width              2 non-null      float64 
     14  Draft              0 non-null      float64 
     15  Cargo              0 non-null      float64 
     16  TranscieverClass   2 non-null      object  
     17  geometry           2 non-null      geometry
     18  distance_from_hyd  2 non-null      float64 
    dtypes: float64(12), geometry(1), int64(1), object(5)
    memory usage: 320.0+ bytes
    


```python
print(isoloated_ship_df)
```

                  MMSI         BaseDateTime       LAT        LON  SOG    COG  \
    2364611  367754810  2021-01-01T10:17:16  36.80796 -121.78519  3.9  342.8   
    2369781  367754810  2021-01-01T10:18:26  36.80857 -121.78669  5.1  250.8   
    
             Heading VesselName         IMO CallSign  VesselType  Status  Length  \
    2364611    511.0     S BASS  IMO9094896  WDJ2053        31.0     0.0    18.0   
    2369781    511.0     S BASS  IMO9094896  WDJ2053        31.0     0.0    18.0   
    
             Width  Draft  Cargo TranscieverClass                     geometry  \
    2364611    7.0    NaN    NaN                A  POINT (-121.78519 36.80796)   
    2369781    7.0    NaN    NaN                A  POINT (-121.78669 36.80857)   
    
             distance_from_hyd  
    2364611       20744.955192  
    2369781       20731.766703  
    


```python
import pandas as pd
from datetime import timedelta

def find_isolated_periods(df, min_isolation_minutes=15):
    """
    Find periods where ships have no temporal overlap with others,
    returning only periods >= min_isolation_minutes.
    """
    # Ensure BaseDateTime is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df['BaseDateTime']):
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
    
    # Get all unique timestamps for all ships
    all_times = df[['BaseDateTime', 'VesselName']].copy()
    
    # Find isolated periods for each ship
    isolated_periods = []
    
    for ship in df['VesselName'].unique():
        # Get times for current ship
        ship_times = set(df[df['VesselName'] == ship]['BaseDateTime'])
        
        # Get times for all other ships
        other_times = set(df[df['VesselName'] != ship]['BaseDateTime'])
        
        # Get sorted list of times for current ship
        ship_times_sorted = sorted(ship_times)
        
        # Initialize variables for finding isolation periods
        current_start = None
        current_end = None
        in_isolation = False
        
        # Examine each timestamp for the current ship
        for i, time in enumerate(ship_times_sorted):
            # Check if any other ship is present at this time
            is_isolated = time not in other_times
            
            # Start new isolation period
            if is_isolated and not in_isolation:
                current_start = pd.Timestamp(time)
                in_isolation = True
            
            # End current isolation period
            elif (not is_isolated and in_isolation) or (is_isolated and i == len(ship_times_sorted) - 1):
                if is_isolated and i == len(ship_times_sorted) - 1:
                    current_end = pd.Timestamp(time)  # Include last timestamp if isolated
                else:
                    current_end = pd.Timestamp(ship_times_sorted[i-1])
                
                # Calculate duration
                try:
                    duration = (current_end - current_start).total_seconds() / 60
                except Exception as e:
                    print(f"Error calculating duration for {ship}:")
                    print(f"Start: {current_start} ({type(current_start)})")
                    print(f"End: {current_end} ({type(current_end)})")
                    raise e
                
                # Only keep periods longer than minimum duration
                if duration >= min_isolation_minutes:
                    isolated_periods.append({
                        'VesselName': ship,
                        'isolation_start': current_start,
                        'isolation_end': current_end,
                        'isolation_duration_minutes': duration
                    })
                
                in_isolation = False
                current_start = None
    
    # Convert results to DataFrame
    if isolated_periods:
        results_df = pd.DataFrame(isolated_periods)
        return results_df.sort_values(['VesselName', 'isolation_start'])
    else:
        return pd.DataFrame(columns=['VesselName', 'isolation_start', 'isolation_end', 'isolation_duration_minutes'])
    
iso_period_df = find_isolated_periods(fast_close_ais_df_2, 15)
iso_period_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 19 entries, 2 to 17
    Data columns (total 4 columns):
     #   Column                      Non-Null Count  Dtype         
    ---  ------                      --------------  -----         
     0   VesselName                  19 non-null     object        
     1   isolation_start             19 non-null     datetime64[ns]
     2   isolation_end               19 non-null     datetime64[ns]
     3   isolation_duration_minutes  19 non-null     float64       
    dtypes: datetime64[ns](2), float64(1), object(1)
    memory usage: 760.0+ bytes
    

    C:\Users\alden\AppData\Local\Programs\Python\Python312\Lib\site-packages\geopandas\geodataframe.py:1819: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      super().__setitem__(key, value)
    


```python
iso_period_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VesselName</th>
      <th>isolation_start</th>
      <th>isolation_end</th>
      <th>isolation_duration_minutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>BLACKFIN</td>
      <td>2021-01-01 18:12:19</td>
      <td>2021-01-01 23:15:06</td>
      <td>302.783333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BLACKFIN</td>
      <td>2021-01-01 23:18:05</td>
      <td>2021-01-01 23:57:34</td>
      <td>39.483333</td>
    </tr>
    <tr>
      <th>18</th>
      <td>COOL CHANGE</td>
      <td>2021-01-01 21:45:56</td>
      <td>2021-01-01 23:47:23</td>
      <td>121.450000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>GODDESS FANTASY</td>
      <td>2021-01-01 20:58:18</td>
      <td>2021-01-01 21:35:17</td>
      <td>36.983333</td>
    </tr>
    <tr>
      <th>15</th>
      <td>GODDESS FANTASY</td>
      <td>2021-01-01 21:40:47</td>
      <td>2021-01-01 22:05:46</td>
      <td>24.983333</td>
    </tr>
  </tbody>
</table>
</div>



## Next Steps
Immediately I intend to visualize this data both in geographic space and in the time domain as well.
Following that I will download the accoustic data from this hydrophone during the time period in question, process that data, and attempt to identify accousitc signals from the ships identified in 'iso_period_df'


```python

```
