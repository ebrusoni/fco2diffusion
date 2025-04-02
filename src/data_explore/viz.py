from matplotlib import pyplot as plt
import pandas as pd
import geopandas as gpd
from loguru import logger


# example of useage: gdf.explore(**shp.GOOGLE_TERRAIN)
_LEAFLET_DEFAULTS = dict()
GOOGLE_TERRAIN = dict(tiles='http://mt0.google.com/vt/lyrs=p&hl=en&x={x}&y={y}&z={z}', attr='Google', **_LEAFLET_DEFAULTS)
GOOGLE_SATELLITE = dict(tiles='http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}', attr='Google', **_LEAFLET_DEFAULTS)


def plot_interactive_cruise_map_lines(df, **kwargs):
    """
    Plot interactive map of cruises using folium. Works well on 
    single years of data. 

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cruise data that contains columns 'lon_005', 'lat_005', 'time_avg', 'expocode'
    
    Returns
    -------
    m : folium.Map
        Interactive map of cruises with day of year as color

    """
    groups = df.groupby(by='expocode')

    logger.info("Preparing data for interactive map")
    df_plotting = pd.concat([
        prep_cruise_data_for_leaflet(grp)  # create LineString instead of markers for each cruise with some metadata
        for _, grp in groups  # iterate over groups
        if len(grp) > 30],  # filtering short cruises
        axis=0)

    t0 = df_plotting.Start
    t1 = df_plotting.End
    df_plotting['doy'] = (t0 + (t1 - t0) / 2).dt.dayofyear  # midpoint of cruise as day of year
    
    # create map using pandas api
    logger.info("Creating interactive map")
    m = df_plotting.explore(column='doy', cmap='hsv')
    
    return m


def plot_cruise_interactive_scatter_map(ser: pd.Series, lat_name='lat_005', lon_name='lon_005', time_name='time_1d', **kwargs):
    name = ser.name
    ser = latlon_to_geometry(ser, lat_name=lat_name, lon_name=lon_name).reset_index()
    props = kwargs | dict(column=name, tiles='cartodbpositron',)
    ser = clean_indicies_for_plotting(ser, lat_name=lat_name, lon_name=lon_name, time_name=time_name)
    map = ser.explore(**props)
    return map


def latlon_to_geometry(df: pd.DataFrame, lat_name='lat_005', lon_name='lon_005') -> gpd.GeoDataFrame:
    """
    Convert lat lon to geometry for plotting in geopandas

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'lon_005', 'lat_005'
    """

    index_names = df.index.names
    df = df.reset_index()

    lon360 = df[lon_name]
    lat = df[lat_name]

    if lon360.diff().abs().max() > 180:
        lon180 = (lon360 - 180) % 360 - 180
        lon = lon180
    else:
        lon = lon360

    geom = gpd.points_from_xy(lon, lat)
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs='EPSG:4326')

    gdf = gdf.set_index(index_names)

    return gdf


def prep_cruise_data_for_leaflet(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Prepare cruise data for plotting in leaflet converting from Points to LineString
    which is easier to display for many cruises (but individual point data is lost)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cruise data that contains columns 'lon_005', 'lat_005', 'time_avg', 'expocode'
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    df_ = latlon_to_geometry(df, lat_name='lat_005', lon_name='lon_005').reset_index()
    
    ser = gpd.GeoSeries(LineString(df_.geometry), crs='EPSG:4326', name='geometry')

    gdf = gpd.GeoDataFrame(ser).reset_index(drop=True)
    gdf.loc[0, 'Expocode'] = df_.expocode.iloc[0]
    gdf.loc[0, 'Start'] = df_['time_1d'].min()
    gdf.loc[0, 'End'] = df_['time_1d'].max()

    return gdf


def clean_indicies_for_plotting(ser: pd.Series, lat_name='lat_005', lon_name='lon_005', time_name='time_1d') -> pd.Series:
    """
    Lat lon may not be nicely rounded and datetime too long for display - this fixes that
    """
    return (
        ser
        .reset_index()
        .assign(  # cleaning up indicies for plotting
            lat=lambda x: x[lat_name].round(3),
            lon=lambda x: x[lon_name].round(3),
            time=lambda x: x[time_name].dt.strftime('%Y-%m-%d'))
        .set_index(['time', 'lat', 'lon'])
    )