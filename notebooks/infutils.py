"""
DEPENDENCIES
------------
- xarray
- cartopy
- matplotlib
- numpy


USAGE EXAMPLE
-------------
import fco2_region_plots as fco2_plots

ds = xr.open_dataset('/Users/luke/Downloads/OceanSODA_ETHZ_HR-v2023.01-dfco2-20200921_20200928.nc')
da = ds.dfco2.compute()

fco2_plots.plot_dfco2_regions(da)
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from cartopy import crs as ccrs


RCMAPS = {
    # these are specific to the geo_subplot function
    "proj": ccrs.PlateCarree(central_longitude=0),
    "land_color": "none",
    "coast_res": "110m",
    "coast_lw": 0.5,
    "robust": True,
    "round": False,
    # you can add any colorbar kwarg here and it will be set as the default
    "colorbar.pad": 0.02,
    "colorbar.fraction": 0.1,
}


REGIONS = dict(
    bengulhas = dict(lat=slice(-42.5, -17), lon=slice(10, 35)),
    califcur = dict(lat=slice(28, 48), lon=slice(-133, -114)),
    capeverd = dict(lat=slice(9, 27), lon=slice(-28, -10)),
    kurioshio = dict(lat=slice(25, 50), lon=slice(130, 150)),
    gapwinds = dict(lat=slice(0, 22), lon=slice(-100, -78)),
    eqpacifc = dict(lat=slice(-25, 20), lon=slice(-164, -69)),
    malvinas = dict(lat=slice(-60, -35), lon=slice(-70, -45.5)),
)


def plot_dfco2_regions(dfco2_single_timestep: xr.DataArray)->tuple:
    """
    Plot a single timestep of the dfco2 data array on a map.

    Parameters
    ----------
    dfco2_single_timestep : xr.DataArray
        A single timestep of the dfco2 data array.

    Returns
    -------
    tuple
        A tuple containing the figure, axes, and the plotted data array.
    """
    
    from cartopy import crs
    from string import ascii_lowercase

    da = dfco2_single_timestep
    
    fig = plt.figure(figsize=[8, 5.7], dpi=200)
    spec = fig.add_gridspec(ncols=3, nrows=2)
    ax = [
        fig.add_subplot(spec[0, 0], projection=crs.PlateCarree()),
        fig.add_subplot(spec[0, 1], projection=crs.PlateCarree()),
        fig.add_subplot(spec[0, 2], projection=crs.PlateCarree()),
        fig.add_subplot(spec[1, 0], projection=crs.PlateCarree()),
        fig.add_subplot(spec[1, 1:], projection=crs.PlateCarree()),
    ]

    img = [
        plot_regional_map(da.sel(**REGIONS['capeverd']),  ax=ax[0]),
        plot_regional_map(da.sel(**REGIONS['califcur']),  ax=ax[1]),
        plot_regional_map(da.sel(**REGIONS['gapwinds']),  ax=ax[2]),
        plot_regional_map(da.sel(**REGIONS['malvinas']),  ax=ax[3]),
        plot_regional_map(da.sel(**REGIONS['eqpacifc']),  ax=ax[4])]

    fig.subplots_adjust(hspace=0.3, wspace=0.11)

    # names of map regions
    names = [
        f'Northwest Africa', 
        f'California Current', 
        f'Tehuano / Papagayo',
        f'Malvinas Current',
        f'Equatorial Pacific'
    ]
    # set axes titles aligned left with figure letters
    for i, a in enumerate(ax):
        a.set_title(f"{ascii_lowercase[i]}) {names[i]}", ha='left', va='center', x=0.01, fontsize=9.7, pad=0, color='k')
        # add a colorbar to each image (below the axes)
        cbar_height = 0.04
        cax = fig.add_axes([a.get_position().x0, a.get_position().y0 - cbar_height, a.get_position().width, cbar_height])
        cbar = plt.colorbar(img[i][2], cax=cax, orientation='horizontal')
        cax.set_title(f"$\\mathit{{f}}\\mathrm{{CO}}_2$ (µatm)", ha='left', va='center', x=0.02, y=0.5, fontsize=9.5, pad=0, color='w')
        cax.xaxis.set_tick_params(color='0.7', labelcolor='0.6', length=0.5, pad=1)


    return fig, ax, img



def plot_regional_map(da, **kwargs):

    props = dict(cmap='inferno', robust=True, add_colorbar=False, interpolation='nearest', rasterized=True)

    if kwargs.get('ax', None) is None:
        print('added subplot')
        props.update(geo_subplot(pos=111, coast_res='50m'))
    else:
        props.update(transform=ccrs.PlateCarree())
        kwargs['ax'].coastlines(resolution='10m', color='k', lw=0.5, zorder=2)
        
    props.update(kwargs)

    da = da.squeeze().ffill('lon', limit=1).bfill('lon', limit=1).ffill('lat', limit=1).bfill('lat', limit=1)
    img = da.plot.imshow(**props)

    img.date = da.time.values.astype('datetime64[D]')
    
    return img.figure, img.axes, img



def geo_subplot(
    pos=111,
    proj=RCMAPS["proj"],
    round=RCMAPS["round"],
    land_color=RCMAPS["land_color"],
    coast_res=RCMAPS["coast_res"],
    fig=None,
    dpi=90,
    figsize=None,
    **kwargs
):
    """
    Makes an axes object with a cartopy projection for the current figure

    Parameters
    ----------
    pos: int/list [111]
        Either a 3-digit integer or three separate integers
        describing the position of the subplot. If the three
        integers are *nrows*, *ncols*, and *index* in order, the
        subplot will take the *index* position on a grid with *nrows*
        rows and *ncols* columns. *index* starts at 1 in the upper left
        corner and increases to the right.

        *pos* is a three digit integer, where the first digit is the
        number of rows, the second the number of columns, and the third
        the index of the subplot. i.e. fig.add_subplot(235) is the same as
        fig.add_subplot(2, 3, 5). Note that all integers must be less than
        10 for this form to work.
    proj: crs.Projection()
        the cartopy coord reference system object to create the projection.
        Defaults to crs.PlateCarree(central_longitude=205) if not given
    round: bool [True]
        If the projection is stereographic, round will cut the corners and
        make the plot round
    land_color: str ['w']
        the color of the land patches
    coast_res: str ['110m']
        the resolution at which coastal lines are plotted. Valid options are
        110m, 50m, 10m
    **kwargs:
        passed to fig.add_subplot(**kwargs)

    """
    from cartopy import feature, crs
    import matplotlib.path as mpath

    fig = plt.gcf()

    is_default_width = fig.get_figwidth() == plt.rcParams["figure.figsize"][0]
    is_default_height = fig.get_figheight() == plt.rcParams["figure.figsize"][1]
    if is_default_width and is_default_height:
        n_row = pos // 100
        n_col = (pos - (n_row * 100)) // 10
        width = n_col * 8
        height = n_row * 3.5
        fig.set_size_inches(width, height)

    ax = fig.add_subplot(pos, projection=proj, **kwargs)
    ax._autoscaleXon = False
    ax._autoscaleYon = False

    # makes maps round
    stereo_maps = (
        crs.Stereographic,
        crs.NorthPolarStereo,
        crs.SouthPolarStereo,
    )
    if isinstance(ax.projection, stereo_maps) & round:

        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.475
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)

        ax.set_boundary(circle, transform=ax.transAxes)

    # adds features
    if coast_res == "110m":
        ax.add_feature(feature.LAND, zorder=4, color=land_color)
    else:
        ax.add_feature(
            feature.NaturalEarthFeature(
                "physical", "land", coast_res, facecolor=land_color
            )
        )

    ax.coastlines(
        resolution=coast_res, color="black", linewidth=RCMAPS["coast_lw"], zorder=5
    )

    ax.spines["geo"].set_lw(RCMAPS["coast_lw"])
    ax.spines["geo"].set_zorder(5)

    return {"ax": ax, "transform": crs.PlateCarree()}