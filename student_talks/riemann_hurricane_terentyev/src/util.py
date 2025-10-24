import matplotlib
import matplotlib.pylab as plt
import jax
import jax.numpy as jnp
import numpy as np
from morphomatics.geom import BezierSpline
try:
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.basemap import Basemap
except: print('mpl_toolkits not available')
from matplotlib import cm
import matplotlib.patches as mpatches

colors = {0: "b", 1: "orange", 2: "r"}
group_0 = mpatches.Patch(color=colors[0], label="Cat 0")
group_1 = mpatches.Patch(color=colors[1], label="Cat 1-3")
group_2 = mpatches.Patch(color=colors[2], label="Cat 4-5")
legend_handle = [group_0, group_1, group_2]
cmap_cat = cm.get_cmap('jet')
cnorm_cat = cm.colors.Normalize(vmin=20, vmax=137)
N_SUBJ, N_SAMPLES = 218, 32  #N_SAMPLES:average = 32, Max = 96

def get_subj_year(subj):
    """
    Given subj array (shape [n_storms, 3]), returns a list of indices where each year starts in subj.
    Assumes storm_id is in format 'AL012010' (last 4 digits = year).
    Returns: years (list of int), where years[i] is the index in subj where a new year starts.
    """
    years = []
    last_year = None
    for i, row in enumerate(subj):
        storm_id = str(row[0]).strip()
        # Extract year from storm_id (last 4 digits)
        year = int(storm_id[-4:])
        if year != last_year:
            years.append(i)
            last_year = year
    years.append(len(subj))  # end index for last year
    return years
def coord_2D3D(lat, lon, h=0.0):
    """
    this function converts latitude,longitude and height above sea level
    to earthcentered xyx coordinates in wgs84, lat and lon in decimal degrees
    e.g. 52.724156(West and South are negative), heigth in meters
    for algoritm see https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates
    for values of a and b see https://en.wikipedia.org/wiki/Earth_radius#Radius_of_curvature
    """

    latr = np.pi*lat/180  # latitude in radians
    lonr = np.pi*lon/180  # longituede in radians

    x = np.cos(latr) * np.cos(lonr)
    y = np.cos(latr) * np.sin(lonr)
    z = np.sin(latr)
    return x, y, z


def coord_3D2D(xyz):
    x, y, z = xyz[0], xyz[1], xyz[2]
    lat = np.sign(z)*180*np.arctan(z/np.sqrt(x**2 + y**2))/np.pi
    lon = 180*np.arctan2(y, x)/np.pi  # West is negative
    return lat, lon


def visEarth(seq_lists, colors, title=None, c_map=cmap_cat):
    _ = plt.figure(figsize=(12, 12))
    plt.rcParams['font.size'] = 14

    # define color maps for water and land
    ocean_map = (plt.get_cmap('ocean'))(210)
    cmap = plt.get_cmap('gist_earth')

    # call the basemap and use orthographic projection at viewing angle
    m = Basemap(projection='ortho', lat_0=10, lon_0=-45)

    # coastlines, map boundary, fill continents/water, fill ocean, draw countries
    m.drawcoastlines()
    m.drawmapboundary(fill_color=ocean_map)
    m.fillcontinents(color=cmap(200), lake_color=ocean_map)
    m.drawcountries()

    # latitude/longitude line vectors
    lat_line_range, lat_lines = [-90, 90], 8
    lat_line_count = (lat_line_range[1] - lat_line_range[0]) / lat_lines

    merid_range, merid_lines = [-180, 180], 8
    merid_count = (merid_range[1] - merid_range[0]) / merid_lines

    m.drawparallels(np.arange(lat_line_range[0], lat_line_range[1], lat_line_count))
    m.drawmeridians(np.arange(merid_range[0], merid_range[1], merid_count))

    # add points
    if title is not None:
        plt.title(title)

    for k, seq_list in enumerate(seq_lists):
        latlons = []
        if seq_list[0].shape[-1] == 3:
            for s in seq_list:
                latlon = np.zeros((s.shape[0], 2))
                for j in range(s.shape[0]):
                    latlon[j, 0], latlon[j, 1] = coord_3D2D(s[j])
                latlons.append(latlon)
        else:
            latlons = seq_list

        for i in range(len(seq_list)):
            lats, lons = latlons[i][:, 0], latlons[i][:, 1]
            norm = cnorm_cat
            x, y = m(lons, lats)

            _ = m.scatter(x, y, marker='.', linewidth=.5, c=np.array([colors[k], ]*len(x)), cmap=c_map, norm=norm)

    plt.clim(0, 137)
    plt.show(block=None)
def sample_spline(B: BezierSpline, n: int = 50) -> np.array:
    """Sample a Bezier spline at n evenly spaced points"""
    return np.array(jax.vmap(B.eval)(jnp.linspace(0, B.nsegments, n)))


def get_seq_date(seq, start, end):
    a = seq[:, 0]
    x = [i for i in range(len(a)) if (a[i] >= start) & (a[i] <= end)]
    return x