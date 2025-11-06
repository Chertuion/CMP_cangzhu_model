import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from shapely.geometry import shape
import rasterio
from rasterio import features
from rasterio.transform import from_origin


def points_to_grid(lon_all, lat_all, value_all):
    df = pd.DataFrame({"lon": lon_all, "lat": lat_all, "val": value_all})
    
    grid = df.pivot(index="lat", columns="lon", values="val")
    grid = grid.sort_index().sort_index(axis=1)
    lat = grid.index.values          
    lon = grid.columns.values        
    Z = grid.values                  
    
    dlon = np.median(np.diff(lon))
    dlat = np.median(np.diff(lat))
    return lon, lat, Z, float(dlon), float(dlat)


def classify_grid(Z, bins=None, n_bins=5):
    if bins is None:
        vmin = np.nanmin(Z)
        vmax = np.nanmax(Z)
        bins = np.linspace(vmin, vmax, n_bins + 1)
    
    cat = np.digitize(Z, bins) - 1           
    cat[np.isnan(Z)] = -1                    
    return cat.astype(np.int16), bins


def categories_to_polygons(cat, lon, lat, dlon, dlat):
    
    cat_topdown = np.flipud(cat)                           
    north = lat.max() + dlat / 2.0
    west  = lon.min() - dlon / 2.0
    transform = from_origin(west=west, north=north, xsize=dlon, ysize=dlat)

    geoms = []
    vals  = []
    for geom, val in features.shapes(cat_topdown, transform=transform):
        if val < 0:   
            continue
        geoms.append(shape(geom))
        vals.append(int(val))
    gdf = gpd.GeoDataFrame({"zone": vals}, geometry=geoms, crs="EPSG:4326")
    
    gdf = gdf.dissolve(by="zone", as_index=False)
    return gdf


def draw_suitability_choropleth_from_points(
    lon_all, lat_all, value_all,
    china_level1_path,
    bins=None, n_bins=5,
    cmap="YlGnBu",
    extent=(73, 135, 18, 54),
    figsize=(10, 8), dpi=300,
    title="Suitability zones (polygon)",
    save_path=None, show=True
):
    
    lon, lat, Z, dlon, dlat = points_to_grid(lon_all, lat_all, value_all)
    
    cat, bins = classify_grid(Z, bins=bins, n_bins=n_bins)
    
    zones_gdf = categories_to_polygons(cat, lon, lat, dlon, dlat)

    
    prov = gpd.read_file(china_level1_path)
    if prov.crs is None or prov.crs.to_epsg() != 4326:
        prov = prov.to_crs("EPSG:4326")
    china_union = prov.geometry.unary_union
    china_gdf = gpd.GeoDataFrame(geometry=[china_union], crs="EPSG:4326")

    zones_clip = gpd.overlay(zones_gdf, china_gdf, how="intersection")

    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])

    K = len(bins) - 1
    cmap_obj = plt.get_cmap(cmap, K)
    zones_clip.plot(column="zone", cmap=cmap_obj, linewidth=0.2, edgecolor="white", ax=ax)

    
    prov.boundary.plot(ax=ax, color="black", linewidth=0.5, alpha=0.9)

    
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=BoundaryNorm(bins, K))
    sm._A = []
    cbar = plt.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Suitability")

    ax.set_title(title)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"zones_gdf": zones_clip, "bins": bins}



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd
from shapely.geometry import Polygon
from tqdm import tqdm
from utils.util import visualize_distribution


def get_equal_bins(predictions, n=5):
    
    visualize_distribution(predictions, save_path=f"/data/home/wxl22/changzu_ans/distribution.png")
    arr = np.asarray(predictions).ravel()
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    bins = np.linspace(vmin, vmax, n + 1)
    
    
    return bins

def get_quantile_bins(predictions, n=4):
    
    arr = np.asarray(predictions).ravel()
    arr = arr[~np.isnan(arr)]
    qs = np.linspace(0, 1, n+1)
    bins = np.quantile(arr, qs)
    return bins

import numpy as np
import mapclassify as mc

def get_natural_breaks_bins(values, k=4):
    
    arr = np.asarray(values).ravel()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("Input data is empty; cannot compute natural breaks")

    nb = mc.NaturalBreaks(arr, k=k)
    edges = nb.bins  
    bins = np.r_[arr.min(), edges]  

    
    zones = np.digitize(values, bins, right=True) - 1
    return bins, zones


import numpy as np
import mapclassify as mc

def get_natural_breaks_bins(values, n=4):
    
    arr = np.asarray(values).ravel()
    arr = arr[~np.isnan(arr)]  
    if arr.size == 0:
        raise ValueError("Input data is empty; cannot compute natural breaks")

    nb = mc.NaturalBreaks(arr, n=4)
    edges = nb.bins
    bins = np.r_[-np.inf, edges]  

    zones = np.digitize(values, bins) - 1
    return bins, zones


    import numpy as np

def compute_zones(
    values,
    method="quantile",   
    k=4,                 
    clip_quantiles=None, 
    fixed_bins=None,     
    kmeans_n_init=10,    
    kmeans_random_state=0,
    zscore_cut=(-1, 0, 1), 
    return_labels=True,
    decimals=3           
):
    
    arr = np.asarray(values).ravel()
    valid = ~np.isnan(arr)
    arr_valid = arr[valid]
    if arr_valid.size == 0:
        raise ValueError("values is empty or entirely NaN")

    
    arr_for_bins = arr_valid.copy()
    if clip_quantiles is not None:
        ql, qh = clip_quantiles
        lo, hi = np.quantile(arr_valid, [ql, qh])
        arr_for_bins = np.clip(arr_valid, lo, hi)

    method = method.lower()

    if method == "equal":
        
        vmin, vmax = arr_valid.min(), arr_valid.max()
        bins = np.linspace(vmin, vmax, k + 1)

    elif method == "quantile":
        
        qs = np.linspace(0, 1, k + 1)
        bins = np.quantile(arr_for_bins, qs)

    elif method == "jenks":
        
        import mapclassify as mc
        nb = mc.NaturalBreaks(arr_for_bins, k=k)
        
        bins = np.r_[arr_valid.min(), nb.bins]  

    elif method == "kmeans":
        
        from sklearn.cluster import KMeans
        x = arr_for_bins.reshape(-1, 1)
        km = KMeans(n_clusters=k, n_init=kmeans_n_init, random_state=kmeans_random_state).fit(x)
        centers = np.sort(km.cluster_centers_.ravel())
        thr = (centers[:-1] + centers[1:]) / 2
        bins = np.r_[arr_valid.min(), thr, arr_valid.max()]

    elif method == "zscore":
        
        mu, sd = float(np.mean(arr_for_bins)), float(np.std(arr_for_bins))
        thr = [mu + t * sd for t in zscore_cut]
        thr = np.array(sorted(thr))
        bins = np.r_[arr_valid.min(), thr, arr_valid.max()]
        k = len(bins) - 1  

    elif method == "fixed":
        
        if fixed_bins is None:
            raise ValueError("method='fixed' requires fixed_bins=[min, ..., max]")
        bins = np.asarray(fixed_bins, dtype=float)
        if np.any(~np.isfinite(bins)) or np.any(np.diff(bins) <= 0):
            raise ValueError("fixed_bins must be a finite, strictly increasing array")
        k = len(bins) - 1

    else:
        raise ValueError("method must be one of 'equal'/'quantile'/'jenks'/'kmeans'/'zscore'/'fixed'")

    
    bins[0]  = min(bins[0],  arr_valid.min())
    bins[-1] = max(bins[-1], arr_valid.max())

    
    zones = np.digitize(values, bins, right=True) - 1
    
    zones = np.clip(zones, 0, k - 1)

    if not return_labels:
        return bins, zones

    
    def _fmt(x):
        return f"{np.round(x, decimals):.{decimals}f}"
    labels = [f"{_fmt(bins[i])}–{_fmt(bins[i+1])}" for i in range(k)]
    return bins, zones, labels



def _mode_int(a):
    a = np.asarray(a, dtype=np.int64)
    if a.size == 0:
        return np.nan
    vals, cnt = np.unique(a, return_counts=True)
    return int(vals[np.argmax(cnt)])


def draw_suitability_map_china(
    lon, lat, predictions,
    bins,
    china_level1_path,         
    gpkg_layer=None,           
    grid_res=0.25,             
    agg="mode",                
    labels=None,
    figsize=(10, 8),
    dpi=200,
    cmap_colors=None,          
    xlim=(73, 135),
    ylim=(18, 54),
    add_insets=None,           
    save_path=None,
    show=True,
    title="Suitability zones (aggregated to grid)"
):
    
    lon = np.asarray(lon).ravel()
    lat = np.asarray(lat).ravel()
    preds = np.asarray(predictions).ravel()
    m = (~np.isnan(lon)) & (~np.isnan(lat)) & (~np.isnan(preds))
    lon, lat, preds = lon[m], lat[m], preds[m]

    zones_raw = np.digitize(preds, bins) - 1
    K = len(bins) - 1
    if labels is None:
        labels = list(range(K))

    
    if china_level1_path.lower().endswith(".gpkg"):
        if gpkg_layer is None:
            raise ValueError("Reading .gpkg requires gpkg_layer (e.g., 'ADM_ADM1')")
        prov = gpd.read_file(china_level1_path, layer=gpkg_layer)
    else:
        prov = gpd.read_file(china_level1_path)

    
    if prov.crs is None or prov.crs.to_epsg() != 4326:
        prov = prov.to_crs("EPSG:4326")

    
    boundary = prov.dissolve()

    
    x0, x1 = xlim
    y0, y1 = ylim
    
    xs = np.arange(x0, x1, grid_res)
    ys = np.arange(y0, y1, grid_res)
    nx, ny = len(xs), len(ys)

    
    ix = np.clip(((lon - x0) / grid_res).astype(int), 0, max(nx - 1, 1) - 1)
    iy = np.clip(((lat - y0) / grid_res).astype(int), 0, max(ny - 1, 1) - 1)
    key = iy * nx + ix

    df = pd.DataFrame({"key": key, "zone": zones_raw, "pred": preds})

    if agg == "mode":
        agg_zone_s = df.groupby("key")["zone"].agg(_mode_int)
    elif agg == "mean":
        mean_pred = df.groupby("key")["pred"].mean()
        agg_zone = np.digitize(mean_pred.values, bins) - 1
        agg_zone_s = pd.Series(agg_zone, index=mean_pred.index)
    else:
        raise ValueError("agg only supports 'mode' or 'mean'")

    
    gcells, zvals = [], []
    for k, z in tqdm(agg_zone_s.dropna().items(), desc="Generating grid polygons", total=len(agg_zone_s.dropna())):
        iy = k // nx
        ix = k % nx
        
        if ix + 1 >= nx or iy + 1 >= ny:
            continue
        poly = Polygon([
            (xs[ix],   ys[iy]),
            (xs[ix+1], ys[iy]),
            (xs[ix+1], ys[iy+1]),
            (xs[ix],   ys[iy+1]),
        ])
        gcells.append(poly)
        zvals.append(int(z))

    grid_gdf = gpd.GeoDataFrame({"zone": zvals}, geometry=gcells, crs="EPSG:4326")

    
    try:
        grid_gdf = gpd.overlay(grid_gdf, boundary[["geometry"]], how="intersection", keep_geom_type=True)
    except Exception:
        
        grid_gdf = gpd.sjoin(grid_gdf, boundary[["geometry"]], predicate="intersects", how="inner")
        grid_gdf = grid_gdf.drop(columns=[c for c in grid_gdf.columns if c.startswith("index_")])

    
    if cmap_colors is None:
        cmap_colors = ["#D9B300", "#5AA1A4", "#C56B5B", "#3F6BAA", "#6FAE5B"][:K]
    cmap = ListedColormap(cmap_colors, name="zones")
    norm = BoundaryNorm(np.arange(-0.5, K + 0.5, 1), cmap.N)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0.06, 0.06, 0.77, 0.88])

    
    boundary.plot(ax=ax, color="#F7F7F7", edgecolor="none")
    grid_gdf.plot(ax=ax, column="zone", cmap=cmap, norm=norm, linewidth=0, alpha=0.95)
    prov.boundary.plot(ax=ax, color="white", linewidth=0.6)
    prov.boundary.plot(ax=ax, color="#333333", linewidth=0.25, alpha=0.7)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)

    
    cax = fig.add_axes([0.86, 0.15, 0.02, 0.7])
    cb = plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        cax=cax,
        ticks=np.arange(0, K, 1),
    )
    cb.ax.set_yticklabels([f"Zone{lab}" for lab in labels])

    
    if add_insets:
        for zone_name, cfg in add_insets.items():
            x0i, y0i, wi, hi, xg, y_hist, y_s1, y_s5 = cfg
            axins = fig.add_axes([x0i, y0i, wi, hi])
            axins.plot(xg, y_hist, label="Historical (1995–2014)")
            axins.plot(xg, y_s1,  label="SSP1–2.6 (2081–2100)")
            axins.plot(xg, y_s5,  label="SSP5–8.5 (2081–2100)")
            axins.set_title(zone_name, fontsize=9, pad=2)
            axins.set_xlabel("Growing days", fontsize=8)
            axins.set_ylabel("GDD", fontsize=8)
            axins.tick_params(labelsize=8)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return grid_gdf

