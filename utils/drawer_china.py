import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import KDTree
from tqdm import tqdm
from shapely import contains
import shapely
import pandas as pd
import os


try:
    from pykrige.ok import OrdinaryKriging
    _HAS_PYKRIGE = True
except ImportError:
    _HAS_PYKRIGE = False


def draw_suitability_interpolation(
    lon, lat, values,
    china_level1_path, gpkg_layer=None,
    method="idw",             
    xlim=(73,135), ylim=(15,54),
    grid_res=0.1,             
    bins=None,                
    cmap_colors=None,         
    figsize=(10,8), dpi=600,
    title="Suitability zones (interpolated)",
    show_uncertainty=False,   
    uncertainty_mode="variance",  
    unc_figsize=(9,7),
    unc_cmap="magma",         
    save_path=None,           
    save_path_unc=None,       
    show=True
):
    

    
    print("Reading provincial and national boundaries")
    if china_level1_path.lower().endswith(".gpkg"):
        prov = gpd.read_file(china_level1_path, layer=gpkg_layer)
    else:
        prov = gpd.read_file(china_level1_path)
    if prov.crs is None or prov.crs.to_epsg() != 4326:
        prov = prov.to_crs(4326)

    
    
    

    
    
    


    boundary = prov.dissolve()
    china_poly = boundary.geometry.unary_union

    
    print("Projecting to metric CRS for distance and semivariogram calculations")
    prov_m = prov.to_crs(3857)
    boundary_m = prov_m.dissolve()
    china_poly_m = boundary_m.geometry.unary_union

    
    print("Preparing point data")
    lon = np.asarray(lon).ravel()
    lat = np.asarray(lat).ravel()
    val = np.asarray(values).ravel()
    m = (~np.isnan(lon)) & (~np.isnan(lat)) & (~np.isnan(val))
    lon, lat, val = lon[m], lat[m], val[m]
    gpt = gpd.GeoDataFrame({"v": val}, geometry=gpd.points_from_xy(lon, lat), crs=4326).to_crs(3857)

    
    print("Building interpolation grid (metric coordinates)")
    x0, y0, x1, y1 = prov_m.total_bounds
    
    nx = max(int((x1 - x0) / (grid_res * 111e3)), 50)
    ny = max(int((y1 - y0) / (grid_res * 111e3)), 50)
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    xx_m, yy_m = np.meshgrid(xs, ys)

    
    print("Interpolating")
    if method == "idw":
        xy = np.c_[gpt.geometry.x, gpt.geometry.y]
        tree = KDTree(xy)
        dist, idx = tree.query(np.c_[xx_m.ravel(), yy_m.ravel()], k=12)
        w = 1.0 / np.maximum(dist, 1e-6)**2
        zi = np.sum(w * val[idx], axis=1) / np.sum(w, axis=1)
        zi = zi.reshape(xx_m.shape)
        ss = None  

    elif method == "kriging":
        if not _HAS_PYKRIGE:
            raise ImportError("Install pykrige via pip before using kriging")
        OK = OrdinaryKriging(
            gpt.geometry.x, gpt.geometry.y, val,
            
            variogram_model="spherical",  
                                
                                
            verbose=False, enable_plotting=False
        )
        zi, ss = OK.execute("grid", xs, ys)   
        zi = np.array(zi)
        ss = np.array(ss)

    else:
        raise ValueError("method must be 'idw' or 'kriging'")

    
    print("Masking regions outside the target country")
    
    
    
    
    
    mask_cache = f"/data/home/wxl22/changzu_ans/mask_cache_{grid_res}.npz"

    if mask_cache and os.path.exists(mask_cache):
        data = np.load(mask_cache)
        mask = data["mask"]
        xx_m, yy_m = data["xx"], data["yy"]
        
        pts_m = shapely.points(xx_m.ravel(), yy_m.ravel())
    else:
        
        pts_m = shapely.points(xx_m.ravel(), yy_m.ravel())
        mask = shapely.contains(china_poly_m, pts_m).reshape(xx_m.shape)
        if mask_cache:
            np.savez(mask_cache, xx=xx_m, yy=yy_m, mask=mask,pts=pts_m)


    zi[~mask] = np.nan
    if method == "kriging":
        ss[~mask] = np.nan



    
    print("Reprojecting to WGS84 for plotting")
    grid_ll = gpd.GeoDataFrame(geometry=pts_m, crs=3857).to_crs(4326)
    xx = grid_ll.geometry.x.to_numpy().reshape(xx_m.shape)
    yy = grid_ll.geometry.y.to_numpy().reshape(yy_m.shape)

    
    print("Colorizing bins for the main map")
    if bins is None:
        vmin, vmax = np.nanmin(val), np.nanmax(val)
        bins = np.linspace(vmin, vmax, 6)
        
    K = len(bins) - 1
    if cmap_colors is None:
        
        cmap_colors = ["#FFFFFF", "#FFFFFF", "#C56B5B", "#3F6BAA", "#6FAE5B"][:K]

    cmap = ListedColormap(cmap_colors, name="zones")
    norm = BoundaryNorm(bins, ncolors=cmap.N)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    pcm = ax.pcolormesh(xx, yy, zi, cmap=cmap, norm=norm, shading="auto")
    prov.boundary.plot(ax=ax, color="white", linewidth=0.6)
    prov.boundary.plot(ax=ax, color="#333333", linewidth=0.3, alpha=0.85)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(title)
    print("Configuring colorbar")
    cbar = plt.colorbar(pcm, ax=ax, ticks=[(bins[i]+bins[i+1])/2 for i in tqdm(range(K),desc="Configuring colorbar", total=K)])
    print("Configuring colorbar")
    cbar.ax.set_yticklabels([f"Zone{i}" for i in tqdm(range(K),desc="Configuring colorbar", total=K)])

    
    print("Adding nine-dash line overlay")
    if os.path.exists("/data/home/wxl22/changzu_ans/utils/nine_line/china_nine_dotted_line.shp"):
        dashes = gpd.read_file("/data/home/wxl22/changzu_ans/utils/nine_line/china_nine_dotted_line.shp")
        if dashes.crs.to_epsg() != 4326:
            dashes = dashes.to_crs(4326)
        dashes.plot(ax=ax, color="gray", linestyle="--", linewidth=0.8)


    add_south_china_sea_inset_raster(
        ax,
        prov_gdf=prov,
        xx=xx, yy=yy, zi=zi,     
        cmap=cmap, norm=norm,
        bbox=(105,125,3,25),
        loc="lower right", size=0.25, borderpad=0.1,
        draw_bbox_on_main=False, label=""
    )
    print("All steps finished; preparing to save figure")




    



    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    
    print("Optional uncertainty map: available only for kriging")
    if show_uncertainty and method == "kriging":
        if uncertainty_mode == "std":
            unc = np.sqrt(ss)
            unc_title = "Kriging Standard Deviation"
            unc_barlabel = "Std"
        else:
            unc = ss
            unc_title = "Kriging Variance"
            unc_barlabel = "Variance"

        fig2, ax2 = plt.subplots(figsize=unc_figsize, dpi=dpi)
        im = ax2.pcolormesh(xx, yy, unc, cmap=unc_cmap, shading="auto")
        prov.boundary.plot(ax=ax2, color="white", linewidth=0.6)
        prov.boundary.plot(ax=ax2, color="#333333", linewidth=0.3, alpha=0.85)
        ax2.set_xlim(xlim); ax2.set_ylim(ylim)
        ax2.set_xlabel("Longitude"); ax2.set_ylabel("Latitude")
        ax2.set_title(unc_title)
        cb2 = plt.colorbar(im, ax=ax2)
        cb2.set_label(unc_barlabel)

        if save_path_unc:
            plt.savefig(save_path_unc, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig2)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"zi": zi, "xx": xx, "yy": yy, "ss": (ss if method == "kriging" else None)}


from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import geopandas as gpd
from shapely.geometry import box
import numpy as np

def add_south_china_sea_inset_raster(
    ax, prov_gdf, xx, yy, zi, cmap, norm,
    bbox=(105,125,3,25), loc="lower right", size=0.05, borderpad=0.1,
    draw_bbox_on_main=False, label=""
):
    xmin, xmax, ymin, ymax = bbox

    
    if draw_bbox_on_main:
        ylo, yhi = ax.get_ylim()
        y0, y1 = max(ymin, ylo), min(ymax, yhi)
        if y0 < y1:
            ax.plot([xmin,xmax,xmax,xmin,xmin],[y0,y0,y1,y1,y0],
                    ls="--", lw=0.5, color="gray", alpha=0.8)

    
    
    

    ax_in = inset_axes(
        ax,
        width="125%", height="125%",   
        bbox_to_anchor=(0.83, 0.02, 0.20, 0.20),  
        bbox_transform=ax.transAxes, 
        loc="lower right",           
        borderpad=0
    )

    
    j = np.where((xx[0] >= xmin) & (xx[0] <= xmax))[0]
    i = np.where((yy[:,0] >= ymin) & (yy[:,0] <= ymax))[0]
    if i.size>1 and j.size>1:
        ax_in.pcolormesh(xx[np.ix_(i,j)], yy[np.ix_(i,j)], zi[np.ix_(i,j)],
                        cmap=cmap, norm=norm, shading="auto")

    
    roi = gpd.GeoDataFrame(geometry=[box(xmin,ymin,xmax,ymax)], crs=4326)
    try:
        prov_clip = gpd.overlay(prov_gdf, roi, how="intersection")
    except Exception:
        prov_clip = gpd.clip(prov_gdf, roi)
    prov_clip.boundary.plot(ax=ax_in, color="black", linewidth=0.6, alpha=0.95)

    
    if os.path.exists("/data/home/wxl22/changzu_ans/utils/nine_line/china_nine_dotted_line.shp"):
        dashes = gpd.read_file("/data/home/wxl22/changzu_ans/utils/nine_line/china_nine_dotted_line.shp")
        if dashes.crs.to_epsg() != 4326:
            dashes = dashes.to_crs(4326)
        dashes_clip = gpd.clip(dashes, roi)
        dashes_clip.plot(ax=ax_in, color="black", linestyle="--", linewidth=0.5)

    
    ax_in.set_xlim(xmin,xmax); ax_in.set_ylim(ymin,ymax)
    ax_in.set_xticks([]); ax_in.set_yticks([])
    for s in ax_in.spines.values():
        s.set_visible(True); s.set_edgecolor("gray"); s.set_linewidth(0.5)
    ax_in.text(0.5, 1.02, label, ha="center", va="bottom",
               transform=ax_in.transAxes, fontsize=9)

    return ax_in
