import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point
import numpy as np
import os

def _get_polygons(gpkg_path, layer):
    if not os.path.exists(gpkg_path): return gpd.GeoDataFrame()
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer)
        polys = []
        for geom in gdf.geometry:
            if isinstance(geom, LineString) and len(geom.coords) >= 3:
                polys.append(Polygon(geom))
            elif isinstance(geom, Polygon):
                polys.append(geom)
        if not polys:
            return gpd.GeoDataFrame(crs=gdf.crs)
        return gpd.GeoDataFrame(geometry=polys, crs=gdf.crs)
    except Exception:
        return gpd.GeoDataFrame()

POLYGON_CACHE = {}

def get_mine_polys(mine_id, base_dir):
    if mine_id in POLYGON_CACHE:
        return POLYGON_CACHE[mine_id]
        
    mine_file = mine_id
    if not "_" in mine_id:
        mine_file = mine_id.replace("mine", "mine_")
        
    path = os.path.join(base_dir, f"{mine_file}_anonymized.gpkg")
    loading = _get_polygons(path, "bench")
    dump_ob = _get_polygons(path, "ob_dump")
    dump_min = _get_polygons(path, "mineral_stock")
    
    dumps = []
    if not dump_ob.empty: dumps.append(dump_ob)
    if not dump_min.empty: dumps.append(dump_min)
    if dumps:
        dumping = pd.concat(dumps).reset_index(drop=True)
    else:
        dumping = gpd.GeoDataFrame()
        
    POLYGON_CACHE[mine_id] = (loading, dumping)
    return loading, dumping

def extract_spatial_features(group_df, loading_gdf, dumping_gdf):
    if "speed" not in group_df.columns or "latitude" not in group_df.columns:
        return pd.Series({"haul_cycles": 0, "avg_load_time": 0, "avg_dump_time": 0, "avg_cycle_dist": 0})
        
    speed = group_df["speed"].values
    ts = group_df["ts"].values
    lat = group_df["latitude"].values
    lon = group_df["longitude"].values
    cumdist = group_df["cumdist"].values if "cumdist" in group_df.columns else np.zeros(len(speed))
    
    stops = []
    in_stop = False
    start_idx = 0
    for i in range(len(speed)):
        if speed[i] == 0:
            if not in_stop:
                in_stop = True
                start_idx = i
        else:
            if in_stop:
                dur = (ts[i] - ts[start_idx]) / np.timedelta64(1, 's')
                if dur >= 60:
                    stops.append({
                        "duration": dur,
                        "lon": lon[start_idx],
                        "lat": lat[start_idx],
                        "cumdist_start": cumdist[start_idx],
                        "cumdist_end": cumdist[i]
                    })
                in_stop = False
                
    if not stops:
        return pd.Series({"haul_cycles": 0, "avg_load_time": 0, "avg_dump_time": 0, "avg_cycle_dist": 0})
        
    stops_df = pd.DataFrame(stops)
    pts = [Point(x, y) for x, y in zip(stops_df["lon"], stops_df["lat"])]
    gdf_stops = gpd.GeoDataFrame(stops_df, geometry=pts, crs="EPSG:4326")
    gdf_stops = gdf_stops.to_crs("EPSG:32645")
    
    # Spatial join
    stops_df["type"] = "unknown"
    
    if not loading_gdf.empty:
        joined_load = gpd.sjoin(gdf_stops, loading_gdf, how="inner", predicate="intersects")
        if not joined_load.empty:
            stops_df.loc[joined_load.index, "type"] = "load"
        
    if not dumping_gdf.empty:
        joined_dump = gpd.sjoin(gdf_stops, dumping_gdf, how="inner", predicate="intersects")
        if not joined_dump.empty:
            stops_df.loc[joined_dump.index, "type"] = "dump"
            
    # Heuristic for unknown stops: if dur > 200s and not near load, likely a dump or bottleneck
    # New: if it's far from everything but has long duration, we consider it a potential "drifted" dump
    mask_unknown = stops_df["type"] == "unknown"
    stops_df.loc[mask_unknown & (stops_df["duration"] >= 180), "type"] = "potential_dump"
        
    haul_cycles = 0
    load_times = []
    dump_times = []
    cycle_dists = []
    
    last_load_row = None
    for i, row in stops_df.iterrows():
        if row["type"] == "load":
            last_load_row = row
            load_times.append(row["duration"])
        elif row["type"] in ["dump", "potential_dump"]:
            dump_times.append(row["duration"])
            if last_load_row is not None:
                dist = row["cumdist_start"] - last_load_row["cumdist_end"]
                if dist > 200: # reasonable minimum haul distance
                    cycle_dists.append(dist)
                    haul_cycles += 1
                    last_load_row = None # Reset after a cycle
                
    return pd.Series({
        "haul_cycles": haul_cycles,
        "avg_load_time": np.mean(load_times) if load_times else 0,
        "avg_dump_time": np.mean(dump_times) if dump_times else 0,
        "avg_cycle_dist": np.mean(cycle_dists) if cycle_dists else 0
    })
