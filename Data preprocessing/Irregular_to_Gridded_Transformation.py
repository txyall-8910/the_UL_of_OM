"""
Description: Spatial Transformation from Irregular Units to a Gridded Format
This script standardizes spatial mobility data by transforming Origin-Destination (OD) flows
from irregular administrative or census units into a uniform regular grid (e.g., 500m x 500m).

This standardization is crucial for cross-city comparisons, ensuring consistent spatial resolution
for subsequent scaling analyses.

Detailed methodological justifications for this area-weighted interpolation approach are
provided in the Materials and Methods section of the main text.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box

# =============================================================================
# 1. PARAMETER INITIALIZATION
# =============================================================================
# Define file paths and spatial parameters
BLOCKS_SHP = "data/Dakar_sites.shp"         # Path to the polygon shapefile of irregular census/administrative blocks
BLOCK_ID = "site_id"                        # Unique identifier field within the block shapefile
OD_CSV = "data/commuting_flows.csv"         # Path to the empirical OD flow matrix
GRID_SIZE = 500                             # Target spatial resolution (grid cell size in meters)
OUTPUT_OD_GRID = "results/DK_od_grid_area_weighted.csv" # Output path for the standardized gridded OD flows
OUTPUT_GRID = "results/DK_grid_500m.shp"    # Output path for the generated grid vector file

# --- Coordinate Reference System (CRS) Configuration ---
# A projected CRS is strictly required to ensure accurate metric area calculations
# for the subsequent areal interpolation process.
GRID_CRS_EPSG = 31028
GRID_CRS_NAME = "Yoff / UTM zone 28N"

# =============================================================================
# 2. SPATIAL DATA INGESTION AND CRS STANDARDIZATION
# =============================================================================
print("1. Loading irregular spatial units (census blocks)...")
blocks = gpd.read_file(BLOCKS_SHP)

# Ensure the spatial data is projected into the correct metric CRS.
# Geographic coordinate systems (e.g., WGS84) will distort area calculations.
if blocks.crs is None:
    print("Warning: Input shapefile lacks a defined CRS.")
    print(f"Assigning default projected CRS: {GRID_CRS_NAME} (EPSG:{GRID_CRS_EPSG})...")
    blocks = blocks.set_crs(epsg=GRID_CRS_EPSG)
elif blocks.crs.is_geographic:
    print(f"Input CRS is geographic ({blocks.crs}). Reprojecting to {GRID_CRS_NAME} (EPSG:{GRID_CRS_EPSG}) for metric accuracy...")
    blocks = blocks.to_crs(epsg=GRID_CRS_EPSG)
else:
    print(f"Input CRS is projected: {blocks.crs}")
    if blocks.crs.to_epsg() != GRID_CRS_EPSG:
        print(f"Reprojecting to target CRS {GRID_CRS_NAME} (EPSG:{GRID_CRS_EPSG}) to ensure spatial consistency...")
        blocks = blocks.to_crs(epsg=GRID_CRS_EPSG)

# Standardize the unique identifier format to prevent merging artifacts
blocks[BLOCK_ID] = blocks[BLOCK_ID].astype(str).str.strip()

# =============================================================================
# 3. STANDARDIZED SPATIAL TESSELLATION (GRID GENERATION)
# =============================================================================
print("2. Generating standardized regular spatial grid...")
# Extract the bounding box of the study area to define the grid extent
minx, miny, maxx, maxy = blocks.total_bounds
x_coords = np.arange(minx, maxx + GRID_SIZE, GRID_SIZE)
y_coords = np.arange(miny, maxy + GRID_SIZE, GRID_SIZE)

# Construct the geometric grid cells
grid_cells = []
for x in x_coords[:-1]:
    for y in y_coords[:-1]:
        cell = box(x, y, x + GRID_SIZE, y + GRID_SIZE)
        grid_cells.append(cell)

# Convert to a GeoDataFrame and assign the standardized CRS
grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=blocks.crs)
grid['grid_id'] = grid.index
grid = grid.to_crs(epsg=GRID_CRS_EPSG)  # Enforce projected CRS

# =============================================================================
# 4. SPATIAL FILTERING (TOPOLOGICAL OPTIMIZATION)
# =============================================================================
print("3. Filtering grid cells via topological intersection...")
# Utilize a spatial index to accelerate intersection queries
blocks_sindex = blocks.sindex

def intersects_any_block(cell):
    """Evaluates whether a given grid cell intersects with any empirical block geometry."""
    possible_matches_index = list(blocks_sindex.intersection(cell.bounds))
    possible_matches = blocks.iloc[possible_matches_index]
    return possible_matches.intersects(cell).any()

# Retain only grid cells that overlap with the study area to optimize computational efficiency
intersect_mask = grid['geometry'].apply(intersects_any_block)
grid_filtered = grid[intersect_mask].copy().reset_index(drop=True)
grid_filtered['grid_id'] = grid_filtered.index  # Re-index filtered grid IDs

# Export the standardized grid for visualization and downstream spatial analysis
grid_filtered.to_file(OUTPUT_GRID)
print(f"   -> Standardized grid saved to {OUTPUT_GRID}")

# =============================================================================
# 5. AREAL INTERPOLATION WEIGHTS CALCULATION
# =============================================================================
print("4. Computing geometric intersections and areal interpolation weights...")
# Perform a geometric intersection between the irregular blocks and the regular grid
block_grid = gpd.overlay(
    blocks[[BLOCK_ID, 'geometry']],
    grid_filtered[['grid_id', 'geometry']],
    how='intersection'
)
# Calculate the absolute area of each intersection fragment
block_grid['area_in_grid'] = block_grid.geometry.area

# Calculate the interpolation weight (area fraction) for each fragment.
block_areas = block_grid.groupby(BLOCK_ID)['area_in_grid'].sum().rename('block_total_area').reset_index()
block_grid = block_grid.merge(block_areas, on=BLOCK_ID)
block_grid['area_fraction'] = block_grid['area_in_grid'] / block_grid['block_total_area']
block_grid = block_grid[[BLOCK_ID, 'grid_id', 'area_fraction']]

# =============================================================================
# 6. EMPIRICAL OD FLOW INTEGRATION AND VALIDATION
# =============================================================================
print("5. Ingesting and validating empirical OD flow data...")
od = pd.read_csv(OD_CSV, dtype={'home': str, 'work': str})
od['home'] = od['home'].str.strip()
od['work'] = od['work'].str.strip()
block_grid[BLOCK_ID] = block_grid[BLOCK_ID].astype(str).str.strip()

# Validate topological consistency between the flow network and spatial geometries
block_ids_in_data = set(block_grid[BLOCK_ID].unique())
home_missing = set(od['home']) - block_ids_in_data
work_missing = set(od['work']) - block_ids_in_data

if home_missing:
    print(f"WARNING: {len(home_missing)} 'home' (origin) IDs in OD data lack corresponding spatial geometries. Example: {list(home_missing)[:5]}")
if work_missing:
    print(f"WARNING: {len(work_missing)} 'work' (destination) IDs in OD data lack corresponding spatial geometries. Example: {list(work_missing)[:5]}")

# Purge topologically orphaned flows to maintain network integrity
od = od[od['home'].isin(block_ids_in_data) & od['work'].isin(block_ids_in_data)].copy()
print(f"   -> Validated OD matrix: {len(od)} flow records retained for transformation.")

# =============================================================================
# 7. FLOW REDISTRIBUTION (BIVARIATE AREAL INTERPOLATION)
# =============================================================================
print("6. Redistributing flows to the standardized grid via areal weighting...")

# Map origin (home) locations to the grid and append origin weights
od_home = od.merge(block_grid, left_on='home', right_on=BLOCK_ID, suffixes=('', '_home'))
od_home = od_home.rename(columns={'grid_id': 'home_grid', 'area_fraction': 'home_fraction'})
od_home = od_home.drop(columns=[BLOCK_ID])

# Map destination (work) locations to the grid and append destination weights
od_home_work = od_home.merge(block_grid, left_on='work', right_on=BLOCK_ID, suffixes=('', '_work'))
od_home_work = od_home_work.rename(columns={'grid_id': 'work_grid', 'area_fraction': 'work_fraction'})
od_home_work = od_home_work.drop(columns=[BLOCK_ID])

# Compute the redistributed flow volume.
od_home_work['weighted_count'] = od_home_work['count'] * od_home_work['home_fraction'] * od_home_work['work_fraction']

# =============================================================================
# 8. MACROSCOPIC AGGREGATION AND EXPORT
# =============================================================================
print("7. Aggregating redistributed flows into the final gridded OD matrix...")
# Aggregate all fractional flows sharing the same origin-destination grid cell pair
od_grid = (
    od_home_work.groupby(['home_grid', 'work_grid'])['weighted_count']
    .sum()
    .reset_index()
)

# Standardize nomenclature for the output matrix
od_grid = od_grid.rename(columns={'weighted_count': 'count'})

print(f"8. Exporting the standardized area-weighted gridded OD flows to {OUTPUT_OD_GRID}")
od_grid.to_csv(OUTPUT_OD_GRID, index=False)
print("SUCCESS: Spatial transformation complete.")