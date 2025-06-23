
## --------------------------------------------------------------
# GLB TO 3DM CONVERTER
# --------------------------------------------------------------

import pygltflib
from pygltflib.validator import validate, summary
import rhino3dm
import os
import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon, LineString
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import json
from supabase import create_client, Client
import requests
import tempfile
import plotly.graph_objects as go

# Fixed scale factor for all GLB files
scale_factor = 1
print(f"Using fixed scale factor: {scale_factor}")
unityAxisFormat = False


def convert_glb_to_3dm(glb_data):
    """
    Loads GLB data from memory, extracts its mesh data, and returns a 3dm model object.
    Only mesh geometry will be converted. Materials, animations, etc., are not translated.
    """
    global unityAxisFormat

    print(f"Loading GLB data from memory")
    
    # Write GLB data to a temporary file for pygltflib to read
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as temp_file:
        temp_file.write(glb_data)
        temp_glb_path = temp_file.name
    
    try:
        gltf = pygltflib.GLTF2().load(temp_glb_path)
        validate(gltf)  # will throw an error depending on the problem
        summary(gltf)  # will pretty print human readable summary of errors
    finally:
        # Clean up the temporary file
        os.unlink(temp_glb_path)

    model_3dm = rhino3dm.File3dm()

    # Iterate through GLTF meshes and add them to the 3dm model
    for mesh_idx, gltf_mesh in enumerate(gltf.meshes):
        print(f"Processing mesh: {gltf_mesh.name if gltf_mesh.name else f'Mesh_{mesh_idx}'}")

        for primitive in gltf_mesh.primitives:
            # Get vertex positions
            accessor_pos = gltf.accessors[primitive.attributes.POSITION]
            buffer_view_pos = gltf.bufferViews[accessor_pos.bufferView]
            buffer_pos = gltf.buffers[buffer_view_pos.buffer]

            # Extract vertices
            vertices_bytes = gltf.get_data_from_buffer_uri(buffer_pos.uri)[
                buffer_view_pos.byteOffset : buffer_view_pos.byteOffset + buffer_view_pos.byteLength
            ]
            vertices = np.frombuffer(vertices_bytes, dtype=np.float32).reshape(-1, 3)

            print(f"Using fixed scale factor: {scale_factor}")

            rhino_mesh = rhino3dm.Mesh()

            for v in vertices:
                if not unityAxisFormat:
                    # Unity's Y-up format: (x, y, z) -> (x, z, y)
                    rhino_mesh.Vertices.Add(v[0] * scale_factor, v[1] * scale_factor, v[2] * scale_factor)
                else:
                    rhino_mesh.Vertices.Add(v[0] * scale_factor, v[2] * scale_factor, v[1] * scale_factor)

            # Get indices (faces)
            if primitive.indices is not None:
                accessor_indices = gltf.accessors[primitive.indices]
                buffer_view_indices = gltf.bufferViews[accessor_indices.bufferView]
                buffer_indices = gltf.buffers[buffer_view_indices.buffer]

                indices_bytes = gltf.get_data_from_buffer_uri(buffer_indices.uri)[
                    buffer_view_indices.byteOffset : buffer_view_indices.byteOffset + buffer_view_indices.byteLength
                ]

                # Determine dtype for indices (UINT8, UINT16, UINT32)
                if accessor_indices.componentType == pygltflib.UNSIGNED_BYTE:
                    indices_dtype = np.uint8
                elif accessor_indices.componentType == pygltflib.UNSIGNED_SHORT:
                    indices_dtype = np.uint16
                elif accessor_indices.componentType == pygltflib.UNSIGNED_INT:
                    indices_dtype = np.uint32
                else:
                    print(f"Warning: Unsupported index component type: {accessor_indices.componentType}. Skipping faces for this primitive.")
                    continue

                indices = np.frombuffer(indices_bytes, dtype=indices_dtype)

                # glTF uses flat arrays for indices, assuming triangles (mode 4)
                if primitive.mode == pygltflib.TRIANGLES:
                    for i in range(0, len(indices), 3):
                        rhino_mesh.Faces.AddFace(int(indices[i]), int(indices[i+1]), int(indices[i+2]))
                else:
                    print(f"Warning: Skipping primitive with unsupported mode: {primitive.mode}. Only triangles (mode 4) are fully supported for faces.")
                    continue
            else:
                print(f"Warning: Primitive has no indices. Assuming sequential triangles, but this might not be correct for complex GLBs.")
                for i in range(0, len(vertices) - 2, 3):
                    rhino_mesh.Faces.AddFace(i, i+1, i+2)

            # Optional: Calculate normals
            rhino_mesh.Normals.ComputeNormals()
            rhino_mesh.Compact()

            model_3dm.Objects.AddMesh(rhino_mesh)

    print("Conversion complete!")
    print("Sample vertices:", vertices[:5])
    
    return model_3dm


# --- Define paths ---
# Prompt the user for the GLB file path

# Initialize Supabase client
SUPABASE_URL = "https://apdbfbjnlsxjfubqahtl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFwZGJmYmpubHN4amZ1YnFhaHRsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NzQ3NjUzOCwiZXhwIjoyMDYzMDUyNTM4fQ.cylQZjLEmtBi507wrJ1KUDyIXTz5H5VAXGj5eKEqDy4"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Storage parameters
bucket_name = "models"
folder_path = "79edaed4-a719-4390-a485-519b68fa68ea"

# List all files in the folder
files = supabase.storage.from_(bucket_name).list(folder_path)
if not files:
    raise RuntimeError(f"No files found in folder '{folder_path}'")

# Pick the latest file by created_at (or updated_at) timestamp
latest_file = sorted(
    files,
    key=lambda f: f.get("created_at", f.get("updated_at", "")),
    reverse=True
)[0]["name"]

# Generate a signed URL for the latest file
storage_path = f"{folder_path}/{latest_file}"
signed = supabase.storage.from_(bucket_name).create_signed_url(storage_path, 60)
download_url = signed.get("signedURL")

# Get GLB data directly into memory
resp = requests.get(download_url)
resp.raise_for_status()
glb_data = resp.content

print(f"✅ Loaded latest GLB '{latest_file}' into memory from Supabase")

# Set Unity Y-up format to False by default (no user prompt)
unityAxisFormat = False 

# Extract filename for later use
file_name_without_ext = os.path.splitext(latest_file)[0]

# --- Run the conversion ---
try:
    model = convert_glb_to_3dm(glb_data)
    print(f"\nConversion successful! The 3DM model is ready in memory.")

except Exception as e:
    print(f"An error occurred during conversion: {e}")



# --------------------------------------------------------------
# MASHALLA FOR FORCED COLUMNS - Now uses the in-memory 3DM model
# --------------------------------------------------------------

# Use the in-memory model directly
print(f"Using in-memory 3DM model for structural analysis")

# Extract geometries and their bounding boxes
building_floor_footprints = []
all_mesh_bboxes = []
roof_meshes_info = []
max_z = 0.0

Z_FLATNESS_TOLERANCE = 3

for obj in model.Objects:
    geom = obj.Geometry
    if geom.ObjectType == rhino3dm.ObjectType.Mesh:
        bbox = geom.GetBoundingBox()
        print(bbox.Min, bbox.Max)
        
        bbox_x_dim = bbox.Max.X - bbox.Min.X
        bbox_y_dim = bbox.Max.Y - bbox.Min.Y
        bbox_z_dim = bbox.Max.Z - bbox.Min.Z
        print (bbox_x_dim, bbox_y_dim, bbox_z_dim)

        if bbox_z_dim < Z_FLATNESS_TOLERANCE and bbox_x_dim > 0.1 and bbox_y_dim > 0.1:
            base_pts = [
                [bbox.Min.X, bbox.Min.Y],
                [bbox.Max.X, bbox.Min.Y],
                [bbox.Max.X, bbox.Max.Y],
                [bbox.Min.X, bbox.Max.Y],
                [bbox.Min.X, bbox.Min.Y]
            ]
            poly = Polygon(base_pts)
            if poly.area > 1e-3:
                building_floor_footprints.append(poly)
                roof_meshes_info.append((obj.Attributes.Id, bbox, poly))

        all_mesh_bboxes.append(bbox)
        
        max_z = max(max_z, bbox.Max.Z)

if not building_floor_footprints:
    raise RuntimeError("No meaningful building floor footprints (meshes flat in Z with area) found in the model.")

print(f"Detected {len(building_floor_footprints)} building floor footprints. Max Z height: {max_z:.2f}m")
print(f"Total meshes contributing to height calculation: {len(all_mesh_bboxes)}")
print(f"Detected {len(roof_meshes_info)} potential roof meshes for comparison.")


# ===== AUTOMATIC FLOOR CALCULATION =====
floor_height = 2.5  # meters per floor
num_floors = max(1, int(round(max_z / floor_height)))
print(f"Automatically calculated number of floors: {num_floors} (based on max height {max_z:.2f}m and {floor_height}m per floor)")

# # --- Then continues with wall thickness input ---
# while True:
#     try:
#         wall_thickness = float(input("Enter desired wall thickness for the perimeter (e.g., 0.3): "))
#         if wall_thickness <= 0:
#             raise ValueError
#         break
#     except ValueError:
#         print("Please enter a valid positive number for wall thickness.")


# Find and print roofs that are peaks (taller than all directly touching roofs)
print("\n--- Analyzing Roof Heights ---")

INTERSECTION_BUFFER_ROOF = 0.1

dominant_roofs_identified = []

for i, (roof1_id, roof1_bbox, roof1_poly_2d) in enumerate(roof_meshes_info):
    roof1_max_z = roof1_bbox.Max.Z
    is_dominant_roof = True
    touching_lower_neighbors = []
    touching_equal_or_higher_neighbors = []

    for j, (roof2_id, roof2_bbox, roof2_poly_2d) in enumerate(roof_meshes_info):
        if i == j:
            continue

        roof2_max_z = roof2_bbox.Max.Z

        intersection_geometry = roof1_poly_2d.buffer(INTERSECTION_BUFFER_ROOF).intersection(roof2_poly_2d.buffer(INTERSECTION_BUFFER_ROOF))
        
        is_touching = not intersection_geometry.is_empty and \
                      intersection_geometry.geom_type in ['LineString', 'MultiLineString', 'Polygon', 'MultiPolygon']
        
        if is_touching:
            if roof1_max_z <= roof2_max_z + 1e-6:
                is_dominant_roof = False
                touching_equal_or_higher_neighbors.append({"Id": str(roof2_id), "Height": round(roof2_max_z, 3)})
            else:
                touching_lower_neighbors.append({"Id": str(roof2_id), "Height": round(roof2_max_z, 3)})
        
    if is_dominant_roof and (touching_lower_neighbors or touching_equal_or_higher_neighbors):
        dominant_roofs_identified.append({
            "RhinoObjectId": str(roof1_id),
            "Height": round(roof1_max_z, 3),
            "Location_Min_X": round(roof1_bbox.Min.X, 3),
            "Location_Min_Y": round(roof1_bbox.Min.Y, 3),
            "Polygon": roof1_poly_2d,
            "TouchingLowerNeighbors": touching_lower_neighbors,
            "TouchingEqualOrHigherNeighbors": touching_equal_or_higher_neighbors
        })

if dominant_roofs_identified:
    print("\nRoofs identified as strictly taller than all their directly touching neighbors:")
    for roof_info in dominant_roofs_identified:
        if not roof_info['TouchingEqualOrHigherNeighbors']:
            print(f"   Roof ID: {roof_info['RhinoObjectId']} (Height: {roof_info['Height']}m)")
            print(f"     Location (Min XY): ({roof_info['Location_Min_X']}, {roof_info['Location_Min_Y']})")
            if roof_info['TouchingLowerNeighbors']:
                neighbor_details = ", ".join([f"ID: {n['Id']} (H: {n['Height']}m)" for n in roof_info['TouchingLowerNeighbors']])
                print(f"     Touching Lower Neighbors: {neighbor_details}")
            else:
                print(f"     No directly touching lower neighbors found (might be isolated or higher than implied).")
else:
    print("\nNo roofs found that are strictly taller than all their directly touching neighbors.")


# # Ask for number of floors
# while True:
#     try:
#         num_floors = int(input("How many floors does the building have? (e.g., 2 for ground + 1 middle + roof): "))
#         if num_floors < 1:
#             raise ValueError
#         break
#     except ValueError:
#         print("Please enter a valid positive integer for the number of floors.")

# --- Section for perimeter line and wall thickness ---

wall_thickness = 0.3  # meters (30 cm)
print(f"Using default wall thickness: {wall_thickness} m")


combined_building_polygon = MultiPolygon(building_floor_footprints)

try:
    exterior_perimeter = combined_building_polygon.buffer(wall_thickness, join_style=1)
except Exception as e:
    print(f"Could not buffer the building outline. Error: {e}")
    exterior_perimeter = None

if exterior_perimeter and exterior_perimeter.geom_type == 'MultiPolygon':
    exterior_perimeter = max(exterior_perimeter.geoms, key=lambda p: p.area)

perimeter_line_coords = []
if exterior_perimeter:
    if exterior_perimeter.geom_type == 'Polygon':
        perimeter_line_coords = list(exterior_perimeter.exterior.coords)
    elif exterior_perimeter.geom_type == 'MultiPolygon':
        perimeter_line_coords = list(exterior_perimeter.geoms[0].exterior.coords)
    else:
        print("Warning: The buffered perimeter is not a Polygon or MultiPolygon. Cannot extract line coordinates.")

# --- End perimeter section ---

detected_rooms = sorted([(poly, poly.area) for poly in building_floor_footprints], key=lambda x: -x[1])
if not detected_rooms:
    raise RuntimeError("No valid rooms detected after filtering by area. Check your Rhino model geometry.")

import rhino3dm
import os
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union # For combining polygons on a floor

# Assume floor_height and Z_FLATNESS_TOLERANCE are defined

# --- Step 1: Categorize Floor Footprints by Height/Floor Level ---
floor_footprints_by_level = {}
floor_z_levels = set() # To store the z-coordinates of each detected floor

for obj in model.Objects:
    geom = obj.Geometry
    if geom.ObjectType == rhino3dm.ObjectType.Mesh:
        bbox = geom.GetBoundingBox()
        
        bbox_x_dim = bbox.Max.X - bbox.Min.X
        bbox_y_dim = bbox.Max.Y - bbox.Min.Y
        bbox_z_dim = bbox.Max.Z - bbox.Min.Z

        if bbox_z_dim < Z_FLATNESS_TOLERANCE and bbox_x_dim > 0.1 and bbox_y_dim > 0.1:
            base_pts = [
                [bbox.Min.X, bbox.Min.Y],
                [bbox.Max.X, bbox.Min.Y],
                [bbox.Max.X, bbox.Max.Y],
                [bbox.Min.X, bbox.Max.Y],
                [bbox.Min.X, bbox.Min.Y]
            ]
            poly = Polygon(base_pts)
            if poly.area > 1e-3:
                # Estimate floor level based on min Z of the bbox
                # You might need to refine this to snap to discrete floor levels
                
                # A more robust way might be to cluster Z values
                # For simplicity, let's just use the rounded Z for grouping for now
                approx_z = round(bbox.Max.Z / floor_height) * floor_height # Snap to nearest floor_height multiple
                floor_z_levels.add(approx_z)
                
                if approx_z not in floor_footprints_by_level:
                    floor_footprints_by_level[approx_z] = []
                floor_footprints_by_level[approx_z].append(poly)

# Sort the Z levels to process floors in order
sorted_floor_z_levels = sorted(list(floor_z_levels))
print(f"Detected Z levels for floors: {sorted_floor_z_levels}")

# --- REVISED CANTILEVER DETECTION LOGIC ---
CANTILEVER_AREA_THRESHOLD = 0.5 # m^2, adjust as needed

print("\n--- REVISING Cantilever Analysis (Ground Floor vs. First Floor) ---")

detected_cantilevers = []

# Store regions where columns should NOT be placed from ground to first floor
cantilever_no_column_zones_ground_to_first = [] 

# Define a small buffer for spatial checks to account for floating point inaccuracies
CANTILEVER_CHECK_BUFFER = 0.05 # meters, a small buffer for point-in-polygon checks

# Initialize a set to store the (x,y) coordinates of columns that should *not* extend from ground to first floor.
# These columns will start from the first floor level instead of the ground.
columns_to_skip_ground_to_first_span = set()

# Ensure we have at least a ground floor and a "first floor" above it
if len(sorted_floor_z_levels) >= 2:
    ground_floor_z = sorted_floor_z_levels[0] # Assuming the lowest Z is the ground floor
    first_floor_above_ground_z = sorted_floor_z_levels[1] # This is the "first floor" to check for cantilevers

    ground_floor_polygons = floor_footprints_by_level.get(ground_floor_z, [])
    first_floor_polygons = floor_footprints_by_level.get(first_floor_above_ground_z, [])

    if not ground_floor_polygons:
        print(f"No ground floor polygons found at Z level {ground_floor_z:.2f}. Cannot check for first floor cantilevers.")
    elif not first_floor_polygons:
        print(f"No first floor polygons found at Z level {first_floor_above_ground_z:.2f}. No cantilevers to detect.")
    else:
        # Create a combined footprint for the ground floor
        combined_ground_floor_footprint = unary_union(ground_floor_polygons)
        
        print(f"\nChecking First Floor (Z={first_floor_above_ground_z:.2f}m) against Ground Floor (Z={ground_floor_z:.2f}m) for cantilevers.")

        for first_floor_poly in first_floor_polygons:
            # Calculate the part of the first floor polygon that extends beyond the ground floor
            cantilever_part = first_floor_poly.difference(combined_ground_floor_footprint)

            if not cantilever_part.is_empty and cantilever_part.geom_type in ['Polygon', 'MultiPolygon']:
                cantilever_area = 0
                if cantilever_part.geom_type == 'Polygon':
                    cantilever_area = cantilever_part.area
                else: # MultiPolygon
                    for geom in cantilever_part.geoms:
                        if geom.geom_type == 'Polygon':
                            cantilever_area += geom.area
                
                if cantilever_area > CANTILEVER_AREA_THRESHOLD:
                    detected_cantilevers.append({
                        "UpperFloorZ": round(first_floor_above_ground_z, 2),
                        "LowerFloorZ": round(ground_floor_z, 2),
                        "CantileverArea": round(cantilever_area, 2),
                        "CantileverGeometry": cantilever_part,
                        "UpperFloorPolygonCenter": [round(first_floor_poly.centroid.x, 2), round(first_floor_poly.centroid.y, 2)]
                    })
                    print(f"  Detected cantilever at First Floor Z={first_floor_above_ground_z:.2f}m with area: {cantilever_area:.2f} m²")
                    print(f"    Originating polygon center: X={first_floor_poly.centroid.x:.2f}, Y={first_floor_poly.centroid.y:.2f}")
else:
    print("\nNot enough floor levels (less than 2) to check for first floor cantilevers.")

if detected_cantilevers:
    print("\n--- Summary of Detected First Floor Cantilevers ---")
    for cantilever in detected_cantilevers:
        print(f"  - Upper Floor Z: {cantilever['UpperFloorZ']}m, Lower Floor Z: {cantilever['LowerFloorZ']}m, Cantilever Area: {cantilever['CantileverArea']} m²")
else:
    print("\nNo significant first floor cantilevers detected based on the defined thresholds.")

# ... (your existing REVISED Cantilever Detection Logic ends here) ...

# --- Identify "No-Column" Zones for Cantilevers (Ground to First Floor) ---
if detected_cantilevers and len(sorted_floor_z_levels) >= 2:
    ground_floor_z = sorted_floor_z_levels[0]
    combined_ground_floor_footprint = unary_union(floor_footprints_by_level.get(ground_floor_z, []))

    if not combined_ground_floor_footprint.is_empty:
        print("\n--- Identifying Cantilever 'No-Column' Zones ---")
        for cantilever_info in detected_cantilevers:
            cantilever_geom = cantilever_info['CantileverGeometry']
            
            # The perimeter of the cantilever geometry
            cantilever_perimeter = cantilever_geom.exterior if cantilever_geom.geom_type == 'Polygon' else unary_union([g.exterior for g in cantilever_geom.geoms if g.geom_type == 'Polygon'])

            if cantilever_perimeter.is_empty:
                continue

            # The part of the cantilever perimeter that does NOT touch the ground floor footprint
            exposed_perimeter_candidate = cantilever_perimeter.difference(combined_ground_floor_footprint.buffer(CANTILEVER_CHECK_BUFFER))

            # Filter out small artifacts and keep only LineString or MultiLineString components
            no_column_lines = []
            if exposed_perimeter_candidate.geom_type == 'LineString':
                if exposed_perimeter_candidate.length > 0.1: # Only consider significant lengths
                    no_column_lines.append(exposed_perimeter_candidate)
            elif exposed_perimeter_candidate.geom_type == 'MultiLineString':
                for line in exposed_perimeter_candidate.geoms:
                    if line.length > 0.1:
                        no_column_lines.append(line)
            elif exposed_perimeter_candidate.geom_type == 'Polygon': # Should ideally not be a polygon unless a very thin part
                if exposed_perimeter_candidate.area > 0.01: # Small area threshold
                    no_column_lines.append(exposed_perimeter_candidate.exterior) # Use its exterior as the line

            for line in no_column_lines:
                # Create a thin rectangular zone along this line
                try:
                    no_column_zone = line.buffer(CANTILEVER_CHECK_BUFFER, cap_style=3) # cap_style=3 for square ends
                    cantilever_no_column_zones_ground_to_first.append(no_column_zone)
                except Exception as e:
                    print(f"Warning: Could not create no-column zone from line. Error: {e}")
        
        # Combine all no-column zones into a single MultiPolygon for efficient checking
        if cantilever_no_column_zones_ground_to_first:
            cantilever_no_column_zones_ground_to_first_combined = unary_union(cantilever_no_column_zones_ground_to_first)
            print(f"  Combined {len(cantilever_no_column_zones_ground_to_first)} no-column zones.")
        else:
            cantilever_no_column_zones_ground_to_first_combined = None
            print("  No significant cantilever no-column zones identified.")
    else:
        cantilever_no_column_zones_ground_to_first_combined = None
        print("  Ground floor footprint is empty, cannot define cantilever no-column zones.")
else:
    cantilever_no_column_zones_ground_to_first_combined = None
    print("  No detected cantilevers or not enough floor levels to define no-column zones.")


# Structural logic
MaxS = 6.0
MinS = 3.0

columns_2d_points = [] # Store raw (x,y) points for columns
beams_2d_lines = []    # Store raw ((x1,y1),(x2,y2)) for beams for 2D plot

added_column_xy = set()
columns_to_skip_ground_to_first_span = set()

# Force columns at the corners of dominant roof footprints (this part is fine, columns go to ground)
# ... (your existing code for forced columns) ...

# Iterate through detected rooms/floor footprints
for approx_z, floor_polygons in floor_footprints_by_level.items():
    current_floor_z = approx_z # Get the Z-level for the current set of polygons

    for room_poly in floor_polygons: # Iterate through individual room polygons on this floor
        minx, miny, maxx, maxy = room_poly.bounds
        width, height = maxx - minx, maxy - miny
        
        divisions_x = max(1, int(np.ceil(width / MaxS)))
        divisions_y = max(1, int(np.ceil(height / MaxS)))
        
        x_points_grid = np.linspace(minx, maxx, divisions_x + 1)
        y_points_grid = np.linspace(miny, maxy, divisions_y + 1)
        
        # Add interior grid columns (this part is fine, columns go to ground unless skipped)
        for x in x_points_grid:
            for y in y_points_grid:
                col_pt = Point(x, y)
                rounded_x = round(x, 5)
                rounded_y = round(y, 5)
                
                if room_poly.contains(col_pt) or room_poly.buffer(1e-6).contains(col_pt):
                    if (rounded_x, rounded_y) not in added_column_xy:
                        if all(np.linalg.norm(np.array((rounded_x, rounded_y)) - np.array(exist_col_xy)) >= MinS for exist_col_xy in added_column_xy):
                            columns_2d_points.append((rounded_x, rounded_y))
                            added_column_xy.add((rounded_x, rounded_y))

                    # Check for cantilever no-column zones for the vertical span
                    if cantilever_no_column_zones_ground_to_first_combined:
                        if cantilever_no_column_zones_ground_to_first_combined.intersects(col_pt.buffer(CANTILEVER_CHECK_BUFFER)):
                            columns_to_skip_ground_to_first_span.add((rounded_x, rounded_y))

        # Add columns at corners of the room polygon (this part is fine)
        for corner_x, corner_y in room_poly.exterior.coords:
            corner_pt_shapely = Point(corner_x, corner_y)
            rounded_corner_x = round(corner_x, 5)
            rounded_corner_y = round(corner_y, 5)
            
            if (rounded_corner_x, rounded_corner_y) not in added_column_xy:
                if all(np.linalg.norm(np.array((corner_x, corner_y)) - np.array(exist_col_xy)) >= MinS * 0.5 for exist_col_xy in added_column_xy):
                    columns_2d_points.append((rounded_corner_x, rounded_corner_y))
                    added_column_xy.add((rounded_corner_x, rounded_corner_y))
            
            # Check for cantilever no-column zones for the vertical span
            if cantilever_no_column_zones_ground_to_first_combined:
                if cantilever_no_column_zones_ground_to_first_combined.intersects(corner_pt_shapely.buffer(CANTILEVER_CHECK_BUFFER)):
                    columns_to_skip_ground_to_first_span.add((rounded_corner_x, rounded_corner_y))

        # IMPORTANT: NEW BEAM GENERATION LOGIC - APPLY CONDITION HERE
        if abs(current_floor_z - ground_floor_z) > 1e-4: # Or 1e-5, depending on precision needed            # Horizontal beams
            for y_fixed in y_points_grid:
                points_on_line = []
                for x_coord in x_points_grid:
                    p = Point(x_coord, y_fixed)
                    if room_poly.buffer(1e-6).contains(p):
                        points_on_line.append((x_coord, y_fixed))
                
                if len(points_on_line) > 1:
                    for i in range(len(points_on_line) - 1):
                        beams_2d_lines.append((points_on_line[i], points_on_line[i+1]))

            # Vertical beams
            for x_fixed in x_points_grid:
                points_on_line = []
                for y_coord in y_points_grid:
                    p = Point(x_fixed, y_coord)
                    if room_poly.buffer(1e-6).contains(p):
                        points_on_line.append((x_fixed, y_coord))
                
                if len(points_on_line) > 1:
                    for i in range(len(points_on_line) - 1):
                        beams_2d_lines.append((points_on_line[i], points_on_line[i+1]))


# Remove duplicate 2D beam lines
unique_beams_2d = set()
for p1, p2 in beams_2d_lines:
    # Ensure consistent order for tuple comparison
    ordered_segment = tuple(sorted((tuple(np.round(p1,5)), tuple(np.round(p2,5)))))
    unique_beams_2d.add(ordered_segment)
beams_2d_lines = [ (np.array(p1), np.array(p2)) for p1,p2 in unique_beams_2d]


# Combine all base columns
all_base_columns = list(added_column_xy)

# --- COLUMN GRID NUMBERING LOGIC ---
grid_xs = sorted(list(set([col[0] for col in all_base_columns])))
grid_ys = sorted(list(set([col[1] for col in all_base_columns])))

x_grid_labels = {x: chr(65 + i) for i, x in enumerate(grid_xs)}
y_grid_labels = {y: i + 1 for i, y in enumerate(grid_ys)}

col_min_x_extent = min(col[0] for col in all_base_columns) if all_base_columns else 0
col_max_x_extent = max(col[0] for col in all_base_columns) if all_base_columns else 0
col_min_y_extent = min(col[1] for col in all_base_columns) if all_base_columns else 0
col_max_y_extent = max(col[1] for col in all_base_columns) if all_base_columns else 0

grid_extent_buffer = 1.0
col_min_x_extent -= grid_extent_buffer
col_max_x_extent += grid_extent_buffer
col_min_y_extent -= grid_extent_buffer
col_max_y_extent += grid_extent_buffer

min_x_plot = min(col_min_x_extent, min([coord[0] for coord in perimeter_line_coords] + [col[0] for col in all_base_columns])) - 3.0 if perimeter_line_coords else col_min_x_extent - 3.0
max_x_plot = max(col_max_x_extent, max([coord[0] for coord in perimeter_line_coords] + [col[0] for col in all_base_columns])) + 1.0 if perimeter_line_coords else col_max_x_extent + 1.0
min_y_plot = min(col_min_y_extent, min([coord[1] for coord in perimeter_line_coords] + [col[1] for col in all_base_columns])) - 1.0 if perimeter_line_coords else col_min_y_extent - 1.0
max_y_plot = max(col_max_y_extent, max([coord[1] for coord in perimeter_line_coords] + [col[1] for col in all_base_columns])) + 2.0 if perimeter_line_coords else col_max_y_extent + 2.0


# --- Utility function for wall height ---
def get_wall_height(x, y, mesh_bboxes, global_max_z):
    pt = Point(x, y)
    relevant_bboxes = []
    for bbox in mesh_bboxes:
        bbox_poly = Polygon([
            [bbox.Min.X, bbox.Min.Y],
            [bbox.Max.X, bbox.Min.Y],
            [bbox.Max.X, bbox.Max.Y],
            [bbox.Min.X, bbox.Max.Y],
            [bbox.Min.X, bbox.Min.Y]
        ])
        if bbox_poly.buffer(1e-4).contains(pt):
            relevant_bboxes.append(bbox)

    if not relevant_bboxes:
        return global_max_z 

    max_relevant_z = 0.0
    for bbox in relevant_bboxes:
        max_relevant_z = max(max_relevant_z, bbox.Max.Z)
        
    return max_relevant_z if max_relevant_z > 0 else global_max_z


# --- DATA GENERATION AND COMBINED JSON EXPORT ---
print("--- Generating data and creating combined JSON ---")

EXPORT_SAVE_PATH = os.path.join(os.getcwd(), "structural_data") # Use current working directory

os.makedirs(EXPORT_SAVE_PATH, exist_ok=True)
print(f"Ensuring directory exists: {os.path.abspath(EXPORT_SAVE_PATH)}")

node_coords = []
node_dict = OrderedDict()

def add_node(pt):
    key = tuple(np.round(pt, 5))
    if key not in node_dict:
        node_id = f"N{len(node_dict)}"
        node_dict[key] = node_id
        node_coords.append([node_id] + list(key))
    return node_dict[key]

column_lines = []
for x, y in all_base_columns:
    current_column_max_z = get_wall_height(x, y, all_mesh_bboxes, max_z)
    
    # Determine the actual floor Z-levels that this specific column should span
    # We need to map the generic 'num_floors' based Z calculation to the actual detected floor Zs.

    # Find the actual ground floor Z and first floor above ground Z
    # Ensure sorted_floor_z_levels is populated and contains these
    local_ground_floor_z = sorted_floor_z_levels[0] if sorted_floor_z_levels else 0.0
    local_first_floor_z = sorted_floor_z_levels[1] if len(sorted_floor_z_levels) >= 2 else local_ground_floor_z # Fallback
    
    # Check if this column needs to skip the ground-to-first-floor span
    skip_ground_to_first = (round(x, 5), round(y, 5)) in columns_to_skip_ground_to_first_span

    # Define the actual Z-levels for this column's segments
    # Start with the local_ground_floor_z
    column_z_levels_for_this_column = [local_ground_floor_z] 

    # Add all other detected floor Z-levels
    # Skip adding local_ground_floor_z if it's already there
    for z in sorted_floor_z_levels:
        if z > local_ground_floor_z + 1e-6: # Add levels above ground, avoiding duplicates
            column_z_levels_for_this_column.append(z)
    
    # Filter out Z-levels that are above the local building's height at this column location
    column_z_levels_for_this_column = [z for z in column_z_levels_for_this_column if z <= current_column_max_z + 1e-6]
    
    # IMPORTANT: If skipping ground-to-first, make sure the first Z level is first_floor_above_ground_z
    # and remove any floor Zs below that (like the actual ground floor).
    if skip_ground_to_first and local_first_floor_z > local_ground_floor_z + 1e-6: # Ensure there IS a first floor
        # Remove any Z-levels below the first floor if skipping
        # This handles cases where local_first_floor_z might be slightly different from column_z_levels_for_this_column[0]
        column_z_levels_for_this_column = [z for z in column_z_levels_for_this_column if z >= local_first_floor_z - 1e-6]
        # Ensure the list is not empty, if it becomes empty, this column might not exist at all,
        # or it should just start at its lowest valid Z.
        if not column_z_levels_for_this_column:
            column_z_levels_for_this_column = [local_first_floor_z] # Make sure there's at least one start point
        elif column_z_levels_for_this_column[0] > local_first_floor_z + 1e-6:
             # If the lowest valid Z is higher than expected, prepend the actual first_floor_z
             column_z_levels_for_this_column.insert(0, local_first_floor_z)

    # Sort and remove duplicates in case of slight floating point differences
    column_z_levels_for_this_column = sorted(list(set(np.round(column_z_levels_for_this_column, 5))))
    
    # Now, generate column segments based on these adjusted Z levels
    for i in range(len(column_z_levels_for_this_column) - 1):
        start_z = column_z_levels_for_this_column[i]
        end_z = column_z_levels_for_this_column[i+1]
        
        # Only create a segment if it has a non-zero height
        if end_z > start_z + 1e-6:
            id_start_node = add_node((x, y, start_z))
            id_end_node = add_node((x, y, end_z))
            column_lines.append((id_start_node, id_end_node))

# Note: The 'num_floors' variable as used in your original loop was implicitly
# assuming evenly spaced floors up to `current_column_max_z`. By using `sorted_floor_z_levels`,
# we are now explicitly using the detected floor heights, which is more robust
# and directly aligns with your floor detection.

# ... (rest of your beam_lines generation, which doesn't need modification for this specific rule) ...
beam_lines = []
unique_beam_tuples_3d = set() # Use a new set for 3D beam uniqueness

for (x1, y1), (x2, y2) in beams_2d_lines:
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    beam_local_max_z = get_wall_height(mid_x, mid_y, all_mesh_bboxes, max_z)

    for floor_z_level_actual in sorted_floor_z_levels:
        if floor_z_level_actual > beam_local_max_z + 1e-6:
            continue

        # 🛑 Skip beams at ground level (Z ≈ 0)
        if abs(floor_z_level_actual) < 1e-4:
            continue

        floor_z_level_rounded = round(floor_z_level_actual, 5)

        id1 = add_node((x1, y1, floor_z_level_rounded))
        id2 = add_node((x2, y2, floor_z_level_rounded))

        ordered_nodes = tuple(sorted((id1, id2)))
        if ordered_nodes not in unique_beam_tuples_3d:
            unique_beam_tuples_3d.add(ordered_nodes)
            beam_lines.append((id1, id2))


# --- Create combined structural data ---
print("--- Creating combined structural data ---")

# Create DataFrames
df_nodes = pd.DataFrame(node_coords, columns=["ID", "X", "Y", "Z"])

# Prepare nodes data
nodes_data = []
for node in node_coords:
    nodes_data.append({
        "ID": node[0],
        "X": node[1],
        "Y": node[2],
        "Z": node[3]
    })

# Prepare columns data  
columns_data = []
for i, (i_node_id, j_node_id) in enumerate(column_lines):
    p1_coords = df_nodes[df_nodes["ID"] == i_node_id][["X", "Y", "Z"]].values[0]
    p2_coords = df_nodes[df_nodes["ID"] == j_node_id][["X", "Y", "Z"]].values[0]
    length = np.linalg.norm(p2_coords - p1_coords)
    columns_data.append({
        "ID": f"C{i}",
        "i_node": i_node_id,
        "j_node": j_node_id,
        "length": round(length, 3)
    })

# Prepare beams data
beams_data = []
for i, (i_node_id, j_node_id) in enumerate(beam_lines):
    p1_coords = df_nodes[df_nodes["ID"] == i_node_id][["X", "Y", "Z"]].values[0]
    p2_coords = df_nodes[df_nodes["ID"] == j_node_id][["X", "Y", "Z"]].values[0]
    length = np.linalg.norm(p2_coords - p1_coords)
    beams_data.append({
        "ID": f"B{i}",
        "i_node": i_node_id,
        "j_node": j_node_id,
        "length": round(length, 3)
    })

# Create DataFrames for later processing
df_columns = pd.DataFrame([(f"C{i}", i_node_id, j_node_id, round(np.linalg.norm(df_nodes[df_nodes["ID"] == j_node_id][["X", "Y", "Z"]].values[0] - df_nodes[df_nodes["ID"] == i_node_id][["X", "Y", "Z"]].values[0]), 3)) for i, (i_node_id, j_node_id) in enumerate(column_lines)], columns=["ID", "i_node", "j_node", "length"])
df_beams = pd.DataFrame([(f"B{i}", i_node_id, j_node_id, round(np.linalg.norm(df_nodes[df_nodes["ID"] == j_node_id][["X", "Y", "Z"]].values[0] - df_nodes[df_nodes["ID"] == i_node_id][["X", "Y", "Z"]].values[0]), 3)) for i, (i_node_id, j_node_id) in enumerate(beam_lines)], columns=["ID", "i_node", "j_node", "length"])


# ---------------------------------------
# Data Processing and Analysis Section
# ---------------------------------------

# Re-load data and re-run identification logic to ensure all variables are defined
# This part processes the structural data to identify and analyze beam patterns

print(f"Using in-memory structural data from previous processing")
print(f"Using in-memory 3DM model for analysis")

# Use the DataFrames from the previous section (already available in memory)
df_nodes_loaded = df_nodes.copy()
df_columns_loaded = df_columns.copy()
df_beams_loaded = df_beams.copy()
print("✅ Structural data ready for update process.")

# Use the in-memory model (already loaded above)

building_floor_footprints = []
all_mesh_bboxes = []
roof_meshes_info = []
max_z_overall_model = 0.0
Z_FLATNESS_TOLERANCE = 0.1

for obj in model.Objects:
    geom = obj.Geometry
    if geom.ObjectType == rhino3dm.ObjectType.Mesh:
        bbox = geom.GetBoundingBox()
        bbox_z_dim = bbox.Max.Z - bbox.Min.Z
        if bbox_z_dim < Z_FLATNESS_TOLERANCE and (bbox.Max.X - bbox.Min.X) > 0.1 and (bbox.Max.Y - bbox.Min.Y) > 0.1:
            base_pts = [[bbox.Min.X, bbox.Min.Y], [bbox.Max.X, bbox.Min.Y], [bbox.Max.X, bbox.Max.Y], [bbox.Min.X, bbox.Max.Y], [bbox.Min.X, bbox.Min.Y]]
            poly = Polygon(base_pts)
            if poly.area > 1e-3:
                building_floor_footprints.append(poly)
                roof_meshes_info.append((obj.Attributes.Id, bbox, poly))
        all_mesh_bboxes.append(bbox)
        max_z_overall_model = max(max_z_overall_model, bbox.Max.Z)

def get_wall_height(x, y, mesh_bboxes, global_max_z):
    pt = Point(x, y)
    relevant_bboxes = []
    for bbox in mesh_bboxes:
        bbox_poly = Polygon([[bbox.Min.X, bbox.Min.Y], [bbox.Max.X, bbox.Min.Y], [bbox.Max.X, bbox.Max.Y], [bbox.Min.X, bbox.Max.Y], [bbox.Min.X, bbox.Min.Y]])
        if bbox_poly.buffer(1e-4).contains(pt):
            relevant_bboxes.append(bbox)
    if not relevant_bboxes: return global_max_z
    max_relevant_z = 0.0
    for bbox in relevant_bboxes:
        max_relevant_z = max(max_relevant_z, bbox.Max.Z)
    return max_relevant_z if max_relevant_z > 0 else global_max_z

# Re-identify dominant roofs
INTERSECTION_BUFFER_ROOF = 0.1
dominant_roofs_identified = []
for i, (roof1_id, roof1_bbox, roof1_poly_2d) in enumerate(roof_meshes_info):
    roof1_max_z = roof1_bbox.Max.Z
    is_dominant_roof = True
    for j, (roof2_id, roof2_bbox, roof2_poly_2d) in enumerate(roof_meshes_info):
        if i == j: continue
        roof2_max_z = roof2_bbox.Max.Z
        intersection_geometry = roof1_poly_2d.buffer(INTERSECTION_BUFFER_ROOF).intersection(roof2_poly_2d.buffer(INTERSECTION_BUFFER_ROOF))
        is_touching = not intersection_geometry.is_empty and intersection_geometry.geom_type in ['LineString', 'MultiLineString', 'Polygon', 'MultiPolygon']
        if is_touching:
            if roof1_max_z <= roof2_max_z + 1e-6:
                is_dominant_roof = False
                break
    if is_dominant_roof and (len(roof_meshes_info) > 1 or len(dominant_roofs_identified) == 0):
        if not is_dominant_roof: continue
        dominant_roofs_identified.append({"RhinoObjectId": str(roof1_id), "Height": round(roof1_max_z, 3), "Polygon_2D": roof1_poly_2d})

# Re-identify peak roof beams
peak_roof_beam_ids = set()
Z_HEIGHT_TOLERANCE = 0.5
node_id_to_coords = df_nodes_loaded.set_index('ID')[['X', 'Y', 'Z']].T.to_dict('list')

for index, row in df_beams_loaded.iterrows():
    beam_id = row['ID']
    n1_id = row['i_node']
    n2_id = row['j_node']
    try:
        p1_coords = np.array(node_id_to_coords[n1_id])
        p2_coords = np.array(node_id_to_coords[n2_id])
    except KeyError:
        continue
    mid_x, mid_y = (p1_coords[0] + p2_coords[0]) / 2, (p1_coords[1] + p2_coords[1]) / 2
    avg_z = (p1_coords[2] + p2_coords[2]) / 2

    # Get the "true" max Z for the area covered by this beam (using its midpoint)
    local_max_z_at_beam_location = get_wall_height(mid_x, mid_y, all_mesh_bboxes, max_z_overall_model)

    # If the beam's average Z is within tolerance of the local max Z, consider it a peak roof beam
    if abs(local_max_z_at_beam_location - avg_z) < Z_HEIGHT_TOLERANCE:
        # Also check if this beam's 2D projection falls within any dominant roof polygon
        beam_line_2d = LineString([(p1_coords[0], p1_coords[1]), (p2_coords[0], p2_coords[1])])
        for roof_info in dominant_roofs_identified:
            roof_poly_2d = roof_info["Polygon_2D"]
            if roof_poly_2d.buffer(0.01).intersects(beam_line_2d): # Small buffer for intersection
                peak_roof_beam_ids.add(beam_id)
                break # Found a matching dominant roof, no need to check other roofs for this beam


# Generate 3D visualization
import plotly.graph_objects as go # Added this import

fig_3d = go.Figure()

# Add columns
for index, row in df_columns_loaded.iterrows():
    n1_coords = df_nodes_loaded[df_nodes_loaded['ID'] == row['i_node']][['X', 'Y', 'Z']].values[0]
    n2_coords = df_nodes_loaded[df_nodes_loaded['ID'] == row['j_node']][['X', 'Y', 'Z']].values[0]
    fig_3d.add_trace(go.Scatter3d(
        x=[n1_coords[0], n2_coords[0]],
        y=[n1_coords[1], n2_coords[1]],
        z=[n1_coords[2], n2_coords[2]],
        mode='lines',
        line=dict(color='blue', width=5),
        name=f'Column {row["ID"]}',
        hoverinfo='text',
        text=f'Column: {row["ID"]}<br>Length: {row["length"]:.2f}m'
    ))

# Add beams
for index, row in df_beams_loaded.iterrows():
    n1_coords = df_nodes_loaded[df_nodes_loaded['ID'] == row['i_node']][['X', 'Y', 'Z']].values[0]
    n2_coords = df_nodes_loaded[df_nodes_loaded['ID'] == row['j_node']][['X', 'Y', 'Z']].values[0]
    
    line_color = 'green'
    line_width = 3
    if row['ID'] in peak_roof_beam_ids:
        line_color = 'red' # Highlight peak roof beams
        line_width = 6

    fig_3d.add_trace(go.Scatter3d(
        x=[n1_coords[0], n2_coords[0]],
        y=[n1_coords[1], n2_coords[1]],
        z=[n1_coords[2], n2_coords[2]],
        mode='lines',
        line=dict(color=line_color, width=line_width),
        name=f'Beam {row["ID"]}',
        hoverinfo='text',
        text=f'Beam: {row["ID"]}<br>Length: {row["length"]:.2f}m'
    ))

# Add nodes as markers
fig_3d.add_trace(go.Scatter3d(
    x=df_nodes_loaded['X'],
    y=df_nodes_loaded['Y'],
    z=df_nodes_loaded['Z'],
    mode='markers',
    marker=dict(size=4, color='black'),
    name='Nodes',
    hoverinfo='text',
    text=df_nodes_loaded['ID']
))

# Update layout for better visualization
fig_3d.update_layout(
    title='3D Structural Model with Columns (Blue), Beams (Green), and Peak Roof Beams (Red)',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data' # Ensure uniform scaling
    ),
    height=800
)

fig_3d.show()

print(f"\nTotal peak roof beams identified: {len(peak_roof_beam_ids)}")
if peak_roof_beam_ids:
    print("IDs of peak roof beams:", ", ".join(sorted(list(peak_roof_beam_ids))))



#--------------------------------------
# Extended Analysis and Data Processing
#-----------------------

import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
import rhino3dm
from shapely.geometry import Polygon, Point, MultiPolygon, LineString
from collections import OrderedDict # Re-import for OrderedDict if not already global

# --- Re-load data and re-run identification logic to ensure all variables are defined ---
# This part is a copy of the previous cell's data loading and identification logic
# to ensure this cell can run independently if the previous one wasn't executed immediately before.

# base_github_path = r"C:\Users\papad\Documents\GitHub\Octopusie"
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
base_github_path = os.path.abspath(os.path.join(script_dir, ".."))

CSV_SAVE_PATH = os.path.join(base_github_path, "eleftheriaexperiment", "structural_data")
os.makedirs(CSV_SAVE_PATH, exist_ok=True)  # <— ensure it exists!

print(f"Using in-memory structural data from previous processing")
print(f"Using in-memory 3DM model for analysis")

# Use the DataFrames from the previous section (already available in memory)
df_nodes_loaded = df_nodes.copy()
df_columns_loaded = df_columns.copy()
df_beams_loaded = df_beams.copy()
print("✅ Structural data ready for update process.")

# Use the in-memory model (already loaded above)

building_floor_footprints = []
all_mesh_bboxes = []
roof_meshes_info = []
max_z_overall_model = 0.0
Z_FLATNESS_TOLERANCE = 0.1

for obj in model.Objects:
    geom = obj.Geometry
    if geom.ObjectType == rhino3dm.ObjectType.Mesh:
        bbox = geom.GetBoundingBox()
        bbox_z_dim = bbox.Max.Z - bbox.Min.Z
        if bbox_z_dim < Z_FLATNESS_TOLERANCE and (bbox.Max.X - bbox.Min.X) > 0.1 and (bbox.Max.Y - bbox.Min.Y) > 0.1:
            base_pts = [[bbox.Min.X, bbox.Min.Y], [bbox.Max.X, bbox.Min.Y], [bbox.Max.X, bbox.Max.Y], [bbox.Min.X, bbox.Max.Y], [bbox.Min.X, bbox.Min.Y]]
            poly = Polygon(base_pts)
            if poly.area > 1e-3:
                building_floor_footprints.append(poly)
                roof_meshes_info.append((obj.Attributes.Id, bbox, poly))
        all_mesh_bboxes.append(bbox)
        max_z_overall_model = max(max_z_overall_model, bbox.Max.Z)

def get_wall_height(x, y, mesh_bboxes, global_max_z):
    pt = Point(x, y)
    relevant_bboxes = []
    for bbox in mesh_bboxes:
        bbox_poly = Polygon([[bbox.Min.X, bbox.Min.Y], [bbox.Max.X, bbox.Min.Y], [bbox.Max.X, bbox.Max.Y], [bbox.Min.X, bbox.Max.Y], [bbox.Min.X, bbox.Min.Y]])
        if bbox_poly.buffer(1e-4).contains(pt):
            relevant_bboxes.append(bbox)
    if not relevant_bboxes: return global_max_z
    max_relevant_z = 0.0
    for bbox in relevant_bboxes:
        max_relevant_z = max(max_relevant_z, bbox.Max.Z)
    return max_relevant_z if max_relevant_z > 0 else global_max_z

# Re-identify dominant roofs
INTERSECTION_BUFFER_ROOF = 0.1
dominant_roofs_identified = []
for i, (roof1_id, roof1_bbox, roof1_poly_2d) in enumerate(roof_meshes_info):
    roof1_max_z = roof1_bbox.Max.Z
    is_dominant_roof = True
    for j, (roof2_id, roof2_bbox, roof2_poly_2d) in enumerate(roof_meshes_info):
        if i == j: continue
        roof2_max_z = roof2_bbox.Max.Z
        intersection_geometry = roof1_poly_2d.buffer(INTERSECTION_BUFFER_ROOF).intersection(roof2_poly_2d.buffer(INTERSECTION_BUFFER_ROOF))
        is_touching = not intersection_geometry.is_empty and intersection_geometry.geom_type in ['LineString', 'MultiLineString', 'Polygon', 'MultiPolygon']
        if is_touching:
            if roof1_max_z <= roof2_max_z + 1e-6:
                is_dominant_roof = False
                break
    if is_dominant_roof and (len(roof_meshes_info) > 1 or len(dominant_roofs_identified) == 0):
        if not is_dominant_roof: continue
        dominant_roofs_identified.append({"RhinoObjectId": str(roof1_id), "Height": round(roof1_max_z, 3), "Polygon_2D": roof1_poly_2d})

# Re-identify peak roof beams
peak_roof_beam_ids = set()
Z_HEIGHT_TOLERANCE = 0.5
node_id_to_coords = df_nodes_loaded.set_index('ID')[['X', 'Y', 'Z']].T.to_dict('list')

for index, row in df_beams_loaded.iterrows():
    beam_id = row['ID']
    n1_id = row['i_node']
    n2_id = row['j_node']
    try:
        p1_coords = np.array(node_id_to_coords[n1_id])
        p2_coords = np.array(node_id_to_coords[n2_id])
    except KeyError:
        continue
    mid_x, mid_y = (p1_coords[0] + p2_coords[0]) / 2, (p1_coords[1] + p2_coords[1]) / 2
    avg_z = (p1_coords[2] + p2_coords[2]) / 2
    beam_midpoint_2d = Point(mid_x, mid_y)
    for roof_info in dominant_roofs_identified:
        roof_polygon_2d = roof_info['Polygon_2D']
        roof_peak_height = roof_info['Height']
        if roof_polygon_2d.buffer(1e-3).contains(beam_midpoint_2d) and abs(avg_z - roof_peak_height) < Z_HEIGHT_TOLERANCE:
            peak_roof_beam_ids.add(beam_id)
            break

# Re-identify low-connectivity linear beams on peak roofs
beam_adj_list = {node_id: [] for node_id in df_nodes_loaded['ID']}
for index, row in df_beams_loaded.iterrows():
    beam_adj_list[row['i_node']].append(row['ID'])
    beam_adj_list[row['j_node']].append(row['ID'])

def angle_between_vectors(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    unit_vector_1 = v1 / (np.linalg.norm(v1) + 1e-9)
    unit_vector_2 = v2 / (np.linalg.norm(v2) + 1e-9)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)

COLLINEARITY_TOLERANCE_DEG = 10 
highlighted_peak_roof_beams = set()

if not df_beams_loaded.empty:
    for index, row in df_beams_loaded.iterrows():
        beam_id = row['ID']
        if beam_id not in peak_roof_beam_ids:
            continue

        n1_id, n2_id = row['i_node'], row['j_node']
        is_low_connectivity_linear = False

        if len(beam_adj_list[n1_id]) == 2:
            other_beam_ids_at_n1 = [b_id for b_id in beam_adj_list[n1_id] if b_id != beam_id]
            if other_beam_ids_at_n1:
                other_beam_id_at_n1 = other_beam_ids_at_n1[0]
                current_beam_other_node = n2_id
                other_beam_row = df_beams_loaded[df_beams_loaded['ID'] == other_beam_id_at_n1].iloc[0]
                other_beam_other_node = other_beam_row['i_node'] if other_beam_row['j_node'] == n1_id else other_beam_row['j_node']
                coords_n1 = np.array(node_id_to_coords.get(n1_id, [0,0,0]))
                coords_current_other = np.array(node_id_to_coords.get(current_beam_other_node, [0,0,0]))
                coords_other_beam_other = np.array(node_id_to_coords.get(other_beam_other_node, [0,0,0]))
                vector_current_beam = coords_current_other - coords_n1
                vector_other_beam = coords_other_beam_other - coords_n1
                angle = angle_between_vectors(vector_current_beam, vector_other_beam)
                if abs(angle) < COLLINEARITY_TOLERANCE_DEG or abs(angle - 180) < COLLINEARITY_TOLERANCE_DEG:
                    is_low_connectivity_linear = True

        if not is_low_connectivity_linear and len(beam_adj_list[n2_id]) == 2:
            other_beam_ids_at_n2 = [b_id for b_id in beam_adj_list[n2_id] if b_id != beam_id]
            if other_beam_ids_at_n2:
                other_beam_id_at_n2 = other_beam_ids_at_n2[0]
                current_beam_other_node = n1_id
                other_beam_row = df_beams_loaded[df_beams_loaded['ID'] == other_beam_id_at_n2].iloc[0]
                other_beam_other_node = other_beam_row['i_node'] if other_beam_row['j_node'] == n2_id else other_beam_row['j_node']
                coords_n2 = np.array(node_id_to_coords.get(n2_id, [0,0,0]))
                coords_current_other = np.array(node_id_to_coords.get(current_beam_other_node, [0,0,0]))
                coords_other_beam_other = np.array(node_id_to_coords.get(other_beam_other_node, [0,0,0]))
                vector_current_beam = coords_current_other - coords_n2
                vector_other_beam = coords_other_beam_other - coords_n2
                angle = angle_between_vectors(vector_current_beam, vector_other_beam)
                if abs(angle) < COLLINEARITY_TOLERANCE_DEG or abs(angle - 180) < COLLINEARITY_TOLERANCE_DEG:
                    is_low_connectivity_linear = True

        if not is_low_connectivity_linear:
            if len(beam_adj_list[n1_id]) == 1 or len(beam_adj_list[n2_id]) == 1:
                is_low_connectivity_linear = True
        
        if is_low_connectivity_linear:
            highlighted_peak_roof_beams.add(beam_id)

print(f"\nIdentified {len(highlighted_peak_roof_beams)} beam segments that are low-connectivity linear ends AND on peak roofs for deletion.")
print("IDs of beams to be deleted:")
for beam_id in sorted(list(highlighted_peak_roof_beams)):
    print(f"- {beam_id}")

# --- END Re-run identification logic ---


# --- DELETE BEAMS AND UPDATE DATA ---
print("\n--- Deleting identified beams and updating structural data ---")

# Filter df_beams_loaded to remove the identified beams
df_beams_updated = df_beams_loaded[~df_beams_loaded['ID'].isin(highlighted_peak_roof_beams)].copy()

# Now, re-prune nodes based on the remaining beams and all columns
connected_nodes_after_deletion = set()

# Add nodes from columns
if not df_columns_loaded.empty:
    connected_nodes_after_deletion.update(df_columns_loaded["i_node"].tolist())
    connected_nodes_after_deletion.update(df_columns_loaded["j_node"].tolist())

# Add nodes from the updated beams
if not df_beams_updated.empty:
    connected_nodes_after_deletion.update(df_beams_updated["i_node"].tolist())
    connected_nodes_after_deletion.update(df_beams_updated["j_node"].tolist())

print(f"Initial nodes count: {len(df_nodes_loaded)}")
print(f"Nodes connected after beam deletion: {len(connected_nodes_after_deletion)}")

# Filter nodes and re-map IDs
df_nodes_updated = df_nodes_loaded[df_nodes_loaded['ID'].isin(connected_nodes_after_deletion)].copy()
new_node_id_mapping = {old_id: f"N{i}" for i, old_id in enumerate(df_nodes_updated['ID'])}
df_nodes_updated['ID'] = df_nodes_updated['ID'].map(new_node_id_mapping)

# Rebuild node_lookup_loaded and node_id_to_coords with updated nodes
node_lookup_loaded = df_nodes_updated.set_index('ID')
node_id_to_coords = df_nodes_updated.set_index('ID')[['X', 'Y', 'Z']].T.to_dict('list')

# Re-map node IDs in columns DataFrame
df_columns_updated = df_columns_loaded.copy()
df_columns_updated['i_node'] = df_columns_updated['i_node'].map(new_node_id_mapping)
df_columns_updated['j_node'] = df_columns_updated['j_node'].map(new_node_id_mapping)
# Filter out columns whose nodes might have been deleted (map result will be NaN for deleted nodes)
df_columns_updated.dropna(subset=['i_node', 'j_node'], inplace=True)
df_columns_updated['ID'] = [f"C{i}" for i in range(len(df_columns_updated))] # Re-index column IDs


# Re-map node IDs in the updated beams DataFrame
df_beams_updated['i_node'] = df_beams_updated['i_node'].map(new_node_id_mapping)
df_beams_updated['j_node'] = df_beams_updated['j_node'].map(new_node_id_mapping)
# Filter out beams whose nodes might have been deleted
df_beams_updated.dropna(subset=['i_node', 'j_node'], inplace=True)
df_beams_updated['ID'] = [f"B{i}" for i in range(len(df_beams_updated))] # Re-index beam IDs


print(f"Final nodes count: {len(df_nodes_updated)}")
print(f"Final columns count: {len(df_columns_updated)}")
print(f"Final beams count: {len(df_beams_updated)}")


# --- FINAL DATA PROCESSING AND UPLOAD ---
print("\n--- Processing final structural data ---")

# --- CREATE AND UPLOAD COMBINED JSON TO SUPABASE ---
print("\n--- Creating and uploading combined structural data to Supabase ---")

# Create combined structural data
combined_structural_data = {
    "metadata": {
        "filename": file_name_without_ext,
        "scale_factor": scale_factor,
        "num_floors": num_floors,
        "wall_thickness": wall_thickness,
        "unity_axis_format": unityAxisFormat,
        "generation_timestamp": pd.Timestamp.now().isoformat()
    },
    "nodes": df_nodes_updated.to_dict(orient='records'),
    "columns": df_columns_updated.to_dict(orient='records'),
    "beams": df_beams_updated.to_dict(orient='records')
}

# Convert to JSON string
combined_json_data = json.dumps(combined_structural_data, indent=4)

# Upload to Supabase
try:
    # Upload the combined JSON to the "analysis-results" bucket
    json_filename = f"{file_name_without_ext}_structural_data.json"
    upload_response = supabase.storage.from_("analysis-results").upload(
        json_filename, 
        combined_json_data.encode('utf-8'),
        {"content-type": "application/json"}
    )
    
    if upload_response:
        print(f"✅ Combined structural data uploaded to Supabase as '{json_filename}'")
        
        # Generate a public URL for the uploaded file
        public_url = supabase.storage.from_("analysis-results").get_public_url(json_filename)
        print(f"📄 Public URL: {public_url}")
    else:
        print("❌ Failed to upload to Supabase")
        
except Exception as e:
    print(f"❌ Error uploading to Supabase: {e}")
    # Fallback: save locally
    local_json_path = os.path.join(CSV_SAVE_PATH, json_filename)
    with open(local_json_path, 'w') as f:
        f.write(combined_json_data)
    print(f"💾 Saved combined structural data locally as fallback: {local_json_path}")

print("\n🎉 Processing complete!")

