## --------------------------------------------------------------
# GLB TO 3DM CONVERTER - FLASK APPLICATION
# --------------------------------------------------------------

import pygltflib
from pygltflib.validator import validate, summary
import rhino3dm
import os
import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon, LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import json
from supabase import create_client, Client
import requests
import tempfile
import plotly.graph_objects as go
from flask import Flask, request, jsonify
import logging

# Configure logging to suppress Flask messages
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# Fixed scale factor for all GLB files
scale_factor = 1
unityAxisFormat = False


def convert_glb_to_3dm(glb_data):
    """
    Loads GLB data from memory, extracts its mesh data, and returns a 3dm model object.
    Only mesh geometry will be converted. Materials, animations, etc., are not translated.
    """
    global unityAxisFormat
    
    # Write GLB data to a temporary file for pygltflib to read
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
                    continue

                indices = np.frombuffer(indices_bytes, dtype=indices_dtype)

                # glTF uses flat arrays for indices, assuming triangles (mode 4)
                if primitive.mode == pygltflib.TRIANGLES:
                    for i in range(0, len(indices), 3):
                        rhino_mesh.Faces.AddFace(int(indices[i]), int(indices[i+1]), int(indices[i+2]))
                else:
                    continue
            else:
                for i in range(0, len(vertices) - 2, 3):
                    rhino_mesh.Faces.AddFace(i, i+1, i+2)

            # Optional: Calculate normals
            rhino_mesh.Normals.ComputeNormals()
            rhino_mesh.Compact()

            model_3dm.Objects.AddMesh(rhino_mesh)
    
    return model_3dm


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


@app.route('/process', methods=['POST'])
def process_glb():
    """
    Flask endpoint to process GLB files and return structural analysis
    """
    try:
        # Initialize Supabase client
        SUPABASE_URL = "https://apdbfbjnlsxjfubqahtl.supabase.co"
        SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFwZGJmYmpubHN4amZ1YnFhaHRsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc0NzY1MzgsImV4cCI6MjA2MzA1MjUzOH0.DPINyYHHUzcuQ6AOcp8hh1W1eIiamOFPKFRMNfHypSU"
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

        # Set Unity Y-up format to False by default (no user prompt)
        global unityAxisFormat
        unityAxisFormat = False 

        # Extract filename for later use
        file_name_without_ext = os.path.splitext(latest_file)[0]

        # --- Run the conversion ---
        model = convert_glb_to_3dm(glb_data)

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
                        building_floor_footprints.append(poly)
                        roof_meshes_info.append((obj.Attributes.Id, bbox, poly))

                all_mesh_bboxes.append(bbox)
                max_z = max(max_z, bbox.Max.Z)

        if not building_floor_footprints:
            raise RuntimeError("No meaningful building floor footprints found in the model.")

        # Automatic floor calculation
        floor_height = 2.5  # meters per floor
        num_floors = max(1, int(round(max_z / floor_height)))
        wall_thickness = 0.3  # meters (30 cm)

        # Process floor footprints by level
        floor_footprints_by_level = {}
        floor_z_levels = set()

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
                        approx_z = round(bbox.Max.Z / floor_height) * floor_height
                        floor_z_levels.add(approx_z)
                        
                        if approx_z not in floor_footprints_by_level:
                            floor_footprints_by_level[approx_z] = []
                        floor_footprints_by_level[approx_z].append(poly)

        sorted_floor_z_levels = sorted(list(floor_z_levels))

        # Structural logic
        MaxS = 6.0
        MinS = 3.0

        columns_2d_points = []
        beams_2d_lines = []
        added_column_xy = set()

        # Generate structural elements
        for approx_z, floor_polygons in floor_footprints_by_level.items():
            current_floor_z = approx_z

            for room_poly in floor_polygons:
                minx, miny, maxx, maxy = room_poly.bounds
                width, height = maxx - minx, maxy - miny
                
                divisions_x = max(1, int(np.ceil(width / MaxS)))
                divisions_y = max(1, int(np.ceil(height / MaxS)))
                
                x_points_grid = np.linspace(minx, maxx, divisions_x + 1)
                y_points_grid = np.linspace(miny, maxy, divisions_y + 1)
                
                # Add interior grid columns
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

                # Add columns at corners
                for corner_x, corner_y in room_poly.exterior.coords:
                    rounded_corner_x = round(corner_x, 5)
                    rounded_corner_y = round(corner_y, 5)
                    
                    if (rounded_corner_x, rounded_corner_y) not in added_column_xy:
                        if all(np.linalg.norm(np.array((corner_x, corner_y)) - np.array(exist_col_xy)) >= MinS * 0.5 for exist_col_xy in added_column_xy):
                            columns_2d_points.append((rounded_corner_x, rounded_corner_y))
                            added_column_xy.add((rounded_corner_x, rounded_corner_y))

                # Generate beams (skip ground level)
                ground_floor_z = sorted_floor_z_levels[0] if sorted_floor_z_levels else 0.0
                if abs(current_floor_z - ground_floor_z) > 1e-4:
                    # Horizontal beams
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
            ordered_segment = tuple(sorted((tuple(np.round(p1,5)), tuple(np.round(p2,5)))))
            unique_beams_2d.add(ordered_segment)
        beams_2d_lines = [ (np.array(p1), np.array(p2)) for p1,p2 in unique_beams_2d]

        # Generate 3D data
        node_coords = []
        node_dict = OrderedDict()

        def add_node(pt):
            key = tuple(np.round(pt, 5))
            if key not in node_dict:
                node_id = f"N{len(node_dict)}"
                node_dict[key] = node_id
                node_coords.append([node_id] + list(key))
            return node_dict[key]

        # Generate columns
        column_lines = []
        all_base_columns = list(added_column_xy)

        for x, y in all_base_columns:
            current_column_max_z = get_wall_height(x, y, all_mesh_bboxes, max_z)
            
            column_z_levels_for_this_column = [sorted_floor_z_levels[0]] if sorted_floor_z_levels else [0.0]
            
            for z in sorted_floor_z_levels:
                if z > column_z_levels_for_this_column[0] + 1e-6:
                    column_z_levels_for_this_column.append(z)
            
            column_z_levels_for_this_column = [z for z in column_z_levels_for_this_column if z <= current_column_max_z + 1e-6]
            column_z_levels_for_this_column = sorted(list(set(np.round(column_z_levels_for_this_column, 5))))
            
            for i in range(len(column_z_levels_for_this_column) - 1):
                start_z = column_z_levels_for_this_column[i]
                end_z = column_z_levels_for_this_column[i+1]
                
                if end_z > start_z + 1e-6:
                    id_start_node = add_node((x, y, start_z))
                    id_end_node = add_node((x, y, end_z))
                    column_lines.append((id_start_node, id_end_node))

        # Generate beams
        beam_lines = []
        unique_beam_tuples_3d = set()

        for (x1, y1), (x2, y2) in beams_2d_lines:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            beam_local_max_z = get_wall_height(mid_x, mid_y, all_mesh_bboxes, max_z)

            for floor_z_level_actual in sorted_floor_z_levels:
                if floor_z_level_actual > beam_local_max_z + 1e-6:
                    continue

                # Skip beams at ground level
                if abs(floor_z_level_actual) < 1e-4:
                    continue

                floor_z_level_rounded = round(floor_z_level_actual, 5)

                id1 = add_node((x1, y1, floor_z_level_rounded))
                id2 = add_node((x2, y2, floor_z_level_rounded))

                ordered_nodes = tuple(sorted((id1, id2)))
                if ordered_nodes not in unique_beam_tuples_3d:
                    unique_beam_tuples_3d.add(ordered_nodes)
                    beam_lines.append((id1, id2))

        # Create DataFrames
        df_nodes = pd.DataFrame(node_coords, columns=["ID", "X", "Y", "Z"])

        # Prepare data
        nodes_data = []
        for node in node_coords:
            nodes_data.append({
                "ID": node[0],
                "X": node[1],
                "Y": node[2],
                "Z": node[3]
            })

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
            "nodes": nodes_data,
            "columns": columns_data,
            "beams": beams_data
        }

        # Convert to JSON string
        combined_json_data = json.dumps(combined_structural_data, indent=4)

        # Upload to Supabase
        try:
            json_filename = f"{file_name_without_ext}_structural_data.json"
            upload_response = supabase.storage.from_("analysis-results").upload(
                json_filename, 
                combined_json_data.encode('utf-8'),
                {"content-type": "application/json"}
            )
            
            if upload_response:
                public_url = supabase.storage.from_("analysis-results").get_public_url(json_filename)
            
        except Exception as e:
            # Fallback: save locally
            local_json_path = os.path.join(os.getcwd(), json_filename)
            with open(local_json_path, 'w') as f:
                f.write(combined_json_data)
        
        return jsonify({
            "status": "success",
            "message": "Processing complete",
            "filename": json_filename,
            "nodes_count": len(nodes_data),
            "columns_count": len(columns_data),
            "beams_count": len(beams_data)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
