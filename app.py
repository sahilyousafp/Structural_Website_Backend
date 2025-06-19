from flask import Flask, jsonify, request
import json
import os
from io import BytesIO
import trimesh
import numpy as np
from shapely.geometry import Polygon, Point
from supabase import create_client

app = Flask(__name__)

# Supabase configuration
SUPABASE_URL = 'https://apdbfbjnlsxjfubqahtl.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFwZGJmYmpubHN4amZ1YnFhaHRsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc0NzY1MzgsImV4cCI6MjA2MzA1MjUzOH0.DPINyYHHUzcuQ6AOcp8hh1W1eIiamOFPKFRMNfHypSU'
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# Structural computation functions
def extract_wall_breps(meshes):
    wall_data = []
    building_polys = []
    max_z = 0.0
    for mesh in meshes:
        bounds = mesh.bounds  # [[minx,miny,minz],[maxx,maxy,maxz]]
        minx, miny, _ = bounds[0]
        maxx, maxy, maxz = bounds[1]
        poly = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
        wall_data.append({'polygon': poly, 'max_z': maxz})
        building_polys.append(poly)
        max_z = max(max_z, maxz)
    return building_polys, wall_data, max_z

def get_wall_height(x, y, wall_data, global_max_z):
    pt = Point(x, y)
    closest = None
    dist0 = float('inf')
    for w in wall_data:
        d = w['polygon'].exterior.distance(pt)
        if d < dist0:
            dist0 = d
            closest = w
    return closest['max_z'] if closest else global_max_z

def compute_structure(meshes, num_floors):
    building_polys, wall_data, global_max_z = extract_wall_breps(meshes)
    MaxS, MinS = 6.0, 3.0
    generated = []
    beams_2d = []
    # sort rooms by footprint area
    rooms = sorted([(poly, poly.area) for poly in building_polys], key=lambda x: -x[1])
    for poly, _ in rooms:
        minx, miny, maxx, maxy = poly.bounds
        dx, dy = maxx - minx, maxy - miny
        nx = int(np.ceil(dx / MaxS))
        ny = int(np.ceil(dy / MaxS))
        xs = np.linspace(minx, maxx, nx + 1)
        ys = np.linspace(miny, maxy, ny + 1)
        # grid points
        for x in xs:
            for y in ys:
                if all(np.linalg.norm([x - gx, y - gy]) >= MinS for gx, gy in generated):
                    generated.append((x, y))
        # beams
        for x in xs:
            beams_2d.append(((x, miny), (x, maxy)))
        for y in ys:
            beams_2d.append(((minx, y), (maxx, y)))
        # corners
        for c in poly.exterior.coords:
            if all(np.linalg.norm([c[0] - gx, c[1] - gy]) >= MinS * 0.5 for gx, gy in generated):
                generated.append(c)
    # assemble JSON
    result = {'num_floors': num_floors, 'columns': [], 'beams': []}
    for x, y in generated:
        h = get_wall_height(x, y, wall_data, global_max_z)
        result['columns'].append({'base': [x, y, 0.0], 'top': [x, y, h]})
    for (x1, y1), (x2, y2) in beams_2d:
        h1 = get_wall_height(x1, y1, wall_data, global_max_z)
        h2 = get_wall_height(x2, y2, wall_data, global_max_z)
        floor_h = min(h1, h2)
        for floor in range(1, num_floors + 1):
            z = floor_h / num_floors * floor
            if z > h1 or z > h2:
                continue
            result['beams'].append({'start': [x1, y1, z], 'end': [x2, y2, z], 'floor': floor})
    return result

# Flask endpoint: load latest model, compute structure, return JSON
@app.route('/')
def root():
    num_floors = int(request.args.get('floors', 1))
    try:
        # list and download latest file in storage folder
        files = sb.storage.from_('models').list('79edaed4-a719-4390-a485-519b68fa68ea/')
        if not files:
            raise RuntimeError('No files in Supabase storage')
        latest_name = files[-1]['name']
        full_path = f"79edaed4-a719-4390-a485-519b68fa68ea/{latest_name}"
        data_bytes = sb.storage.from_('models').download(full_path)
        # load mesh (OBJ or GLB)
        ext = latest_name.split('.')[-1].lower()
        scene = trimesh.load(BytesIO(data_bytes), file_type=ext, force='scene')
        meshes = list(scene.geometry.values()) if scene.geometry else []
        # compute structure JSON
        result = compute_structure(meshes, num_floors)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)
