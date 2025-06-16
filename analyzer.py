import os
import tempfile
from supabase import create_client
import rhino3dm
import numpy as np
from shapely.geometry import Polygon, Point

# Initialize Supabase client
def _init_supabase():
    url = os.getenv('SUPABASE_URL', 'https://apdbfbjnlsxjfubqahtl.supabase.co')
    key = os.getenv('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFwZGJmYmpubHN4amZ1YnFhaHRsIiwicm9zZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NzQ3NjUzOCwiZXhwIjoyMDYzMDUyNTM4fQ.cylQZjLEmtBi507wrJ1KUDyIXTz5H5VAXGj5eKEqDy4')
    if not url or not key:
        raise RuntimeError('SUPABASE_URL and SUPABASE_KEY must be set')
    return create_client(url, key)

supabase = _init_supabase()
BUCKET = os.getenv('SUPABASE_BUCKET', 'models')


def analyze_structure(file_key: str, num_floors: int) -> dict:
    """
    Downloads a .3dm file, analyzes structure, and returns data.
    """
    # Download model
    table = supabase.storage.from_(BUCKET)
    result = table.download(file_key)
    if result.error:
        raise RuntimeError(f"Error downloading {file_key}: {result.error.message}")
    data = result.data

    # Write to temp .3dm
    tmp_3dm = tempfile.NamedTemporaryFile(delete=False, suffix='.3dm')
    tmp_3dm.write(data)
    tmp_3dm.flush()
    tmp_3dm.close()

    # Read model
    model = rhino3dm.File3dm.Read(tmp_3dm.name)
    layers = {layer.Name.lower(): layer.Index for layer in model.Layers}
    if 'building' not in layers or ('column' not in layers and 'columns' not in layers):
        raise RuntimeError("Missing required layers: 'building' and 'column' or 'columns'.")
    column_layer = 'columns' if 'columns' in layers else 'column'

    # Extract building volumes and columns
    building_volumes = []
    imported_columns = []
    max_z = 0.0
    wall_breps = []

    for obj in model.Objects:
        layer_idx = obj.Attributes.LayerIndex
        geom = obj.Geometry
        if layer_idx == layers['building'] and isinstance(geom, rhino3dm.Brep):
            bbox = geom.GetBoundingBox()
            base_pts = [[bbox.Min.X, bbox.Min.Y], [bbox.Max.X, bbox.Min.Y],
                        [bbox.Max.X, bbox.Max.Y], [bbox.Min.X, bbox.Max.Y]]
            poly = Polygon(base_pts)
            building_volumes.append(poly)
            wall_breps.append({'polygon': poly, 'bbox': bbox})
            max_z = max(max_z, bbox.Max.Z)
        elif layer_idx == layers[column_layer] and isinstance(geom, rhino3dm.Brep):
            bbox = geom.GetBoundingBox()
            cx = (bbox.Min.X + bbox.Max.X) / 2
            cy = (bbox.Min.Y + bbox.Max.Y) / 2
            imported_columns.append([cx, cy])

    if not building_volumes:
        raise RuntimeError("No valid building geometry found on the 'building' layer.")

    # Prepare rooms
    detected_rooms = sorted([(poly, poly.area) for poly in building_volumes], key=lambda x: -x[1])
    MaxS, MinS = 6.0, 3.0
    columns, corrected, existing = [], [], imported_columns.copy()
    beams = []

    for poly, _ in detected_rooms:
        minx, miny, maxx, maxy = poly.bounds
        dx = int(np.ceil((maxx-minx)/MaxS))
        dy = int(np.ceil((maxy-miny)/MaxS))
        xs = np.linspace(minx, maxx, dx+1)
        ys = np.linspace(miny, maxy, dy+1)
        candidates = [[x, y] for x in xs for y in ys]
        for c in candidates:
            if any(np.linalg.norm(np.array(c)-np.array(ic))<MinS for ic in existing):
                continue
            snap = False
            for ic in imported_columns:
                if np.linalg.norm(np.array(c)-np.array(ic))<MinS:
                    corrected.append(c); existing.append(c); snap=True; break
            if not snap:
                columns.append(c); existing.append(c)
        for x in xs:
            beams.append([[x, miny], [x, maxy]])
        for y in ys:
            beams.append([[minx, y], [maxx, y]])
        # corners
        for x,y in poly.exterior.coords:
            if all(np.linalg.norm(np.array([x,y])-np.array(ec))>=MinS/2 for ec in existing):
                columns.append([x,y]); existing.append([x,y])

    all_columns = columns + corrected

    return {
        'imported_columns': imported_columns,
        'generated_columns': columns,
        'snapped_columns': corrected,
        'beams': beams,
        'num_floors': num_floors
    }
