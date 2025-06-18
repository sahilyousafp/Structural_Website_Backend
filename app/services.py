import rhino3dm
import numpy as np
from shapely.geometry import Polygon
from .utils import mesh_brep

class RhinoLoader:
    """Loads Rhino model and extracts building and column geometry."""
    def __init__(self, path):
        self.model = rhino3dm.File3dm.Read(path)
        self.layers = {lay.Name.lower(): lay.Index for lay in self.model.Layers}
        self._validate_layers()
        self.building_volumes, self.wall_breps, self.imported_columns, self.max_z = self._extract_geometry()

    def _validate_layers(self):
        if 'building' not in self.layers or ('column' not in self.layers and 'columns' not in self.layers):
            raise RuntimeError("Missing required layers: 'building' and 'column(s)'.")

    def _extract_geometry(self):
        vols, breps, cols = [], [], []
        max_z = 0.0
        col_layer = 'columns' if 'columns' in self.layers else 'column'
        for obj in self.model.Objects:
            idx = obj.Attributes.LayerIndex
            geom = obj.Geometry
            if idx == self.layers['building'] and geom.ObjectType == rhino3dm.ObjectType.Brep:
                bbox = geom.GetBoundingBox()
                pts2d = [[bbox.Min.X, bbox.Min.Y], [bbox.Max.X, bbox.Min.Y],
                         [bbox.Max.X, bbox.Max.Y], [bbox.Min.X, bbox.Max.Y]]
                poly = Polygon(pts2d)
                vols.append(poly)
                breps.append({'polygon': poly, 'bbox': bbox})
                max_z = max(max_z, bbox.Max.Z)
            elif idx == self.layers[col_layer] and geom.ObjectType == rhino3dm.ObjectType.Brep:
                bb = geom.GetBoundingBox()
                cx, cy = (bb.Min.X + bb.Max.X) / 2, (bb.Min.Y + bb.Max.Y) / 2
                cols.append((cx, cy))
        if not vols:
            raise RuntimeError("No building geometry found.")
        return vols, breps, cols, max_z

class GridGenerator:
    """Generates column and beam placements based on building volumes."""
    def __init__(self, volumes, imported_cols, wall_breps, max_z, num_floors, MaxS=6.0, MinS=3.0):
        self.volumes = volumes
        self.imported = list(imported_cols)
        self.wall_breps = wall_breps
        self.max_z = max_z
        self.num_floors = num_floors
        self.MaxS = MaxS
        self.MinS = MinS
        self.detected_rooms = sorted([(p, p.area) for p in volumes], key=lambda x: -x[1])
        self.columns, self.corrected, self.beams = [], [], []
        self._generate()
        self.all_base = self.columns + self.corrected

    def _generate(self):
        for poly, _ in self.detected_rooms:
            minx, miny, maxx, maxy = poly.bounds
            divx = int(np.ceil((maxx - minx) / self.MaxS))
            divy = int(np.ceil((maxy - miny) / self.MaxS))
            xs = np.linspace(minx, maxx, divx + 1)
            ys = np.linspace(miny, maxy, divy + 1)
            pts = [(x, y) for x in xs for y in ys]
            self._snap_and_add(pts)
            for x in xs:
                self.beams.append(((x, miny), (x, maxy)))
            for y in ys:
                self.beams.append(((minx, y), (maxx, y)))
            for x, y in poly.exterior.coords:
                if all(np.linalg.norm(np.array((x, y)) - np.array(c)) >= self.MinS * 0.5 for c in self.all_base):
                    self.columns.append((x, y))
                    self.all_base.append((x, y))

    def _snap_and_add(self, pts):
        for p in pts:
            snap = False
            for imp in self.imported:
                if np.linalg.norm(np.array(p) - np.array(imp)) < self.MinS:
                    self.corrected.append(p)
                    self.all_base.append(p)
                    self.imported.remove(imp)
                    snap = True
                    break
            if not snap and all(np.linalg.norm(np.array(p) - np.array(c)) >= self.MinS for c in self.all_base):
                self.columns.append(p)
                self.all_base.append(p)


def generate_structure(rhino_path, num_floors):
    loader = RhinoLoader(rhino_path)
    grid = GridGenerator(loader.building_volumes, loader.imported_columns, loader.wall_breps, loader.max_z, num_floors = 2)
    return {
        "columns": [[float(x), float(y)] for x, y in grid.all_base],
        "beams": [
            {"start": [float(coord) for coord in s], "end": [float(coord) for coord in e]}
            for s, e in grid.beams
        ]
    }
